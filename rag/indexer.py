from pathlib import Path
from rag.db import get_total_chunks
import faiss, math
import numpy as np


INDEX_DIR = Path("./data/index")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

FLAT_INDEX_PATH = INDEX_DIR / "faiss_flat.index"

IVFPQ_INDEX_PATH = INDEX_DIR / "faiss_ivfpq.index"
MIN_TRAIN_SIZE = 5000
TRAIN_SIZE_CAP = 100000
BACKFILL_BATCH_SIZE = 50000

# Flat Index
def _load_or_create_flat_index(dim):
    if FLAT_INDEX_PATH.exists():
        return faiss.read_index(str(FLAT_INDEX_PATH))
    
    index = faiss.IndexFlatIP(dim)
    return faiss.IndexIDMap2(index)

def _get_all_ids_and_vectors_from_flat_index(flat_index):
    idmap = faiss.downcast_index(flat_index)
    core = faiss.downcast_index(idmap.index)
    ids = faiss.vector_to_array(idmap.id_map).astype("int64")

    try:
        vecs = core.reconstruct_n(0, core.ntotal)
        vecs = np.ascontiguousarray(vecs, dtype="float32")
        return ids, vecs
    except Exception:
        pass
    return None

def add_to_flat_index(vecs, ids, dim):
    index = _load_or_create_flat_index(dim)
    index.add_with_ids(vecs.astype("float32"), ids.astype("int64"))
    faiss.write_index(index, str(FLAT_INDEX_PATH))


# IVFPQ Index
def _get_m(dim): 
    # Choose m that divides the dim  
    for m in range(64, 7, -1):
        if dim % m == 0:
            return m
    
    return 64

def _get_nlist(total_vectors):
    # Choose sqrt of total_vectors
    if total_vectors <= 0:
        return 1024
    return min(65536, int(math.sqrt(total_vectors)))

def _build_ivfpq_index(dim, nlist, m, nbits=8, nprobe = 16):
    # Using compression and non-exhaustive search strategy for scalability
    # Note: for this simple implementation this is probably overkill, 
    # but in case tons of pdfs are uploaded, this strategy is used
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits)

    if hasattr(index, "metric_type"):
        index.metric_type = faiss.METRIC_INNER_PRODUCT
    index.nprobe = nprobe
    return index


def _load_or_create_ivfpq(dim):
    if IVFPQ_INDEX_PATH.exists():
        # This does not handle model change right now in a case where the new model embeddings 
        # are of different dimension. Need to re-index everything if that happens but keeping it
        # simple for now
        return faiss.read_index(str(IVFPQ_INDEX_PATH))

    nlist = _get_nlist(get_total_chunks() or TRAIN_SIZE_CAP)  # number of clusters 
    m = _get_m(dim)  

    return _build_ivfpq_index(dim, nlist, m)

def _sample_training_vectors(vecs):
    if vecs.shape[0] < TRAIN_SIZE_CAP:
        return vecs
    return vecs[np.random.choice(vecs.shape[0], TRAIN_SIZE_CAP, replace=False)]

def _train_ivfpq_index_from_flat(flat_index, dim):
    idmap = faiss.downcast_index(flat_index)
    core = faiss.downcast_index(idmap.index)
    if core.ntotal == 0:
        return None
    index = _load_or_create_ivfpq(dim)
    _, vecs = _get_all_ids_and_vectors_from_flat_index(flat_index)
    train = _sample_training_vectors(vecs)
    print("This is the nlist", {index.nlist})
    index.train(train)
    faiss.write_index(index, str(IVFPQ_INDEX_PATH))
    return index

def _backfill_ivfpq_index(flat_index, ivfpq_index):
    ids, vecs = _get_all_ids_and_vectors_from_flat_index(flat_index)
    for i in range(0, vecs.shape[0], BACKFILL_BATCH_SIZE):
        ivfpq_index.add_with_ids(vecs[i:i+BACKFILL_BATCH_SIZE], ids[i:i+BACKFILL_BATCH_SIZE])
    faiss.write_index(ivfpq_index, str(IVFPQ_INDEX_PATH))
    return ivfpq_index

def _try_load_trained_ivfpq_index(dim: int):
    if not IVFPQ_INDEX_PATH.exists():
        return None
    index = faiss.read_index(str(IVFPQ_INDEX_PATH))
    if getattr(index, "is_trained", False) and index.d == dim:
        return index

    try: 
        IVFPQ_INDEX_PATH.unlink(missing_ok=True)
    except Exception: 
        pass
    return None

def add_to_ivfpq_index(dim, ids, vecs):
    index = _try_load_trained_ivfpq_index(dim)
    if index is not None:
        if ids is not None and vecs is not None and len(ids) > 0:
            index.add_with_ids(vecs.astype("float32"), ids.astype("int64"))
            faiss.write_index(index, str(IVFPQ_INDEX_PATH))
        return index
    
    if not FLAT_INDEX_PATH.exists():
        return None
    
    flat_index = faiss.read_index(str(FLAT_INDEX_PATH))
    idmap = faiss.downcast_index(flat_index)
    core = faiss.downcast_index(idmap.index)

    if core.ntotal == 0 or core.ntotal < MIN_TRAIN_SIZE:
        return None
    
    index = _train_ivfpq_index_from_flat(flat_index, dim)
    if index is None:
        return None
    
    index = _backfill_ivfpq_index(flat_index, index)
    return index
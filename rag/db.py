import sqlite3, os
from enum import Enum

class DocumentStatus(Enum):
    UPLOADED = "UPLOADED"
    PROCESSING = "PROCESSING"
    INDEXED = "INDEXED"
    FAILED = "FAILED"

class JobStatus(Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    FAILED = "FAILED"
    DONE = "DONE"

# hard coded for now for simple control but could be taken in using env
DB_PATH = "data/db.sqlite3"

# could live in a separate file, but left here for now
SCHEMA = """
-- Table to store the pdf documents
CREATE TABLE IF NOT EXISTS documents(
          id TEXT PRIMARY KEY,
          original_name TEXT NOT NULL,
          storage_path TEXT NOT NULL,
          sha256 TEXT NOT NULL,
          bytes INTEGER NOT NULL,
          pages INTEGER,
          status TEXT NOT NULL,        -- UPLOADED|PROCESSING|INDEXED|FAILED
          error_msg TEXT,
          created_at TEXT DEFAULT CURRENT_TIMESTAMP,
          updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        -- To ensure that if the same pdf is uploaded, it does not have to go through the heavy work again
        CREATE UNIQUE INDEX IF NOT EXISTS ux_doc_sha ON documents(sha256);

        -- Table to store index jobs to queue documents to index
        CREATE TABLE IF NOT EXISTS jobs(
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          document_id TEXT NOT NULL,
          document_sha TEXT NOT NULL,
          type TEXT NOT NULL,          -- INDEX_DOCUMENT is the only purpose for now, but this could be used for other jobs
          status TEXT NOT NULL,        -- QUEUED|RUNNING|DONE|FAILED
          created_at TEXT DEFAULT CURRENT_TIMESTAMP,
          updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
          error_msg TEXT
        );

        CREATE UNIQUE INDEX IF NOT EXISTS ux_jobs_sha ON jobs(document_sha);

        -- Table to store the meta data for all the chunks and link it to embeddings
        CREATE TABLE IF NOT EXISTS chunk_meta(
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          document_id TEXT NOT NULL,
          ordinal INT,
          page_num INT,
          start_char INT,
          end_char INT,
          embed_model TEXT
        );

        -- Table to store the actual text chunks so that we can also performm keyword search
        -- Uses SQLite's built-in full-text index and English Porter stemmer to find keywords
        CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
          text, tokenize='porter'
        );
"""
os.makedirs("./data", exist_ok=True)

def _connect():
    con = sqlite3.connect(DB_PATH, isolation_level=None)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con

def init_schema():
    with _connect() as con:
        con.executescript(SCHEMA)
    return True

def create_document(*, id, original_name, storage_path, sha256, bytes, pages):
    with _connect() as con:
        con.execute(
            """INSERT OR IGNORE INTO documents
            (id, original_name, storage_path, sha256, bytes, pages, status)
            VALUES(?,?,?,?,?,?,?)""",
            (id, original_name, storage_path, sha256, bytes, pages, DocumentStatus.UPLOADED.value))
        
def get_document(document_id):
    with _connect() as con:
        return con.execute("SELECT * FROM documents WHERE id=?", (document_id, )).fetchone()
        
def enqueue_index_job(document_id, document_sha):
    with _connect() as con:
        con.execute(
            """INSERT INTO jobs
            (document_id, document_sha, status, type)
            VALUES(?,?,?, 'INDEX_DOCUMENT')""",
            (document_id, document_sha, JobStatus.QUEUED.value, )
        )

def update_document_status(document_id, status, pages=None, error=None):
    with _connect() as con:
        con.execute(
            """UPDATE documents SET status=?, pages=COALESCE(?, pages), 
            error_msg=COALESCE(?, error_msg), updated_at=CURRENT_TIMESTAMP
            WHERE id=?""",
            (status, pages, error, document_id)
        )

def get_job():
    with _connect() as con:
        row = con.execute(
            """SELECT id FROM jobs WHERE status=?
                ORDER BY created_at LIMIT 1""",
                (JobStatus.QUEUED.value,)
        ).fetchone()

        if not row:
            return None
        
        job_id = row["id"]
        current = con.execute(
            """UPDATE jobs
                SET status=?, updated_at=CURRENT_TIMESTAMP
                WHERE id=? AND status=?""",
            (JobStatus.RUNNING.value, job_id, JobStatus.QUEUED.value)
        )

        if current.rowcount == 0:
            return None
        
        return con.execute("SELECT * FROM jobs WHERE id=?", (job_id, )).fetchone()
    
def mark_job_done(job_id):
    with _connect() as con:
        con.execute(
            """UPDATE jobs SET status=?, 
                updated_at=CURRENT_TIMESTAMP WHERE id=?""",
            (JobStatus.DONE.value, job_id)
        )

def mark_job_failed(job_id, error_msg):
    with _connect() as con:
        con.execute(
            """UPDATE jobs SET status=?, 
                error_msg=?, updated_at=CURRENT_TIMESTAMP WHERE id=?""",
            (JobStatus.FAILED.value, error_msg[:2000], job_id)
        )

def insert_chunks(doc_id, chunks):
    ids = []
    with _connect() as con:
        # Ensure that the insertion is atomic
        con.execute("BEGIN")
        try:
            for c in chunks:
                # Meta Data
                row_id = con.execute(
                    """INSERT INTO chunk_meta
                    (document_id, ordinal, page_num, start_char, end_char, embed_model)
                    VALUES(?,?,?,?,?,?) RETURNING id""",
                    (doc_id, c["ordinal"], c["page_num"], c["start"], c["end"], c["embed_model"])
                ).fetchone()[0]
                ids.append(row_id)

                # Actual text
                con.execute("INSERT INTO chunk_fts(rowid, text) VALUES(?,?)", (row_id, c["text"]))
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
    
    return ids


def get_total_chunks():
    with _connect() as con:
        return con.execute("SELECT COUNT(*) FROM chunk_meta").fetchone()[0]
    
def match_fts_query(fts_query, top_k):
    with _connect() as con:
        cur = con.execute(
            "SELECT rowid, bm25(chunk_fts) AS s "
            "FROM chunk_fts WHERE chunk_fts MATCH ? "
            "ORDER BY s LIMIT ?",
            (fts_query, int(top_k))
        )
        res = [(int(r[0]), float(r[1])) for r in cur.fetchall()] 
        return res
    
def get_chunk_meta(ids):
    res = []
    with _connect() as con:
        cur = con.execute(
            f""" SELECT m.id, m.document_id, m.page_num, m.start_char, m.end_char,
            d.original_name, t.text
            FROM chunk_meta m
            JOIN documents d ON d.id = m.document_id
            JOIN chunk_fts t  ON t.rowid = m.id
            WHERE m.id IN ({ids})"""
        )
        for r in cur:
            res.append({
                "chunk_id": r[0],
                "document_id": r[1],
                "page_num": r[2],
                "start_char": r[3],
                "end_char": r[4],
                "document_name": r[5],
                "text": r[6],
            })
        return res
.PHONY: help install dev api worker run killport reset-db

# venv paths
VENV := .venv
PY := $(VENV)/bin/python

# config
PORT ?= 8000
STREAMLIT_PORT ?= 8501
STREAMLIT_APP ?= $(PWD)/ui/app.py

help:
	@echo "make install   - create venv and install deps"
	@echo "make dev       - run API (reload) + worker together"
	@echo "make api       - run only API"
	@echo "make worker    - run only worker"
	@echo "make run       - run both (no reload)"
	@echo "make reset-db  - delete SQLite DB"

install:
	$(PY) -m venv $(VENV)
	$(PY) -m pip install -U pip
	$(PY) -m pip install -r requirements.txt
	mkdir -p data/uploads data/index         # This is done in the application as well so this is not entirely necessary

# This is just done for dev purpose as sometimes the port is still held for listening by prev runs
# Note: If any important service runs in your device in this port, it will be killed. Hence, only meant to be
# ran in dev
killport:
	@PID=$$(lsof -tiTCP:$(PORT) -sTCP:LISTEN 2>/dev/null); \
	if [ -n "$$PID" ]; then echo "Killing PID $$PID on port $(PORT)"; kill $$PID || true; fi


# This kills the job that is listening to the port necessary for the api, and runs both api and worker
dev: killport
	set -a; [ -f .env ] && . ./.env; set +a; \
	export DEBUG=True; \
	$(PY) -m uvicorn app.main:app --reload --port $(PORT) & \
	$(PY) -u worker.py & \
	$(PY) -m streamlit run $(STREAMLIT_APP) --server.port $(STREAMLIT_PORT) --server.headless true & \
	wait

run: 
	set -a; [ -f .env ] && . ./.env; set +a; \
	export DEBUG=False; \
	$(PY) -m uvicorn app.main:app --port $(PORT) & \
	$(PY) -u worker.py & \
	$(PY) -m streamlit run $(STREAMLIT_APP) --server.port $(STREAMLIT_PORT) --server.headless true & \
	wait

# In case you want to remove the existing db for testing or otherwise
reset-db:
	rm -f data/db.sqlite3

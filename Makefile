SHELL := /bin/bash
.ONESHELL:
.PHONY: help install dev api worker run killport reset-db

# venv paths
VENV := .venv
PY_SYS := $(shell command -v python3 2>/dev/null || command -v python)
PY := $(VENV)/bin/python
PIP := $(PY) -m pip

# config
PORT ?= 8000
export RAG_DB_PATH := $(PWD)/data/db.sqlite3

help:
	@echo "make install   - create venv and install deps"
	@echo "make dev       - run API (reload) + worker together"
	@echo "make api       - run only API"
	@echo "make worker    - run only worker"
	@echo "make run       - run both (no reload)"
	@echo "make reset-db  - delete SQLite DB"

install:
	$(PY_SYS) -m venv $(VENV)
	$(PY) -m pip install -U pip
	$(PIP) install -r requirements.txt
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
	$(PY) -m uvicorn app.main:app --reload --port $(PORT) & \
	$(PY) -u worker.py & \
	wait

# If you want to just run the api
api:
	set -a; [ -f .env ] && . ./.env; set +a; \
	$(PY) -m uvicorn app.main:app --reload --port $(PORT)

# If you want to just run the worker
worker:
	set -a; [ -f .env ] && . ./.env; set +a; \
	$(PY) -u worker.py

run: 
	set -a; [ -f .env ] && . ./.env; set +a; \
	$(PY) -m uvicorn app.main:app --port $(PORT) & \
	$(PY) -u worker.py & \
	wait

# In case you want to remove the existing db for testing or otherwise
reset-db:
	rm -f data/db.sqlite3

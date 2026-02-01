SHELL := /bin/bash
ENV_FILE := .env
COMPOSE := docker compose

include $(ENV_FILE)
export $(shell sed 's/=.*//' $(ENV_FILE))

up:
	$(COMPOSE) up -d --build

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f rag-api

restart:
	$(COMPOSE) restart rag-api

ingest:
	@if [ -z "$(path)" ]; then echo "Usage: make ingest path=./docs"; exit 1; fi
	curl -fsS -H "Authorization: Bearer $(API_KEY)"         	     -H "Content-Type: application/json"         	     -d "{"path": "$$PWD/$(path)"}"         	     http://localhost:8000/ingest

query:
	@if [ -z "$(q)" ]; then echo "Usage: make query q='your question'"; exit 1; fi
	curl -fsS -H "Authorization: Bearer $(API_KEY)"         	     -H "Content-Type: application/json"         	     -d "{"question": "$(q)"}"         	     http://localhost:8000/query | jq

stream:
	@if [ -z "$(q)" ]; then echo "Usage: make stream q='your question'"; exit 1; fi
	curl -N -H "Authorization: Bearer $(API_KEY)"         	     -H "Content-Type: application/json"         	     -d "{"question": "$(q)"}"         	     http://localhost:8000/stream

test:
	$(COMPOSE) exec rag-api pytest -q

pull-model:
	# (Online) Pre-pull default LLM on Ollama
	curl -fsS -X POST http://localhost:11434/api/pull -d "{"name":"$(OLLAMA_MODEL)"}"

help:
	@echo "Targets:"
	@echo "  up, down, logs, restart"
	@echo "  ingest path=./docs"
	@echo "  query q='...' | stream q='...'"
	@echo "  test | pull-model | help"
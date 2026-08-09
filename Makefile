# Makefile
# ─────────────────────────────────────────────────────────────────────────────
# Convenience wrapper around docker compose.
#
#   make up      → build (if needed) + start the whole system in the background
#   make down    → stop and remove the container
#   make logs    → tail logs
#   make restart → down + up
#   make shell   → open a shell inside the running container
#   make build   → rebuild the image without starting it
#   make clean   → stop everything and remove the built image too
#
# Portability note: recipes deliberately avoid POSIX-shell-only syntax
# (`if [ ... ]; then ... fi`, backslash line continuations, etc.) because
# on Windows, GNU Make runs recipes through cmd.exe by default, not
# bash/sh — cmd.exe doesn't understand that syntax at all. Every recipe
# below is either a single external command (works identically in
# cmd.exe/PowerShell/bash) or delegates any real logic to a small Python
# script (scripts/check_env.py), since Python already runs identically
# everywhere and is a hard requirement of this project anyway.
#
# The app's port is a single source of truth: whatever API_PORT is set
# to in your .env (default 8000) — see docker-compose.yml, which reads
# the same .env for its port mapping. Change the port by editing .env
# only, not this file or docker-compose.yml.
# ─────────────────────────────────────────────────────────────────────────────

COMPOSE := docker compose

# Read .env (if present) as Make variables too, purely so `make up` can
# print the right URL back to you. `-include` silently does nothing if
# .env doesn't exist yet (first run, before check-env creates it).
-include .env
API_PORT ?= 8000

.PHONY: up down build logs restart shell clean check-env

check-env:
	@python scripts/check_env.py

up: check-env
	$(COMPOSE) up --build -d
	@echo Running at http://localhost:$(API_PORT)  (Web UI, Sandbox, Library tabs)
	@echo API docs at http://localhost:$(API_PORT)/docs
	@echo Logs: make logs

down:
	$(COMPOSE) down

build:
	$(COMPOSE) build

logs:
	$(COMPOSE) logs -f app

restart: down up

shell:
	$(COMPOSE) exec app /bin/bash

clean:
	$(COMPOSE) down --rmi local -v

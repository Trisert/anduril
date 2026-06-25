# Anduril developer workflow.
#
# The targets mirror what CI runs (see .github/workflows/ci.yml) so
# a developer can reproduce the green/red signal locally before
# pushing. Keep them in sync when CI changes.

# Use bash for the shell (better error handling than POSIX sh).
SHELL := /usr/bin/env bash

# uv is the default package installer / venv manager.
UV    := uv
PY    := $(UV) run python
RUFF  := $(UV) run ruff

.PHONY: help install lint format test test-web ci clean clean-all

# Default target: print the help.
help:
	@echo "anduril — developer targets:"
	@echo "  make install   Create venv + install with dev + web extras (uv)"
	@echo "  make lint      Run ruff check + ruff format --check"
	@echo "  make format    Auto-format with ruff format"
	@echo "  make test      Run the test suite (pytest)"
	@echo "  make test-web  Run the test suite + the web-skill import smoke"
	@echo "  make ci        Run everything CI runs (lint + test + build)"
	@echo "  make clean     Remove build artefacts (build/, dist/, *.egg-info/)"
	@echo "  make clean-all Remove .venv and all build artefacts"

# Install the package with all extras. Idempotent: uv skips
# already-satisfied deps.
install:
	$(UV) venv
	$(UV) pip install -e ".[dev,web]"

# Lint: ruff check (rules) + ruff format --check (style). CI runs
# both, so we run both here. Use `make format` to auto-fix style.
lint:
	$(RUFF) check anduril test_anduril.py
	$(RUFF) format --check anduril test_anduril.py

# Auto-format. This is the only target that modifies files outside
# of build outputs.
format:
	$(RUFF) format anduril test_anduril.py

# Run the test suite. Verbose-ish by default; pass PYTEST_ARGS=... to
# override (e.g. `make test PYTEST_ARGS=-k fuzzy`).
test:
	$(UV) run pytest -ra --strict-markers $(PYTEST_ARGS)

# Same as `test` but also imports the web skill to catch a missing-
# dependency regression early. CI runs this as part of the matrix
# because the `.[web]` extra is installed there.
test-web:
	$(UV) run python -c "import anduril.contrib.skills.web" && \
		$(UV) run pytest -ra --strict-markers $(PYTEST_ARGS)

# The full CI loop. Used locally as a final pre-push check; CI does
# the same three jobs (lint, test, build).
ci: lint test
	@echo "--- build ---"
	$(UV) pip install --quiet build twine
	$(UV) run python -m build
	$(UV) run python -m twine check dist/*

# Remove build artefacts. Keeps .venv intact.
clean:
	rm -rf build/ dist/ *.egg-info anduril.egg-info/ \
	       anduril/__pycache__/ anduril/*/__pycache__/ \
	       .pytest_cache/ .ruff_cache/
	find . -name '*.pyc' -delete

# Nuke the venv too. After this, `make install` is the recovery.
clean-all: clean
	rm -rf .venv

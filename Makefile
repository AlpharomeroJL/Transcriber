# Development tasks for transcriber. Tools are addressed through .venv/bin so
# make works without activating the virtualenv.

VENV := .venv/bin

.PHONY: install lint type test cov check build gauntlet

install:
	$(VENV)/pip install -e ".[dev]"

lint:
	$(VENV)/ruff check .
	$(VENV)/ruff format --check .

type:
	$(VENV)/mypy src

test:
	$(VENV)/pytest

cov:
	$(VENV)/pytest --cov=transcriber --cov-report=term-missing --cov-fail-under=85

check: lint type test

build:
	$(VENV)/python -m build

gauntlet: check cov build
	@if [ -x scripts/smoke_install.sh ]; then \
		scripts/smoke_install.sh; \
	else \
		echo "scripts/smoke_install.sh not present; skipping install smoke"; \
	fi

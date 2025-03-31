.PHONY = \
    build-dist \
    default \
    check-all \
    check-lint \
    check-lock \
    clean \
    fix-all \
    fix-format \
    fix-lint \
    fix-lint-unsafe \
    help \
    install \
    install-pre-commit \
    lock \
    run-tests \
    sync

default: check-all run-tests

build-dist: sync
	uv build --verbose --sdist

check-all: check-lock check-lint

check-lint: check-lock
	uv run ruff check

check-lock:
	uv lock --locked

clean:
	rm -rf .ruff_cache .venv build .cache *.egg-info dist

fix-all: fix-format fix-lint lock

fix-format: check-lock
	uv run ruff format

fix-lint: check-lock
	uv run ruff check --fix

fix-lint-unsafe: check-lock
	uv run ruff check --fix --unsafe-fixes

help:
	@echo ${.PHONY}

install: check-lock build-dist
	uv pip install .

install-pre-commit:
	uv run pre-commit install

lock:
	uv lock

run-tests: check-lock
	uv run pytest .

sync: check-lock
	uv sync --no-install-workspace

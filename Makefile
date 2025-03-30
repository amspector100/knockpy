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
    install-pre-commit \
    lock \
    run-tests \
    sync

default: check-all test

build-dist:
	uv build

check-all: check-lint check-lock

check-lint:
	uv run ruff check

check-lock:
	uv lock --locked

clean:
	rm -rf .ruff_cache .venv build .cache *.egg-info

fix-all: fix-format fix-lint lock

fix-format:
	uv run ruff format

fix-lint:
	uv run ruff check --fix

fix-lint-unsafe:
	uv run ruff check --fix --unsafe-fixes

help:
	@echo ${.PHONY}

install-pre-commit:
	uv run pre-commit install

lock:
	uv lock

run-tests:
	uv run pytest .

sync:
	uv sync --no-install-workspace


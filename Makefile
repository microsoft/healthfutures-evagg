# Typical usages of make for this project are listed below.
# 	make				: see `make ci`
#	make ci				: run most checks required for completing pull requests
#	make test			: run all tests and generate coverage report
#	make lint			: run all linter rules required for completing pull requests

PY_DIRS_MAIN := lib
PY_DIRS_EXTRA := sandbox
PY_DIRS_TEST := test
PY_DIRS_ALL := $(PY_DIRS_MAIN) $(PY_DIRS_EXTRA) $(PY_DIRS_TEST)

ANSI_GREEN := \033[0;32m
ANSI_RESET := \033[0;0m

# Run all CI checks.
.DEFAULT_GOAL := ci
.PHONY: ci
ci: lint test
	@echo
	@echo "$(ANSI_GREEN)====== All linters and tests PASS ======$(ANSI_RESET)"

# Run all linters.
.PHONY: lint
lint: lint-flake8 type sec
	@echo
	@echo "$(ANSI_GREEN)====== All linters PASS ======$(ANSI_RESET)"

# Individual task definitions.

.PHONY: test
test:
	@pytest $(PY_DIRS_TEST) --cov=$(PY_DIRS_MAIN) --cov-report=term-missing

.PHONY: lint-flake8
lint-flake8:
	@flake8 $(PY_DIRS_ALL)

.PHONY: type
type:
	@mypy $(PY_DIRS_MAIN) $(PY_DIRS_EXTRA)

.PHONY: sec
sec:
	@bandit -c pyproject.toml -r $(PY_DIRS_ALL)

.PHONY: typecheck
typecheck:
	pyright

.PHONY: check
check: typecheck
	ruff check --fix
	ruff format

.PHONY: test
test:
	pytest

# The LiteLLM guardrail is a package of its own, tested and checked in an
# environment of its own with one LiteLLM version, which this repository's
# environment never has. Its tests start a real sidecar from this repository's
# environment, so run `uv sync` first.
LITELLM ?= 1.105.0
LITELLM_ENV = cd src/proxies/litellm && uv run --isolated --no-project \
	--with "litellm[proxy]==$(LITELLM)" --with-editable . \
	--with pytest --with pytest-timeout --with jsonschema --with httpx

.PHONY: test-litellm
test-litellm:
	$(LITELLM_ENV) pytest

.PHONY: check-litellm
check-litellm:
	$(LITELLM_ENV) --with pyright pyright

# The Middleman hook is a package of its own too, which runs in Middleman's
# process. Middleman is not a package, so its tests hand the hook what
# Middleman would. Its tests start a real sidecar from this repository's
# environment, so run `uv sync` first. `LOWEST=1` tests it with the lowest
# versions of its dependencies that its pyproject.toml allows, which are
# Middleman's own.
MIDDLEMAN_LOWEST = $(if $(LOWEST),--with "aiohttp==3.9.5" --with "fastapi==0.92.0")
MIDDLEMAN_ENV = cd src/proxies/middleman && uv run --isolated --no-project \
	--with-editable . $(MIDDLEMAN_LOWEST) --with pytest --with pytest-timeout \
	--with jsonschema --with uvicorn

.PHONY: test-middleman
test-middleman:
	$(MIDDLEMAN_ENV) pytest

.PHONY: check-middleman
check-middleman:
	$(MIDDLEMAN_ENV) --with pyright pyright

.PHONY: docs
docs:
	cd docs && PATH="$(CURDIR)/.venv/bin:$$PATH" ../.venv/bin/quarto render

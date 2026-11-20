# Alvos de uso humano. tests/test_makefile.py confere que eles existem e que
# cada um pode passar pelo que seleciona.

PY := .venv/bin/python
PYTEST := $(PY) -m pytest

.PHONY: setup test test-golden test-all lint check clean

## setup — cria o ambiente e verifica a integridade dos dados de referência
#
# Instala também o grupo `desktop`: sem PySide6 e pytest-qt os testes da janela
# caem todos em `skip`, e a suíte passaria sem exercitar a janela.
setup:
	python3.12 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev,desktop]"
	bash scripts/fetch_reference_data.sh

## test — unitários e de propriedade
test:
	$(PYTEST) -m "not golden and not integration and not synthetic and not slow"

## test-golden — comparação contra as saídas do pipeline R
#
# `--no-cov` porque o piso de 85% é sobre a suíte inteira, e uma seleção parcial
# não o alcança: sem isto o alvo reprovaria com todos os testes verdes. Quem mede
# cobertura é `make check`.
test-golden:
	$(PYTEST) -m golden --no-cov

## test-all — a suíte inteira, inclusive os testes lentos
test-all:
	$(PYTEST)

## lint — ruff e mypy
lint:
	$(PY) -m ruff check src tests scripts
	$(PY) -m ruff format --check src tests scripts
	$(PY) -m mypy src/refino

## check — lint e a suíte sem os testes lentos
check: lint
	$(PYTEST) -m "not slow"

## clean — apaga caches e relatórios de cobertura
clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml htmlcov
	find src tests -name __pycache__ -type d -exec rm -rf {} +

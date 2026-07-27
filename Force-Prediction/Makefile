.PHONY: setup lint typecheck test collect run viewer prepare-viewer all clean

PY ?= python

setup:
	$(PY) -m pip install -e ".[dev]"

lint:
	ruff check modules scripts tests

typecheck:
	mypy modules

test:
	pytest

# Generate a synthetic dataset on the mock bench; descriptions use Gemini.
collect:
	$(PY) -m modules.collect --mock --dataset mock --n 40 --confirm-gemini-cost

# E1-E4 and E6 require --confirm-gemini-cost; E5 is fully local.
E ?= e4
run:
	$(PY) scripts/run_experiment.py --exp $(E) $(if $(filter e5,$(E)),,--confirm-gemini-cost)

run-all:
	$(PY) scripts/run_experiment.py --all --confirm-gemini-cost --out results.json

prepare-viewer:
	$(PY) scripts/prepare_expforce_viewer.py --confirm-gemini-cost

viewer:
	$(PY) -m streamlit run app.py

# End-to-end Gemini smoke: dataset -> full experiment suite.
smoke: collect run-all

all: lint typecheck test

clean:
	rm -rf data/cache/mock data/mock results.json

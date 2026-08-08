.PHONY: setup lint typecheck test run viewer all clean

PY ?= python

setup:
	$(PY) -m pip install -e ".[dev]"

lint:
	ruff check modules scripts streamlit_app tests

typecheck:
	mypy modules

test:
	pytest

# Active experiments use the contiguous IDs E1-E5.
E ?= e5
run:
	$(PY) scripts/run_experiment.py --exp $(E) --confirm-gemini-cost

run-all:
	$(PY) scripts/run_experiment.py --all --confirm-gemini-cost --out results.json

viewer:
	$(PY) -m streamlit run app.py

all: lint typecheck test

clean:
	rm -rf data/cache/mock data/mock/results data/mock/runs data/mock/run_images data/mock/suites results.json

.PHONY: setup lint typecheck test collect run viewer prepare-viewer all clean

PY ?= python

setup:
	$(PY) -m pip install -e ".[dev]"

lint:
	ruff check force_prediction scripts tests

typecheck:
	mypy force_prediction

test:
	pytest

# Generate a synthetic dataset on the mock bench (no hardware, no API).
collect:
	$(PY) -m force_prediction.collect --mock --n 40

# Run the paired-retrieval VLM method (E4) offline. Use E=e5 for calibrated physics.
E ?= e4
run:
	$(PY) scripts/run_experiment.py --exp $(E) --dry-run

run-all:
	$(PY) scripts/run_experiment.py --all --dry-run --out results.json

prepare-viewer:
	$(PY) scripts/prepare_expforce_viewer.py

viewer:
	$(PY) -m streamlit run app.py

# End-to-end smoke: dataset -> full experiment suite, all offline.
smoke: collect run-all

all: lint typecheck test

clean:
	rm -rf data/cache/*.json data/experiences.jsonl data/splits.json data/images/*.png results.json

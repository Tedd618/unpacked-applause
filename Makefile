PY := .venv/bin/python
APP := $(PY) -m unpacked_applause

.PHONY: all catalog audio captions detect validate features market study analyze figures report notebooks test lint

all: catalog audio captions detect validate features market study analyze figures notebooks report

catalog:      ## merge + validate the hand-curated keynote catalog
	$(APP) catalog
audio:        ## download keynote audio (cached in data/raw/audio, not committed)
	$(APP) fetch-audio
captions:     ## download YouTube captions used as detector labels
	$(APP) fetch-captions
detect:       ## PANNs sound-event detection -> data/processed/timelines/*.parquet
	$(APP) detect
validate:     ## tune/evaluate episode thresholds against human caption tags
	$(APP) validate
features:     ## event-level applause measures
	$(APP) features
market:       ## Yahoo Finance prices + Fama-French factors
	$(APP) market
study:        ## daily event study (abnormal returns, CARs, abnormal volume)
	$(APP) study
analyze:      ## pre-registered cross-sectional analysis -> reports/tables
	$(APP) analyze
figures:      ## report figures -> reports/figures
	$(APP) figures
notebooks:    ## execute the analysis notebooks in place
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
report:       ## HTML + PDF report -> reports/
	$(APP) report
test:
	$(PY) -m pytest -q
lint:
	$(PY) -m ruff check src tests

PYTHON ?= python
CONFIG ?= config/user_inputs.local.yaml

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev,analysis]"

print-questions:
	$(PYTHON) -m father_longrun print-questions

check-config:
	$(PYTHON) -m father_longrun check-config --config $(CONFIG)

inspect-nlsy:
	$(PYTHON) -m father_longrun inspect-nlsy --config $(CONFIG)

doctor:
	$(PYTHON) -m father_longrun doctor --config $(CONFIG)

build-nlsy-pilot:
	$(PYTHON) -m father_longrun build-nlsy-pilot --config $(CONFIG)

build-phase0:
	$(PYTHON) -m father_longrun build-phase0 --config $(CONFIG)

build-merge-contract:
	$(PYTHON) -m father_longrun build-merge-contract --config $(CONFIG)

build-backbone-scaffold:
	$(PYTHON) -m father_longrun build-backbone-scaffold --config $(CONFIG)

build-reviewed-layers:
	$(PYTHON) -m father_longrun build-reviewed-layers --config $(CONFIG)

build-refresh-spec:
	$(PYTHON) -m father_longrun build-refresh-spec --config $(CONFIG)

refresh-nlsy-treatment-extracts:
	$(PYTHON) -m father_longrun refresh-nlsy-treatment-extracts --config $(CONFIG)

build-treatment-candidate-layers:
	$(PYTHON) -m father_longrun build-treatment-candidate-layers --config $(CONFIG)

build-analysis-ready-treatments:
	$(PYTHON) -m father_longrun build-analysis-ready-treatments --config $(CONFIG)

build-fatherlessness-profiles:
	$(PYTHON) -m father_longrun build-fatherlessness-profiles --config $(CONFIG)

build-nlsy97-longitudinal-panel:
	$(PYTHON) -m father_longrun build-nlsy97-longitudinal-panel --config $(CONFIG)

build-quasi-causal-scaffold:
	$(PYTHON) -m father_longrun build-quasi-causal-scaffold --config $(CONFIG)

build-ml-benchmarks:
	$(PYTHON) -m father_longrun build-ml-benchmarks --config $(CONFIG)

build-psid-intake:
	$(PYTHON) -m father_longrun build-psid-intake --config $(CONFIG)

build-add-health-intake:
	$(PYTHON) -m father_longrun build-add-health-intake --config $(CONFIG)

build-ffcws-intake:
	$(PYTHON) -m father_longrun build-ffcws-intake --config $(CONFIG)

build-benchmarks:
	$(PYTHON) -m father_longrun build-benchmarks --config $(CONFIG)

build-ipums-workflow:
	$(PYTHON) -m father_longrun build-ipums-workflow --config $(CONFIG)

build-public-microdata:
	$(PYTHON) -m father_longrun build-public-microdata --config $(CONFIG)

build-public-benchmark-profiles:
	$(PYTHON) -m father_longrun build-public-benchmark-profiles --config $(CONFIG)

build-cross-cohort-benchmarks:
	$(PYTHON) -m father_longrun build-cross-cohort-benchmarks --config $(CONFIG)

build-results-appendix:
	$(PYTHON) -m father_longrun build-results-appendix --config $(CONFIG)

build-synthesis:
	$(PYTHON) -m father_longrun build-synthesis --config $(CONFIG)

public-results:
	$(PYTHON) -m father_longrun build-fatherlessness-profiles --config $(CONFIG)
	$(PYTHON) -m father_longrun build-nlsy97-longitudinal-panel --config $(CONFIG)
	$(PYTHON) -m father_longrun build-quasi-causal-scaffold --config $(CONFIG)
	$(PYTHON) -m father_longrun build-benchmarks --config $(CONFIG)
	$(PYTHON) -m father_longrun build-public-microdata --config $(CONFIG)
	$(PYTHON) -m father_longrun build-public-benchmark-profiles --config $(CONFIG)
	$(PYTHON) -m father_longrun build-cross-cohort-benchmarks --config $(CONFIG)
	$(PYTHON) -m father_longrun build-results-appendix --config $(CONFIG)
	$(PYTHON) -m father_longrun build-synthesis --config $(CONFIG)

prepush:
	$(PYTHON) -m pytest -q
	$(PYTHON) -m father_longrun doctor --config $(CONFIG)
	$(PYTHON) -m father_longrun --help >/dev/null

test:
	pytest -q

.PHONY: install test run
install:
	pip install -e ".[dev]"
test:
	pytest
run:
	python -m cpf.run

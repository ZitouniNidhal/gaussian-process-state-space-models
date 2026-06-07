PYTHON = python

.PHONY: install test docs clean

install:
	$(PYTHON) -m pip install -e .

test:
	$(PYTHON) -m pytest

docs:
	sphinx-build -b html docs/source docs/build

clean:
	-rm -rf __pycache__
	-rm -rf gpssm/__pycache__
	-rm -rf docs/build
	-rm -rf docs/source/*.pyc

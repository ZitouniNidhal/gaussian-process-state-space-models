PYTHON = python

.PHONY: install test docs clean

install:
	$(PYTHON) -m pip install -e .

test:
	$(PYTHON) -m pytest

docs:
	sphinx-build -b html docs/source docs/build

clean:
	$(PYTHON) -c "import shutil, pathlib; import os; paths=['__pycache__','gpssm/__pycache__','docs/build']; [shutil.rmtree(path, ignore_errors=True) for path in [pathlib.Path(p) for p in paths]]; [p.unlink() for p in pathlib.Path('docs/source').glob('*.pyc') if p.exists()]"

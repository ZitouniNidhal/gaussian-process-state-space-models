import os
import sys

sys.path.insert(0, os.path.abspath("../../"))

project = "gpssm"
release = "0.1.0"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]
master_doc = "index"
exclude_patterns = ["_build"]
html_theme = "alabaster"

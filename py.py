from pathlib import Path

# Project directory structure
directories = [
    "gpssm/models",
    "gpssm/kernels",
    "gpssm/filters",
    "gpssm/inference",
    "gpssm/utils",
    "gpssm/datasets",
    "gpssm/examples",
    "gpssm/tests/unit",
    "gpssm/tests/integration",
    "benchmarks/synthetic",
    "benchmarks/baselines",
    "notebooks",
    "tutorials",
    "docs/source",
    "scripts/training",
    "scripts/evaluation",
    "scripts/visualization",
    ".github/workflows",
]

# Files to create
files = [
    # Models
    "gpssm/models/base.py",
    "gpssm/models/gpssm.py",
    "gpssm/models/linear.py",
    "gpssm/models/variational.py",
    "gpssm/models/online.py",
    "gpssm/models/free_form.py",

    # Kernels
    "gpssm/kernels/base.py",
    "gpssm/kernels/rbf.py",
    "gpssm/kernels/matern.py",
    "gpssm/kernels/spectral_mixture.py",
    "gpssm/kernels/polynomial.py",

    # Filters
    "gpssm/filters/kalman.py",
    "gpssm/filters/particle.py",

    # Inference
    "gpssm/inference/elbo.py",
    "gpssm/inference/recognition.py",
    "gpssm/inference/inducing.py",

    # Utilities
    "gpssm/utils/metrics.py",
    "gpssm/utils/diagnostics.py",
    "gpssm/utils/visualization.py",
    "gpssm/utils/config.py",

    # Datasets
    "gpssm/datasets/synthetic.py",
    "gpssm/datasets/real_world.py",

    # Benchmarks
    "benchmarks/synthetic/pendulum_benchmark.py",
    "benchmarks/baselines/ekf.py",

    # Notebooks
    "notebooks/01_quickstart.ipynb",
    "notebooks/02_model_comparison.ipynb",
    "notebooks/03_advanced_kernels.ipynb",

    # Tutorials
    "tutorials/01_getting_started.md",
    "tutorials/02_model_selection.md",
    "tutorials/03_uncertainty_quantification.md",
    "tutorials/04_advanced_topics.md",

    # Documentation
    "docs/source/conf.py",
    "docs/source/index.rst",
    "docs/source/api.rst",

    # Scripts
    "scripts/training/train.py",
    "scripts/evaluation/eval.py",
    "scripts/visualization/plot_results.py",

    # GitHub Actions
    ".github/workflows/ci.yml",
    ".github/workflows/release.yml",

    # Root files
    "README.md",
    "pyproject.toml",
    "Dockerfile",
    "docker-compose.yml",
    "Makefile",
    ".pre-commit-config.yaml",
    "LICENSE",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    ".gitignore",
]

# Create directories
for directory in directories:
    Path(directory).mkdir(parents=True, exist_ok=True)

# Create files
for file in files:
    path = Path(file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)

# Create __init__.py files for Python packages
packages = [
    "gpssm",
    "gpssm/models",
    "gpssm/kernels",
    "gpssm/filters",
    "gpssm/inference",
    "gpssm/utils",
    "gpssm/datasets",
]

for package in packages:
    init_file = Path(package) / "__init__.py"
    init_file.touch(exist_ok=True)

print("Project structure created successfully!")
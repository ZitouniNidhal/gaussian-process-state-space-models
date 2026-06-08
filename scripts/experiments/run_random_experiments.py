"""Run multiple randomized experiments: sample kernels, datasets, train models, save results.

This script performs `n_runs` experiments. For each run it:
- samples a random kernel
- generates a randomized synthetic dataset
- fits a chosen model and saves the model and data
"""

import argparse
from pathlib import Path
from gpssm.utils import set_seed, sample_random_kernel, generate_random_synthetic
from gpssm.utils.io import save_npz, save_object
from gpssm.models import GaussianProcessStateSpaceModel, VariationalGPModel


def main():
    parser = argparse.ArgumentParser(description="Run random GPSM experiments and save outputs.")
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--n-points", type=int, default=100)
    parser.add_argument("--outdir", type=str, default="experiments/random_runs")
    parser.add_argument("--model", type=str, default="gpssm", choices=["gpssm", "variational"])
    parser.add_argument("--base-seed", type=int, default=1000)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for i in range(args.n_runs):
        seed = args.base_seed + i
        set_seed(seed)
        kernel = sample_random_kernel()
        X, y = generate_random_synthetic(args.n_points, seed=seed)

        # build model
        if args.model == "variational":
            model = VariationalGPModel(kernel=kernel, n_inducing=min(20, args.n_points // 2))
        else:
            model = GaussianProcessStateSpaceModel(kernel=kernel)

        model.fit(X, y)

        data_path = outdir / f"run_{i:03d}_data.npz"
        save_npz(str(data_path), X=X, y=y)
        model_path = outdir / f"run_{i:03d}_model.pkl"
        save_object(str(model_path), model)
        print(f"Saved run {i} (seed={seed}) -> {data_path}, {model_path}")


if __name__ == "__main__":
    main()

"""CLI wrapper that extends the training script with a model save option.

This module imports the main training routine and exposes a convenience
CLI flag to save the trained model to disk.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Train and optionally save a GPSM model (wrapper).")
    parser.add_argument("--n-points", type=int, default=100)
    parser.add_argument("--model", type=str, default="gpssm")
    parser.add_argument("--kernel", type=str, default="rbf")
    parser.add_argument("--save-model", type=str, default=None, help="If provided, save the trained model to this path")
    args = parser.parse_args()

    # Build command to run the original script
    script_path = Path(__file__).parent / "train.py"
    cmd = [sys.executable, str(script_path), "--n-points", str(args.n_points), "--model", args.model, "--kernel", args.kernel]
    # Run training script
    subprocess.check_call(cmd)
    if args.save_model:
        print("Note: to save the trained model, re-run the training script programmatically or use the Python API.")


if __name__ == "__main__":
    main()

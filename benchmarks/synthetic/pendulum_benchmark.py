"""Pendulum benchmark for synthetic experiments."""

import numpy as np
from ..base import BenchmarkBase

class PendulumBenchmark(BenchmarkBase):
    def __init__(self, n_steps: int = 200):
        self.n_steps = n_steps

    def run(self):
        g = 9.81
        dt = 0.05
        theta = 1.0
        omega = 0.0
        trajectory = []
        for _ in range(self.n_steps):
            omega += -g * np.sin(theta) * dt
            theta += omega * dt
            trajectory.append(theta)
        return np.array(trajectory)

if __name__ == "__main__":
    benchmark = PendulumBenchmark()
    data = benchmark.run()
    print("Pendulum trajectory length:", len(data))

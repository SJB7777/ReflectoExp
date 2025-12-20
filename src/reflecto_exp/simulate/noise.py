import numpy as np


def apply_poisson_noise(arr: np.ndarray, s: float) -> np.ndarray:
    """Apply Poisson noise to an array."""
    expected_counts = s * arr
    noisy_counts = np.random.poisson(expected_counts)
    return noisy_counts / s

def get_background_noise(count: int, b_min: float, b_max: float) -> np.ndarray:
    """Generate background noise."""
    b = pow(10, np.random.uniform(b_min, b_max))
    return np.random.normal(b, 0.1 * b, count)

def add_noise(R):
    N = len(R)
    R_poisson = apply_poisson_noise(R, s=10 ** 8)
    uniform_noise = 1 + np.random.uniform(-0.1, 0.1, N)
    background_noise = get_background_noise(N, -8, -6)
    curve_scaling = np.random.uniform(0.99, 1.01)
    return R_poisson * uniform_noise * curve_scaling + background_noise

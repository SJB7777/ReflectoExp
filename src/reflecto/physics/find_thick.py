import numpy as np
import torch
from scipy.signal import savgol_filter


def flatten_trend(arr):
    R_smooth = savgol_filter(torch.log10(arr), window_length=int(len(arr)), polyorder=4, mode='interp')
    savgol = 10**R_smooth
    return arr / savgol


def estimate_thickness_from_gradient(
    reflectivity_curve: torch.Tensor,
    q_values: torch.Tensor,
    n_layers: int,
) -> torch.Tensor:
    try:
        pass
    except Exception:
        return torch.zeros(n_layers)

def main():
    arr = np.ones(100)
    torch.asarray(arr)

if __name__ == "__main__":
    main()

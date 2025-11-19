import numpy as np


def fom_log(ref_exp, ref_calc):
    """Figure of Merit based on log-scale reflectivity difference."""
    return np.mean(np.abs(np.log10(ref_exp) - np.log10(ref_calc)))

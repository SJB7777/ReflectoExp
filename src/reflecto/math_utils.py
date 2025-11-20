import warnings

import numpy as np


def fom_log(ref_exp, ref_calc):
    """
    Calculates the Figure of Merit (FOM) based on the log-scale reflectivity difference.
    """
    return np.mean(np.abs(np.log10(ref_exp) - np.log10(ref_calc)))

def i0_normalize(arr):
    """
    Normalizes the input array by dividing all elements by the maximum value (I0 normalization).

    This process scales the array values into the range [0, 1]. This is commonly used
    to normalize raw intensity data in XRR measurements to a reference intensity (I0).
    """
    max_val = arr.max()

    if max_val == 0:
        # The RuntimeWarning is used here as it indicates an issue encountered
        # during runtime that doesn't halt execution but requires attention.
        warnings.warn(
            "Input array contains only zero values (max value is 0). Normalization skipped. Returning original array.",
            RuntimeWarning, stacklevel=2
        )
        return arr
    return arr / arr.max()

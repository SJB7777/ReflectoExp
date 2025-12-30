import numpy as np

from scipy.special import erf


rad = np.pi / 180.0
sqrt2 = np.sqrt(2.0)

def GaussIntensity(alpha, s1, s2, sigma_x):
    sinalpha = np.sin(alpha * rad)
    if s1 == s2:
        return erf(s2 / sqrt2 / sigma_x * sinalpha)
    else:
        common = sinalpha / sqrt2 / sigma_x
        return (erf(s2 * common) + erf(s1 * common)) / 2.0

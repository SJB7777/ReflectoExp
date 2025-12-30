import torch
from torch import Tensor

from .abeles import abeles, abeles_compiled
from .kinematical import kinematical_approximation
from .memory_eff import abeles_memory_eff
from .numpy_implementations import (
    abeles_np,
    kinematical_approximation_np,
)
from .smearing import abeles_constant_smearing

__all__ = [
    "simulate_reflectivity",
    "abeles",
    "abeles_compiled",
    "abeles_memory_eff",
    "abeles_np",
    "kinematical_approximation",
    "kinematical_approximation_np",
    "abeles_constant_smearing",
]


def simulate_reflectivity(
    q: Tensor,
    thickness: Tensor,
    roughness: Tensor,
    sld: Tensor,
    dq_q: Tensor = None,
    gauss_num: int = 51,
    xrr_dq: bool = False,
    abeles_func=None,
    q_offset: Tensor = 0.0,
    bkg: Tensor = 0.0,
    r_scale: Tensor = 1.0,
):
    abeles_func = abeles_func or abeles
    q = torch.atleast_2d(q) + q_offset
    q = torch.clamp(q, min=0.0)

    if dq_q is None:
        reflectivity_curves = abeles_func(q, thickness, roughness, sld)
    else:
        reflectivity_curves = abeles_constant_smearing(
            q,
            thickness,
            roughness,
            sld,
            dq=dq_q,
            gauss_num=gauss_num,
            xrr_dq=xrr_dq,
            abeles_func=abeles_func,
        )

    reflectivity_curves = reflectivity_curves * r_scale + bkg

    return reflectivity_curves


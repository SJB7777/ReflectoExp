from typing import Final

from genx.models.spec_nx import Coords, FootType, Instrument, Layer, Probe, ResType

XRAY_TUBE: Final[Instrument] = Instrument(
    probe=Probe.xray,
    wavelength=1.54,
    coords=Coords.q,
    I0=1.0,
    Ibkg=1e-10,
    res=0.005,
    restype=ResType.fast_conv, # 분해능 컨볼루션
    footype=FootType.gauss
)

AIR: Final[Layer] = Layer(
    d=0.0,
    f=complex(0, 0),
    dens=0.0,
    sigma=0.0
)

SURFACE_SIO2: Final[Layer] = Layer(
    d=15.0,
    f=complex(14, 0.1),
    dens=0.05,
    sigma=2.0
)

SUBSTRATE_SI: Final[Layer] = Layer(
    d=0.0,
    f=complex(13.5, 0.05),
    dens=0.05,
    sigma=2.0
)

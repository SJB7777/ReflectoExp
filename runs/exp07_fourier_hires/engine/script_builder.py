class GenXScriptBuilder:
    """GenX 모델 내부에서 실행될 파이썬 스크립트를 생성합니다."""
    
    def build(self, init_params: dict[str, float]) -> str:
        return f"""
import numpy as np
from genx.models.spec_nx import Sample, Stack, Layer, Instrument, Specular
from genx.models.spec_nx import Probe, Coords, ResType, FootType
from genx.models.lib.physical_constants import r_e

def to_f(sld_val):
    return complex((sld_val * 1e-6) / r_e, 0)

class Sim_Vars:
    def __init__(self):
        # Film (AI Guess)
        self.f_d   = {init_params['f_d']}
        self.f_sig = {init_params['f_sig']}
        self.f_sld = {init_params['f_sld']}
        
        # SiO2 (Fixed/Guess)
        self.s_d   = 15.0
        self.s_sig = 3.0
        self.s_sld = 18.8

        # Instrument
        self.i0    = 1.0
        self.s_len = {init_params['s_len']}
        self.beam_w = {init_params['beam_w']}

    # =================================================================
    # [FIX] GenX requires BOTH Setters AND Getters for every parameter
    # =================================================================

    # --- Film ---
    def set_f_d(self, v): self.f_d = float(v)
    def get_f_d(self): return self.f_d

    def set_f_sig(self, v): self.f_sig = float(v)
    def get_f_sig(self): return self.f_sig

    def set_f_sld(self, v): self.f_sld = float(v)
    def get_f_sld(self): return self.f_sld
    
    # --- SiO2 ---
    def set_s_d(self, v): self.s_d = float(v)
    def get_s_d(self): return self.s_d

    def set_s_sig(self, v): self.s_sig = float(v)
    def get_s_sig(self): return self.s_sig

    def set_s_sld(self, v): self.s_sld = float(v)
    def get_s_sld(self): return self.s_sld
    
    # --- Instrument ---
    def set_i0(self, v): self.i0 = float(v)
    def get_i0(self): return self.i0

    def set_s_len(self, v): self.s_len = float(v)
    def get_s_len(self): return self.s_len

    def set_beam_w(self, v): self.beam_w = float(v)
    def get_beam_w(self): return self.beam_w

v = Sim_Vars()

# Layer Definitions
Amb = Layer(d=0, f=0, dens=0)
Sub = Layer(d=0, f=to_f(20.07), dens=1, sigma=3.0)
Film = Layer(d=v.f_d, sigma=v.f_sig, f=to_f(v.f_sld), dens=1)
SiO2 = Layer(d=v.s_d, sigma=v.s_sig, f=to_f(v.s_sld), dens=1)

sample = Sample(Stacks=[Stack(Layers=[Film, SiO2])], Ambient=Amb, Substrate=Sub)

# Instrument Definitions
inst = Instrument(
    probe=Probe.xray, wavelength=1.54, coords=Coords.q,
    I0=v.i0, Ibkg=1e-10, res=0.002,
    restype=ResType.fast_conv,
    footype=FootType.gauss,
    samplelen=v.s_len,
    beamw=v.beam_w
)

def Sim(data):
    # Dynamic update hook
    Film.d, Film.sigma, Film.f = v.f_d, v.f_sig, to_f(v.f_sld)
    SiO2.d, SiO2.sigma, SiO2.f = v.s_d, v.s_sig, to_f(v.s_sld)
    inst.I0, inst.samplelen, inst.beamw = v.i0, v.s_len, v.beam_w
    return [Specular(d.x, sample, inst) for d in data]
"""
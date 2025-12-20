import matplotlib.pyplot as plt
import numpy as np
from genx import fom_funcs
from genx.data import DataList, DataSet
from genx.model import Model
from genx.parameters import Parameters

from reflecto_exp.simulate.simul_genx import ParamSet


class GenXFitter:
    def __init__(self, q: np.ndarray, R: np.ndarray, nn_params: ParamSet):
        self.q = q
        self.R = R / R.max()

        self.init_d = float(nn_params.thickness)
        self.init_sigma = float(nn_params.roughness)
        self.init_sld = float(nn_params.sld)

        self.model = self._build_model()
        self.R_sim = None

    def _build_model(self):
        ds = DataSet(name="XRR_Data")
        ds.x_raw = self.q
        ds.y_raw = self.R
        ds.error_raw = np.maximum(self.R * 0.1, 1e-9)
        ds.run_command()

        model = Model()
        model.data = DataList([ds])

        # 초기값
        sio_d, sio_sig, sio_sld = 15.0, 3.0, 18.8

        script = rf"""
import numpy as np
from genx.models.spec_nx import Sample, Stack, Layer, Instrument, Specular
from genx.models.spec_nx import Probe, Coords, ResType, FootType
from genx.models.lib.physical_constants import r_e

class Sim_Vars:
    def __init__(self):
        # Film
        self.f_d   = {self.init_d}
        self.f_sig = {self.init_sigma}
        self.f_sld = {self.init_sld}
        # SiO2
        self.s_d   = {sio_d}
        self.s_sig = {sio_sig}
        self.s_sld = {sio_sld}

        # [추가됨] 기기 관련 변수
        self.i0    = 1.0
        self.s_len = 10.0  # 샘플 길이 (mm)
        self.beam_w = 0.1  # 빔 폭 (mm) - 보통 슬릿 크기

    # --- Film ---
    def set_f_d(self, v):   self.f_d = float(v)
    def get_f_d(self):      return self.f_d
    def set_f_sig(self, v): self.f_sig = float(v)
    def get_f_sig(self):    return self.f_sig
    def set_f_sld(self, v): self.f_sld = float(v)
    def get_f_sld(self):    return self.f_sld

    # --- SiO2 ---
    def set_s_d(self, v):   self.s_d = float(v)
    def get_s_d(self):      return self.s_d
    def set_s_sig(self, v): self.s_sig = float(v)
    def get_s_sig(self):    return self.s_sig
    def set_s_sld(self, v): self.s_sld = float(v)
    def get_s_sld(self):    return self.s_sld

    # --- Instrument (I0, Footprint) ---
    def set_i0(self, v):    self.i0 = float(v)
    def get_i0(self):       return self.i0

    def set_s_len(self, v): self.s_len = float(v)
    def get_s_len(self):    return self.s_len

    def set_beam_w(self, v): self.beam_w = float(v)
    def get_beam_w(self):    return self.beam_w

v = Sim_Vars()

def to_f(sld_val):
    return complex((sld_val * 1e-6) / r_e, 0)

Amb = Layer(d=0, f=0, dens=0)
Sub = Layer(d=0, f=to_f(20.07), dens=1, sigma=3.0)

Film = Layer(d=v.get_f_d(), sigma=v.get_f_sig(), f=to_f(v.get_f_sld()), dens=1)
SiO2 = Layer(d=v.get_s_d(), sigma=v.get_s_sig(), f=to_f(v.get_s_sld()), dens=1)

sample = Sample(Stacks=[Stack(Layers=[Film, SiO2])], Ambient=Amb, Substrate=Sub)

# [핵심 변경] Instrument에 samplelen과 beamw 연결
inst = Instrument(
    probe=Probe.xray, wavelength=1.54, coords=Coords.q,
    I0=v.get_i0(), Ibkg=1e-10, res=0.002,
    restype=ResType.fast_conv,
    footype=FootType.gauss,     # Footprint 보정 켜짐
    samplelen=v.get_s_len(),    # 샘플 길이 변수 연결
    beamw=v.get_beam_w()        # 빔 폭 변수 연결
)

def Sim(data):
    Film.d     = v.get_f_d()
    Film.sigma = v.get_f_sig()
    Film.f     = to_f(v.get_f_sld())

    SiO2.d     = v.get_s_d()
    SiO2.sigma = v.get_s_sig()
    SiO2.f     = to_f(v.get_s_sld())

    inst.I0    = v.get_i0()
    inst.samplelen = v.get_s_len() # 동기화
    inst.beamw = v.get_beam_w()    # 동기화

    return [Specular(d.x, sample, inst) for d in data]
"""
        model.set_script(script)
        model.compile_script()
        return model

    def run(self, verbose=True):
        pars = Parameters()
        model = self.model

        # --- Parameters ---
        # 1. Material Params
        p_f_d = pars.append("v.set_f_d", model); p_f_d.min=1.0; p_f_d.max=self.init_d*2
        p_f_sig = pars.append("v.set_f_sig", model); p_f_sig.min=0.0; p_f_sig.max=30.0
        p_f_sld = pars.append("v.set_f_sld", model); p_f_sld.min=0.1; p_f_sld.max=50.0

        p_s_d = pars.append("v.set_s_d", model); p_s_d.value=15.0; p_s_d.min=5.0; p_s_d.max=50.0
        p_s_sig = pars.append("v.set_s_sig", model); p_s_sig.value=3.0; p_s_sig.min=0.0; p_s_sig.max=15.0
        p_s_sld = pars.append("v.set_s_sld", model); p_s_sld.value=18.8; p_s_sld.min=10.0; p_s_sld.max=23.0

        # 2. Instrument Params (I0 + Footprint)
        p_i0 = pars.append("v.set_i0", model)
        p_i0.value = 1.0; p_i0.min = 0.1; p_i0.max = 5.0

        # [추가] 샘플 길이 피팅 (저각도 모양 결정)
        p_slen = pars.append("v.set_s_len", model)
        p_slen.value = 10.0  # 초기값 10mm
        p_slen.min = 2.0     # 최소 2mm
        p_slen.max = 50.0    # 최대 50mm

        # [추가] 빔 폭 (보통 고정하거나 미세 조정)
        p_beam = pars.append("v.set_beam_w", model)
        p_beam.value = 0.1   # 초기값 0.1mm (슬릿 크기)
        p_beam.min = 0.01
        p_beam.max = 0.5
        p_beam.fit = False   # 일단 고정 (원하면 True)

        model.parameters = pars

        # --- Step 1: I0 & Footprint Fitting (Linear) ---
        if verbose:
            print("\n[GenX] Step 1: Fitting I0 & Sample Length...")
        model.set_fom_func(fom_funcs.diff) # Linear Scale

        # I0와 Sample Length는 서로 스케일에 영향을 주므로 같이 피팅
        p_i0.fit = True
        p_slen.fit = True

        # 나머지는 끔
        p_f_d.fit = False
        p_f_sig.fit = False
        p_f_sld.fit = False
        p_s_d.fit = False
        p_s_sig.fit = False
        p_s_sld.fit = False

        res1 = model.bumps_fit(method="de", steps=300)
        model.bumps_update_parameters(res1)
        if verbose:
            print(f"  -> I0: {p_i0.value:.3f}, Sample Len: {p_slen.value:.2f} mm")

        # --- Step 2: Full Fitting (Log) ---
        if verbose:
            print("[GenX] Step 2: Fitting All Params (Log)...")
        model.set_fom_func(fom_funcs.log)

        # 미세조정 위해 I0, SampleLen 켜둠 (원하면 꺼도 됨)
        p_i0.fit = True
        p_slen.fit = True

        p_f_d.fit = True
        p_f_sig.fit = True
        p_f_sld.fit = True
        p_s_d.fit = True
        p_s_sig.fit = True
        p_s_sld.fit = True

        res2 = model.bumps_fit(method="de", steps=800, pop=20, tol=0.002)
        model.bumps_update_parameters(res2)

        model.evaluate_sim_func()

        self.R_sim = self.model.data[0].y_sim
        results = {p.name.replace("v.", ""): p.value for p in pars if p.fit}
        return results

    def plot(self):
        q = self.model.data[0].x
        R_meas = self.model.data[0].y
        R_sim = self.model.data[0].y_sim

        plt.figure(figsize=(8, 6))
        plt.plot(q, R_meas, 'ko', label='Measured', markersize=4, alpha=0.6)
        plt.plot(q, R_sim, 'r-', label='Fit', linewidth=2)
        plt.yscale('log')
        plt.xlabel(r'q [$\AA^{-1}$]')
        plt.ylabel('Reflectivity')
        plt.title(f'Fit Result (FOM: {self.model.fom:.4e})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

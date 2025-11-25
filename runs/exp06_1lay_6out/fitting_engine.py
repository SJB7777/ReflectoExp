import matplotlib.pyplot as plt
import numpy as np
from genx import fom_funcs
from genx.data import DataList, DataSet
from genx.model import Model
from genx.parameters import Parameters

from reflecto.simulate.simul_genx import ParamSet


class GenXFitter:
    """
    Clean & Robust XRR Fitting Engine
    - Input: Readable SLD values (e.g., 2.07, 18.8)
    - Internal: Automatically handles physical unit conversion (1e-6 * r_e)
    """
    def __init__(self, q: np.ndarray, R: np.ndarray, film_params: ParamSet, sio2_params: ParamSet):
        """
        Args:
            q (array): Momentum transfer [1/A]
            R (array): Reflectivity
            nn_params (ParamSet)
        """
        self.q = q
        self.R = R / R.max()

        # 1. 입력값 정리 (Human Readable Scale로 통일)
        self.init_d = float(film_params.thickness)
        self.init_sigma = float(film_params.roughness)
        self.init_sld = float(film_params.sld)
        self.init_sio2_d = float(sio2_params.thickness)
        self.init_sio2_sigma = float(sio2_params.roughness)
        self.init_sio2_sld = float(sio2_params.sld)
        self.model = self._build_model()

    def _build_model(self):
        # --- Data Load ---
        ds = DataSet(name="XRR_Data")
        ds.x_raw = self.q
        ds.y_raw = self.R
        ds.error_raw = np.maximum(self.R * 0.1, 1e-9)
        ds.run_command()

        model = Model()
        model.data = DataList([ds])

        # --- Script Generation ---
        script = rf"""
import numpy as np
from genx.models.spec_nx import Sample, Stack, Layer, Instrument, Specular
from genx.models.spec_nx import Probe, Coords, ResType, FootType
from genx.models.lib.physical_constants import r_e

# [1] 파라미터 제어 클래스 (Setter/Getter 필수)
class Sim_Vars:
    def __init__(self):
        # 내부 변수 초기화
        self.f_d   = {self.init_d}
        self.f_sig = {self.init_sigma}
        self.f_sld = {self.init_sld}

        self.s_d   = {self.init_sio2_d}
        self.s_sig = {self.init_sio2_sigma}
        self.s_sld = {self.init_sio2_sld}

        self.i0    = 1.0

    # --- Film Setter/Getter ---
    def set_f_d(self, val):   self.f_d = float(val)
    def get_f_d(self):        return self.f_d

    def set_f_sig(self, val): self.f_sig = float(val)
    def get_f_sig(self):      return self.f_sig

    def set_f_sld(self, val): self.f_sld = float(val)
    def get_f_sld(self):      return self.f_sld

    # --- SiO2 Setter/Getter ---
    def set_s_d(self, val):   self.s_d = float(val)
    def get_s_d(self):        return self.s_d

    def set_s_sig(self, val): self.s_sig = float(val)
    def get_s_sig(self):      return self.s_sig

    def set_s_sld(self, val): self.s_sld = float(val)
    def get_s_sld(self):      return self.s_sld

    # --- Instrument Setter/Getter ---
    def set_i0(self, val):    self.i0 = float(val)
    def get_i0(self):         return self.i0

v = Sim_Vars()

# [2] 물리 변환 함수 (Human Readable -> Physical)
def to_f(sld_val):
    # SLD(10^-6 A^-2) -> f (Scattering Factor)
    # Formula: f = SLD_real / (density * r_e)
    # Here we assume density=1.0 for simplicity in Layer def
    return complex((sld_val * 1e-6) / r_e, 0)

# [3] 레이어 정의
Amb = Layer(d=0, f=0, dens=0)
Sub = Layer(d=0, f=to_f(20.07), dens=1, sigma=3.0)

# 초기값은 변수 v에서 가져옴
Film = Layer(d=v.get_f_d(), sigma=v.get_f_sig(), f=to_f(v.get_f_sld()), dens=1)
SiO2 = Layer(d=v.get_s_d(), sigma=v.get_s_sig(), f=to_f(v.get_s_sld()), dens=1)

sample = Sample(Stacks=[Stack(Layers=[Film, SiO2])], Ambient=Amb, Substrate=Sub)

inst = Instrument(probe=Probe.xray, wavelength=1.54, coords=Coords.q,
    I0=v.get_i0(), Ibkg=1e-10, res=0.002,
    restype=ResType.fast_conv, footype=FootType.gauss)

# [4] 시뮬레이션 루프 (파라미터 동기화)
def Sim(data):
    # Optimizer가 값을 변경하면(v._val), 실제 객체(Layer)에 반영
    Film.d     = v.get_f_d()
    Film.sigma = v.get_f_sig()
    Film.f     = to_f(v.get_f_sld())

    SiO2.d     = v.get_s_d()
    SiO2.sigma = v.get_s_sig()
    SiO2.f     = to_f(v.get_s_sld())

    inst.I0    = v.get_i0()

    return [Specular(d.x, sample, inst) for d in data]
"""
        model.set_script(script)
        model.compile_script()
        return model

    def run(self, verbose=True):
        """2-Step Fitting: I0 (Linear) -> All (Log)"""
        pars = Parameters()
        model = self.model

        # --- Parameters Registration ---
        # 1. Film (NN Prediction based)
        p_f_d = pars.append("v.set_f_d", model)
        p_f_d.min = max(1.0, self.init_d * 0.5)
        p_f_d.max = self.init_d * 1.5

        p_f_sig = pars.append("v.set_f_sig", model)
        p_f_sig.min = 0.0
        p_f_sig.max = 30.0

        p_f_sld = pars.append("v.set_f_sld", model)
        p_f_sld.min = 0.1
        p_f_sld.max = 50.0

        # 2. SiO2
        p_s_d = pars.append("v.set_s_d", model)
        p_s_d.value = 15.0
        p_s_d.min = 5.0
        p_s_d.max = 50.0

        p_s_sig = pars.append("v.set_s_sig", model)
        p_s_sig.value = 3.0
        p_s_sig.min = 0.0
        p_s_sig.max = 10.0

        p_s_sld = pars.append("v.set_s_sld", model)
        p_s_sld.value = 18.8
        p_s_sld.min = 10.0
        p_s_sld.max = 23.0

        # 3. Instrument
        p_i0 = pars.append("v.set_i0", model)
        p_i0.value = 1.0
        p_i0.min = 0.1
        p_i0.max = 3.0

        model.parameters = pars

        # --- Step 1: I0 Fitting ---
        if verbose:
            print("\n[GenX] Step 1: Fitting I0 (Linear)...")
        model.set_fom_func(fom_funcs.diff)

        p_i0.fit = True

        p_f_d.fit = False
        p_f_sig.fit = False
        p_f_sld.fit = False
        p_s_d.fit = False
        p_s_sig.fit = False
        p_s_sld.fit = False

        res1 = model.bumps_fit(method="amoeba", steps=200)
        model.bumps_update_parameters(res1)
        if verbose:
            print(f"  -> I0 Fitted: {p_i0.value:.4f}")

        # --- Step 2: Full Fitting ---
        if verbose:
            print("[GenX] Step 2: Fitting All Params (Log)...")
        model.set_fom_func(fom_funcs.log)

        p_i0.fit = False
        p_f_d.fit = True
        p_f_sig.fit = True
        p_f_sld.fit = True
        p_s_d.fit = True
        p_s_sig.fit = True
        p_s_sld.fit = True
        # Differential Evolution
        res2 = model.bumps_fit(method="de", steps=800, pop=15, tol=0.002)
        model.bumps_update_parameters(res2)

        model.evaluate_sim_func()

        # Return clean dictionary
        results = {p.name.replace("v.", ""): p.value for p in pars if p.fit}
        return results

    def plot(self):
        q = self.model.data[0].x
        R_meas = self.model.data[0].y
        R_sim = self.model.data[0].y_sim

        plt.figure(figsize=(8, 6))
        plt.plot(q, R_meas, 'ko', label='Measured', markersize=4, alpha=0.6)
        plt.plot(q, R_sim, 'r-', label='GenX Fit', linewidth=2)
        plt.yscale('log')
        plt.xlabel(r'q [$\AA^{-1}$]')
        plt.ylabel('Reflectivity')
        plt.title(f'GenX Fit Result (FOM: {self.model.fom:.4e})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

import matplotlib.pyplot as plt
import numpy as np
from genx import fom_funcs
from genx.data import DataList, DataSet
from genx.model import Model
from genx.parameters import Parameters


class GenXFitter:
    """
    Clean & Robust XRR Fitting Engine
    - Input: Readable SLD values (e.g., 2.07, 18.8)
    - Internal: Automatically handles physical unit conversion (1e-6 * r_e)
    """
    def __init__(self, q, R, nn_params):
        """
        Args:
            q (array): Momentum transfer [1/A]
            R (array): Reflectivity
            nn_params (dict): {'d', 'sigma', 'sld'}
                              If sld is small (e.g. 2e-6), it auto-scales to 2.0
        """
        self.q = q
        self.R = R

        # 1. 입력값 정리 (Human Readable Scale로 통일)
        self.init_d = float(nn_params['d'])
        self.init_sigma = float(nn_params['sigma'])

        # SLD가 1e-6 단위(물리값)로 들어오면 -> 보기 편한 값(예: 2.0)으로 변환
        # SLD가 이미 2.0 처럼 들어오면 -> 그대로 사용
        raw_sld = float(nn_params['sld'])
        if raw_sld < 0.01:
            self.init_sld = raw_sld * 1e6  # 2.07e-6 -> 2.07
        else:
            self.init_sld = raw_sld        # 2.07 -> 2.07

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

        # --- Hardcoded Initial Guesses for SiO2 ---
        sio_d, sio_sig, sio_sld = 15.0, 3.0, 18.8

        # --- Script Generation ---
        script = rf"""
import numpy as np
from genx.models.spec_nx import Sample, Stack, Layer, Instrument, Specular
from genx.models.spec_nx import Probe, Coords, ResType, FootType
from genx.models.lib.physical_constants import r_e

# [1] Optimizer가 조절할 변수들 (Human Readable Units)
class Sim_Vars:
    def __init__(self):
        # Film
        self.f_d   = {self.init_d}
        self.f_sig = {self.init_sigma}
        self.f_sld = {self.init_sld}  # ex: 2.07

        # SiO2
        self.s_d   = {sio_d}
        self.s_sig = {sio_sig}
        self.s_sld = {sio_sld}        # ex: 18.8

        # Inst
        self.i0    = 1.0

v = Sim_Vars()

# [2] 내부 단위 변환 함수 (핵심)
def sld_to_f(sld_val):
    # Input: 2.07 (meaning 2.07e-6 A^-2)
    # Output: Scattering Factor f (dimensionless complex number)
    real_sld = sld_val * 1e-6
    return complex(real_sld / r_e, 0)

# [3] 레이어 정의 (여기서 변환 함수 사용)
Amb = Layer(d=0, f=0, dens=0)
Sub = Layer(d=0, f=sld_to_f(20.07), dens=1, sigma=3.0) # Si Substrate

# Film & SiO2 (변수는 v에서 가져오고, 물리적 변환은 sld_to_f가 담당)
Film = Layer(d=v.f_d, sigma=v.f_sig, f=sld_to_f(v.f_sld), dens=1)
SiO2 = Layer(d=v.s_d, sigma=v.s_sig, f=sld_to_f(v.s_sld), dens=1)

sample = Sample(Stacks=[Stack(Layers=[Film, SiO2])], Ambient=Amb, Substrate=Sub)

inst = Instrument(probe=Probe.xray, wavelength=1.54, coords=Coords.q, 
    I0=v.i0, Ibkg=1e-10, res=0.002,
    restype=ResType.fast_conv, footype=FootType.gauss)

# [4] 시뮬레이션 함수
def Sim(data):
    # Update Objects from Vars
    Film.d     = v.f_d
    Film.sigma = v.f_sig
    Film.f     = sld_to_f(v.f_sld)  # 변환 적용

    SiO2.d     = v.s_d
    SiO2.sigma = v.s_sig
    SiO2.f     = sld_to_f(v.s_sld)  # 변환 적용

    inst.I0    = v.i0

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
        p_f_d = pars.append("v.f_d", model)
        p_f_d.min = max(1.0, self.init_d * 0.5); p_f_d.max = self.init_d * 1.5

        p_f_sig = pars.append("v.f_sig", model)
        p_f_sig.min = 0.0; p_f_sig.max = 30.0

        p_f_sld = pars.append("v.f_sld", model)
        p_f_sld.min = 0.1; p_f_sld.max = 100.0 # 넉넉하게

        # 2. SiO2
        p_s_d = pars.append("v.s_d", model)
        p_s_d.value = 15.0
        p_s_d.min = 5.0
        p_s_d.max = 50.0

        p_s_sig = pars.append("v.s_sig", model)
        p_s_sig.value = 3.0
        p_s_sig.min = 0.0
        p_s_sig.max = 10.0

        p_s_sld = pars.append("v.s_sld", model)
        p_s_sld.value = 18.8
        p_s_sld.min = 10.0
        p_s_sld.max = 25.0

        # 3. Instrument
        p_i0 = pars.append("v.i0", model)
        p_i0.value = 1.0
        p_i0.min = 0.1
        p_i0.max = 100.0

        model.parameters = pars

        # --- Step 1: I0 Fitting ---
        if verbose:
            print("\n[GenX] Step 1: Fitting I0 (Linear)...")
        model.set_fom_func(fom_funcs.diff)

        for p in pars:
            p.fit = False
        p_i0.fit = True

        res1 = model.bumps_fit(method="amoeba", steps=200)
        model.bumps_update_parameters(res1)
        if verbose:
            print(f"  -> I0 Fitted: {p_i0.value:.4f}")

        # --- Step 2: Full Fitting ---
        if verbose:
            print("[GenX] Step 2: Fitting All Params (Log)...")
        model.set_fom_func(fom_funcs.log)

        for p in pars:
            p.fit = True

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

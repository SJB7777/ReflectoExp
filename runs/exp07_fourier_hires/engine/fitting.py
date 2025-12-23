import time
from dataclasses import dataclass

import numpy as np
from engine.script_builder import GenXScriptBuilder
from genx import fom_funcs
from genx.data import DataList, DataSet
from genx.model import Model
from genx.parameters import Parameters


@dataclass
class XRRConfig:
    """범용 XRR 분석을 위한 물리학적 파라미터 설정"""
    wavelength: float = 1.5406
    beam_width: float = 0.1
    sample_len_init: float = 10.0

    steps_instrument: int = 300
    steps_thickness: int = 800
    steps_structure: int = 1000
    steps_fine: int = 1200
    pop_size: int = 40

class GenXFitter:
    def __init__(self, q: np.ndarray, R: np.ndarray, nn_params, config: XRRConfig = None):
        self.q = q
        max_r = np.nanmax(R) if np.nanmax(R) > 0 else 1.0
        self.R = np.nan_to_num(R / max_r, nan=1e-12, posinf=1e-12)
        self.nn_params = nn_params
        self.config = config if config else XRRConfig()
        self.model = self._initialize_genx_model()
        self.pars_map = {}

    def _initialize_genx_model(self) -> Model:
        ds = DataSet(name="Reflecto_Ultimate_Fitter")
        ds.x_raw, ds.y_raw = self.q, self.R
        ds.error_raw = np.maximum(self.R * 0.1, 1e-9)
        ds.run_command()

        model = Model()
        model.data = DataList([ds])

        init_vals = {
            'f_d': float(self.nn_params.thickness),
            'f_sig': float(self.nn_params.roughness),
            'f_sld': float(self.nn_params.sld),
            's_len': self.config.sample_len_init,
            'beam_w': self.config.beam_width,
            'i0': 1.0,
            'ibkg': 1e-7
        }
        builder = GenXScriptBuilder()
        model.set_script(builder.build(init_vals))
        model.compile_script()
        return model

    def _setup_parameters(self):
        pars = Parameters()
        model, cfg = self.model, self.config
        d_ai = float(self.nn_params.thickness)
        sld_ai = float(self.nn_params.sld)
        sig_ai = float(self.nn_params.roughness)
        def add_par(name, val, min_v, max_v, fit=True):
            p = pars.append(name, model)
            p.min, p.max = min_v, max_v
            p.value = np.clip(val, min_v, max_v)
            p.fit = fit
            self.pars_map[name.replace("v.set_", "set_").replace("v.", "")] = p
            return p

        # [1] Main Film: ±150A 윈도우 전략
        add_par("v.set_f_d", d_ai, max(10.0, d_ai*0.9), d_ai*1.1)
        add_par("v.set_f_sig", sig_ai, max(0, sig_ai*0.9), sig_ai*1.1)
        add_par("v.set_f_sld", sld_ai, max(5.0, sld_ai*0.9), min(150.0, sld_ai*1.1))

        # [2] Substrate Oxide (SiO2): 저각 위상 조절의 핵심
        add_par("v.set_s_d", 15.0, 0.01, 55.0)
        add_par("v.set_s_sig", 3.0, 0.5, 15.0)
        add_par("v.set_s_sld", 16, 5.0, 20.0)

        # [3] Instrument: I0 범위를 넓혀 강도 보상 허용
        add_par("v.set_i0", 1.0, 0.1, 20.0)
        add_par("v.set_ibkg", 1e-7, 1e-10, 1e-4)

        # [교정] L 파라미터가 40mm를 넘으면 1인치 시료에서 오차가 발생함
        # 실제 시료 크기가 작다면 max_v를 25.0 정도로 줄이는 것이 저각에 유리합니다.
        add_par("v.set_s_len", cfg.sample_len_init, 5.0, 45.0)

        model.parameters = pars

    def run(self, verbose=True):
        self._setup_parameters()
        model, cfg = self.model, self.config
        if verbose: self._print_header()

        # [Step 1] INITIAL ANCHORING: I0만 맞추기
        # start = time.time()
        # model.set_fom_func(fom_funcs.R1)
        # self._set_active_params(["set_i0", "set_ibkg"])
        # model.bumps_update_parameters(model.bumps_fit(method="de", steps=200, pop=15))
        # if verbose: self._print_status("ANCHORING", time.time() - start)

        # [Step 2] MICRO-REFINEMENT: 아주 좁은 범위에서만 허용
        # start = time.time()
        # model.set_fom_func(fom_funcs.logR2)
        # self._set_active_params(["set_f_d", "set_f_sld", "set_f_sig", "set_i0"])
        # model.bumps_update_parameters(model.bumps_fit(method="de", steps=400, pop=cfg.pop_size))
        # if verbose: self._print_status("REFINEMENT", time.time() - start)

        # [Step 3] SUBSTRATE MATCHING: 산화막 위상 조정
        start = time.time()
        model.set_fom_func(fom_funcs.logR2)
        self._set_active_params(["set_s_d", "set_s_sig", "set_s_d", "set_i0"])
        model.bumps_update_parameters(model.bumps_fit(method="de", steps=400, pop=cfg.pop_size))
        if verbose:
            self._print_status("SUBSTRATE", time.time() - start)

        # [Step 4] FINAL POLISHING: 전체 미세 보정
        start = time.time()
        self._set_active_params(["set_f_d", "set_f_sld", "set_f_sig", "set_s_d", "set_s_sld", "set_i0", "set_ibkg", "set_s_len"])
        model.bumps_update_parameters(model.bumps_fit(method="de", steps=600, pop=cfg.pop_size))

        if verbose:
            self._print_status("FINAL", time.time() - start)
            print("="*185 + "\n")

        return self._collect_results()

    def _set_active_params(self, active_names: list[str]):
        for p in self.model.parameters:
            p.fit = False
        for name in active_names:
            if name in self.pars_map:
                self.pars_map[name].fit = True

    def _print_header(self):
        print("\n" + "="*185)
        print(f"{'Fitting Step':^15} | {'FOM (logR1)':^20} | {'Thick':^7} | {'Rough':^5} | {'SLD':^5} | {'SiO2_d':^6} | {'SiO2_s':^6} | {'SiO2_sld':^8} | {'I0':^5} | {'L':^4} | {'Bkg':^5} | {'Time':^7}")
        print("-" * 185)

    def _print_status(self, step_name: str, elapsed: float):
        self.model.evaluate_sim_func()
        p = self.pars_map
        fom_val = self.model.fom
        log_fom = np.log10(fom_val) if fom_val > 0 else -np.inf
        b_val = np.log10(max(1e-12, p['set_ibkg'].value))
        fom_str = f"{fom_val:.4e} ({log_fom:5.2f})"

        print(f"   >> [{step_name:^12}] {fom_str:^20} | "
              f"{p['set_f_d'].value:7.2f} | {p['set_f_sig'].value:5.2f} | {p['set_f_sld'].value:5.2f} | "
              f"{p['set_s_d'].value:6.2f} | {p['set_s_sig'].value:6.2f} | {p['set_s_sld'].value:8.2f} | "
              f"{p['set_i0'].value:5.2f} | {p['set_s_len'].value:4.1f} | {b_val:.1f} | {elapsed:6.2f}s")

    def _collect_results(self) -> dict:
        self.model.evaluate_sim_func()
        results = {name: p.value for name, p in self.pars_map.items()}
        results.update({'R_sim': self.model.data[0].y_sim, 'fom': self.model.fom})
        return results

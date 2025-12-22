import numpy as np
from dataclasses import dataclass
from genx import fom_funcs
from genx.data import DataList, DataSet
from genx.model import Model
from genx.parameters import Parameters
from engine.script_builder import GenXScriptBuilder

@dataclass
class XRRConfig:
    """XRR 분석을 위한 물리학적 범용 하이퍼파라미터 설정"""
    wavelength: float = 1.54
    beam_width: float = 0.1
    sample_len_init: float = 10.0
    
    # [수정] 물리학적 절대 탐색 범위 (Physical Absolute Bounds)
    thickness_bounds: tuple[float, float] = (10.0, 1200.0)
    sld_bounds: tuple[float, float] = (5.0, 100.0)
    # [핵심] Roughness 상한을 15A로 강제하여 "진동 포기" 현상 방지
    roughness_bounds: tuple[float, float] = (0.0, 15.0) 
    
    # 최적화 강도
    steps_instrument: int = 200
    steps_structure: int = 800
    steps_fine: int = 1000
    pop_size: int = 30


class GenXFitter:
    def __init__(self, q: np.ndarray, R: np.ndarray, nn_params, config: XRRConfig = None):
        self.q = q
        self.R = R / R.max()
        self.nn_params = nn_params
        self.config = config if config else XRRConfig()
        
        self.model = self._initialize_genx_model()
        self.pars_map = {}

    def _initialize_genx_model(self) -> Model:
        ds = DataSet(name="XRR_Data")
        ds.x_raw, ds.y_raw = self.q, self.R
        ds.error_raw = np.maximum(self.R * 0.1, 1e-9)
        ds.run_command()

        model = Model()
        model.data = DataList([ds])
        
        # [조치] AI 예측값을 물리적 범위 내로 Clipping
        init_vals = {
            'f_d': np.clip(float(self.nn_params.thickness), *self.config.thickness_bounds),
            'f_sig': np.clip(float(self.nn_params.roughness), *self.config.roughness_bounds),
            'f_sld': np.clip(float(self.nn_params.sld), *self.config.sld_bounds), 
            's_len': self.config.sample_len_init,
            'beam_w': self.config.beam_width
        }
        
        builder = GenXScriptBuilder()
        model.set_script(builder.build(init_vals))
        model.compile_script()
        return model

    def _setup_parameters(self):
        pars = Parameters()
        model = self.model
        cfg = self.config
        
        def add_par(name, val=None, min_v=None, max_v=None, fit=False):
            p = pars.append(name, model)
            if val is not None: p.value = val
            if min_v is not None: p.min = min_v
            if max_v is not None: p.max = max_v
            p.fit = fit
            clean_name = name.replace("v.set_", "set_").replace("v.", "")
            self.pars_map[clean_name] = p 
            return p

        # Film Layer: 절대 범위 적용
        add_par("v.set_f_d", min_v=cfg.thickness_bounds[0], max_v=cfg.thickness_bounds[1])
        add_par("v.set_f_sig", min_v=cfg.roughness_bounds[0], max_v=cfg.roughness_bounds[1])
        add_par("v.set_f_sld", min_v=cfg.sld_bounds[0], max_v=cfg.sld_bounds[1])

        # SiO2 & Substrate
        add_par("v.set_s_d",   val=15.0, min_v=5.0, max_v=60.0)
        add_par("v.set_s_sig", val=3.0,  min_v=1.0, max_v=15.0)
        add_par("v.set_s_sld", val=18.8, min_v=10.0, max_v=25.0) 

        # Instrument
        add_par("v.set_i0",    val=1.0, min_v=0.1, max_v=10.0)
        add_par("v.set_s_len", val=cfg.sample_len_init, min_v=2.0, max_v=100.0)
        add_par("v.set_beam_w", val=cfg.beam_width, fit=False)

        model.parameters = pars

    def _set_active_params(self, active_names: list[str]):
        for p in self.model.parameters:
            p.fit = False
        for name in active_names:
            clean = name.replace("v.set_", "set_").replace("v.", "")
            if clean in self.pars_map:
                self.pars_map[clean].fit = True

    def run(self, verbose=True):
        self._setup_parameters()
        model = self.model
        cfg = self.config

        # [NEW] Step 0: Harmonic Check (배수 두께 에러 방지)
        # AI 예측값이 d일 때, 0.5d와 2d 지점의 FOM을 광속으로 계산하여 가장 유망한 곳에서 시작합니다.
        d_guess = float(self.nn_params.thickness)
        potential_ds = [d_guess * 0.5, d_guess, d_guess * 2.0]
        best_d = d_guess
        min_fom = 1e9

        if verbose: print(f"[GenX] Harmonic Pre-check (Guess: {d_guess:.1f}Å)...")
        model.set_fom_func(fom_funcs.log) # 주기성은 Log 스케일에서 명확함
        for td in potential_ds:
            if cfg.thickness_bounds[0] <= td <= cfg.thickness_bounds[1]:
                self.pars_map['set_f_d'].value = td
                model.evaluate_sim_func()
                if model.fom < min_fom:
                    min_fom = model.fom
                    best_d = td
        
        self.pars_map['set_f_d'].value = best_d
        if verbose: print(f"      > Best starting thickness selected: {best_d:.1f}Å")

        # Step 1: Instrument (Intensity & Footprint)
        if verbose: print("\n[GenX] Step 1: Instrumental Fitting...")
        model.set_fom_func(fom_funcs.diff)
        self._set_active_params(["set_i0", "set_s_len"])
        res1 = model.bumps_fit(method="de", steps=cfg.steps_instrument, pop=15)
        model.bumps_update_parameters(res1)

        # Step 2: Structure (Global Search)
        if verbose: print("[GenX] Step 2: Global Structure Search (Log FOM)...")
        model.set_fom_func(fom_funcs.log)
        self._set_active_params(["set_f_d", "set_f_sld", "set_s_d"])
        res2 = model.bumps_fit(method="de", steps=cfg.steps_structure, pop=cfg.pop_size)
        model.bumps_update_parameters(res2)

        # Step 3: Fine Tuning
        if verbose: print("[GenX] Step 3: Final Refinement...")
        self._set_active_params([
            "set_f_d", "set_f_sld", "set_f_sig",
            "set_s_d", "set_s_sld", "set_s_sig", "set_i0"
        ])
        res3 = model.bumps_fit(method="de", steps=cfg.steps_fine, pop=cfg.pop_size)
        model.bumps_update_parameters(res3)

        model.evaluate_sim_func()
        return self._collect_results()

    def _collect_results(self) -> dict:
        results = {name: p.value for name, p in self.pars_map.items()}
        results['R_sim'] = self.model.data[0].y_sim
        results['fom'] = self.model.fom
        return results
from dataclasses import dataclass

import numpy as np
from engine.script_builder import GenXScriptBuilder
from genx import fom_funcs
from genx.data import DataList, DataSet
from genx.model import Model
from genx.parameters import Parameters


@dataclass
class XRRConfig:
<<<<<<< HEAD
    """XRR 분석을 위한 범용 하이퍼파라미터 설정"""

    # 1. 기기 기본값
=======
    """XRR 분석을 위한 물리학적 범용 하이퍼파라미터 설정"""
>>>>>>> b0d3c75701673a03fd014559260eecd1e7185489
    wavelength: float = 1.54
    beam_width: float = 0.1
    sample_len_init: float = 10.0

<<<<<<< HEAD
    # 2. 범용 탐색 범위 (Universal Bounds)
    # 어떤 물질이 들어와도 커버 가능한 물리적 한계치 설정

    # [두께] AI 예측값 대비 비율 (예: 0.2배 ~ 5.0배까지 탐색)
    thickness_search_ratio: tuple[float, float] = (0.2, 5.0)

    # [SLD] 절대 범위 (단위: x10^-6 A^-2)
    # 0 (진공) ~ 150 (금/백금 등 고밀도 금속)
    sld_absolute_bounds: tuple[float, float] = (5.0, 150.0)

    # [거칠기] 절대 범위 (Angstrom)
    # 0 (완벽한 표면) ~ 50 (매우 거친 표면)
    roughness_absolute_bounds: tuple[float, float] = (0.0, 50.0)

    # 3. 최적화 강도 (범위가 넓어진 만큼 steps를 조금 늘리는 게 안전함)
=======
    # [수정] 물리학적 절대 탐색 범위 (Physical Absolute Bounds)
    thickness_bounds: tuple[float, float] = (10.0, 1200.0)
    sld_bounds: tuple[float, float] = (5.0, 100.0)
    # [핵심] Roughness 상한을 15A로 강제하여 "진동 포기" 현상 방지
    roughness_bounds: tuple[float, float] = (0.0, 15.0)

    # 최적화 강도
>>>>>>> b0d3c75701673a03fd014559260eecd1e7185489
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

<<<<<<< HEAD
        # 초기값 설정 (AI 예측값 사용)
        # 단, 초기값이 너무 터무니없으면 안전한 값(Safe Default)으로 보정
        ai_sld = float(self.nn_params.sld)

        init_vals = {
            'f_d': float(self.nn_params.thickness),
            'f_sig': float(self.nn_params.roughness),
            'f_sld': ai_sld,
=======
        # [조치] AI 예측값을 물리적 범위 내로 Clipping
        init_vals = {
            'f_d': np.clip(float(self.nn_params.thickness), *self.config.thickness_bounds),
            'f_sig': np.clip(float(self.nn_params.roughness), *self.config.roughness_bounds),
            'f_sld': np.clip(float(self.nn_params.sld), *self.config.sld_bounds),
>>>>>>> b0d3c75701673a03fd014559260eecd1e7185489
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

<<<<<<< HEAD
        # AI 예측값 가져오기
        guess_d = float(self.nn_params.thickness)

        # 헬퍼 함수
=======
>>>>>>> b0d3c75701673a03fd014559260eecd1e7185489
        def add_par(name, val=None, min_v=None, max_v=None, fit=False):
            p = pars.append(name, model)
            if val is not None: p.value = val
            if min_v is not None: p.min = min_v
            if max_v is not None: p.max = max_v
            p.fit = fit
            clean_name = name.replace("v.set_", "set_").replace("v.", "")
            self.pars_map[clean_name] = p
            return p

<<<<<<< HEAD
        # ==========================================================
        # 1. Film Layer (범용 설정 적용)
        # ==========================================================
        # 두께: AI 예측값 기준 비율 범위 (예: 0.2 ~ 5.0배)
        add_par("v.set_f_d",
                min_v=max(5.0, guess_d * cfg.thickness_search_ratio[0]),
                max_v=guess_d * cfg.thickness_search_ratio[1])

        # 거칠기: 범용 절대 범위 (0 ~ 50 A)
        add_par("v.set_f_sig",
                min_v=cfg.roughness_absolute_bounds[0],
                max_v=cfg.roughness_absolute_bounds[1])

        # SLD: 범용 절대 범위 (5 ~ 150) -> 물질 몰라도 다 커버됨
        add_par("v.set_f_sld",
                min_v=cfg.sld_absolute_bounds[0],
                max_v=cfg.sld_absolute_bounds[1])
=======
        # Film Layer: 절대 범위 적용
        add_par("v.set_f_d", min_v=cfg.thickness_bounds[0], max_v=cfg.thickness_bounds[1])
        add_par("v.set_f_sig", min_v=cfg.roughness_bounds[0], max_v=cfg.roughness_bounds[1])
        add_par("v.set_f_sld", min_v=cfg.sld_bounds[0], max_v=cfg.sld_bounds[1])
>>>>>>> b0d3c75701673a03fd014559260eecd1e7185489

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
<<<<<<< HEAD

        # 켜기
=======
>>>>>>> b0d3c75701673a03fd014559260eecd1e7185489
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
<<<<<<< HEAD

        # 핵심: 두께, SLD, 기판 두께를 동시에 넓은 범위에서 탐색
        self._set_active_params(["set_f_d", "set_f_sld", "set_s_d"])

        res2 = model.bumps_fit(method="de", steps=cfg.steps_structure, pop=25, tol=0.005)
=======
        self._set_active_params(["set_f_d", "set_f_sld", "set_s_d"])
        res2 = model.bumps_fit(method="de", steps=cfg.steps_structure, pop=cfg.pop_size)
>>>>>>> b0d3c75701673a03fd014559260eecd1e7185489
        model.bumps_update_parameters(res2)

        # Step 3: Fine Tuning
        if verbose: print("[GenX] Step 3: Final Refinement...")
        self._set_active_params([
            "set_f_d", "set_f_sld", "set_f_sig",
            "set_s_d", "set_s_sld", "set_s_sig", "set_i0"
        ])
<<<<<<< HEAD

        res3 = model.bumps_fit(method="de", steps=cfg.steps_fine, pop=25, tol=0.001)
=======
        res3 = model.bumps_fit(method="de", steps=cfg.steps_fine, pop=cfg.pop_size)
>>>>>>> b0d3c75701673a03fd014559260eecd1e7185489
        model.bumps_update_parameters(res3)

        model.evaluate_sim_func()
        return self._collect_results()

    def _collect_results(self) -> dict:
        results = {name: p.value for name, p in self.pars_map.items()}
        results['R_sim'] = self.model.data[0].y_sim
        results['fom'] = self.model.fom
        return results

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import h5py

# 모듈 경로 설정
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from config import CONFIG
from dataset import XRRPreprocessor
from reflecto_exp.math_utils import powerspace

def inspect_statistics(stats_file):
    print("\n=== [1] Statistics Check (stats.pt) ===")
    if not stats_file.exists():
        print(f"❌ Stats file not found: {stats_file}")
        return None, None

    stats = torch.load(stats_file, map_location='cpu')
    
    # Check Param Stats
    p_mean = stats['param_mean'].numpy()
    p_std = stats['param_std'].numpy()
    
    print(f"Parameter Mean: {p_mean}")
    print(f"Parameter Std : {p_std}")
    
    # Check if values are reasonable
    if np.any(np.isnan(p_mean)) or np.any(p_std == 0):
        print("⚠️ WARNING: NaN or Zero Std detected in parameters!")
    else:
        print("✅ Parameter stats look valid.")
        
    return p_mean, p_std

def inspect_preprocessing(h5_file, stats_file, sample_idx=0):
    print(f"\n=== [2] Preprocessing Check (Sample {sample_idx}) ===")
    
    # 1. Load Raw Data
    with h5py.File(h5_file, 'r') as hf:
        q_raw = hf['q'][:]
        R_raw = hf['R'][sample_idx]
        
        # Load params
        d = hf['thickness'][sample_idx]
        sig = hf['roughness'][sample_idx]
        sld = hf['sld'][sample_idx]
        print(f"Target Params: d={d}, sig={sig}, sld={sld}")

    # 2. Setup Preprocessor
    # Config에서 설정 가져오기
    sim_conf = CONFIG['simulation']
    
    # Power Grid 생성 (학습 때와 동일해야 함)
    target_qs = powerspace(
        sim_conf['q_min'], 
        sim_conf['q_max'], 
        sim_conf['q_points'], 
        power=sim_conf['power']
    ).astype(np.float32)
    
    processor = XRRPreprocessor(
        qs=target_qs,
        stats_file=stats_file
    )
    
    # 3. Step-by-Step Preprocessing & Visualization
    
    # Step A: Log & Normalize (Max=1)
    R_max = np.max(R_raw)
    R_norm = R_raw / (R_max + 1e-15)
    R_log = np.log10(np.maximum(R_norm, 1e-15))
    
    # Step B: Interpolation (Linear q -> Power q)
    # Padding Check: -15.0 vs 0.0
    padding_val = -15.0  # 우리가 원하는 바닥값
    R_interp = np.interp(target_qs, q_raw, R_log, left=padding_val, right=padding_val)
    
    # Step C: Masking
    mask = (target_qs >= np.min(q_raw)) & (target_qs <= np.max(q_raw))
    
    # Step D: Final Tensor (What model sees)
    input_tensor = processor.process_input(q_raw, R_raw)
    model_input_R = input_tensor[0].numpy()
    model_input_M = input_tensor[1].numpy()
    
    # --- Visualization ---
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Raw Linear Data
    plt.subplot(2, 2, 1)
    plt.plot(q_raw, np.log10(np.maximum(R_raw, 1e-15)), 'k-', label='Raw Data (Linear Grid)')
    plt.title(f"Raw Input (Linear q)\nRange: {q_raw.min():.2f} ~ {q_raw.max():.2f}")
    plt.xlabel("q")
    plt.ylabel("log10(R)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 2: Model Input (Resampled)
    plt.subplot(2, 2, 2)
    plt.plot(target_qs, model_input_R, 'b.-', label='Processed Input (Power Grid)')
    plt.plot(target_qs, model_input_M * np.min(model_input_R), 'r-', alpha=0.3, label='Mask')
    
    plt.title(f"Model Input (Resampled)\nRange: {target_qs.min():.2f} ~ {target_qs.max():.2f}")
    plt.xlabel("q (Power Space)")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Check for "The Cliff" (Zero Padding Issue)
    # 마스크가 0인 구간(데이터 밖)의 값이 -15 근처인지, 0인지 확인
    out_of_range_indices = np.where(model_input_M == 0)[0]
    if len(out_of_range_indices) > 0:
        bg_val = np.mean(model_input_R[out_of_range_indices])
        plt.text(0.5, 0.5, f"Background Level: {bg_val:.2f}", 
                 transform=plt.gca().transAxes, fontsize=12, color='red', fontweight='bold')
        if bg_val > -5.0:
             print(f"🚨 CRITICAL WARNING: Background level is too high ({bg_val:.2f})! It should be around -15.0.")
             print("   -> Fix 'dataset.py' np.interp left/right arguments.")

    # Plot 3: Grid Density Check
    plt.subplot(2, 2, 3)
    plt.plot(target_qs[:-1], np.diff(target_qs), 'g.-')
    plt.title(f"Grid Spacing (dq)\nPower={sim_conf['power']}")
    plt.xlabel("q")
    plt.ylabel("dq")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    base_path = Path(CONFIG["base_dir"]) / CONFIG["exp_name"]
    h5_path = base_path / "dataset.h5"
    stats_path = base_path / "stats.pt"
    
    inspect_statistics(stats_path)
    
    for i in [0, 100, 500]:

            inspect_preprocessing(h5_path, stats_path, sample_idx=i)

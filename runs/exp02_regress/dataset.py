# dataset.py
class DatasetH5(Dataset):
    def __init__(self, ..., normalize_targets=True):
        # 타겟 값을 0~1 범위로 정규화
        if normalize_targets:
            self.tg_min = [thick.min(), rough.min(), sld.min()]
            self.tg_max = [thick.max(), rough.max(), sld.max()]
    
    def __getitem__(self, idx):
        # ...
        # 정규화된 타겟: 0~1 범위
        thick_norm = (thick - self.tg_min[0]) / (self.tg_max[0] - self.tg_min[0])
        
        return refl_t, torch.tensor([thick_norm, rough_norm, sld_norm], dtype=torch.float32)
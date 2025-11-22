import re
from datetime import datetime
from pathlib import Path

import pandas as pd
from numpy import ndarray

def append_timestamp(path: Path | str) -> Path:
    """
    Append a timestamp to a path (file or directory) to make it unique.
    The exact naming format is not fixed.
    """
    path = Path(path)
    new_name = f"{path.name}_{datetime.now():%Y%m%d_%H%M%S}"

    if path.suffix:
        return path.parent / (new_name + path.suffix)
    return path.parent / new_name


def next_unique_file(path: Path | str) -> Path:
    """
    Generate a unique file path.
    - If no file exists with the same name, return the original path.
    - If files exist, append (n) where n = max existing number + 1.

    Example:
        file.txt       -> file.txt        (if doesn't exist)
        file.txt       -> file(1).txt     (if file.txt exists)
        file(1).txt    -> file(2).txt     (if file(1).txt exists)
    """
    path = Path(path)
    parent = path.parent
    stem = path.stem
    suffix = path.suffix

    # Regular expression: Find '(number)' end of the stem.
    pattern = re.compile(rf"^{re.escape(stem)}(?:\((\d+)\))?$")

    # Find same patterns
    existing = [f for f in parent.glob(f"{stem}*{suffix}") if pattern.match(f.stem)]

    if not existing:
        return path

    max_n = 0
    for f in existing:
        m = pattern.match(f.stem)
        if m and m.group(1):
            n = int(m.group(1))
            if n > max_n:
                max_n = n

    return parent / f"{stem}({max_n + 1}){suffix}"

def load_xrr_dat(file: Path | str) -> tuple[ndarray, ndarray]:
    """
    Load 2 column .dat file.
    Any non-numeric values (including '(1.00)') are converted to 0.0.
    """
    # 1. 모든 데이터를 일단 문자열로 읽어옵니다 (파싱 에러 방지)
    df = pd.read_csv(file, header=None, sep=r"\s+", names=["tth", "R"], dtype=str)

    # 2. 숫자로 변환 시도
    # errors='coerce': 숫자가 아닌 것((1.00) 포함)은 전부 NaN(Not a Number)으로 변환
    tth_series = pd.to_numeric(df["tth"], errors='coerce')
    R_series = pd.to_numeric(df["R"], errors='coerce')

    # 3. NaN을 0.0으로 채우고 numpy 배열로 변환
    tth = tth_series.fillna(0.0).to_numpy()
    R = R_series.fillna(0.0).to_numpy()

    return tth, R

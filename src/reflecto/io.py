import re
from datetime import datetime
from pathlib import Path


def append_timestamp(path: str | Path) -> Path:
    """
    Append a timestamp to a path (file or directory) to make it unique.
    The exact naming format is not fixed.
    """
    path = Path(path)
    new_name = f"{path.name}_{datetime.now():%Y%m%d_%H%M%S}"

    if path.suffix:
        return path.parent / new_name + path.suffix
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

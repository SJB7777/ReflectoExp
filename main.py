import sys
import platform
import subprocess
import argparse
from pathlib import Path

def get_venv_python() -> Path:
    """
    현재 프로젝트의 .venv 내 Python 실행 파일 경로를 반환합니다.
    """
    base_path = Path.cwd()  # 현재 작업 경로

    # OS에 따른 실행 파일 경로 분기
    if platform.system() == "Windows":
        venv_python = base_path / ".venv" / "Scripts" / "python.exe"
    else:
        venv_python = base_path / ".venv" / "bin" / "python"

    if not venv_python.exists():
        print(f"[Error] 가상환경 Python을 찾을 수 없습니다.")
        print(f"  경로 확인: {venv_python}")
        sys.exit(1)

    return venv_python

def find_exp_folder(exp_num: int, runs_dir_path: Path = Path("./runs")) -> Path:
    """
    runs 폴더 내에서 exp{num} 또는 exp0{num}으로 시작하는 폴더를 찾습니다.
    """
    # runs 폴더 존재 확인
    if not runs_dir_path.exists():
        print(f"❌ [Error] '{runs_dir_path}' 디렉토리가 존재하지 않습니다.")
        sys.exit(1)

    # 검색할 접두사 패턴 (예: exp5_, exp05_)
    target_prefixes = [f"exp{exp_num}_", f"exp{exp_num:02d}_"]

    # iterdir()로 폴더 순회하며 찾기
    found_folder = None
    for item in runs_dir_path.iterdir():
        if item.is_dir():
            # 폴더 이름이 접두사 중 하나로 시작하는지 확인
            if any(item.name.startswith(prefix) for prefix in target_prefixes):
                found_folder = item
                break

    if found_folder is None:
        print(f"[Error] 실험 번호 {exp_num}번에 해당하는 폴더를 못 찾았습니다.")
        print(f"   탐색 위치: {runs_dir_path.resolve()}")
        sys.exit(1)

    return found_folder

def run_main(target_folder: Path):
    """
    찾은 폴더 내부의 main.py를 가상환경 Python으로 실행합니다.
    """
    venv_python = get_venv_python()
    target_script = target_folder / "main.py"

    if not target_script.exists():
        print(f"[Error] '{target_folder.name}' 폴더 안에 'main.py'가 없습니다.")
        sys.exit(1)

    print(f"[Start] Experiment: {target_folder.name}")
    print(f"   Path: {target_folder}")
    print("-" * 50)

    try:
        # subprocess에는 Path 객체를 문자열로 변환(str)해서 넘기는 것이 안전합니다.
        subprocess.run(
            [str(venv_python), "main.py"], 
            cwd=target_folder,  # 작업 디렉토리를 해당 실험 폴더로 변경
            check=True
        )
    except subprocess.CalledProcessError as e:
        print("-" * 50)
        print(f"💥 [Fail] 실행 중 오류 발생 (Exit code: {e.returncode})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run experiment based on ID')
    parser.add_argument('exp_num', type=int, help='Experiment number (e.g., 5)')

    try:
        args = parser.parse_args()
    except SystemExit:
        print("사용법: python main.py <실험번호>")
        sys.exit(1)

    # 1. 대상 폴더 찾기
    target_folder = find_exp_folder(args.exp_num)

    # 2. 실행
    run_main(target_folder)

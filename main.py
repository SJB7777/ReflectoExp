import os
import platform
import subprocess
import sys

if __name__ == "__main__":
    base_path = os.getcwd()

    if platform.system() == "Windows":
        # Windows일 경우: .venv/Scripts/python.exe
        venv_python = os.path.join(base_path, ".venv", "Scripts", "python.exe")
    else:
        # Mac/Linux일 경우: .venv/bin/python
        venv_python = os.path.join(base_path, ".venv", "bin", "python")

    # 가상환경 파일이 실제로 있는지 확인
    if not os.path.exists(venv_python):
        print(f"Error: Can not find virtual environment python.\nCheck directory: {venv_python}")
        sys.exit(1)

    print(f"Virtual python detected: {venv_python}")

    # ---------------------------------------------------------
    # 2. 실행할 타겟 스크립트 설정
    # ---------------------------------------------------------
    target_folder = "./runs/exp05_1layer_mask"
    target_script = "main.py"

    # ---------------------------------------------------------
    # 3. 실행 (subprocess)
    # ---------------------------------------------------------
    # cwd=target_folder : 실행 위치를 exp05 폴더 내부로 변경하여 실행
    # venv_python : 시스템 python이 아닌, 위에서 찾은 가상환경 python을 사용
    subprocess.run([venv_python, target_script], cwd=target_folder)
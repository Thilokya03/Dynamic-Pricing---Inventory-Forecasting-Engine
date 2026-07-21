from pathlib import Path
import pandas as pd
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts"

def get_scripts_name():
    script_files = [f for f in os.listdir(SCRIPT_DIR) if f.endswith(".py")]
    script_names = [os.path.splitext(f)[0] for f in script_files]
    return script_names

def run_script(script_name):
    script_path = SCRIPT_DIR / f"{script_name}.py"
    if not script_path.exists():
        print(f"Script '{script_name}' does not exist.")
        return

    print(f"Running script: {script_name}")
    os.system(f"python {script_path}")


if __name__ == "__main__":
    scripts = get_scripts_name()
    print("Run Available scripts:")
    for script in scripts:
        print(f"- {script}")
        run_script(script)
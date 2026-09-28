import subprocess
import sys
from pathlib import Path

from profilometer_comparison import __version__

RUN = Path(__file__).resolve().parents[1] / "run.py"


def test_run_script_works_from_another_folder(tmp_path):
    out = subprocess.run([sys.executable, str(RUN), "--version"], cwd=tmp_path, capture_output=True, text=True)
    assert out.returncode == 0 and __version__ in out.stdout


def test_run_script_menu_quits(tmp_path):
    out = subprocess.run([sys.executable, str(RUN)], cwd=tmp_path, input="q\n", capture_output=True, text=True)
    assert out.returncode == 0 and "Prepare" in out.stdout

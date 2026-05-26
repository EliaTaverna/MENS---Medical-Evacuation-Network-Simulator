"""
Canonical path constants for every notebook in this repo.

Bootstrap — paste as the first cell of any notebook:

    import sys
    from pathlib import Path
    _root = Path(".").resolve()
    while not (_root / "paths.py").exists():
        _root = _root.parent
    sys.path.insert(0, str(_root))
    import paths

Then use e.g.  paths.SINGLE_RUN_OUTPUT / FOLDER_NAME / "patient_log.xlsx"
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent

SINGLE_RUN_INPUT  = ROOT / "data" / "single_run" / "input"
SINGLE_RUN_OUTPUT = ROOT / "data" / "single_run" / "output"

MULTI_RUN_INPUT   = ROOT / "data" / "multi_run" / "input"
MULTI_RUN_OUTPUT  = ROOT / "data" / "multi_run" / "output"

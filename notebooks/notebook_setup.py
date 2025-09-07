# notebook_setup.py (save this in notebooks/)
import sys
from pathlib import Path

PROJECT_ROOT = Path(r"D:/0) Abhay/04) SRH University Study Docs/Advance Programming/Python Files/Case Study Files")


if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print(f"Project root set to: {PROJECT_ROOT}")
print("sys.path updated, project-level imports ready.")

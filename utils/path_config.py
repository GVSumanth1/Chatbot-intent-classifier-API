
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from pathlib import Path

PROJECT_ROOT = Path(r"D:/0) Abhay/04) SRH University Study Docs/Advance Programming/Python Files/Case Study Files")

def get_data_path(filename):
    return PROJECT_ROOT / "data" / filename

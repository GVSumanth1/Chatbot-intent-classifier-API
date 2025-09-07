"""
utils/data_saving.py

Reusable function for rolling/rotating DataFrame backups.
Backups are stored in /data/backup/ under your project root.
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path
from utils.path_config import get_data_path

def save_with_rolling_backup(
    df,
    base_filename,     # e.g. "rf_model_df_v", "knn_features_v"
    max_versions=3,
    backup_subdir="backup"
):
    """
    Saves DataFrame as a rolling backup, keeping up to max_versions,
    in /data/backup/. Overwrites oldest version each time.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        base_filename (str): The prefix for the file (no version/extension).
        max_versions (int): How many versions to keep (default 3).
        backup_subdir (str): Subfolder under data for backups (default "backup").
    """
    # Full backup path: /data/backup/
    backup_dir = get_data_path(backup_subdir)
    os.makedirs(backup_dir, exist_ok=True)

    # Create versioned filenames
    version_numbers = [i+1 for i in range(max_versions)]
    files = [os.path.join(backup_dir, f"{base_filename}{i}.csv") for i in version_numbers]

    # Find oldest (or missing) file to overwrite
    times = [os.path.getmtime(f) if os.path.exists(f) else 0 for f in files]
    idx_to_overwrite = times.index(min(times))
    file_to_save = files[idx_to_overwrite]

    df.to_csv(file_to_save, index=False)
    print(f"[save_with_rolling_backup] Saved to: {file_to_save}")

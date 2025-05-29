import os
import re

def remove_timestamp_from_filenames_recursive(root_dir):
    # Regex to match _YYYYMMDD_HHMMSS before the file extension
    pattern = re.compile(r"_(\d{8}_\d{6})(?=\.)")

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            old_path = os.path.join(dirpath, filename)
            new_name = pattern.sub("", filename)
            if new_name != filename:
                new_path = os.path.join(dirpath, new_name)
                print(f"Renaming: {old_path} -> {new_path}")
                os.rename(old_path, new_path)

if __name__ == "__main__":
    root_folder = "/folder/"  # Change to your directory path
    remove_timestamp_from_filenames_recursive(root_folder)
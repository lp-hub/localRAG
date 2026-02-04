import os
import shutil
import subprocess

ramdisk_root = os.getenv("RAMDISK_ROOT") # if you prefer, import from config.py

# ========== Mount RAM Disk ==========
def mount_ramdisk():
    script_path = os.path.join(os.path.dirname(__file__), 'mount_ramdisk.sh')
    print(f"Mounting ramdisk using script at: {script_path}")
    try:
        subprocess.run(['sudo', script_path], check=True)
        print("RAM disk mounted successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to mount ramdisk: {e}")
        exit(1)
# If you want to run it without sudo prompts, you can allow passwordless execution via /etc/sudoers:
# your_username ALL=(ALL) NOPASSWD: /full/path/to/mount_ramdisk.sh
# Then change the Python call to:
# subprocess.run(['sudo', '/full/path/to/mount_ramdisk.sh'], check=True)

# ========== Copy Specific Directories to RAM Disk ==========
def copy_to_ramdisk(env_vars, ramdisk_path=ramdisk_root):
    for var in env_vars:
        original_path = os.getenv(var)
        if original_path is None:
            print(f"[Warning] Environment variable {var} is not set; skipping.")
            continue

        if not os.path.exists(original_path):
            print(f"[SKIP] {var} does not point to a real path: {original_path}")
            continue

        base_name = os.path.basename(original_path.rstrip("/"))
        ram_path = os.path.join(ramdisk_path, base_name)

        print(f"Copying {var} from {original_path} to RAM disk at {ram_path}...")
        # If destination exists, remove it first to avoid copytree errors
        try:

            if os.path.exists(ram_path):
                print(f"RAM path {ram_path} exists, removing before copy...")
                shutil.rmtree(ram_path)

            shutil.copytree(original_path, ram_path)
                        
            ram_var = "RAM_" + var # Set separate RAM var
            os.environ[ram_var] = ram_path # Override environment variable to RAM disk => RAM_
            print(f"{ram_var} set to {ram_path}")
            print(f"{var} copied to RAM disk successfully.")
        except Exception as e:
            print(f"Failed to copy {original_path} to RAM disk: {e}")

# ========== Fallback to HDD if RAM Disk Fails ==========
def safe_load(path_var, fallback_env):
    path = os.getenv(path_var)
    print(f"Trying to load from {path_var}: {path}")
    if path and os.path.exists(path):
        print(f"Successfully found {path_var} at: {path}")
        return path
    else:
        print(f"[Warning] {path_var} not found or path does not exist: {path}")
        fallback_path = os.getenv(fallback_env)
        print(f"Trying fallback path {fallback_env}: {fallback_path}")
        if fallback_path and os.path.exists(fallback_path):
            print(f"Successfully found fallback {fallback_env} at: {fallback_path}")
            return fallback_path
        else:
            print(f"[Error] Neither {path_var} nor fallback {fallback_env} paths exist.")
            return None
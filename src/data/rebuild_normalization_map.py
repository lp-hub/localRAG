def rebuild_normalization_map():
    ensure_normalization_json(force=True)
    print(f"Normalization map created at {JSON_PATH}")
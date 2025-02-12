# modules/utils/config_loader.py
import json
import os

def load_config(config_file="config.json"):
    if os.path.exists(config_file):
        with open(config_file, "r") as f:
            return json.load(f)
    else:
        return {}

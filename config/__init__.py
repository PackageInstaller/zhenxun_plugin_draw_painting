import os
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))


class PathConfig:
    PLUGIN_DIR = ROOT_DIR
    MODEL_DIR = os.path.join(PLUGIN_DIR, "model")
    FONT_FOLDER = os.path.join(PLUGIN_DIR, "font")
    FONT_PATH = os.path.join(FONT_FOLDER, "Sarasa-Regular.ttc")
    HUSBANDS_IMAGES_FOLDER = os.path.join(PLUGIN_DIR, "husbands")
    WIVES_IMAGES_FOLDER = os.path.join(PLUGIN_DIR, "wives")
    DROP_FOLDER = os.path.join(PLUGIN_DIR, "drop")

class DeviceConfig:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

paths = PathConfig()
device = DeviceConfig()

def ensure_directories():
    directories = [
        paths.MODEL_DIR,
        paths.FONT_FOLDER,
        paths.HUSBANDS_IMAGES_FOLDER,
        paths.WIVES_IMAGES_FOLDER,
        paths.DROP_FOLDER
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
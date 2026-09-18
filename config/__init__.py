import os

import torch

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))


class PathConfig:
    PLUGIN_DIR = ROOT_DIR
    DATA_DIR = os.path.join(PLUGIN_DIR, "data")
    MODEL_DIR = os.path.join(PLUGIN_DIR, "model")
    IMAGE_FEATURES_DB = os.path.join(DATA_DIR, "image_features.db")
    FONT_FOLDER = os.path.join(PLUGIN_DIR, "font")
    FONT_PATH = os.path.join(FONT_FOLDER, "STSONG.TTF")
    HUSBANDS_IMAGES_FOLDER = os.path.join(PLUGIN_DIR, "husbands")
    WIVES_IMAGES_FOLDER = os.path.join(PLUGIN_DIR, "wives")
    OTHERS_IMAGES_FOLDER = os.path.join(PLUGIN_DIR, "others")
    GAME_ALIASES_PATH = os.path.join(PLUGIN_DIR, "utils", "game_aliases.yaml")


class DeviceConfig:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


paths = PathConfig()
device = DeviceConfig()


def ensure_directories():
    directories = [
        paths.DATA_DIR,
        paths.MODEL_DIR,
        paths.FONT_FOLDER,
        paths.HUSBANDS_IMAGES_FOLDER,
        paths.WIVES_IMAGES_FOLDER,
        paths.OTHERS_IMAGES_FOLDER,
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)

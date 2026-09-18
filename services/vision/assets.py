from dataclasses import dataclass
from pathlib import Path

from ...config import paths


@dataclass(frozen=True)
class ModelAsset:
    repo: str
    revision: str
    folder: str
    filename: str
    size: int
    sha256: str

    @property
    def path(self) -> Path:
        return Path(paths.MODEL_DIR) / self.folder / self.filename


WD_REVISION = "627aef95638667ddcaa3ac8ae625e88ea5b02f51"
CAMIE_REVISION = "7d40c1b85b86ab4f607b2caf26b1b50c99db743e"
PERSON_REVISION = "e39c744c22432ad01f91dd254fe2b02c8d878b8c"
WD_MODEL = ModelAsset(
    "SmilingWolf/wd-swinv2-tagger-v3",
    WD_REVISION,
    "wd-swinv2-tagger-v3",
    "model.onnx",
    467_460_978,
    "e6774bff34d43bd49f75a47db4ef217dce701c9847b546523eb85ff6dbba1db1",
)
WD_TAGS = ModelAsset(
    WD_MODEL.repo,
    WD_REVISION,
    WD_MODEL.folder,
    "selected_tags.csv",
    308_468,
    "298633d94d0031d2081c0893f29c82eab7f0df00b08483ba8f29d1e979441217",
)
CAMIE_MODEL = ModelAsset(
    "Camais03/camie-tagger-v2",
    CAMIE_REVISION,
    "camie-tagger-v2",
    "camie-tagger-v2.onnx",
    788_983_561,
    "ab0aaf253e3d546090001bec9bebc776c354ab6800f442ab9167af87b4a953ac",
)
CAMIE_TAGS = ModelAsset(
    CAMIE_MODEL.repo,
    CAMIE_REVISION,
    CAMIE_MODEL.folder,
    "camie-tagger-v2-metadata.json",
    7_771_946,
    "de9f962eb0fd86b7e30d0af4e8c7990205200d70e955d8ecae60f87d14eae66b",
)
PERSON_MODEL = ModelAsset(
    "deepghs/anime_person_detection",
    PERSON_REVISION,
    "anime-person",
    "person_detect_v1.3_s/model.onnx",
    44_583_231,
    "6da88929438cd442e31e45ff4f934dd2d7eb9cf7a423c22885d650ee52550f90",
)
FACE_MODEL = ModelAsset(
    "deepghs/anime_face_detection",
    "784dc4c0bb692351ddcdbe6131a050b17d3025d5",
    "anime-face-detection",
    "face_detect_v1.4_s/model.onnx",
    44583229,
    "403b5bc93b6ff789b7d183418df4a1364049bac00c24acd927604a7ff6891483",
)
# Optional, query-only character instance segmentation.  Keep it outside ASSETS:
# adding a diagnostic model must not invalidate or block the 80k-image library
# index that uses MODEL_VERSION.
INSTANCE_MODEL = ModelAsset(
    "Faor-Mati/anime-character-segmentation",
    "57bdbddee92e1e18838bbc2e3bd4262d41bc6252",
    "anime-instance",
    "anime_segmentor_rtmdet_e60_simplified.onnx",
    238_686_077,
    "8826c0b7f1785f78c8a9f5f4804f227c7cadc7770b7ecb629a8a7da394703433",
)
INSTANCE_REFINER = ModelAsset(
    INSTANCE_MODEL.repo,
    INSTANCE_MODEL.revision,
    INSTANCE_MODEL.folder,
    "mask_refiner_isnetdis_refine_last_simplified.onnx",
    176_197_192,
    "17b50ed2958289fd6678f1915bd11594e5e5d53f20eed2ddedc5d05a9daf3f9f",
)
FOREGROUND_MODEL = ModelAsset(
    "onnx-community/anime-seg-ONNX",
    "25f233dc9f60e116dc79a5ba7e9575585388a36f",
    "anime-foreground",
    "onnx/model.onnx",
    176_068_431,
    "6a92a19a47e8197fb6dbcf85be14600806019831fedfe7f86eeeeffd4c40dbba",
)
INSTANCE_ASSETS = (INSTANCE_MODEL, INSTANCE_REFINER, FOREGROUND_MODEL)
INSTANCE_VERSION = f"animeinsseg-rtmdet:{INSTANCE_MODEL.revision[:12]}"
ASSETS = (WD_TAGS, CAMIE_TAGS, WD_MODEL, CAMIE_MODEL, PERSON_MODEL, FACE_MODEL)
# Includes the policy revision: changing thresholds/cropping requires re-indexing.
MODEL_VERSION = (
    f"subject-fusion-v2:wd-{WD_REVISION[:12]}:"
    f"camie-{CAMIE_REVISION[:12]}:person-{PERSON_REVISION[:12]}"
)

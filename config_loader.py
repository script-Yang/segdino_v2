import json
from dataclasses import dataclass, replace
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = ROOT_DIR / "config.yaml"


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    data_dir: str
    img_size: int = 256


@dataclass(frozen=True)
class ModelConfig:
    dino_size: str = "s"
    dino_repo: str = ""
    dino_ckpt: str = ""
    decoder_dim: int = 256
    use_bn: bool = False
    num_classes: int = 1
    patch_size: int = 16


@dataclass(frozen=True)
class TrainConfig:
    name: str
    dataset: DatasetConfig
    model: ModelConfig
    batch_size: int = 4
    epochs: int = 50
    lr: float = 1e-4
    save_dir: str = ""
    train_workers: int = 4
    val_workers: int = 2


@dataclass(frozen=True)
class TestConfig:
    name: str
    dataset: DatasetConfig
    model: ModelConfig
    ckpt_path: str = ""
    batch_size: int = 1
    num_workers: int = 8
    dice_thr: float = 0.5
    save_root: str = ""


def _load_raw_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


RAW_CONFIG = _load_raw_config()
EXPERIMENT_ENV_VAR = RAW_CONFIG["experiment_env_var"]
DEFAULT_TRAIN_EXPERIMENT = RAW_CONFIG["default_train_experiment"]
DEFAULT_TEST_EXPERIMENT = RAW_CONFIG["default_test_experiment"]


def _resolve_path(value: str) -> str:
    path = Path(value)
    if not path.is_absolute():
        path = (ROOT_DIR / path).resolve()
    return str(path)


def _build_dataset_config(raw_dataset):
    return DatasetConfig(
        name=raw_dataset["name"],
        data_dir=_resolve_path(raw_dataset["data_dir"]),
        img_size=raw_dataset.get("img_size", 256),
    )


DATASETS = {
    name: _build_dataset_config(raw_dataset)
    for name, raw_dataset in RAW_CONFIG["datasets"].items()
}

BASE_MODEL = ModelConfig(
    dino_size=RAW_CONFIG["base_model"].get("dino_size", "s"),
    dino_repo=_resolve_path(RAW_CONFIG["base_model"]["dino_repo"]),
    dino_ckpt=_resolve_path(RAW_CONFIG["base_model"]["dino_ckpt"]),
    decoder_dim=RAW_CONFIG["base_model"].get("decoder_dim", 256),
    use_bn=RAW_CONFIG["base_model"].get("use_bn", False),
    num_classes=RAW_CONFIG["base_model"].get("num_classes", 1),
    patch_size=RAW_CONFIG["base_model"].get("patch_size", 16),
)


def _build_model_config(overrides):
    return replace(
        BASE_MODEL,
        dino_size=overrides.get("dino_size", BASE_MODEL.dino_size),
        dino_repo=_resolve_path(overrides["dino_repo"]) if "dino_repo" in overrides else BASE_MODEL.dino_repo,
        dino_ckpt=_resolve_path(overrides["dino_ckpt"]) if "dino_ckpt" in overrides else BASE_MODEL.dino_ckpt,
        decoder_dim=overrides.get("decoder_dim", BASE_MODEL.decoder_dim),
        use_bn=overrides.get("use_bn", BASE_MODEL.use_bn),
        num_classes=overrides.get("num_classes", BASE_MODEL.num_classes),
        patch_size=overrides.get("patch_size", BASE_MODEL.patch_size),
    )


TRAIN_EXPERIMENTS = {}
for experiment_name, raw_train in RAW_CONFIG["train_experiments"].items():
    TRAIN_EXPERIMENTS[experiment_name] = TrainConfig(
        name=experiment_name,
        dataset=DATASETS[raw_train["dataset"]],
        model=_build_model_config(raw_train.get("model", {})),
        batch_size=raw_train.get("batch_size", 4),
        epochs=raw_train.get("epochs", 50),
        lr=raw_train.get("lr", 1e-4),
        save_dir=_resolve_path(raw_train.get("save_dir", "./checkpoints")),
        train_workers=raw_train.get("train_workers", 4),
        val_workers=raw_train.get("val_workers", 2),
    )


TEST_EXPERIMENTS = {}
for experiment_name, raw_test in RAW_CONFIG["test_experiments"].items():
    TEST_EXPERIMENTS[experiment_name] = TestConfig(
        name=experiment_name,
        dataset=DATASETS[raw_test["dataset"]],
        model=_build_model_config(raw_test.get("model", {})),
        ckpt_path=_resolve_path(raw_test["ckpt_path"]),
        batch_size=raw_test.get("batch_size", 1),
        num_workers=raw_test.get("num_workers", 8),
        dice_thr=raw_test.get("dice_thr", 0.5),
        save_root=_resolve_path(raw_test.get("save_root", "./runs_test")),
    )


def get_train_config(name: str) -> TrainConfig:
    try:
        return TRAIN_EXPERIMENTS[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown train experiment '{name}'. Available: {', '.join(TRAIN_EXPERIMENTS)}"
        ) from exc


def get_test_config(name: str) -> TestConfig:
    try:
        return TEST_EXPERIMENTS[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown test experiment '{name}'. Available: {', '.join(TEST_EXPERIMENTS)}"
        ) from exc


def resolve_encoder_size(dino_size: str) -> str:
    mapping = {"s": "small", "b": "base"}
    try:
        return mapping[dino_size]
    except KeyError as exc:
        raise ValueError(f"Unsupported dino_size '{dino_size}'") from exc

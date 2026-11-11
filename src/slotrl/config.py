from pathlib import Path
from typing import Any
import yaml

def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dirs(cfg: dict[str, Any], root: str | Path = ".") -> None:
    root = Path(root)
    for key in ("processed_dir", "model_dir", "results_dir"):
        (root / cfg["paths"][key]).mkdir(parents=True, exist_ok=True)


from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    project_root: Path
    data_root: Path

    raw_data: Path
    processed_data: Path
    assets: Path

    body_models: Path


def get_paths() -> Paths:
    # src/motion_retargeting/config/paths.py -> repository root
    project_root = Path(__file__).resolve().parents[3]

    data_root =  project_root / "data"

    return Paths(
        project_root=project_root,
        data_root=data_root,

        raw_data=data_root / "raw",
        processed_data=data_root / "processed",
        assets=data_root / "assets",

        body_models=data_root / "assets" / "body_models",
    )


PATHS = get_paths()

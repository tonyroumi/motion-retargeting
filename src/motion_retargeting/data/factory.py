# """
# Builds one MotionDataset per dataset group (motion domain) from the `dataset`
# section of the config.
# """

# from __future__ import annotations

# from typing import Any, Dict

# from .adapters.factory import build_adapter
# from .builder import MotionDatasetBuilder, MotionDatasetBuilderConfig
# from .dataset import MotionDataset


# def build_dataset(group_cfg: Dict[str, Any]) -> MotionDataset:
#     """ Build one domain from a `dataset.group_*` block of the config. """
#     adapters = [build_adapter(name) for name in group_cfg["sources"]]
#     config = MotionDatasetBuilderConfig(**(group_cfg.get("params") or {}))
#     return MotionDatasetBuilder(config).build(adapters)


# def build_datasets(dataset_cfg: Dict[str, Any]) -> Dict[str, MotionDataset]:
#     return {name: build_dataset(group_cfg) for name, group_cfg in dataset_cfg.items()}

"""
Replay a PAN training-step capture, <run>/captures/step_<N>_<kind>.pt: restore the weights, optimizer states,
replay buffers and RNG from before the step and rebuild its batch from the training datasets. Then print each
generator term with its gradient norm, and run the whole step again to compare its gradient norms with the
captured ones. Nothing is written to disk.

    python scripts/replay_pan_capture.py --config configs/methods/pan.yaml --capture <capture.pt>

The config must build the same datasets as the captured run. On the device it was captured on, the replay
matches the step up to that device's floating-point nondeterminism.
"""

from __future__ import annotations

import argparse
import dataclasses
import json

import torch
import yaml

from motion_retargeting.methods.pan.config import DomainSpec, PANConfig, PANModelConfig
from motion_retargeting.methods.pan.data import build_pan_splits, rebuild_batch
from motion_retargeting.methods.pan.method import PANMethod
from motion_retargeting.training.utils import move_to_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--capture", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    device = torch.device(cfg["device"])
    capture = torch.load(args.capture, map_location=device, weights_only=False)

    train_datasets, _ = build_pan_splits(cfg["dataset"])

    # The captured run's PAN settings, without further spike reports.
    fields = {field.name for field in dataclasses.fields(PANConfig)}
    method_cfg = PANConfig(**{key: value for key, value in capture["config"].items() if key in fields})
    method_cfg.grad_spike_factor = 0.0

    method = PANMethod(
        model_cfg=PANModelConfig.from_dict(cfg["method"]["model"]),
        domain_specs={name: DomainSpec.from_dataset(dataset) for name, dataset in train_datasets.items()},
        device=device,
        cfg=method_cfg,
    )
    method.train()
    batch = move_to_device(rebuild_batch(train_datasets, capture["batch"]), device)

    captured = capture["report"]
    print(f"Captured step {captured['step']} ({capture['kind']}): {captured['reason']}")
    print(json.dumps(captured["domains"], indent=2))

    # Each generator term and its own gradient norm, from the step's forward.
    method.restore_capture(capture)
    try:
        state = method.forward_state(batch)
        method.set_discriminator_requires_grad(False)
        losses = method.compute_generator_losses(state)
        grads = method.loss_grad_norms(losses, list(method.model.generator_parameters()))
        print(f"{'generator term':<40} {'loss':>12} {'gradient norm':>14}")
        for name, loss in losses.items():
            print(f"{name:<40} {loss.item():>12.6g} {grads[f'loss_grad/{name}']:>14.6g}")
    except Exception as error:
        print(f"Generator terms raised {type(error).__name__}: {error}")

    # The whole step, from the same state.
    method.restore_capture(capture)
    try:
        metrics = method.train_step(batch)
    except Exception as error:
        print(f"Replayed step raised {type(error).__name__}: {error}")
        return
    print(f"{'replayed step':<40} {'replayed':>12} {'captured':>14}")
    for name, value in metrics.items():
        if name in captured["grad_norms"] or name in captured["losses"]:
            reference = captured["grad_norms"].get(name, captured["losses"].get(name))
            print(f"{name:<40} {value:>12.6g} {reference:>14.6g}")


if __name__ == "__main__":
    main()

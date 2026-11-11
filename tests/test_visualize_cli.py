"""Headless plots, explicit units/axes, and read-only coordinate-stage previews."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from motion_retargeting.data.representations.motion import MotionData
from motion_retargeting.data.representations.skeleton import SkeletonTopology
from motion_retargeting.data.visualization.datasets import dataset_panels
from motion_retargeting.methods.pan.human_reference import HUMAN_PARENTS


@pytest.fixture
def viewer():
    path = Path(__file__).resolve().parents[1] / "scripts/visualize.py"
    spec = importlib.util.spec_from_file_location("dataset_viewer", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    yield module
    plt.close("all")


def motion(units="meters", fps=30.):
    positions = np.array([[[0., 1., 2.], [0., 2., 2.]], [[.1, 1., 2.], [.1, 2., 2.]],
                          [[.2, 1., 2.], [.2, 2., 2.]]])
    if units == "centimeters":
        positions *= 100
    return {"positions": positions, "position_parents": np.array([-1, 0]), "fps": fps,
            "checkpoint_physical_units": units, "role": "original", "domain": "human", "path": Path("input.npz")}


def test_units_y_up_and_shared_scale(viewer):
    a, b = viewer.validated_motion(motion()), viewer.validated_motion(motion("centimeters"))
    np.testing.assert_array_equal(a["positions"], b["positions"])
    fig, update, count = viewer.make_figure("units", [a, b])
    assert count == 3
    update(0)
    for ax in fig.axes:
        assert ax.get_xlabel() == "X (m)" and ax.get_ylabel() == "Z (m)" and ax.get_zlabel() == "Y up (m)"
        # First line is the bone, with matplotlib vertical displaying source Y.
        x, z, y = ax.lines[0].get_data_3d()
        np.testing.assert_array_equal(y, [1., 2.])
        np.testing.assert_array_equal(z, [2., 2.])
    assert fig.axes[0].get_zlim() == fig.axes[1].get_zlim()
    del a["checkpoint_physical_units"]
    with pytest.raises(ValueError, match="units"):
        viewer.validated_motion(a)
    with pytest.raises(ValueError, match="parent-before-child"):
        viewer.validated_motion({**motion(), "position_parents": np.array([-1, 1])})


def test_source_z_up_is_labeled_without_reinterpreting_coordinates(viewer):
    raw = motion(); raw["up_axis"] = "Z"
    fig, update, _ = viewer.make_figure("source", [viewer.validated_motion(raw)])
    update(0)
    assert fig.axes[0].get_ylabel() == "Y (m)"
    assert fig.axes[0].get_zlabel() == "Z up (m)"
    _, horizontal_y, vertical_z = fig.axes[0].lines[0].get_data_3d()
    np.testing.assert_array_equal(horizontal_y, [1., 2.])
    np.testing.assert_array_equal(vertical_z, [2., 2.])


def test_npz_still_gif_and_overwrite_protection(viewer, tmp_path):
    path = tmp_path / "input.npz"
    data = motion("centimeters"); data.pop("path")
    np.savez(path, **data)
    original = path.read_bytes()
    still = tmp_path / "still"
    viewer.main(["outputs", str(path), "--save-dir", str(still), "--frame", "1"])
    assert (still / "outputs_frame0001.png").stat().st_size > 1000
    with pytest.raises(FileExistsError, match="overwrite"):
        viewer.main(["outputs", str(path), "--save-dir", str(still), "--frame", "1"])
    animation = tmp_path / "animation"
    viewer.main(["outputs", str(path), "--save-dir", str(animation)])
    assert (animation / "outputs.gif").stat().st_size > 1000
    assert path.read_bytes() == original


def test_time_alignment_respects_input_rates(viewer):
    a, b = viewer.validated_motion(motion(fps=30)), viewer.validated_motion(motion(fps=60))
    _, update, count = viewer.make_figure("rates", [a, b], fps=30)
    assert count == 2  # common duration 2/60 sec, not min(frame counts)/30
    update(1)
    with pytest.raises(IndexError):
        update(2)


def test_dataset_world_and_training_coordinates_are_not_conflated(monkeypatch, viewer):
    source = "amass"; parents = torch.tensor(HUMAN_PARENTS[source])
    offsets = torch.zeros(22, 3, dtype=torch.float64); offsets[1:, 1] = .1
    offsets[1] = torch.tensor([.1, -.1, 0.]); offsets[2] = torch.tensor([-.1, -.1, 0.])
    q = torch.zeros(65, 22, 4, dtype=torch.float64); q[..., 0] = 1
    root = torch.zeros(65, 3, dtype=torch.float64); root[:, 0] = torch.arange(65)*.01; root[:, 1] = 1.
    clip = MotionData(offsets, q, 30., root)
    adapter = SimpleNamespace(_read_clip=lambda path, dtype: clip,
                              _topology=SkeletonTopology(parents))
    monkeypatch.setattr("motion_retargeting.data.visualization.datasets.human_adapter", lambda *args: adapter)
    world, training = dataset_panels(source, Path("fixture.npz"), space="both")
    assert world["units"] == "meters" and training["units"] == "centimeters"
    np.testing.assert_allclose(world["positions"][:, 0], root[:-1])
    assert training["positions"][0, 0, 0] == pytest.approx(1., abs=1e-6)  # inclusive first step in cm
    assert training["positions"][0, 0, 1] == 0  # training root starts from local displacement, not absolute height
    assert "EXPERIMENTAL" in training["role"]
    plotted = [viewer.validated_motion(panel) for panel in (world, training)]
    assert plotted[1]["positions"][0, 0, 0] == pytest.approx(.01, abs=1e-7)

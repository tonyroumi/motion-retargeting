"""Adapter unit contracts and raw PAN loading; no training or external data required."""
from pathlib import Path
from types import SimpleNamespace
import os

import numpy as np
import pytest
import torch

from motion_retargeting.data.adapters import base
from motion_retargeting.data.adapters.amass import AMASSAdapter, zup_to_yup_vector
from motion_retargeting.data.adapters.bandai import BANDAIAdapter
from motion_retargeting.data.adapters.cmu import CMUAdapter, INCHES_TO_METERS, parse_asf
from motion_retargeting.data.adapters.factory import ADAPTERS, build_adapter
from motion_retargeting.data.adapters.lafan1 import DogAdapter, LAFAN1Adapter
from motion_retargeting.data.adapters.lafan1_bvh import REFERENCE_HIERARCHIES, read_lafan_reference_bvh
from motion_retargeting.data.adapters.lafan1_splits import REFERENCE_SPLITS
from motion_retargeting.data.representations.motion import MotionData
from motion_retargeting.data.representations.skeleton import SkeletonTopology
from motion_retargeting.methods.pan.data import build_pan_splits
from motion_retargeting.methods.pan.raw_reference import load_raw_reference_splits
from motion_retargeting.methods.pan.reference import reference_channels


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    monkeypatch.setattr(base, "PATHS", SimpleNamespace(data_root=tmp_path))
    monkeypatch.setattr(base, "label_joint_bodies", lambda *args: {})
    return tmp_path


def write_bvh(path, names, frames=86, ends=1, offset=10., angle=0., fps=30., parents=None, end_parents=None):
    """Small BVH in original cm; defaults to a chain for generic adapter fixtures."""
    lines = ["HIERARCHY"]
    parents = [-1] + list(range(len(names)-1)) if parents is None else parents
    end_parents = [len(names)-1]*ends if end_parents is None else end_parents

    def visit(i):
        lines.extend([f"{'ROOT' if i == 0 else 'JOINT'} {names[i]}", "{",
                      f"OFFSET {0 if i == 0 else offset} 0 0",
                      "CHANNELS 6 Xposition Yposition Zposition Xrotation Yrotation Zrotation" if i == 0
                      else "CHANNELS 3 Xrotation Yrotation Zrotation"])
        for child, parent in enumerate(parents):
            if parent == i:
                visit(child)
        for _ in range(end_parents.count(i)):
            lines.extend(["End Site", "{", "OFFSET 1 0 0", "}"])
        lines.append("}")

    visit(0)
    lines += ["MOTION", f"Frames: {frames}", f"Frame Time: {1/fps:.9f}"]
    for t in range(frames):
        values = [100 + t*t*.01, 200, 300 + t*.1, angle, 0, 0] + [0.] * (3 * (len(names)-1))
        lines.append(" ".join(map(str, values)))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


@pytest.mark.parametrize("name", list(ADAPTERS))
def test_factory_does_not_create_cache_on_inspection(data_root, name):
    adapter = build_adapter(name)
    assert not adapter.cache_dir.exists()
    assert adapter.raw_dir.parent == data_root / "raw"


@pytest.mark.parametrize("name", ["amass", "cmu", "lafan1", "dog"])
def test_legacy_caches_are_preserved_and_not_reused(data_root, monkeypatch, name):
    adapter = build_adapter(name)
    legacy = data_root / "processed" / f"{name}.pt"
    legacy.parent.mkdir()
    torch.save("old units or split", legacy)
    original = legacy.read_bytes()
    adapter._topology = SkeletonTopology(parents=torch.tensor([-1]), body_joints={0: [0]})
    clip = MotionData(torch.zeros(1, 3), torch.tensor([[[1., 0., 0., 0.]]]), 30., torch.zeros(1, 3))
    monkeypatch.setattr(adapter, "_extract", lambda: [clip])
    result = adapter.load()
    assert len(result) == 1
    assert adapter.motion_cache_path != legacy
    assert legacy.read_bytes() == original
    monkeypatch.setattr(adapter, "_extract", lambda: pytest.fail("new cache should be used"))
    torch.testing.assert_close(adapter.load()[0].offsets, clip.offsets)


def test_amass_offsets_and_root_are_both_metres(data_root, monkeypatch):
    adapter = AMASSAdapter()
    adapter.raw_dir.mkdir(parents=True)
    np.savez(adapter.raw_dir / "clip.npz", gender="neutral", poses=np.zeros((3, 66)),
             trans=np.array([[1., 2., 3.]] * 3), betas=np.zeros(16), mocap_framerate=60.)
    joints = torch.zeros(22, 3); joints[:, 1] = torch.arange(22) * .1

    class BodyModel:
        kintree_table = torch.tensor([[-1] + list(range(21))])

        def __call__(self, **kwargs):
            return SimpleNamespace(Jtr=joints[None])

    monkeypatch.setattr(adapter, "_body_model", lambda gender: BodyModel())
    clip = adapter._extract()[0]
    torch.testing.assert_close(clip.offsets[1:, 1], torch.full((21,), .1))
    torch.testing.assert_close(clip.root_position, zup_to_yup_vector(torch.tensor([[1., 2., 3.]] * 3)))
    assert clip.fps == 60


def test_cmu_asf_lengths_and_amc_translation_are_metres(data_root):
    adapter = CMUAdapter(); adapter.raw_dir.mkdir(parents=True)
    asf_path = adapter.raw_dir / "01.asf"
    asf_path.write_text(""":units
length 0.45
:root
order TX TY TZ RX RY RZ
axis XYZ
orientation 0 0 0
:bonedata
begin
id 1
name leg
direction 0 1 0
length 2
axis 0 0 0 XYZ
dof rx ry rz
end
:hierarchy
begin
root leg
end
""")
    (adapter.raw_dir / "01_01.amc").write_text(":DEGREES\n1\nroot 1 2 3 0 0 0\nleg 0 0 0\n")
    asf = parse_asf(asf_path)
    scale = INCHES_TO_METERS / .45
    np.testing.assert_allclose(asf.offsets[-1], [0, 2*scale, 0])
    clip = adapter._extract()[0]
    torch.testing.assert_close(clip.root_position, torch.tensor([[3*scale, 2*scale, -scale]]))
    torch.testing.assert_close(clip.offsets[-1], torch.tensor([0., 2*scale, 0.]))


def test_bandai_remains_in_metres(data_root):
    adapter = BANDAIAdapter(); adapter.raw_dir.mkdir(parents=True)
    # Ground anchor is dropped, its translation still contributes to Hips.
    write_bvh(adapter.raw_dir / "clip.bvh", ["joint_Root", "Hips", "Spine"], frames=3)
    clip = adapter._extract()[0]
    assert clip.offsets.shape == (2, 3)
    assert float(clip.offsets[1].norm()) == pytest.approx(.1)
    torch.testing.assert_close(clip.root_position[0], torch.tensor([3., 2., -1.1]))


@pytest.mark.parametrize("name", ["mixamo_a", "mixamo_b", "lafan1", "dog"])
def test_generic_bvh_adapters_keep_metre_contract(data_root, name):
    adapter = build_adapter(name)
    path = data_root / "fixture.bvh"
    write_bvh(path, adapter.joint_names, frames=3)
    clip = adapter._read_clip(path)
    torch.testing.assert_close(clip.root_position[0], torch.tensor([1., 2., 3.]))
    torch.testing.assert_close(clip.offsets[1:].norm(dim=-1), torch.full((len(adapter.joint_names)-1,), .1))
    assert clip.joint_rotations.shape == (3, len(adapter.joint_names), 4)


@pytest.mark.parametrize("kind,folder", [("lafan1", "LAFAN1"), ("dog", "Dog")])
def test_capitalized_directories_and_exact_membership(data_root, monkeypatch, kind, folder):
    directory = data_root / "raw" / folder; directory.mkdir(parents=True)
    monkeypatch.setitem(REFERENCE_SPLITS[kind], "train", ["train"])
    monkeypatch.setitem(REFERENCE_SPLITS[kind], "test", ["heldout"])
    for stem in ("train", "heldout", "extra"):
        (directory / f"{stem}.bvh").touch()
    adapter = build_adapter(kind)
    assert adapter.raw_dir == directory
    assert [p.stem for p in adapter._files()] == ["train"]
    (directory / "train.bvh").unlink()
    with pytest.raises(FileNotFoundError, match="missing reference clips"):
        adapter._files()
    (directory / "train.bvh").touch()
    (directory / "duplicate").mkdir(); (directory / "duplicate/train.bvh").touch()
    with pytest.raises(ValueError, match="duplicate reference clip"):
        adapter._files()


def test_float64_bvh_reader_preserves_offsets_channels_and_ends(tmp_path):
    path = tmp_path / "clip.bvh"
    write_bvh(path, ["Root", "Child"], frames=3, offset=10.000000123, angle=90.)
    clip = read_lafan_reference_bvh(path)
    assert clip.rotations.dtype == clip.root_position.dtype == clip.offsets.dtype == torch.float64
    assert clip.offsets[1, 0] == 10.000000123
    assert clip.parents.tolist() == [-1, 0]
    assert clip.parents_withend.tolist() == [-1, 0, 1]
    assert clip.not_end.tolist() == [0, 1]
    torch.testing.assert_close(clip.rotations[:, 0], torch.tensor([[2**-.5, 2**-.5, 0., 0.]] * 3, dtype=torch.float64))
    torch.testing.assert_close(clip.root_position[0], torch.tensor([100., 200., 300.], dtype=torch.float64))
    path.write_text(path.read_text().replace("CHANNELS 3 Xrotation", "CHANNELS 3 Xposition"))
    with pytest.raises(ValueError, match="common rotation order"):
        read_lafan_reference_bvh(path)


@pytest.mark.parametrize("kind,cls", [("lafan1", LAFAN1Adapter), ("dog", DogAdapter)])
def test_raw_loader_domain_stats_subject_banks_splits_and_no_cache(data_root, monkeypatch, kind, cls):
    directory = data_root / "raw" / kind
    train = ["a_subject2", "b_subject2"] if kind == "lafan1" else ["D1_a", "D1_b"]
    test = ["c_subject1"] if kind == "lafan1" else ["D1_c"]
    monkeypatch.setitem(REFERENCE_SPLITS[kind], "train", train)
    monkeypatch.setitem(REFERENCE_SPLITS[kind], "test", test)
    parents, expanded, mapping = REFERENCE_HIERARCHIES[kind]
    end_parents = [mapping.index(expanded[i]) for i in range(27) if i not in mapping]
    for stem in train + test + ["unlisted"]:
        write_bvh(directory / f"{stem}.bvh", cls.joint_names, frames=86 if kind == "lafan1" else 172,
                  parents=parents, end_parents=end_parents, angle=90. if kind == "lafan1" else 0.,
                  fps=30. if kind == "lafan1" else 60., offset=10.00001 if stem == train[1] else 10.)
    training, heldout = load_raw_reference_splits(kind, directory)
    assert (len(training), len(heldout)) == (4, 2)
    assert training.num_banks == (2 if kind == "lafan1" else 1)
    assert training.bank_ids.tolist() == [0]*4
    assert training.reference is heldout.reference
    assert training.reference.offsets_withend.shape[1:] == (27, 3)
    samples = torch.cat((training.motion, heldout.motion)).numpy().reshape(-1, training.motion.shape[-1])
    expected_std = samples.std(0); expected_std[expected_std == 0] = 1
    np.testing.assert_array_equal(training.mean[0].numpy(), samples.mean(0))
    np.testing.assert_array_equal(training.std[0].numpy(), expected_std)
    assert not (data_root / "processed").exists()
    assert training[0]["motion"].dtype == torch.float32
    assert torch.isfinite(training[0]["motion"]).all()
    # Matching names and counts alone are insufficient: loss End Site indices
    # have fixed semantic meanings and must not be silently remapped.
    write_bvh(directory / f"{train[0]}.bvh", cls.joint_names, ends=27-len(cls.joint_names))
    with pytest.raises(ValueError, match="hierarchy/End Site mapping"):
        load_raw_reference_splits(kind, directory)


def test_raw_source_routing_and_conflicting_options(monkeypatch):
    calls = []
    def loader(**kwargs):
        calls.append(kwargs)
        return "train", "test"
    monkeypatch.setattr("motion_retargeting.methods.pan.data.load_raw_reference_splits", loader)
    train, test = build_pan_splits({"a": {"sources": ["lafan1"]}, "b": {"sources": ["dog"]}})
    assert calls == [{"kind": "lafan1"}, {"kind": "dog"}]
    assert train == {"a": "train", "b": "train"}
    assert test == {"a": "test", "b": "test"}
    with pytest.raises(ValueError, match="explicit reference splits"):
        build_pan_splits({"a": {"sources": ["lafan1"], "params": {"holdout_percent": 5}}})
    with pytest.raises(ValueError, match="no raw mixing"):
        build_pan_splits({"a": {"sources": ["lafan1"]}, "b": {"sources": ["cmu"]}})
    with pytest.raises(ValueError, match="cannot be combined"):
        build_pan_splits({"a": {"sources": ["lafan1", "dog"]}})


@pytest.mark.parametrize("kind,folder,prefix,counts", [
    ("lafan1", "LAFAN1", "hum", (6566, 2074)), ("dog", "Dog", "dog", (2769, 779)),
])
def test_real_raw_corpus_against_reference_archives(kind, folder, prefix, counts):
    root, bvh_root = os.environ.get("PAN_REFERENCE_ROOT"), os.environ.get("PAN_REFERENCE_BVH_ROOT")
    if not root or not bvh_root:
        pytest.skip("Set PAN_REFERENCE_ROOT and PAN_REFERENCE_BVH_ROOT for full raw corpus checks")
    directory = Path(root) / "data_preprocess/Lafan1_and_dog"
    for split in ("train", "test"):
        manifest = directory / f"{kind}_{split}.txt"
        assert set(REFERENCE_SPLITS[kind][split]) == set(manifest.read_text().split())
    training, heldout = load_raw_reference_splits(kind, Path(bvh_root) / folder)
    assert (len(training), len(heldout)) == counts
    assert training.num_banks == (5 if kind == "lafan1" else 1)
    with np.load(directory / f"{prefix}stats.npz") as stats:
        mean, std = stats["mean"], stats["std"]
        np.testing.assert_allclose((training.mean[0].numpy()-mean)/std, 0, atol=1e-7)
        np.testing.assert_allclose(training.std[0].numpy()/std, 1, atol=1e-8, rtol=0)
        assert training.reference.min_velocity == pytest.approx(float(stats["min_vel"]), abs=1e-12)
        assert training.reference.max_velocity == pytest.approx(float(stats["max_vel"]), abs=1e-12)
        np.testing.assert_array_equal(training.parents, stats["parents"])
        np.testing.assert_array_equal(training.reference.parents_withend, stats["parents_withend"])
        np.testing.assert_array_equal(training.reference.not_end, stats["not_end"])
    for split, dataset in (("train", training), ("test", heldout)):
        with np.load(directory / f"{prefix}{split}.npz", allow_pickle=True) as archive:
            oracle = reference_channels(torch.from_numpy(archive["Q"]), torch.from_numpy(archive["V"][:, :, 0]),
                                        torch.from_numpy(archive["yrot"]))
            # Upstream os.listdir order differs. Match by first pose, then compare
            # every channel/frame of evenly spaced samples including held-out clips.
            for i in torch.linspace(0, len(dataset)-1, 12).long().tolist():
                distances = (oracle[:, 0, :-4]-dataset.motion[i, 0, :-4]).abs().amax(1)
                match = int(distances.argmin())
                torch.testing.assert_close(dataset.motion[i], oracle[match], rtol=1e-8, atol=1e-8)
                expected = ((oracle[match]-torch.from_numpy(mean))/torch.from_numpy(std)).float()
                torch.testing.assert_close(dataset[i]["motion"], expected, rtol=1e-6, atol=1e-6)
                offsets = archive["skel_offsets"].item()[archive["skel_names"][match]].reshape(-1, 3)
                np.testing.assert_allclose(dataset.offsets[dataset.bank_ids[i], 1:], offsets, rtol=0, atol=7e-5)

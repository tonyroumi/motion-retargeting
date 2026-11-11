"""Bounded CMU/AMASS experiment checks, distinct from published dataset parity."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from motion_retargeting.data.representations.motion import MotionData
from motion_retargeting.methods.pan.config import DomainSpec, PANConfig, PANModelConfig
from motion_retargeting.methods.pan.data import build_pan_splits
from motion_retargeting.methods.pan.human_reference import (
    HUMAN_PARENTS, HUMAN_PARTS, HUMAN_ROTATION_IDS, anatomical_forward, human_files, load_human_reference_splits,
)
from motion_retargeting.methods.pan.losses import mapped_root_velocity
from motion_retargeting.methods.pan.method import PANMethod
from motion_retargeting.methods.pan.reference import preprocess_reference_window
from motion_retargeting.training.utils import PrimaryRecycledLoaders
from motion_retargeting.training.trainer import Trainer, TrainerConfig
from test_pan_reference import upstream, test_upstream_objective_discriminator_and_generator_gradients as compare_upstream_objective


def synthetic_clip(source):
    parents = HUMAN_PARENTS[source]
    offsets = torch.zeros(len(parents), 3, dtype=torch.float64); offsets[1:, 1] = .1
    left, right = (1, 2) if source == "amass" else (1, 5)
    offsets[left] = torch.tensor([.1, -.1, 0.]); offsets[right] = torch.tensor([-.1, -.1, 0.])
    q = torch.zeros(86, len(parents), 4, dtype=torch.float64); q[..., 0] = 1
    t = torch.arange(86, dtype=torch.float64)
    root = torch.zeros(86, 3, dtype=torch.float64); root[:, 2] = (t-5).clamp_min(0).square()*.001
    return MotionData(offsets, q, 30., root)


@pytest.fixture
def humans(tmp_path, monkeypatch):
    import motion_retargeting.methods.pan.human_reference as module
    monkeypatch.setattr(module, "human_adapter", lambda source, body_models=None: SimpleNamespace(
        _read_clip=lambda path, dtype: synthetic_clip(source),
        _asf_model=lambda path: SimpleNamespace(bone_vectors={name: np.array([0., 0., .06])
                                                            for name in ("ltoes", "rtoes", "lwrist", "rwrist")})))
    datasets = {}
    for source in ("amass", "cmu"):
        directory = tmp_path / source; directory.mkdir()
        suffix = "_poses.npz" if source == "amass" else ".amc"
        for subject in ("01", "02"):
            (directory / f"{subject}_01{suffix}").touch()
        datasets[source] = load_human_reference_splits(source, ["02"], directory)
    return datasets


def test_anatomy_axis_is_a_root_frame_invariant():
    # LAFAN-style root frame: local +X is up, hips left/right are +/-Z.
    offsets = torch.tensor([[0., 0., 0.], [0., 0., 1.], [0., 0., -1.], [1., 0., 0.]])
    axis = anatomical_forward(offsets, torch.tensor([-1, 0, 0, 0]), 1, 2, 3)
    torch.testing.assert_close(axis, torch.tensor([0., 1., 0.]))
    with pytest.raises(ValueError, match="degenerate"):
        anatomical_forward(offsets*0, torch.tensor([-1, 0, 0, 0]), 1, 2, 3)


def test_human_forward_option_matches_published_math_when_axis_matches():
    clip = synthetic_clip("amass")
    # Supplying exactly the published +Y axis must not change LAFAN math.
    q = clip.joint_rotations[:65].clone()
    q[:, 0] = torch.tensor([2**-.5, 2**-.5, 0., 0.])
    args = (q, clip.root_position[:65]*100, clip.offsets*100, torch.tensor(HUMAN_PARENTS["amass"]), "lafan1")
    expected, heading = preprocess_reference_window(*args)
    actual, actual_heading = preprocess_reference_window(*args, forward_axis=torch.tensor([0., 1., 0.]))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_heading, heading, rtol=0, atol=0)


def test_stable_human_heading_handles_exact_half_turn():
    clip = synthetic_clip("amass")
    q = clip.joint_rotations[:65].clone(); q[:, 0] = torch.tensor([0., 0., 1., 0.])
    channels, heading = preprocess_reference_window(
        q, clip.root_position[:65]*100, clip.offsets*100, torch.tensor(HUMAN_PARENTS["amass"]),
        "lafan1", forward_axis=torch.tensor([0., 0., 1.]), stable_heading=True,
    )
    torch.testing.assert_close(heading.norm(dim=-1), torch.ones(64, 1, dtype=torch.float64))
    torch.testing.assert_close(channels[:, :4].norm(dim=-1), torch.ones(64, dtype=torch.float64))
    assert torch.isfinite(channels).all()


def test_human_splits_cm_domain_statistics_available_joint_mapping(humans):
    for source, (train, test) in humans.items():
        assert (len(train), len(test)) == (2, 2)
        assert train.num_banks == 2
        assert set(train.bank_ids.tolist()).isdisjoint(test.bank_ids.tolist())
        ref = train.reference
        assert ref.protocol == "human_lafan_style_v1"
        assert ref.parts == HUMAN_PARTS[source]
        assert len(ref.rotation_indices) == (14 if source == "amass" else 15)
        assert len(ref.position_indices) == (14 if source == "amass" else 18)
        assert len(ref.parents_withend) == (22 if source == "amass" else 28)
        if source == "cmu":
            torch.testing.assert_close(ref.offsets_withend[0, -4:, 2], torch.full((4,), 6.))
            assert ref.parents_withend[-4:].tolist() == [4, 8, 19, 23]
        assert train.provenance["zero_speed_frames"] > 0
        samples = torch.cat([train.motion, test.motion]).numpy().reshape(-1, train.motion.shape[-1])
        np.testing.assert_array_equal(train.mean[0], samples.mean(0))
        std = samples.std(0); std[std == 0] = 1
        np.testing.assert_array_equal(train.std[0], std)
        torch.testing.assert_close(train.offsets[0].double(), synthetic_clip(source).offsets[HUMAN_ROTATION_IDS[source]]*100)
        assert train.motion.shape[-1] == (92 if source == "amass" else 96)
        assert len(train.parents) == (22 if source == "amass" else 23)
        assert train[0]["motion"].dtype == torch.float32
        assert torch.isfinite(train[0]["motion"]).all()


def test_zero_speed_defined_only_for_explicit_human_experiment():
    velocity = torch.tensor([[0., 0., 0.], [3., 0., 4.]], requires_grad=True)
    actual = mapped_root_velocity(velocity, 1., 9., zero_safe=True)
    torch.testing.assert_close(actual, torch.tensor([[0., 0., 0.], [.3, 0., .4]]))
    actual.square().sum().backward()
    assert torch.isfinite(velocity.grad).all()
    assert torch.isnan(mapped_root_velocity(velocity[:1], 1., 9.)).all()


def human_method_and_batch(humans):
    torch.manual_seed(0)
    datasets = {name: pair[0] for name, pair in humans.items()}
    config = PANModelConfig.from_dict({
        "motion_encoder": {"embed_dim": 8, "embed_hidden_dim": 16, "num_layers": 1, "num_heads": 1,
                           "feedforward_dim": 16, "kernel_size": 3},
        "motion_decoder": {"kernel_size": 3}, "motion_discriminator": {"kernel_size": 3, "hidden_dim": 16},
    })
    method = PANMethod(config, {name: DomainSpec.from_dataset(d) for name, d in datasets.items()},
                       learning_rate=1e-4, cfg=PANConfig(position_lambda=.01, cycle_lambda=.001))
    batch = {name: {"motion": torch.stack([d[i]["motion"] for i in (0, 1)]), "bank_id": d.bank_ids}
             for name, d in datasets.items()}
    return method, batch


def test_human_objective_routes_update_and_checkpoint(humans):
    method, batch = human_method_and_batch(humans)
    state = method.forward_state(batch)
    assert set(state.pairs) == {("cmu", "amass"), ("amass", "cmu")}
    assert method.buffers == {}
    for name, source in state.sources.items():
        domain = method.model.domains[name]
        assert domain.ee_ids == [] and domain.heights is None
        torch.testing.assert_close(domain.discriminator_input(source.bones), domain.fk.world_positions(source.bones))
    metrics = method.train_step(batch)
    assert all(np.isfinite(value) for value in metrics.values())
    assert all(torch.isfinite(p).all() for p in method.model.parameters())
    checkpoint = method.state_dict(); method.load_state_dict(checkpoint)
    assert checkpoint["metadata"]["representation"] == "cmu_amass_lafan_style_v1"
    checkpoint["metadata"]["representation"] = "lafan1_dog_reference_raw_v1"
    with pytest.raises(ValueError, match="incompatible"):
        method.load_state_dict(checkpoint)


def test_nonzero_human_objective_and_gradients_against_reference(upstream, humans):
    method, batch = human_method_and_batch(humans)
    # The experiment intentionally fixes the reference's exact-zero singularity.
    # Compare the actual upstream objective/gradients away from that exception,
    # using our 22/23 rotation joints and 22/28 position-joint mappings.
    for name, item in batch.items():
        domain = method.model.domains[name]
        item["motion"][..., -4:-1] = (torch.tensor([1., 2., 3.])-domain.mean[0, -4:-1])/domain.std[0, -4:-1]
    compare_upstream_objective(upstream, (method, batch))


def test_shared_split_and_multi_actor_session_guards(tmp_path):
    for stem in ("18_01", "22_01", "01_01"):
        (tmp_path / f"{stem}.amc").touch()
    with pytest.raises(ValueError, match="18--23"):
        human_files("cmu", tmp_path, ["22"])
    split = human_files("cmu", tmp_path, [str(i) for i in range(18, 24)])
    assert {p.stem for p in split["test"]} == {"18_01", "22_01"}
    with pytest.raises(ValueError, match="same held-out"):
        build_pan_splits({"a": {"human_reference_style": {"source": "amass", "test_subjects": ["01"]}},
                          "b": {"human_reference_style": {"source": "cmu", "test_subjects": ["02"]}}})


def test_exact_update_budget_and_partial_epoch_checkpoint(tmp_path):
    calls = []
    method = SimpleNamespace(train=lambda: None, train_step=lambda batch: (calls.append(batch) or {"loss": 1.}),
                             state_dict=lambda: {"steps": len(calls)}, load_state_dict=lambda state: None)
    trainer = Trainer(method, [{"x": torch.ones(1)} for _ in range(5)], None,
                      TrainerConfig(num_epochs=10, max_steps=3, checkpoint_interval_steps=2,
                                    log_dir=str(tmp_path), use_tensorboard=False, verbose=False, device="cpu"))
    trainer.train()
    assert len(calls) == trainer.global_step == 3
    saved = torch.load(trainer.checkpoint_dir / "step_2.pt", weights_only=True)
    assert saved["global_step"] == 2 and not saved["epoch_complete"]
    final = torch.load(trainer.checkpoint_dir / "final.pt", weights_only=True)
    assert final["global_step"] == 3 and not final["epoch_complete"]
    trainer.load_checkpoint(trainer.checkpoint_dir / "final.pt")
    assert trainer.current_epoch == 0  # partial epoch is not silently marked complete


def test_primary_recycles_secondary_without_caching_batches():
    a = DataLoader(TensorDataset(torch.arange(10)), batch_size=2, drop_last=True)
    b = DataLoader(TensorDataset(torch.arange(6)), batch_size=2, drop_last=True)
    loader = PrimaryRecycledLoaders({"a": a, "b": b}, "a")
    first = list(loader); second = list(loader)
    assert len(loader) == len(first) == 5
    assert [item["a"][0].tolist() for item in first] == [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    assert [item["b"][0].tolist() for item in first] == [[0, 1], [2, 3], [4, 5], [0, 1], [2, 3]]
    assert second[0]["b"][0].tolist() == [4, 5]  # continues across primary epochs
    with pytest.raises(ValueError, match="drop_last"):
        PrimaryRecycledLoaders({"a": DataLoader(TensorDataset(torch.arange(1)), batch_size=2, drop_last=True)}, "a")

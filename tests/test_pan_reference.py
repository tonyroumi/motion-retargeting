"""Bounded reference semantics: synthetic regressions and optional upstream oracles.

PAN_REFERENCE_ROOT enables differential checks against a read-only reference checkout.
"""
from pathlib import Path
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from motion_retargeting.kinematics.quaternions import quat_mul
from motion_retargeting.methods.pan.config import DomainSpec, PANConfig, PANModelConfig
from motion_retargeting.methods.pan.data import build_pan_splits
from motion_retargeting.methods.pan.kinematics import JointForwardKinematics
from motion_retargeting.methods.pan.losses import mapped_root_velocity
from motion_retargeting.methods.pan.method import PANMethod
from motion_retargeting.methods.pan.reference import (
    PARTS, load_reference_splits, preprocess_reference_window, reference_channels, reference_window_starts,
)

PARENTS = {
    "lafan1": [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 11, 14, 15, 16, 11, 18, 19, 20],
    "dog": [-1, 0, 1, 2, 3, 2, 5, 6, 7, 2, 9, 10, 11, 0, 13, 14, 0, 16, 17, 0, 19],
}
EXPANDED = {
    "lafan1": [-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 15, 13, 17, 18, 19, 20, 13, 22, 23, 24, 25],
    "dog": [-1, 0, 1, 2, 3, 4, 2, 6, 7, 8, 9, 2, 11, 12, 13, 14, 0, 16, 17, 18, 0, 20, 21, 22, 0, 24, 25],
}
NOT_END = {
    "lafan1": [0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 15, 17, 18, 19, 20, 22, 23, 24, 25],
    "dog": [0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25],
}


@pytest.fixture
def archives(tmp_path):
    rng = np.random.default_rng(123)
    for kind, prefix in (("lafan1", "hum"), ("dog", "dog")):
        j = len(PARENTS[kind])
        mean = np.zeros(4*j+4)
        mean[-4:-1] = [1., 2., 3.]  # Nonzero mapped speed even for a zero network output
        for split, n in (("train", 2), ("test", 1)):
            q = rng.normal(size=(n, 64, j, 4))
            q /= np.linalg.norm(q, axis=-1, keepdims=True)
            q[:, :, 4, 1] = 2e-8  # precision-sensitive saved statistic below old floor
            v = np.ones((n, 64, j, 3))
            heading = np.zeros((n, 64, 1, 4)); heading[..., 0] = 1
            names = [f"{split}{i}" for i in range(n)]
            expanded = {name: rng.normal(size=78) for name in names}
            offsets = {name: np.concatenate((np.zeros((1, 3)), expanded[name].reshape(26, 3)))[NOT_END[kind]][1:].ravel() for name in names}
            np.savez(tmp_path / f"{prefix}{split}.npz", Q=q, V=v, yrot=heading,
                     parents=PARENTS[kind], skel_names=names, skel_offsets=offsets, skel_offsets_withend=expanded)
        std = np.ones(4*j+4); std[17] = 1e-8
        mean[17] = 1e-8
        np.savez(tmp_path / f"{prefix}stats.npz", mean=mean, std=std, min_vel=.01, max_vel=10.,
                 parents=PARENTS[kind], parents_withend=EXPANDED[kind], not_end=NOT_END[kind])
    return tmp_path


def test_archive_splits_precision_signs_and_validation_banks(archives):
    train, test = load_reference_splits("lafan1", archives)
    assert (len(train), len(test)) == (2, 1)
    assert train.num_banks == 3
    assert test.bank_ids.tolist() == [2]
    assert train.reference is test.reference
    assert train.reference.position_indices == [0, 3, 4, 8, 9, 1, 2, 6, 7, 11, 12, 13, 14, 15, 5, 10, 16]
    assert train.body_joints == dict(enumerate(PARTS["lafan1"]))
    assert (train.motion[..., :-4].reshape(2, 64, 22, 4)[..., 0] >= 0).all()
    assert train[0]["motion"].dtype == torch.float32
    assert train.std[0, 17] == 1e-8  # no floor or fitting on these windows
    expected = ((train.motion[0] - train.mean[0]) / train.std[0]).float()
    torch.testing.assert_close(train[0]["motion"], expected, rtol=0, atol=0)
    assert train.subset(torch.tensor([1])).reference is train.reference
    with pytest.raises(ValueError, match="cannot be combined"):
        build_pan_splits({
            "a": {"reference_archive": {"kind": "lafan1", "directory": archives}, "sources": ["lafan1"]},
            "b": {"reference_archive": {"kind": "dog", "directory": archives}},
        })
    with pytest.raises(ValueError, match="no raw mixing"):
        build_pan_splits({"a": {"reference_archive": {"kind": "lafan1", "directory": archives}},
                          "b": {"sources": ["cmu"]}})


def test_strict_windows_signs_and_final_yaw():
    assert list(reference_window_starts(65)) == []
    assert list(reference_window_starts(66)) == [0]
    assert list(reference_window_starts(85)) == [0]
    assert list(reference_window_starts(86)) == [0, 20]
    with pytest.raises(ValueError):
        reference_window_starts(100, stride=0)
    angles = torch.tensor([0., .1, .4, .9], dtype=torch.float64)
    heading = torch.zeros(1, 4, 1, 4, dtype=torch.float64)
    heading[0, :, 0, 0] = (angles/2).cos(); heading[0, :, 0, 2] = (angles/2).sin()
    q = torch.zeros(1, 4, 2, 4, dtype=torch.float64); q[..., 0] = -1
    result = reference_channels(q, torch.zeros(1, 4, 3), heading)
    assert (result[..., [0, 4]] == 1).all()
    torch.testing.assert_close(result[0, :, -1], torch.tensor([.1, .3, .5, .5], dtype=torch.float64))
    flipped = reference_channels(-q, torch.zeros(1, 4, 3), -heading)
    torch.testing.assert_close(result, flipped)


def raw_window(kind, frames=65):
    angles = torch.linspace(-.7, 1.2, frames, dtype=torch.float64).square()
    q = torch.zeros(frames, len(PARENTS[kind]), 4, dtype=torch.float64); q[..., 0] = 1
    yaw = torch.stack(((angles/2).cos(), angles*0, (angles/2).sin(), angles*0), -1)
    base = torch.tensor([2**-.5, 2**-.5, 0., 0.], dtype=torch.float64) if kind == "lafan1" else torch.tensor([1., 0., 0., 0.], dtype=torch.float64)
    q[:, 0] = quat_mul(yaw, base)
    q[::2] *= -1
    offsets = torch.zeros(len(PARENTS[kind]), 3, dtype=torch.float64); offsets[1:, 2] = 2
    root = torch.stack((angles.sin(), angles*0, angles.cos()), -1).cumsum(0)
    return q, root, offsets, torch.tensor(PARENTS[kind])


@pytest.mark.parametrize("kind", ["lafan1", "dog"])
def test_heading_removed_velocity_local_and_window_local_smoothing(kind):
    q, root, offsets, parents = raw_window(kind)
    motion, heading = preprocess_reference_window(q, root, offsets, parents, kind)
    assert motion.shape == (64, 4*len(parents)+4)
    assert (motion[..., :-4].reshape(64, -1, 4)[..., 0] >= 0).all()
    assert not torch.allclose(motion[..., -4:-1], root[1:]-root[:-1])
    assert motion[-1, -1] == motion[-2, -1]
    if kind == "lafan1":
        canonical = motion[..., :4]
        torch.testing.assert_close(canonical, canonical[:1].expand_as(canonical), atol=1e-7, rtol=1e-7)
    else:
        _, shorter_heading = preprocess_reference_window(q[:33], root[:33], offsets, parents, kind)
        assert not torch.allclose(shorter_heading[-8:], heading[24:32], atol=1e-4, rtol=1e-4)


def test_inclusive_fk_end_sites_and_velocity_mapping():
    fk = JointForwardKinematics([-1, 0, 1], inclusive_velocity=True, not_end=[0, 1])
    motion = torch.zeros(1, 3, 12); motion[..., [0, 4]] = 1; motion[..., -4] = 2
    offsets = torch.tensor([[[0., 0., 0.], [0., 1., 0.], [0., 1., 0.]]])
    position = fk.world_positions(fk.bone_vectors_with_absolute_root(motion, offsets))
    torch.testing.assert_close(position[0, :, 0, 0], torch.tensor([2., 4., 6.]))
    torch.testing.assert_close(position[0, :, 2, 1], torch.tensor([2., 2., 2.]))
    motion.requires_grad_()
    fk.world_positions(fk.bone_vectors_with_absolute_root(motion, offsets)).sum().backward()
    assert torch.isfinite(motion.grad).all()
    v = torch.tensor([[3., 0., 4.]], requires_grad=True)
    torch.testing.assert_close(mapped_root_velocity(v, 1., 9.), torch.tensor([[.3, 0., .4]]))
    torch.testing.assert_close(mapped_root_velocity(v * 2, 2., 18.), mapped_root_velocity(v, 1., 9.))
    assert torch.isnan(mapped_root_velocity(torch.zeros(1, 3), 1., 9.)).all()


@pytest.fixture
def method_and_batch(archives):
    old_threads = torch.get_num_threads(); torch.set_num_threads(1)
    torch.manual_seed(3)
    datasets = {kind: load_reference_splits(kind, archives)[0] for kind in PARTS}
    specs = {name: DomainSpec.from_dataset(data) for name, data in datasets.items()}
    config = PANModelConfig.from_dict({
        "motion_encoder": {"embed_dim": 8, "embed_hidden_dim": 16, "num_layers": 1, "num_heads": 1,
                           "feedforward_dim": 16, "kernel_size": 3, "dropout": 0.},
        "motion_decoder": {"kernel_size": 3}, "motion_discriminator": {"kernel_size": 3, "hidden_dim": 16},
    })
    method = PANMethod(config, specs, cfg=PANConfig(position_lambda=.01, cycle_lambda=.001))
    batch = {name: {"motion": torch.stack([data[i]["motion"] for i in [1, 0]]), "bank_id": torch.tensor([1, 0])} for name, data in datasets.items()}
    yield method, batch
    torch.set_num_threads(old_threads)


def test_exact_routes_raw_discriminator_inputs_and_optimizer(method_and_batch):
    method, batch = method_and_batch
    state = method.forward_state(batch)
    assert set(state.pairs) == {("lafan1", "dog"), ("dog", "lafan1")}
    for (src, dst), pair in state.outputs.pairs.items():
        assert torch.equal(pair.target_ids, batch[dst]["bank_id"])
        domain = method.model.domains[dst]
        assert domain.heights is None and domain.ee_ids == []
        bones = state.pairs[(src, dst)].fake_bones
        value = domain.discriminator_input(bones)
        assert value.shape == (2, 64, 27, 3)
        assert not torch.allclose(value, bones)
        torch.testing.assert_close(value, domain.fk.world_positions(bones))
        with torch.no_grad():
            domain.discriminator.conv3.weight.zero_(); domain.discriminator.conv3.bias.fill_(2.)
        assert (domain.discriminator(value) == 2).all()  # no sigmoid
    assert method.buffers == {}
    metrics = method.train_step(batch)
    assert all(np.isfinite(x) for x in metrics.values())
    assert all(torch.isfinite(p).all() for p in method.model.parameters())
    assert all(np.isfinite(x) for x in method.validation_step(batch).values())
    checkpoint = method.state_dict()
    method.load_state_dict(checkpoint)
    checkpoint["metadata"] = {"representation": "legacy"}
    with pytest.raises(ValueError, match="incompatible"):
        method.load_state_dict(checkpoint)
    batch["dog"]["motion"] = batch["dog"]["motion"][:1]
    with pytest.raises(ValueError, match="equally sized"):
        method.model.forward_generator(batch)


def test_selected_losses_separate_root_and_yaw_and_mse_cycle(method_and_batch):
    method, batch = method_and_batch
    state = method.forward_state(batch)
    losses = method.compute_generator_losses(state)
    for name, source in state.sources.items():
        ref = method.domain_specs[name].reference
        shape = (*source.motion.shape[:2], -1, 4)
        expected = F.mse_loss(source.reconstruction[..., :-4].reshape(shape)[:, :, ref.rotation_indices],
                              source.motion[..., :-4].reshape(shape)[:, :, ref.rotation_indices])
        torch.testing.assert_close(losses[f"{name}/reconstruction"], expected)
        torch.testing.assert_close(losses[f"{name}/root"], 100 * F.mse_loss(source.reconstruction[..., -4:], source.motion[..., -4:]))
        # Perturb an omitted quaternion: direct normalized reconstruction must not change.
        omitted = next(j for j in range(method.domain_specs[name].num_joints) if j not in ref.rotation_indices)
        original = source.reconstruction.clone()
        source.reconstruction = original.clone(); source.reconstruction[..., omitted*4:omitted*4+4] += 100
        changed = method.compute_generator_losses(state)
        torch.testing.assert_close(changed[f"{name}/reconstruction"], losses[f"{name}/reconstruction"])
        source.reconstruction = original
    for (src, dst), output in state.outputs.pairs.items():
        route = f"{src}_to_{dst}"
        torch.testing.assert_close(losses[f"{route}/latent_cycle"], .001 * F.mse_loss(output.fake_code, state.outputs.motion_codes[src]))
        fake = method.model.domains[dst].denormalize(output.fake, output.target_ids)
        torch.testing.assert_close(losses[f"{route}/yaw"], 1000 * F.mse_loss(state.sources[src].denormalized[..., -1:], fake[..., -1:]))
    assert all(torch.isfinite(loss) for loss in losses.values())


@pytest.fixture
def upstream(monkeypatch):
    root = os.environ.get("PAN_REFERENCE_ROOT")
    if not root:
        pytest.skip("Set PAN_REFERENCE_ROOT for upstream differential checks")
    monkeypatch.syspath_prepend(root)
    # The reference uses top-level `utils`; isolate from earlier optional oracle tests.
    existing = {key: value for key, value in sys.modules.items() if key == "utils" or key.startswith("utils.")}
    for key in existing:
        del sys.modules[key]
    yield Path(root)
    for key in list(sys.modules):
        if key == "utils" or key.startswith("utils."):
            del sys.modules[key]
    sys.modules.update(existing)


@pytest.mark.parametrize("kind", ["lafan1", "dog"])
def test_upstream_preprocessing_and_archive_channels(upstream, kind):
    from config import Configuration
    from utils import data_utils as du
    from utils.rotation import wrap, quat2pivots, qmultipy, qinv
    key = "hum_joints" if kind == "lafan1" else "dog_joints"
    assert PARTS[kind] == [item[key] for item in Configuration.correspondence]
    q, root, offsets, parents = raw_window(kind)
    local = offsets[None].expand(len(q), -1, -1).clone(); local[:, 0] = root
    rotate = du.rotate_at_each_frame if kind == "lafan1" else du.rotate_at_each_dog_frame
    _, canonical, heading, _ = rotate(local[None, :-1].numpy(), q[None, :-1].numpy(), parents.numpy())
    velocity = du.extract_local_velocities((root[1:]-root[:-1])[None, :, None].numpy(), heading)[:, :, 0]
    expected = reference_channels(torch.from_numpy(canonical), torch.from_numpy(velocity), torch.from_numpy(heading))[0]
    actual, actual_heading = preprocess_reference_window(q, root, offsets, parents, kind)
    torch.testing.assert_close(actual, expected, rtol=1e-7, atol=1e-7)
    torch.testing.assert_close(actual_heading, torch.from_numpy(heading[0]), rtol=1e-12, atol=1e-12)
    prefix, index = ("hum", 5097) if kind == "lafan1" else ("dog", 1114)
    directory = upstream / "data_preprocess/Lafan1_and_dog"
    with np.load(directory / f"{prefix}train.npz", allow_pickle=True) as a:
        q, v, y = a["Q"][index:index+1], a["V"][index:index+1, :, 0], a["yrot"][index:index+1]
    rv = wrap(quat2pivots, wrap(qmultipy, wrap(qinv, y[:, :-1]), y[:, 1:]))
    rv = np.concatenate((rv, rv[:, -1:]), 1).reshape(1, 64, 1)
    oracle = np.concatenate((np.where(q[..., :1] < 0, -q, q).reshape(1, 64, -1), v, rv), -1)
    actual = reference_channels(torch.from_numpy(q), torch.from_numpy(v), torch.from_numpy(y))
    torch.testing.assert_close(actual, torch.from_numpy(oracle), rtol=0, atol=0)
    train, test = load_reference_splits(kind, directory)
    assert (len(train), len(test)) == ((6566, 2074) if kind == "lafan1" else (2769, 779))
    with np.load(directory / f"{prefix}stats.npz") as stats:
        normalized = torch.from_numpy((oracle[0]-stats["mean"])/stats["std"]).float()
    torch.testing.assert_close(train[index]["motion"], normalized, rtol=0, atol=0)
    # Expanded FK parity on real archive motion, including every End Site.
    from utils.utils import ForwardKinematics
    spec = DomainSpec.from_dataset(train)
    ref = spec.reference
    expected_ends = Configuration.hum_end if kind == "lafan1" else Configuration.dog_end
    assert ref.position_indices == ref.not_end[ref.rotation_indices].tolist() + expected_ends
    offsets = ref.offsets_withend[train.bank_ids[index:index+1]]
    fk = JointForwardKinematics(ref.parents_withend.tolist(), True, ref.not_end.tolist())
    motion = actual.float().requires_grad_()
    ours = fk.world_positions(fk.bone_vectors_with_absolute_root(motion, offsets))
    theirs, _ = ForwardKinematics(ref.parents_withend.numpy(), 27, ref.not_end.numpy()).forward(
        motion, offsets[:, None].expand(-1, 64, -1, -1))
    torch.testing.assert_close(ours, theirs, rtol=1e-5, atol=2e-4)
    ours.square().mean().backward()
    ours_grad = motion.grad.clone(); motion.grad = None
    theirs.square().mean().backward()
    torch.testing.assert_close(ours_grad, motion.grad, rtol=1e-4, atol=2e-3)


@pytest.mark.parametrize("kind,folder,stem,start,index", [
    ("lafan1", "LAFAN1", "run1_subject2", 1000, 5097),
    ("dog", "Dog", "D1_010_KAN01_004", 0, 1114),
])
def test_raw_bvh_window_matches_shipped_archive(upstream, tmp_path, kind, folder, stem, start, index):
    root = os.environ.get("PAN_REFERENCE_BVH_ROOT")
    if not root:
        pytest.skip("Set PAN_REFERENCE_BVH_ROOT to the read-only raw data directory")
    from utils.bvh_utils import read_bvh
    source = Path(root) / folder / f"{stem}.bvh"
    frames = 65 if kind == "lafan1" else 130
    # Parse only the controlled window/lookahead, never modify the input BVH.
    with source.open() as stream:
        lines = []
        for line in stream:
            lines.append(f"Frames: {frames}\n" if line.startswith("Frames:") else line)
            if line.startswith("Frame Time:"):
                break
        for _ in range(start):
            next(stream)
        lines.extend(next(stream) for _ in range(frames))
    sliced = tmp_path / "window.bvh"
    sliced.write_text("".join(lines))
    anim = read_bvh(str(sliced), downsample_rate=2 if kind == "dog" else None)
    if kind == "dog":
        anim = anim[0]
    from motion_retargeting.data.adapters.lafan1_bvh import read_lafan_reference_bvh
    clip = read_lafan_reference_bvh(sliced)
    step = 2 if kind == "dog" else 1
    torch.testing.assert_close(clip.rotations[::step], torch.from_numpy(anim.quats), rtol=0, atol=0)
    torch.testing.assert_close(clip.root_position[::step], torch.from_numpy(anim.pos[:, 0]), rtol=0, atol=0)
    actual, _ = preprocess_reference_window(clip.rotations[::step], clip.root_position[::step],
                                            clip.offsets, clip.parents, kind)
    prefix = "hum" if kind == "lafan1" else "dog"
    with np.load(upstream / f"data_preprocess/Lafan1_and_dog/{prefix}train.npz", allow_pickle=True) as archive:
        expected = reference_channels(torch.from_numpy(archive["Q"][index:index+1]),
                                      torch.from_numpy(archive["V"][index:index+1, :, 0]),
                                      torch.from_numpy(archive["yrot"][index:index+1]))[0]
    torch.testing.assert_close(actual, expected, rtol=1e-7, atol=1e-7)


def test_upstream_objective_discriminator_and_generator_gradients(upstream, method_and_batch):
    """Execute the actual reference backward_G on common controlled forward tensors."""
    import ast
    from types import SimpleNamespace
    import loss_function as lf
    from models.functions import get_recloss_input, get_discriminator_input
    from models.networks import LatentDiscriminator

    # Importing architecture_humdog also imports unused Mixamo code that requires
    # removed numpy.core.umath_tests. Execute its UNMODIFIED backward_G AST instead.
    path = upstream / "models/architecture_humdog.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PAN_model")
    backward = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "backward_G")
    namespace = {"torch": torch, "get_recloss_input": get_recloss_input, "get_discriminator_input": get_discriminator_input}
    exec(compile(ast.Module(body=[backward], type_ignores=[]), str(path), "exec"), namespace)
    from motion_retargeting.methods.pan.losses import discriminator_loss

    method, batch = method_and_batch
    method.set_discriminator_requires_grad(False)
    state = method.forward_state(batch)
    names = list(state.sources)
    recorded, upstream_models = {}, []
    for name in names:
        domain = method.model.domains[name]
        ref_dis = LatentDiscriminator(3, 3, len(domain.fk.parents) * 3, 16)
        for dst, src in zip(ref_dis.layers, (domain.discriminator.conv1, domain.discriminator.conv2, domain.discriminator.conv3)):
            dst[0].load_state_dict(src.state_dict())
        ref_dis.requires_grad_(False)
        positions = domain.discriminator_input(state.sources[name].bones).detach().requires_grad_()
        actual = domain.discriminator(positions)
        expected = ref_dis(positions.flatten(2).transpose(1, 2)).squeeze(1)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        ours_grad = torch.autograd.grad(actual.square().mean(), positions, retain_graph=True)[0]
        ref_grad = torch.autograd.grad(expected.square().mean(), positions)[0]
        torch.testing.assert_close(ours_grad, ref_grad, rtol=0, atol=0)
        reference = method.domain_specs[name].reference
        upstream_models.append(SimpleNamespace(indices=reference.rotation_indices,
                                               indices_withend=reference.position_indices, discriminator=ref_dis))
    sources = [state.sources[name] for name in names]
    pairs = list(state.pairs)
    fake_denorm = [method.model.domains[dst].denormalize(state.outputs.pairs[(src, dst)].fake, state.pairs[(src, dst)].target_ids) for src, dst in pairs]
    oracle = SimpleNamespace(
        args=SimpleNamespace(rec_loss_type="norm_rec", retar_vel_matching="mapping", dis=True,
                             dis_mode="denorm_pos", lambda_rec=1., lambda_cycle=.001, lambda_retar_vel=1000.),
        n_topology=2, device="cpu", models=upstream_models,
        datasets=[SimpleNamespace(njoints=method.domain_specs[name].num_joints,
                                  min_vel=np.array(method.domain_specs[name].reference.min_velocity),
                                  max_vel=np.array(method.domain_specs[name].reference.max_velocity)) for name in names],
        criterion_rec=lf.caloutputloss, criterion_root=F.mse_loss, criterion_kine=lf.calposloss,
        criterion_cycle=lf.cycle_motions, criterion_cycle_latent=lf.cycle_latents,
        criterion_root_v=F.mse_loss, mse=F.mse_loss, criterion_gan=lf.dis_loss_l2,
        loss_recoder=SimpleNamespace(add_scalar=lambda key, value: recorded.update({key: value})),
        motion=[source.motion.transpose(1, 2) for source in sources],
        motion_denorm=[source.denormalized for source in sources],
        rec=[source.reconstruction for source in sources],
        gt_pos=[method.model.domains[name].fk.world_positions(state.sources[name].bones) for name in names],
        rec_pos=[method.model.domains[name].fk.world_positions(state.sources[name].reconstruction_bones) for name in names],
        cyc_pos=[method.model.domains[src].fk.world_positions(state.pairs[(src, dst)].cycle_bones) for src, dst in pairs],
        retar_latents=[state.outputs.motion_codes[name] for name in names],
        cyc_latents=[state.outputs.pairs[key].fake_code for key in pairs],
        fake_retar_denorm=fake_denorm,
        fake_pos=[method.model.domains[dst].discriminator_input(state.pairs[(src, dst)].fake_bones) for src, dst in pairs],
    )
    losses = method.compute_generator_losses(state)
    parameters = list(method.model.generator_parameters())
    gradients = torch.autograd.grad(sum(losses.values()), parameters, retain_graph=True)
    namespace["backward_G"](oracle)
    torch.testing.assert_close(sum(losses.values()), oracle.loss_G_total, rtol=2e-6, atol=2e-5)
    for param, gradient in zip(parameters, gradients):
        torch.testing.assert_close(param.grad, gradient, rtol=2e-4, atol=1e-4)
    for index, name in enumerate(names):
        torch.testing.assert_close(losses[f"{name}/reconstruction"], recorded[f"rec_loss_quater_{index}"])
        torch.testing.assert_close(losses[f"{name}/root"], 100 * recorded[f"rec_loss_global_{index}"])
        torch.testing.assert_close(losses[f"{name}/position"], .01 * recorded[f"rec_loss_position_{index}"])
    for index, (src, dst) in enumerate(pairs):
        i, j = names.index(src), names.index(dst)
        route = f"{src}_to_{dst}"
        torch.testing.assert_close(losses[f"{route}/position_cycle"], .001 * recorded[f"cycle_loss_{i}_{j}"])
        torch.testing.assert_close(losses[f"{route}/latent_cycle"], .001 * recorded[f"cycle_latent_loss_{i}_{j}"])
        torch.testing.assert_close(losses[f"{route}/root_velocity"] + losses[f"{route}/yaw"], 1000 * recorded[f"retar_root_v_loss_{i}_{j}"])
        torch.testing.assert_close(losses[f"{route}/adversarial"], recorded[f"G_loss_{i}_{j}"])
        real_scores = oracle.models[j].discriminator(oracle.gt_pos[j].flatten(2).transpose(1, 2))
        fake_scores = oracle.models[j].discriminator(oracle.fake_pos[index].detach().flatten(2).transpose(1, 2))
        torch.testing.assert_close(discriminator_loss(real_scores, fake_scores),
                                   .5 * (lf.dis_loss_l2(real_scores, True) + lf.dis_loss_l2(fake_scores, False)))

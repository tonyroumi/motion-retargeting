"""
SAN reference recipe's routes, objective and update order on synthetic domains, with reference scalar
parity checks that run when SAN_REFERENCE_ROOT is set.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import matplotlib

matplotlib.use("Agg")

import pytest
import torch
import torch.nn.functional as F
import yaml

from motion_retargeting.data.adapters import MotionSourceAdapter, base
from motion_retargeting.data.representations.motion import MotionCollection, MotionData
from motion_retargeting.data.representations.skeleton import SkeletonTopology
from motion_retargeting.methods.pan.kinematics import END_EFFECTOR_VELOCITY_SCALE
from motion_retargeting.methods.san.config import (
    DomainSpec,
    SANConfig,
    SANModelConfig,
    SANReferenceConfig,
    SANReferenceDomainSpec,
)
from motion_retargeting.methods.san.method import SANMethod, SANReferenceMethod
from motion_retargeting.methods.san.model import SANReferenceModel
from motion_retargeting.methods.san.utils import (
    SANReferenceGeneratorOutputs,
    SANReferencePairOutputs,
    select_banks,
)
from motion_retargeting.training.trainer import Trainer
from san_reference_support import (
    assert_close,
    copy_reference_weights,
    discriminator_parameter_pairs,
    generator_parameter_pairs,
    load_reference,
    reference_edges,
    requires_reference,
)

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "scripts" / "train.py"
REFERENCE_CONFIG = ROOT / "configs" / "methods" / "san_reference.yaml"
LEGACY_CONFIG = ROOT / "configs" / "methods" / "san.yaml"

# The AMASS and CMU skeletons of configs/methods/san_reference.yaml.
AMASS_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
CMU_PARENTS = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 13, 14, 11, 16, 17, 18, 11, 20, 21, 22]

BATCH_SIZE = 4
WINDOW = 64
NUM_BANKS = {"group_a": 3, "group_b": 2}
PARENTS = {"group_a": AMASS_PARENTS, "group_b": CMU_PARENTS}


def make_spec(parents, num_banks: int, generator: torch.Generator) -> SANReferenceDomainSpec:
    num_joints = len(parents)
    channels = 4 * (num_joints - 1) + 3

    offsets = 0.1 * torch.randn(num_banks, num_joints, 3, generator=generator)
    offsets[:, 0] = 0.0
    # Rotations stay near identity once denormalized, with banks visibly different from each other.
    mean = torch.zeros(num_banks, channels)
    mean[:, ::4] = 1.0
    mean[:, -3:] = torch.randn(num_banks, 3, generator=generator)
    # Small enough that an untrained decoder's generated velocities include contact frames.
    std = 0.001 + 0.002 * torch.rand(num_banks, channels, generator=generator)

    return SANReferenceDomainSpec(parents=torch.tensor(parents), offsets=offsets, mean=mean, std=std)


def make_specs(seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    return {name: make_spec(PARENTS[name], NUM_BANKS[name], generator) for name in PARENTS}


def make_batch(specs, batch_size: int = BATCH_SIZE, window: int = WINDOW, seed: int = 1):
    generator = torch.Generator().manual_seed(seed)
    batch = {}
    for name, spec in specs.items():
        batch[name] = {
            "motion": torch.randn(batch_size, window, spec.mean.shape[1], generator=generator),
            "bank_id": torch.randint(spec.num_banks, (batch_size,), generator=generator),
        }
    return batch


def make_model(specs=None, seed: int = 0) -> SANReferenceModel:
    torch.manual_seed(seed)
    return SANReferenceModel(SANModelConfig(), specs if specs is not None else make_specs())


def structure_config() -> SANModelConfig:
    cfg = SANModelConfig()
    cfg.motion_discriminator.train_structure = True
    return cfg


def make_method(specs=None, seed: int = 0, **cfg) -> SANReferenceMethod:
    torch.manual_seed(seed)
    return SANReferenceMethod(
        model_cfg=structure_config(),
        domain_specs=specs if specs is not None else make_specs(),
        device="cpu",
        cfg=SANReferenceConfig(**cfg),
    )


def fixed_outputs(specs, seed: int = 2) -> SANReferenceGeneratorOutputs:
    """ Generator outputs that do not depend on the networks, so losses can be pinned on their own. """
    generator = torch.Generator().manual_seed(seed)

    def motion(name):
        return torch.randn(BATCH_SIZE, WINDOW, specs[name].mean.shape[1], generator=generator)

    def code():
        return torch.randn(BATCH_SIZE, 112, WINDOW // 4, generator=generator)

    return SANReferenceGeneratorOutputs(
        motion_codes={name: code() for name in specs},
        reconstructions={name: motion(name) for name in specs},
        pairs={
            (source, destination): SANReferencePairOutputs(
                # Not palindromic, and more than one bank, so a wrong or reordered bank changes the result.
                target_ids=torch.arange(BATCH_SIZE) % specs[destination].num_banks,
                fake=motion(destination),
                fake_code=code(),
            )
            for source in specs
            for destination in specs
        },
    )


def losses_on_fixed_outputs(method: SANReferenceMethod, batch, outputs) -> dict:
    with mock.patch.object(method.model, "forward_generator", return_value=outputs):
        return method.compute_generator_losses(method.forward_state(batch))


class TestRoutes:
    def test_all_four_routes_exist(self):
        model = make_model()
        outputs = model.forward_generator(make_batch(model.specs))

        assert set(outputs.pairs) == {
            ("group_a", "group_a"), ("group_a", "group_b"), ("group_b", "group_a"), ("group_b", "group_b"),
        }
        pair = outputs.pairs[("group_a", "group_b")]
        assert pair.fake.shape == (BATCH_SIZE, WINDOW, 95)
        assert pair.fake_code.shape == (BATCH_SIZE, 112, WINDOW // 4)
        assert outputs.reconstructions["group_a"].shape == (BATCH_SIZE, WINDOW, 87)

    def test_routes_have_no_decode_back_to_the_source(self):
        model = make_model()
        calls = {name: 0 for name in model.domains}
        for name, domain in model.domains.items():
            domain.decoder.register_forward_pre_hook(
                lambda module, args, name=name: calls.__setitem__(name, calls[name] + 1)
            )

        model.forward_generator(make_batch(model.specs))

        # One reconstruction per domain, then each domain's decoder as the destination of two routes.
        assert calls == {"group_a": 3, "group_b": 3}

    def test_each_domains_banks_are_encoded_once(self):
        model = make_model()
        calls = []
        for domain in model.domains.values():
            domain.offset_encoder.register_forward_hook(lambda module, args, output: calls.append(args[0].shape[0]))

        model.forward_generator(make_batch(model.specs))

        assert calls == [NUM_BANKS["group_a"], NUM_BANKS["group_b"]]

    def test_every_route_decodes_and_re_encodes_on_its_drawn_banks(self):
        model = make_model()
        model.eval()
        batch = make_batch(model.specs, batch_size=16)

        with torch.no_grad():
            outputs = model.forward_generator(batch)
            for (source, destination), pair in outputs.pairs.items():
                domain = model.domains[destination]
                static = select_banks(domain.encode_offsets(), pair.target_ids)
                fake = domain.decode(outputs.motion_codes[source], static)
                assert torch.equal(pair.fake, fake)
                assert torch.equal(pair.fake_code, domain.encode(fake, static))

    def test_same_domain_routes_draw_a_bank_rather_than_reuse_the_sources(self):
        model = make_model()
        batch = make_batch(model.specs, batch_size=32)
        outputs = model.forward_generator(batch)

        for name in model.domains:
            assert not torch.equal(outputs.pairs[(name, name)].target_ids, batch[name]["bank_id"])
            assert not torch.equal(outputs.pairs[(name, name)].fake, outputs.reconstructions[name])

    def test_seeded_draws_are_uniform_over_banks_and_repeatable(self):
        model = make_model()
        batch = make_batch(model.specs, batch_size=600, window=16)
        # A lopsided source batch must not move the target draw.
        batch["group_a"]["bank_id"] = torch.zeros(600, dtype=torch.long)

        with torch.no_grad():
            torch.manual_seed(3)
            outputs = model.forward_generator(batch)
            torch.manual_seed(3)
            repeated = model.forward_generator(batch)

        counts = outputs.pairs[("group_b", "group_a")].target_ids.bincount(minlength=3).float()
        assert ((counts - 200).abs() < 60).all()
        for route, pair in outputs.pairs.items():
            assert torch.equal(pair.target_ids, repeated.pairs[route].target_ids)

    def test_no_code_or_fake_is_detached(self):
        model = make_model()
        outputs = model.forward_generator(make_batch(model.specs))

        for code in outputs.motion_codes.values():
            assert code.grad_fn is not None
        for pair in outputs.pairs.values():
            assert pair.fake.grad_fn is not None and pair.fake_code.grad_fn is not None


class TestObjective:
    def test_terms_and_keys(self):
        method = make_method()
        losses = method.compute_generator_losses(method.forward_state(make_batch(method.domain_specs)))

        expected = {f"{name}/{term}" for name in PARENTS for term in ("reconstruction", "root", "position")}
        routes = [f"{s}_to_{d}" for s in PARENTS for d in PARENTS]
        expected |= {f"{route}/{term}" for route in routes for term in ("latent", "end_effector", "contact")}
        expected |= {"group_a_to_group_b/adversarial", "group_b_to_group_a/adversarial"}
        assert set(losses) == expected

    def test_each_weight_is_pinned_separately(self):
        specs = make_specs()
        batch = make_batch(specs)
        outputs = fixed_outputs(specs)
        unit = losses_on_fixed_outputs(
            make_method(
                specs,
                reconstruction_lambda=1, root_lambda=1, position_lambda=1, latent_lambda=1,
                end_effector_lambda=1, contact_lambda=1, adversarial_lambda=1,
                contact_threshold=1e6,
            ),
            batch,
            outputs,
        )
        weighted = losses_on_fixed_outputs(make_method(specs, contact_threshold=1e6), batch, outputs)

        weights = {
            "reconstruction": 5, "root": 1250, "position": 500, "latent": 2.5,
            "end_effector": 50, "contact": 5000, "adversarial": 1,
        }
        for key, value in weighted.items():
            assert value.item() == pytest.approx(weights[key.split("/")[1]] * unit[key].item(), rel=1e-5), key

    def test_unit_terms_match_hand_computation(self):
        specs = make_specs()
        batch = make_batch(specs)
        outputs = fixed_outputs(specs)
        method = make_method(specs, contact_threshold=1e6)
        losses = losses_on_fixed_outputs(method, batch, outputs)

        name = "group_a"
        spec = specs[name]
        fk = method.model.domains[name].fk
        ids = batch[name]["bank_id"]
        motion, reconstruction = batch[name]["motion"], outputs.reconstructions[name]
        mean, std = spec.mean[ids][:, None], spec.std[ids][:, None]
        real_denormalized, rec_denormalized = motion * std + mean, reconstruction * std + mean
        heights = spec.heights[ids]
        h2, h3 = heights[:, None, None], heights[:, None, None, None]

        assert losses["group_a/reconstruction"].item() == pytest.approx(
            5 * ((reconstruction - motion) ** 2).mean().item(), rel=1e-5
        )
        assert losses["group_a/root"].item() == pytest.approx(
            1250 * (((rec_denormalized[..., -3:] - real_denormalized[..., -3:]) / h2) ** 2).mean().item(), rel=1e-5
        )

        real_bones = fk.bone_vectors_with_absolute_root(real_denormalized, spec.offsets[ids])
        rec_bones = fk.bone_vectors_with_absolute_root(rec_denormalized, spec.offsets[ids])
        difference = fk.root_relative_with_absolute_root(rec_bones) - fk.root_relative_with_absolute_root(real_bones)
        assert losses["group_a/position"].item() == pytest.approx(500 * ((difference / h3) ** 2).mean().item(), rel=1e-5)

        # A fake is measured over the target bank's height and the source over its own.
        pair = outputs.pairs[("group_a", "group_b")]
        target = specs["group_b"]
        target_fk = method.model.domains["group_b"].fk
        fake_denormalized = pair.fake * target.std[pair.target_ids][:, None] + target.mean[pair.target_ids][:, None]
        fake_world = target_fk.world_positions(
            target_fk.bone_vectors_with_absolute_root(fake_denormalized, target.offsets[pair.target_ids])
        )

        def velocity(world, ee_ids, h):
            ends = world[:, :, ee_ids]
            return (ends[:, 1:] - ends[:, :-1]) * END_EFFECTOR_VELOCITY_SCALE / h

        fake_velocity = velocity(fake_world, target.ee_ids, target.heights[pair.target_ids][:, None, None, None])
        real_velocity = velocity(fk.world_positions(real_bones), spec.ee_ids, h3)
        assert losses["group_a_to_group_b/end_effector"].item() == pytest.approx(
            50 * ((fake_velocity - real_velocity) ** 2).mean().item(), rel=1e-5
        )
        # Every sample is a contact at this threshold, so the contact term repeats the transfer's error.
        assert losses["group_a_to_group_b/contact"].item() == pytest.approx(
            100 * losses["group_a_to_group_b/end_effector"].item(), rel=1e-5
        )
        assert losses["group_a_to_group_b/latent"].item() == pytest.approx(
            2.5 * F.l1_loss(outputs.motion_codes["group_a"], pair.fake_code).item(), rel=1e-5
        )

    def test_contacts_are_selected_on_the_generated_velocity(self):
        method = make_method()
        state = method.forward_state(make_batch(method.domain_specs))
        pair = state.pairs[("group_a", "group_b")]
        selected = torch.linalg.vector_norm(pair.fake_velocity, dim=-1) < method.cfg.contact_threshold

        source = state.sources["group_a"]
        source_velocity = method.normalized_end_effector_velocity("group_a", source.bones, source.bank_ids)
        expected = F.mse_loss(pair.fake_velocity[selected], source_velocity[selected])

        losses = method.compute_generator_losses(state)
        assert losses["group_a_to_group_b/contact"].item() == pytest.approx(5000 * expected.item(), rel=1e-5)

    def test_scaling_offsets_and_root_together_leaves_position_terms_unchanged(self):
        specs = make_specs()
        scale = 3.0

        def scaled(spec):
            mean, std = spec.mean.clone(), spec.std.clone()
            mean[:, -3:] *= scale
            std[:, -3:] *= scale
            return SANReferenceDomainSpec(parents=spec.parents, offsets=spec.offsets * scale, mean=mean, std=std)

        batch = make_batch(specs)
        outputs = fixed_outputs(specs)
        original = losses_on_fixed_outputs(make_method(specs, contact_threshold=1e6), batch, outputs)
        rescaled = losses_on_fixed_outputs(
            make_method({name: scaled(spec) for name, spec in specs.items()}, contact_threshold=1e6),
            batch,
            outputs,
        )

        for key in original:
            if key.split("/")[1] in ("root", "position", "end_effector", "contact"):
                assert rescaled[key].item() == pytest.approx(original[key].item(), rel=1e-4), key

    def test_contact_fraction_is_reported_per_route(self):
        method = make_method()
        state = method.forward_state(make_batch(method.domain_specs))
        fractions = method.contact_fractions(state)

        assert set(fractions) == {f"{s}_to_{d}/contact_fraction" for s in PARENTS for d in PARENTS}
        assert all(0.0 <= value <= 1.0 for value in fractions.values())

    def test_replay_holds_bones_and_real_bones_are_scored_first(self):
        method = make_method()
        state = method.forward_state(make_batch(method.domain_specs))
        losses = method.compute_discriminator_losses(state)

        assert set(losses) == {"group_a/discriminator", "group_b/discriminator"}
        assert method.buffers["group_a"].items[0].shape == (1, WINDOW, len(AMASS_PARENTS), 3)


def snapshot(parameters) -> list:
    return [parameter.detach().clone() for parameter in parameters]


def changed(parameters, before) -> int:
    return sum(not torch.equal(parameter, old) for parameter, old in zip(parameters, before))


class TestUpdateOrder:
    def test_optimizers_are_reference_adam(self):
        method = make_method()
        for optimizer in (method.generator_optimizer, method.discriminator_optimizer):
            group = optimizer.param_groups[0]
            assert (group["lr"], tuple(group["betas"])) == (2e-4, (0.9, 0.999))
            assert (group["weight_decay"], group["amsgrad"]) == (0, False)

    def test_each_step_changes_only_its_own_networks(self):
        method = make_method()
        method.train()
        generator = list(method.model.generator_parameters())
        discriminator = list(method.model.discriminator_parameters())
        assert (len(generator), len(discriminator)) == (40, 18)

        before = (snapshot(generator), snapshot(discriminator))
        after_generator_step = {}
        step = method.generator_optimizer.step

        def spy():
            step()
            after_generator_step["generator"] = snapshot(generator)
            after_generator_step["discriminator"] = snapshot(discriminator)
            after_generator_step["discriminator_grads"] = [p.grad for p in discriminator]
            after_generator_step["frozen"] = [p.requires_grad for p in discriminator]

        with mock.patch.object(method.generator_optimizer, "step", side_effect=spy):
            method.train_step(make_batch(method.domain_specs))

        # The generator's 4 unchanged tensors are the static encoder's dead last layer, in both domains.
        assert changed(after_generator_step["generator"], before[0]) == 36
        assert changed(after_generator_step["discriminator"], before[1]) == 0
        assert all(grad is None for grad in after_generator_step["discriminator_grads"])
        assert not any(after_generator_step["frozen"])

        assert changed(discriminator, after_generator_step["discriminator"]) == 18
        assert changed(generator, after_generator_step["generator"]) == 0
        assert all(parameter.requires_grad for parameter in discriminator)

    def test_dead_static_layers_get_no_gradient(self):
        method = make_method()
        method.train_step(make_batch(method.domain_specs))
        for domain in method.model.domains.values():
            for parameter in domain.offset_encoder.block2.parameters():
                assert parameter.grad is None

    def test_one_generator_forward_per_step(self):
        method = make_method()
        with mock.patch.object(
            method.model, "forward_generator", wraps=method.model.forward_generator
        ) as forward_generator:
            method.train_step(make_batch(method.domain_specs))
        assert forward_generator.call_count == 1

    def test_fakes_entering_the_buffers_keep_their_values_from_before_the_generator_step(self):
        method = make_method()
        forward_state = method.forward_state
        captured = {}

        def spy(batch):
            state = forward_state(batch)
            captured.update({
                destination: pair.fake_bones.detach().clone()
                for (source, destination), pair in state.pairs.items()
                if source != destination
            })
            return state

        entered = {}
        for name, buffer in method.buffers.items():
            query = buffer.query
            buffer.query = lambda items, query=query, name=name: (entered.__setitem__(name, items.clone()), query(items))[1]

        with mock.patch.object(method, "forward_state", side_effect=spy):
            method.train_step(make_batch(method.domain_specs))

        assert set(entered) == {"group_a", "group_b"}
        for name, fake in entered.items():
            assert torch.equal(fake, captured[name])
            assert not fake.requires_grad

    def test_empty_contact_selection_raises_before_either_optimizer_changes_a_parameter(self):
        method = make_method(contact_threshold=0.0)
        parameters = list(method.model.parameters())
        before = snapshot(parameters)

        with pytest.raises(RuntimeError, match="contact threshold"):
            method.train_step(make_batch(method.domain_specs))

        assert changed(parameters, before) == 0
        assert all(buffer.num_items == 0 for buffer in method.buffers.values())

    def test_metrics(self):
        method = make_method(loss_grad_norm_interval=1)
        metrics = method.train_step(make_batch(method.domain_specs))

        for key in (
            "total/generator",
            "total/discriminator",
            "group_a_to_group_b/contact_fraction",
            "loss_grad/group_a/reconstruction",
            "loss_grad/group_a/discriminator",
            "grad/group_a/motion_encoder",
            "grad/group_b/offset_encoder",
            "grad/group_a/decoder",
            "grad/group_b/discriminator",
        ):
            assert key in metrics, key
        assert all(isinstance(value, float) and value == value for value in metrics.values())


class TestValidation:
    def test_metrics_per_source_domain_leave_parameters_alone(self):
        method = make_method()
        method.eval()
        parameters = list(method.model.parameters())
        before = snapshot(parameters)

        metrics = method.validation_step(make_batch(method.domain_specs))

        assert set(metrics) == {
            f"{name}/val/{metric}"
            for name in PARENTS
            for metric in (
                "reconstruction_mse",
                "reconstruction_mpjpe_m",
                "reconstruction_squared_error_over_height2",
                "retarget_latent_l1",
                "retarget_realism",
                "retarget_ee_velocity",
            )
        }
        assert all(isinstance(value, float) for value in metrics.values())
        assert changed(parameters, before) == 0

    def test_validation_does_not_need_a_generated_contact(self):
        method = make_method(contact_threshold=0.0)
        method.eval()
        method.validation_step(make_batch(method.domain_specs))

    def test_metric_fixture_tells_mpjpe_from_squared_error_over_height(self):
        specs = make_specs()
        method = make_method(specs)
        method.eval()
        batch = make_batch(specs)
        batch["group_a"]["bank_id"] = torch.zeros(BATCH_SIZE, dtype=torch.long)

        # Shift the reconstructed root by 3 cm along X and 4 cm along Y: every joint is 5 cm off.
        spec = specs["group_a"]
        shift = torch.tensor([0.03, 0.04, 0.0])
        reconstruction = batch["group_a"]["motion"].clone()
        reconstruction[..., -3:] += shift / spec.std[0, -3:]

        outputs = fixed_outputs(specs)
        outputs.reconstructions["group_a"] = reconstruction

        with mock.patch.object(method.model, "forward_generator", return_value=outputs):
            metrics = method.validation_step(batch)

        height = spec.heights[0].item()
        assert metrics["group_a/val/reconstruction_mpjpe_m"] == pytest.approx(0.05, rel=1e-3)
        # Mean over the three coordinates of the squared error, over height squared.
        assert metrics["group_a/val/reconstruction_squared_error_over_height2"] == pytest.approx(
            (0.03 ** 2 + 0.04 ** 2) / 3 / height ** 2, rel=1e-3
        )


class TestCheckpoints:
    def test_save_and_reload_gives_equal_outputs_with_the_trained_structure(self):
        specs = make_specs()
        method = make_method(specs)
        batch = make_batch(specs)
        method.train_step(batch)

        restored = make_method(specs, seed=9)
        restored.load_state_dict(method.state_dict())

        method.eval()
        restored.eval()
        with torch.no_grad():
            torch.manual_seed(4)
            ours = method.forward_state(batch)
            torch.manual_seed(4)
            theirs = restored.forward_state(batch)

        assert torch.equal(ours.sources["group_a"].reconstruction, theirs.sources["group_a"].reconstruction)
        pair, restored_pair = ours.pairs[("group_a", "group_b")], theirs.pairs[("group_a", "group_b")]
        assert torch.equal(pair.fake_bones, restored_pair.fake_bones)
        domain, restored_domain = method.model.domains["group_b"], restored.model.domains["group_b"]
        assert torch.equal(domain.discriminate(pair.fake_bones), restored_domain.discriminate(pair.fake_bones))

    def test_metadata_records_what_the_weights_mean(self):
        metadata = make_method().state_dict()["recipe"]
        assert metadata["recipe"] == "mixamo_reference"
        assert metadata["reference_commit"].startswith("74fff8f")
        assert metadata["representation"] and metadata["physical_units"] and metadata["normalization_scope"]
        assert "discriminator_masks_and_pools_trained" in metadata["kept_accidents"]
        assert metadata["domains"]["group_a"] == {
            "parents": AMASS_PARENTS,
            "num_skeleton_banks": 3,
            "end_effector_ids": [10, 11, 15, 20, 21],
        }

    def test_legacy_checkpoint_gives_a_clear_error(self):
        legacy = SANMethod(
            SANModelConfig(),
            {
                name: DomainSpec(parents=torch.tensor(PARENTS[name]), num_joints=len(PARENTS[name]))
                for name in PARENTS
            },
            cfg=SANConfig(),
        )
        with pytest.raises(ValueError, match="legacy SAN recipe"):
            make_method().load_state_dict(legacy.state_dict())

    def test_incompatible_setup_is_refused(self):
        state = make_method().state_dict()
        generator = torch.Generator().manual_seed(0)
        other = {name: make_spec(PARENTS[name], 5, generator) for name in PARENTS}
        with pytest.raises(ValueError, match="domains"):
            make_method(other).load_state_dict(state)


def smooth_clip(num_joints: int, offset_scale: float, seed: int) -> MotionData:
    """ Near-rest motion, slow enough to hold contact frames. """
    generator = torch.Generator().manual_seed(seed)
    frames = 130

    rotations = torch.zeros(frames, num_joints, 4)
    rotations[..., 0] = 1.0
    phase = torch.linspace(0, 6.28, frames)[:, None, None]
    rotations = rotations + 0.01 * torch.randn(1, num_joints, 4, generator=generator) * phase.sin()
    rotations = F.normalize(rotations, dim=-1)

    offsets = offset_scale * 0.1 * torch.ones(num_joints, 3)
    offsets[0] = 0.0
    root = torch.linspace(0, 1, frames)[:, None] * torch.tensor([0.1, 0.0, 0.1]) + torch.tensor([0.0, 0.9, 0.0])

    return MotionData(offsets=offsets, joint_rotations=rotations, fps=30.0, root_position=root)


def write_config(path: Path, tmp_path: Path) -> Path:
    cfg = yaml.safe_load(path.read_text())
    cfg["device"] = "cpu"
    cfg["trainer"].update(batch_size=2, num_epochs=1, validation_interval=1, checkpoint_interval=1, log_dir=str(tmp_path / "logs"), use_tensorboard=False, verbose=False)
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(cfg))
    return config


def run_training(config: Path, monkeypatch) -> SimpleNamespace:
    """ Runs scripts/train.py as a script, returning the method and trainer it built. """
    captured = SimpleNamespace()
    train = Trainer.train

    def spy(self):
        captured.method, captured.trainer = self.method, self
        return train(self)

    monkeypatch.setattr(Trainer, "train", spy)
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", str(config)])
    runpy.run_path(str(TRAIN_SCRIPT), run_name="__main__")
    return captured


class TestEndToEnd:
    def test_reference_config_reaches_the_reference_classes(self, tmp_path, monkeypatch, capsys):
        class Source:
            def __init__(self, name):
                parents = {"amass": AMASS_PARENTS, "cmu": CMU_PARENTS}[name]
                self.collection = MotionCollection(
                    topology=SkeletonTopology(parents=torch.tensor(parents)),
                    motions=[smooth_clip(len(parents), scale, seed) for seed, scale in enumerate((1.0, 1.0, 1.5))],
                )

            def load(self):
                return self.collection

        monkeypatch.setattr("motion_retargeting.methods.pan.data.build_adapter", Source)

        run = run_training(write_config(REFERENCE_CONFIG, tmp_path), monkeypatch)

        assert type(run.method) is SANReferenceMethod
        assert isinstance(run.method.domain_specs["group_a"], SANReferenceDomainSpec)
        assert isinstance(run.method.cfg, SANReferenceConfig)
        assert run.method.domain_specs["group_a"].num_banks == 2
        assert len(list(run.method.model.discriminator_parameters())) == 18

        # Two clips share a bank (3 windows each) and the third has its own.
        assert "group_a: windows per bank: least 3, median 3, most 6" in capsys.readouterr().out

        checkpoint = torch.load(run.trainer.checkpoint_dir / "final.pt", weights_only=True)
        assert checkpoint["method"]["recipe"]["recipe"] == "mixamo_reference"

        gifs = list(run.trainer.visualization_dir.rglob("*.gif"))
        assert {gif.name for gif in gifs} == {"group_a_to_group_b.gif", "group_b_to_group_a.gif"}

        run.trainer.load_checkpoint(run.trainer.checkpoint_dir / "final.pt")

    def test_existing_san_config_still_runs_the_legacy_method(self, tmp_path, monkeypatch):
        monkeypatch.setattr(base, "PATHS", SimpleNamespace(data_root=tmp_path))
        monkeypatch.setattr(base, "label_joint_bodies", lambda topology, offsets: {
            0: list(range(topology.num_joints // 2)),
            1: list(range(topology.num_joints // 2, topology.num_joints)),
        })

        class Source(MotionSourceAdapter):
            def __init__(self, name):
                super().__init__(name)
                self.parents = {"amass": AMASS_PARENTS, "cmu": CMU_PARENTS}[name]

            def _extract(self):
                clips = [smooth_clip(len(self.parents), 1.0, seed) for seed in range(30)]
                return [
                    self._construct_motion(
                        parents=self.parents,
                        offsets=clip.offsets,
                        joint_rotations=clip.joint_rotations,
                        root_position=clip.root_position,
                        fps=clip.fps,
                    )
                    for clip in clips
                ]

        monkeypatch.setattr("motion_retargeting.data.factory.build_adapter", Source)

        run = run_training(write_config(LEGACY_CONFIG, tmp_path), monkeypatch)

        assert type(run.method) is SANMethod
        assert run.method.cfg.recipe == "legacy"
        assert "recipe" not in torch.load(run.trainer.checkpoint_dir / "final.pt", weights_only=True)["method"]

        batch = next(iter(run.trainer.train_loader))
        assert {"motion", "offsets", "q_heading"} <= set(batch["group_a"])


def reached(method: SANReferenceMethod, loss: torch.Tensor) -> set:
    """ The modules a loss term has a nonzero gradient on, named like the reference report's table. """
    modules = {}
    for letter, name in (("A", "group_a"), ("B", "group_b")):
        domain = method.model.domains[name]
        modules[f"enc_{letter}"] = domain.motion_encoder
        modules[f"dec_{letter}"] = domain.decoder
        modules[f"static_{letter}"] = domain.offset_encoder

    found = set()
    for label, module in modules.items():
        grads = torch.autograd.grad(loss, list(module.parameters()), retain_graph=True, allow_unused=True)
        if any(grad is not None and grad.abs().sum() > 0 for grad in grads):
            found.add(label)
    return found


class TestGradientReach:
    """ The per-term table of the reference report, measured there on the real graph at HEAD. """

    A = {"enc_A", "dec_A", "static_A"}
    B = {"enc_B", "dec_B", "static_B"}

    @pytest.mark.parametrize(
        "term, expected",
        [
            ("group_a/reconstruction", A),
            ("group_a/root", A),
            ("group_a/position", A),
            ("group_b/reconstruction", B),
            ("group_b/root", B),
            ("group_b/position", B),
            ("group_a_to_group_a/latent", A),
            ("group_a_to_group_a/end_effector", A),
            ("group_a_to_group_b/latent", {"enc_A", "static_A", "enc_B", "dec_B", "static_B"}),
            ("group_a_to_group_b/end_effector", {"enc_A", "static_A", "dec_B", "static_B"}),
            ("group_a_to_group_b/adversarial", {"enc_A", "static_A", "dec_B", "static_B"}),
            ("group_b_to_group_a/latent", {"enc_A", "dec_A", "static_A", "enc_B", "static_B"}),
            ("group_b_to_group_a/end_effector", {"dec_A", "static_A", "enc_B", "static_B"}),
            ("group_b_to_group_a/adversarial", {"dec_A", "static_A", "enc_B", "static_B"}),
            ("group_b_to_group_b/latent", B),
            ("group_b_to_group_b/end_effector", B),
        ],
    )
    def test_term_reaches_exactly_its_modules(self, term, expected):
        method = make_method()
        losses = method.compute_generator_losses(method.forward_state(make_batch(method.domain_specs)))
        assert reached(method, losses[term]) == expected


def run_reference_objective(reference, method: SANReferenceMethod, batch, target_ids) -> SimpleNamespace:
    """
    Runs the reference's own `GAN_model.forward` and `backward_G` with copied weights and the same target
    draws. Its FK and discriminators see metres like ours; only its static encoder gets the centimetres
    our domains feed theirs.
    """
    args = reference.args
    names = list(method.domain_specs)
    models, topologies, parameter_pairs = [], [], {}
    for name in names:
        spec = method.domain_specs[name]
        topology = (0, *spec.parents.tolist()[1:])
        edges = reference_edges(reference, topology)
        auto_encoder = reference.AE(args, edges)
        static_encoder = reference.StaticEncoder(args, edges)
        discriminator = reference.Discriminator(args, edges)
        domain = method.model.domains[name]
        copy_reference_weights(domain, auto_encoder, static_encoder, discriminator)
        parameter_pairs[name] = (
            generator_parameter_pairs(domain, auto_encoder, static_encoder),
            discriminator_parameter_pairs(domain, discriminator),
        )

        topologies.append(topology)
        models.append(SimpleNamespace(
            auto_encoder=auto_encoder,
            static_encoder=lambda offsets, encoder=static_encoder: encoder(offsets * 100),
            discriminator=discriminator,
            fk=reference.ForwardKinematics(args, edges),
            height=spec.heights[:, None],
            real_height=spec.heights,
        ))

    specs = [method.domain_specs[name] for name in names]
    recorded = {}
    gan = SimpleNamespace(
        args=args,
        n_topology=len(names),
        character_names=[list(range(spec.num_banks)) for spec in specs],
        is_train=True,
        device=torch.device("cpu"),
        models=models,
        dataset=SimpleNamespace(
            offsets=[spec.offsets for spec in specs],
            joint_topologies=topologies,
            ee_ids=[spec.ee_ids for spec in specs],
            denorm=lambda i, ids, motion: motion * specs[i].std[ids][..., None] + specs[i].mean[ids][..., None],
        ),
        motions_input=[[batch[name]["motion"].transpose(1, 2), batch[name]["bank_id"]] for name in names],
        criterion_rec=torch.nn.MSELoss(),
        criterion_cycle=torch.nn.L1Loss(),
        criterion_ee=reference.Criterion_EE(args, torch.nn.MSELoss()),
        criterion_gan=reference.GAN_loss(args.gan_mode),
        fake_pools=[reference.ImagePool(args.pool_size) for _ in names],
        loss_recoder=SimpleNamespace(add_scalar=lambda name, value: recorded.__setitem__(name, float(value))),
    )
    gan.backward_D_basic = lambda *arguments: reference.GAN_model.backward_D_basic(gan, *arguments)

    # GAN_model.optimize_parameters, without its optimizer steps: these weights must stay comparable.
    draws = iter(target_ids)
    with mock.patch.object(torch, "randint", side_effect=lambda *a, **k: next(draws)):
        reference.GAN_model.forward(gan)
    reference.GAN_model.discriminator_requires_grad_(gan, False)
    reference.GAN_model.backward_G(gan)
    reference.GAN_model.discriminator_requires_grad_(gan, True)
    reference.GAN_model.backward_D(gan)

    return SimpleNamespace(recorded=recorded, total=gan.loss_G_total.item(), gan=gan, parameter_pairs=parameter_pairs)


def parity_specs():
    """ Rotations move enough that a fake's velocity is not lost beside the source's, with contacts left. """
    specs = {}
    for name, spec in make_specs().items():
        specs[name] = SANReferenceDomainSpec(parents=spec.parents, offsets=spec.offsets, mean=spec.mean, std=spec.std * 10)
    return specs


def parity_target_ids():
    # Not palindromic, so a reordered or wrong bank changes the result.
    return [
        torch.arange(BATCH_SIZE) % NUM_BANKS[destination]
        for _ in PARENTS
        for destination in PARENTS
    ]


@requires_reference
class TestAgainstReference:
    def test_intermediate_tensors_equal_the_references(self):
        reference = load_reference()
        method = make_method(parity_specs())
        batch = make_batch(method.domain_specs)

        target_ids = parity_target_ids()
        gan = run_reference_objective(reference, method, batch, target_ids).gan

        draws = iter(target_ids)
        with mock.patch.object(torch, "randint", side_effect=lambda *a, **k: next(draws)):
            state = method.forward_state(batch)

        for index, (name, source) in enumerate(state.sources.items()):
            assert_close(state.outputs.motion_codes[name], gan.latents[index], f"{name} latent", 1e-3)
            assert_close(source.reconstruction.transpose(1, 2), gan.res[index], f"{name} reconstruction", 1e-3)
            assert_close(source.bones, gan.pos_ref[index], f"{name} source bones", 1e-3)
            assert_close(source.reconstruction_bones, gan.res_pos[index], f"{name} reconstruction bones", 1e-3)
            velocity = method.normalized_end_effector_velocity(name, source.bones, source.bank_ids)
            assert_close(velocity, gan.ee_ref[index], f"{name} source velocity", 1e-3)

        for index, (route, pair) in enumerate(state.pairs.items()):
            outputs = state.outputs.pairs[route]
            assert_close(outputs.fake.transpose(1, 2), gan.fake_res[index], f"{route} fake", 1e-3)
            assert_close(outputs.fake_code, gan.fake_latent[index], f"{route} fake code", 1e-3)
            assert_close(pair.fake_bones, gan.fake_pos[index], f"{route} fake bones", 1e-3)
            assert_close(pair.fake_velocity, gan.fake_ee[index], f"{route} fake velocity", 1e-2)

    def test_every_gradient_of_a_train_step_equals_the_references(self):
        reference = load_reference()
        method = make_method(parity_specs())
        batch = make_batch(method.domain_specs)

        target_ids = parity_target_ids()
        run = run_reference_objective(reference, method, batch, target_ids)

        # The optimizers do not step, so both sides' gradients are at the same weights.
        draws = iter(target_ids)
        with (
            mock.patch.object(torch, "randint", side_effect=lambda *a, **k: next(draws)),
            mock.patch.object(method.generator_optimizer, "step"),
            mock.patch.object(method.discriminator_optimizer, "step"),
        ):
            metrics = method.train_step(batch)

        assert metrics["total/generator"] == pytest.approx(run.total, rel=1e-4)
        assert metrics["group_a/discriminator"] == pytest.approx(run.recorded["D_loss_0"], rel=1e-4)
        assert metrics["group_b/discriminator"] == pytest.approx(run.recorded["D_loss_1"], rel=1e-4)

        for name, (generator, discriminator) in run.parameter_pairs.items():
            for label, ours, theirs in generator + discriminator:
                assert ours.grad is not None and theirs.grad is not None, f"{name} {label}"
                assert_close(ours.grad, theirs.grad, f"{name} {label} gradient")

    def test_every_scalar_equals_the_references(self):
        reference = load_reference()
        method = make_method(parity_specs())
        batch = make_batch(method.domain_specs)

        target_ids = parity_target_ids()

        draws = iter(target_ids)
        run = run_reference_objective(reference, method, batch, target_ids)
        with mock.patch.object(torch, "randint", side_effect=lambda *a, **k: next(draws)):
            state = method.forward_state(batch)
        losses = method.compute_generator_losses(state)

        recorded = run.recorded
        for index, name in enumerate(PARENTS):
            assert losses[f"{name}/reconstruction"].item() / 5 == pytest.approx(recorded[f"rec_loss_quater_{index}"], rel=1e-4)
            assert losses[f"{name}/root"].item() / 1250 == pytest.approx(recorded[f"rec_loss_global_{index}"], rel=1e-4)
            assert losses[f"{name}/position"].item() / 500 == pytest.approx(recorded[f"rec_loss_position_{index}"], rel=1e-4)

        reconstruction_total = cycle_total = ee_total = adversarial_total = 0.0
        for index, name in enumerate(PARENTS):
            reconstruction_total += recorded[f"rec_loss_quater_{index}"] + 100 * (
                2.5 * recorded[f"rec_loss_global_{index}"] + recorded[f"rec_loss_position_{index}"]
            )
        for s, source in enumerate(PARENTS):
            for d, destination in enumerate(PARENTS):
                route = f"{source}_to_{destination}"
                assert losses[f"{route}/latent"].item() / 2.5 == pytest.approx(recorded[f"cycle_loss_{s}_{d}"], rel=1e-4)
                ours_ee = (losses[f"{route}/end_effector"].item() + losses[f"{route}/contact"].item()) / 50
                assert ours_ee == pytest.approx(recorded[f"ee_loss_{s}_{d}"], rel=1e-4)
                cycle_total += recorded[f"cycle_loss_{s}_{d}"]
                ee_total += recorded[f"ee_loss_{s}_{d}"]
                if source != destination:
                    assert losses[f"{route}/adversarial"].item() == pytest.approx(recorded[f"G_loss_{s}_{d}"], rel=1e-4)
                    adversarial_total += recorded[f"G_loss_{s}_{d}"]

        assert sum(loss.item() for loss in losses.values()) == pytest.approx(run.total, rel=1e-4)
        assert run.total == pytest.approx(
            5 * reconstruction_total + 2.5 * cycle_total + 50 * ee_total + adversarial_total, rel=1e-4
        )

    def test_heights_equal_the_references_on_both_domains(self):
        reference = load_reference()
        for spec in make_specs().values():
            parents = (0, *spec.parents.tolist()[1:])
            for bank in range(spec.num_banks):
                expected = reference.BVH_file.get_height(SimpleNamespace(
                    offset=spec.offsets[bank].numpy(),
                    topology=parents,
                    ee_id=spec.ee_ids,
                ))
                assert spec.heights[bank].item() == pytest.approx(float(expected), rel=1e-5)

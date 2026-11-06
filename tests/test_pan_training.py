"""
PAN's routes, objective and update order on synthetic domains, with scalar parity checks against the
official code that run when PAN_REFERENCE_ROOT is set.
"""

from __future__ import annotations

import copy
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import pytest
import torch
import yaml

from motion_retargeting.data.representations.motion import MotionCollection, MotionData
from motion_retargeting.data.representations.skeleton import SkeletonTopology
from motion_retargeting.methods.pan import config as pan_config
from motion_retargeting.methods.pan import data as pan_data
from motion_retargeting.methods.pan import losses as pan_losses
from motion_retargeting.methods.pan import method as pan_method
from motion_retargeting.methods.pan import model as pan_model
from motion_retargeting.methods.pan import utils as pan_utils
from motion_retargeting.methods.pan.config import DomainSpec, PANConfig, PANModelConfig
from motion_retargeting.methods.pan.losses import end_effector_terms
from motion_retargeting.methods.pan.method import PANMethod
from motion_retargeting.methods.pan.utils import mpjpe, squared_error_over_height2
from motion_retargeting.modules import decoders, discriminators, encoders
from motion_retargeting.training.trainer import Trainer
from pan_reference_support import AMASS_BODY_JOINTS, ARGS, CMU_BODY_JOINTS, load_reference, requires_reference

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "scripts" / "train.py"
PAN_CONFIG = ROOT / "configs" / "methods" / "pan.yaml"

# The AMASS and CMU skeletons of configs/methods/pan.yaml.
AMASS_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
CMU_PARENTS = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 13, 14, 11, 16, 17, 18, 11, 20, 21, 22]

BATCH_SIZE = 4
WINDOW = 64
NUM_BANKS = {"group_a": 3, "group_b": 2}
PARENTS = {"group_a": AMASS_PARENTS, "group_b": CMU_PARENTS}
BODY_JOINTS = {"group_a": AMASS_BODY_JOINTS, "group_b": CMU_BODY_JOINTS}


def make_spec(parents, body_joints, num_banks: int, generator: torch.Generator) -> DomainSpec:
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

    return DomainSpec(parents=torch.tensor(parents), offsets=offsets, mean=mean, std=std, body_joints=body_joints)


def make_specs(seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    return {name: make_spec(PARENTS[name], BODY_JOINTS[name], NUM_BANKS[name], generator) for name in PARENTS}


def make_batch(specs, batch_size: int = BATCH_SIZE, window: int = WINDOW, seed: int = 1):
    generator = torch.Generator().manual_seed(seed)
    batch = {}
    for name, spec in specs.items():
        batch[name] = {
            "motion": torch.randn(batch_size, window, spec.mean.shape[1], generator=generator),
            "bank_id": torch.randint(spec.num_banks, (batch_size,), generator=generator),
        }
    return batch


def make_method(specs=None, seed: int = 0, **cfg) -> PANMethod:
    torch.manual_seed(seed)
    return PANMethod(
        model_cfg=PANModelConfig(),
        domain_specs=specs if specs is not None else make_specs(),
        learning_rate=1e-3,
        device="cpu",
        cfg=PANConfig(**cfg),
    )


class TestRoutes:
    def test_all_four_pairs_exist(self):
        method = make_method()
        state = method.forward_state(make_batch(method.domain_specs))
        assert set(state.pairs) == {
            ("group_a", "group_a"), ("group_a", "group_b"), ("group_b", "group_a"), ("group_b", "group_b"),
        }
        pair = state.outputs.pairs[("group_a", "group_b")]
        assert pair.fake.shape == (BATCH_SIZE, WINDOW, 4 * 23 + 3)
        assert pair.cycle.shape == (BATCH_SIZE, WINDOW, 4 * 21 + 3)

    def test_every_decode_sees_the_fused_code(self):
        method = make_method()
        batch = make_batch(method.domain_specs)
        method.eval()

        calls = {"group_a": [], "group_b": []}
        for name, domain in method.model.domains.items():
            domain.decoder.register_forward_pre_hook(lambda module, args, name=name: calls[name].append(args[0]))

        codes = []
        real_decode = method.model.decode

        def spy(domain, motion_code, skeleton_code):
            codes.append((domain, motion_code + skeleton_code[:, None] / 100))
            return real_decode(domain, motion_code, skeleton_code)

        method.model.decode = spy
        method.forward_state(batch)

        # Two reconstructions + four routes of (retarget + cycle).
        assert len(codes) == 10
        seen = {"group_a": 0, "group_b": 0}
        for domain, fused in codes:
            assert torch.equal(calls[domain][seen[domain]], fused)
            seen[domain] += 1

    def test_decoder_input_is_z_plus_s_over_100_on_reconstruction(self):
        method = make_method()
        batch = make_batch(method.domain_specs)
        method.eval()

        domain = method.model.domains["group_a"]
        inputs = []
        domain.decoder.register_forward_pre_hook(lambda module, args: inputs.append(args[0]))

        outputs = method.model.forward_generator(batch)

        codes = domain.body_part_encoder(domain.offsets)[batch["group_a"]["bank_id"]]
        expected = outputs.motion_codes["group_a"] + codes[:, None] / 100
        assert torch.allclose(inputs[0], expected)

    def test_target_draws_cover_banks_uniformly_regardless_of_clips(self):
        method = make_method()
        batch = make_batch(method.domain_specs, batch_size=1)
        counts = torch.zeros(NUM_BANKS["group_a"])
        with torch.no_grad():
            torch.manual_seed(3)
            for _ in range(60):
                outputs = method.model.forward_generator(batch)
                counts[outputs.pairs[("group_b", "group_a")].target_ids] += 1
        assert (counts > 5).all()

    def test_target_statistics_offsets_and_height_are_coupled(self):
        method = make_method()
        state = method.forward_state(make_batch(method.domain_specs))
        domain = method.model.domains["group_b"]
        pair = state.outputs.pairs[("group_a", "group_b")]

        denormalized = pair.fake * domain.std[pair.target_ids][:, None] + domain.mean[pair.target_ids][:, None]
        expected = domain.fk.bone_vectors_with_absolute_root(denormalized, domain.offsets[pair.target_ids])
        assert torch.equal(state.pairs[("group_a", "group_b")].fake_bones, expected)
        assert pair.target_ids.equal(state.pairs[("group_a", "group_b")].target_ids)

    def test_same_domain_routes_use_the_drawn_target_too(self):
        method = make_method()
        state = method.forward_state(make_batch(method.domain_specs, batch_size=32))
        same = state.pairs[("group_a", "group_a")]
        assert not torch.equal(same.target_ids, state.sources["group_a"].bank_ids)

    def test_target_morphology_never_changes_source_cycle_offsets(self):
        method = make_method()
        batch = make_batch(method.domain_specs)
        method.eval()

        state = method.forward_state(batch)
        source = method.model.domains["group_a"]
        pair = state.outputs.pairs[("group_a", "group_b")]
        bank_ids = batch["group_a"]["bank_id"]

        denormalized = pair.cycle * source.std[bank_ids][:, None] + source.mean[bank_ids][:, None]
        expected = source.fk.bone_vectors_with_absolute_root(denormalized, source.offsets[bank_ids])
        assert torch.equal(state.pairs[("group_a", "group_b")].cycle_bones, expected)

    def test_retarget_and_latent_are_not_detached(self):
        method = make_method()
        state = method.forward_state(make_batch(method.domain_specs))
        pair = state.outputs.pairs[("group_a", "group_b")]
        assert pair.fake.requires_grad and pair.fake_code.requires_grad and pair.cycle.requires_grad
        assert state.outputs.motion_codes["group_a"].requires_grad

    def test_real_motion_positions_are_detached(self):
        method = make_method()
        state = method.forward_state(make_batch(method.domain_specs))
        assert not state.sources["group_a"].bones.requires_grad

    def test_visualization_uses_absolute_root_world_positions(self, tmp_path):
        method = make_method()
        method.eval()
        method.visualize(make_batch(method.domain_specs, batch_size=1), tmp_path, fps=30.0)
        assert (tmp_path / "group_a_to_group_b.gif").exists()
        assert (tmp_path / "group_b_to_group_a.gif").exists()


class TestGANRoutes:
    def test_cross_routes_only(self):
        method = make_method()
        losses = method.compute_adversarial_losses(method.forward_state(make_batch(method.domain_specs)))
        assert set(losses) == {"group_a_to_group_b/adversarial", "group_b_to_group_a/adversarial"}

    def test_generator_gan_reaches_source_encoder_target_decoder_and_static_encoder_only(self):
        method = make_method()
        state = method.forward_state(make_batch(method.domain_specs))
        loss = method.compute_adversarial_losses(state)["group_a_to_group_b/adversarial"]

        a, b = method.model.domains["group_a"], method.model.domains["group_b"]
        reached = lambda module: any(  # noqa: E731
            grad is not None and grad.abs().sum() > 0
            for grad in torch.autograd.grad(loss, list(module.parameters()), retain_graph=True, allow_unused=True)
        )

        assert reached(a.motion_encoder)
        assert reached(b.decoder)
        assert reached(b.body_part_encoder)
        assert not reached(b.motion_encoder)
        assert not reached(a.decoder)
        assert not reached(a.body_part_encoder)

    def test_discriminator_loss_cannot_reach_the_generator(self):
        method = make_method()
        state = method.forward_state(make_batch(method.domain_specs))
        losses = method.compute_discriminator_losses(state)
        assert set(losses) == {"group_a/discriminator", "group_b/discriminator"}

        grads = torch.autograd.grad(
            sum(losses.values()), list(method.model.generator_parameters()), retain_graph=True, allow_unused=True
        )
        assert all(grad is None for grad in grads)

    def test_replay_buffer_stores_discriminator_positions(self):
        method = make_method()
        batch = make_batch(method.domain_specs)
        state = method.forward_state(batch)
        method.compute_discriminator_losses(state)

        stored = torch.cat(method.buffers["group_b"].items)
        assert stored.shape == (BATCH_SIZE, WINDOW, 24, 3)
        assert not stored.requires_grad
        assert torch.equal(stored, state.pairs[("group_a", "group_b")].fake_bones.detach())


def expected_generator_terms() -> set:
    terms = {f"{d}/{term}" for d in PARENTS for term in ("reconstruction", "root", "position")}
    for source in PARENTS:
        for destination in PARENTS:
            route = f"{source}_to_{destination}"
            terms |= {f"{route}/{term}" for term in ("latent_cycle", "position_cycle", "end_effector", "contact")}
            if source != destination:
                terms.add(f"{route}/adversarial")
    return terms


class TestObjective:
    def setup_method(self):
        self.method = make_method()
        self.method.eval()
        self.batch = make_batch(self.method.domain_specs)
        self.state = self.method.forward_state(self.batch)
        self.losses = self.method.compute_generator_losses(self.state)

    def test_term_set_has_no_feature_cycle_or_cycle_fk_multiplier(self):
        assert set(self.losses) == expected_generator_terms()
        assert all(torch.isfinite(loss) for loss in self.losses.values())

    def test_reconstruction_is_unweighted_channel_mse(self):
        source = self.state.sources["group_a"]
        expected = torch.mean((source.reconstruction - source.motion) ** 2)
        assert self.losses["group_a/reconstruction"].item() == pytest.approx(expected.item())

    @pytest.mark.parametrize("field, term, factor", [
        ("root_lambda", "group_a/root", 100),
        ("position_lambda", "group_a/position", 100),
        ("cycle_lambda", "group_a_to_group_b/latent_cycle", 2.5),
        ("cycle_lambda", "group_a_to_group_b/position_cycle", 2.5),
        ("end_effector_lambda", "group_b_to_group_a/end_effector", 50),
        ("contact_lambda", "group_b_to_group_a/contact", 100),
    ])
    def test_factors(self, field, term, factor):
        assert getattr(PANConfig(), field) == factor

        self.method.cfg = PANConfig(**{field: 2 * factor})
        changed = self.method.compute_generator_losses(self.state)
        assert changed[term].item() == pytest.approx(2 * self.losses[term].item(), rel=1e-5)

    def test_latent_cycle_is_l1_and_differentiable_on_both_sides(self):
        pair = self.state.outputs.pairs[("group_a", "group_b")]
        source_code = self.state.outputs.motion_codes["group_a"]
        expected = 2.5 * (pair.fake_code - source_code).abs().mean()
        assert self.losses["group_a_to_group_b/latent_cycle"].item() == pytest.approx(expected.item())

        assert pair.fake_code.requires_grad and source_code.requires_grad

    def test_no_direct_paired_retarget_rotation_target(self):
        """ The only terms depending on a fake's rotations go through positions, latent or D. """
        pair = self.state.outputs.pairs[("group_a", "group_b")]
        loss = self.losses["group_a/reconstruction"]
        assert torch.autograd.grad(loss, pair.fake, retain_graph=True, allow_unused=True)[0] is None

    def test_contacts_select_on_generated_velocity(self):
        source = torch.ones(1, 1, 2, 3)
        fake = torch.zeros(1, 1, 2, 3)
        fake[:, :, 0] = 0.001  # Generated contact, where the source is moving.
        fake[:, :, 1] = 1.0  # Generated motion, not selected.

        transfer, contact = end_effector_terms(source, fake, 0.008)
        assert contact.item() == pytest.approx((1 - 0.001) ** 2)
        assert transfer.item() == pytest.approx(((1 - 0.001) ** 2 + 0.0) / 2)

    def test_height_doubling_with_proportionate_geometry_leaves_geometry_terms_invariant(self):
        specs = make_specs()
        scaled = {}
        for name, spec in specs.items():
            mean, std = spec.mean.clone(), spec.std.clone()
            mean[:, -3:] *= 2
            std[:, -3:] *= 2
            scaled[name] = DomainSpec(
                parents=spec.parents, offsets=2 * spec.offsets, mean=mean, std=std, body_joints=spec.body_joints
            )

        original = make_method(specs)
        doubled = make_method(scaled)
        original.eval()
        doubled.eval()

        batch = make_batch(specs)
        original.model.load_state_dict(
            {key: value for key, value in doubled.model.state_dict().items() if "encoder" in key or "decoder" in key},
            strict=False,
        )
        torch.manual_seed(5)
        original_losses = original.compute_generator_losses(original.forward_state(batch))
        torch.manual_seed(5)
        doubled_losses = doubled.compute_generator_losses(doubled.forward_state(batch))

        for term in ("group_a/root", "group_b/position", "group_a_to_group_b/position_cycle", "group_b_to_group_a/end_effector"):
            assert doubled_losses[term].item() == pytest.approx(original_losses[term].item(), rel=1e-3), term

    def test_empty_contact_selection_raises(self):
        self.method.cfg = PANConfig(contact_threshold=0.0)
        with pytest.raises(RuntimeError, match="contact threshold"):
            self.method.compute_generator_losses(self.state)


@requires_reference
class TestObjectiveAgainstReference:
    def test_scalars_and_gradients_match_the_reference_ledger(self):
        reference = load_reference()
        method = make_method()
        method.eval()
        batch = make_batch(method.domain_specs)
        state = method.forward_state(batch)
        losses = method.compute_generator_losses(state)

        topologies = {name: [0] + PARENTS[name][1:] for name in PARENTS}
        kinematics = {
            name: reference.ForwardKinematics(ARGS, reference.build_edge_topology(topologies[name], torch.zeros(len(topologies[name]), 3)))
            for name in PARENTS
        }
        criterion_rec = torch.nn.MSELoss()
        criterion_latent = torch.nn.L1Loss()
        criterion_ee = reference.Criterion_EE(ARGS, torch.nn.MSELoss())
        criterion_gan = reference.GAN_loss("lsgan")

        def remote_fk(name, motion, bank_ids):
            domain = method.model.domains[name]
            denormalized = motion * domain.std[bank_ids][:, None] + domain.mean[bank_ids][:, None]
            return denormalized, kinematics[name].forward_from_raw(denormalized.transpose(1, 2), domain.offsets[bank_ids])

        def remote_ee(name, positions, bank_ids):
            domain = method.model.domains[name]
            velocity = reference.get_ee(positions, topologies[name], domain.ee_ids, velo=1, from_root=1)
            return velocity / domain.heights[bank_ids].reshape(-1, 1, 1, 1)

        def world(name, positions, bank_ids):
            height = method.model.domains[name].heights[bank_ids].reshape(-1, 1, 1, 1)
            return kinematics[name].from_local_to_world(positions) / height

        expected = {}
        references = {}
        for name in PARENTS:
            bank_ids = batch[name]["bank_id"]
            motion = batch[name]["motion"]
            reconstruction = state.outputs.reconstructions[name]
            height = method.model.domains[name].heights[bank_ids].reshape(-1, 1)

            denormalized, positions = remote_fk(name, motion, bank_ids)
            positions = positions.detach()
            reconstruction_denormalized, reconstruction_positions = remote_fk(name, reconstruction, bank_ids)
            references[name] = (bank_ids, positions)

            expected[f"{name}/reconstruction"] = criterion_rec(motion, reconstruction)
            expected[f"{name}/root"] = 100 * criterion_rec(
                denormalized[..., -3:] / height.unsqueeze(-1),
                reconstruction_denormalized[..., -3:] / height.unsqueeze(-1),
            )
            expected[f"{name}/position"] = 100 * criterion_rec(
                world(name, positions, bank_ids),
                world(name, reconstruction_positions, bank_ids),
            )

        for (source, destination), pair in state.outputs.pairs.items():
            route = f"{source}_to_{destination}"
            bank_ids, positions = references[source]

            _, cycle_positions = remote_fk(source, pair.cycle, bank_ids)
            _, fake_positions = remote_fk(destination, pair.fake, pair.target_ids)

            expected[f"{route}/latent_cycle"] = 2.5 * criterion_latent(
                state.outputs.motion_codes[source].flatten(2, 3).transpose(1, 2),
                pair.fake_code.flatten(2, 3).transpose(1, 2),
            )
            expected[f"{route}/position_cycle"] = 2.5 * criterion_rec(
                world(source, positions, bank_ids),
                world(source, cycle_positions, bank_ids),
            )

            ee_total = criterion_ee(
                remote_ee(source, positions, bank_ids),
                remote_ee(destination, fake_positions, pair.target_ids),
            )
            expected[f"{route}/end_effector_total"] = 50 * ee_total

            if source != destination:
                expected[f"{route}/adversarial"] = criterion_gan(
                    method.model.domains[destination].discriminator(fake_positions), True
                )

        for term, value in expected.items():
            if term.endswith("end_effector_total"):
                route = term.rsplit("/", 1)[0]
                local = losses[f"{route}/end_effector"] + losses[f"{route}/contact"]
            else:
                local = losses[term]
            assert local.item() == pytest.approx(value.item(), rel=1e-3, abs=1e-5), term

        parameters = [p for p in method.model.generator_parameters() if p.requires_grad]
        local_grads = torch.autograd.grad(sum(losses.values()), parameters, retain_graph=True, allow_unused=True)
        remote_grads = torch.autograd.grad(sum(expected.values()), parameters, retain_graph=True, allow_unused=True)
        for local, remote in zip(local_grads, remote_grads):
            assert (local is None) == (remote is None)
            if local is not None:
                assert torch.allclose(local, remote, atol=1e-3, rtol=1e-2)


def snapshot(parameters) -> list:
    return [parameter.detach().clone() for parameter in parameters]


def unchanged(parameters, before) -> bool:
    return all(torch.equal(parameter, old) for parameter, old in zip(parameters, before))


class RecordingOptimizer:
    """ Wraps one optimizer's step to record order, and the parameters around the step. """

    def __init__(self, optimizer, name: str, log: list, on_step=None):
        self.step, self.name, self.log = optimizer.step, name, log
        self.on_step = on_step
        optimizer.step = self

    def __call__(self, *args, **kwargs):
        self.log.append(self.name)
        if self.on_step is not None:
            self.on_step()
        return self.step(*args, **kwargs)


class TestUpdateOrder:
    def setup_method(self):
        self.method = make_method()
        self.batch = make_batch(self.method.domain_specs)
        self.generator = list(self.method.model.generator_parameters())
        self.discriminator = list(self.method.model.discriminator_parameters())

    def test_generator_steps_before_discriminator_with_one_forward(self):
        log = []
        RecordingOptimizer(self.method.generator_optimizer, "generator", log)
        RecordingOptimizer(self.method.discriminator_optimizer, "discriminator", log)

        calls = []
        forward = self.method.model.forward_generator
        self.method.model.forward_generator = lambda batch: (calls.append(1), forward(batch))[1]

        self.method.train_step(self.batch)
        assert log == ["generator", "discriminator"]
        assert len(calls) == 1

    def test_generator_step_changes_only_the_generator_and_leaves_no_discriminator_gradients(self):
        discriminator_before = snapshot(self.discriminator)
        checks = {}

        def at_generator_step():
            checks["no_discriminator_gradients"] = all(p.grad is None for p in self.discriminator)
            checks["frozen"] = not any(p.requires_grad for p in self.discriminator)
            checks["generator_has_gradients"] = any(p.grad is not None for p in self.generator)

        RecordingOptimizer(self.method.generator_optimizer, "generator", [], on_step=at_generator_step)

        def at_discriminator_step():
            checks["discriminator_untouched_by_generator"] = unchanged(self.discriminator, discriminator_before)
            checks["restored"] = all(p.requires_grad for p in self.discriminator)

        RecordingOptimizer(self.method.discriminator_optimizer, "discriminator", [], on_step=at_discriminator_step)

        generator_before = snapshot(self.generator)
        self.method.train_step(self.batch)

        assert all(checks.values()), checks
        assert not unchanged(self.generator, generator_before)
        assert not unchanged(self.discriminator, discriminator_before)

    def test_discriminator_step_changes_only_the_discriminator(self):
        checks = {}

        def at_discriminator_step():
            checks["generator"] = snapshot(self.generator)

        RecordingOptimizer(self.method.discriminator_optimizer, "discriminator", [], on_step=at_discriminator_step)

        self.method.train_step(self.batch)
        assert unchanged(self.generator, checks["generator"])

    def test_generator_keeps_an_adversarial_gradient_through_the_frozen_discriminator(self):
        self.method.set_discriminator_requires_grad(False)
        state = self.method.forward_state(self.batch)
        loss = self.method.compute_adversarial_losses(state)["group_a_to_group_b/adversarial"]
        grad = torch.autograd.grad(loss, list(self.method.model.domains["group_b"].decoder.parameters()), allow_unused=True)
        assert any(g is not None and g.abs().sum() > 0 for g in grad)

    def test_pool_receives_the_pre_update_fake_values(self):
        recorded = {}
        forward_state = self.method.forward_state

        def spy(batch):
            state = forward_state(batch)
            recorded["fake"] = state.pairs[("group_a", "group_b")].fake_bones.detach().clone()
            return state

        self.method.forward_state = spy
        self.method.train_step(self.batch)

        assert torch.equal(torch.cat(self.method.buffers["group_b"].items), recorded["fake"])

    def test_empty_contact_stops_before_either_optimizer_changes_parameters(self):
        self.method.cfg = PANConfig(contact_threshold=0.0)
        generator_before, discriminator_before = snapshot(self.generator), snapshot(self.discriminator)

        with pytest.raises(RuntimeError, match="contact threshold"):
            self.method.train_step(self.batch)

        assert unchanged(self.generator, generator_before)
        assert unchanged(self.discriminator, discriminator_before)

    def test_nonfinite_objective_stops_before_either_optimizer_changes_parameters(self):
        self.batch["group_a"]["motion"][0, 0, -1] = float("nan")
        generator_before, discriminator_before = snapshot(self.generator), snapshot(self.discriminator)

        with pytest.raises(RuntimeError):
            self.method.train_step(self.batch)

        assert unchanged(self.generator, generator_before)
        assert unchanged(self.discriminator, discriminator_before)

    def test_diagnostics_are_finite(self):
        self.method.cfg = PANConfig(loss_grad_norm_interval=1)
        metrics = self.method.train_step(self.batch)

        assert torch.isfinite(torch.tensor(list(metrics.values()))).all()
        assert metrics["grad/group_a/decoder"] > 0
        assert metrics["grad/group_b/discriminator"] > 0
        assert "loss_grad/group_a/reconstruction" in metrics
        assert "loss_grad/group_a/discriminator" in metrics


class TestMetrics:
    def test_euclidean_mpjpe_is_distinct_from_squared_coordinate_error(self):
        predicted = torch.zeros(1, 1, 1, 3)
        target = torch.tensor([[[[3.0, 4.0, 0.0]]]])
        heights = torch.tensor([2.0])

        assert mpjpe(predicted, target) == pytest.approx(5.0)
        # Averaged over coordinates: (9 + 16 + 0) / 3, then over height^2.
        assert squared_error_over_height2(predicted, target, heights) == pytest.approx(25 / 3 / 4)

    def test_validation_names_metrics_for_what_they_are(self):
        method = make_method()
        method.eval()
        metrics = method.validation_step(make_batch(method.domain_specs))

        for name in (
            "retarget_latent_l1",
            "retarget_realism",
            "reconstruction_mse",
            "reconstruction_mpjpe_m",
            "reconstruction_squared_error_over_height2",
            "cycle_mpjpe_m",
            "cycle_squared_error_over_height2",
        ):
            assert torch.isfinite(torch.tensor(metrics[f"group_a/val/{name}"])), name
        assert not any("paired" in name or "retarget_accuracy" in name for name in metrics)

    def test_realism_reads_discriminator_positions(self):
        method = make_method()
        method.eval()
        batch = make_batch(method.domain_specs)

        inputs = []
        method.model.domains["group_b"].discriminator.register_forward_pre_hook(lambda module, args: inputs.append(args[0]))
        method.validation_step(batch)
        assert inputs[0].shape == (BATCH_SIZE, WINDOW, 24, 3)


class TestCheckpoints:
    def test_round_trip_preserves_banks_masks_and_decoding(self, tmp_path):
        specs = make_specs()
        method = make_method(specs, seed=0)
        method.eval()

        path = tmp_path / "checkpoint.pt"
        torch.save(method.state_dict(), path)
        state = torch.load(path, weights_only=True)
        assert state["metadata"]["official_commit"].startswith("5a51d92")
        assert state["metadata"]["physical_units"] == "metres"
        assert state["metadata"]["domains"]["group_a"]["end_effector_ids"] == [10, 11, 15, 20, 21]

        restored = make_method(specs, seed=1)
        restored.eval()
        assert not torch.equal(restored.model.domains["group_a"].decoder.conv1.weight, method.model.domains["group_a"].decoder.conv1.weight)
        restored.load_state_dict(state)

        for key, value in method.model.state_dict().items():
            assert torch.equal(restored.model.state_dict()[key], value), key
        assert any(key.endswith("attention_mask") for key in state["model"])
        assert any(key.endswith("conv2.mask") for key in state["model"])
        assert any(key.endswith("mean") for key in state["model"])

        batch = make_batch(specs)
        torch.manual_seed(9)
        original = method.forward_state(batch)
        torch.manual_seed(9)
        reloaded = restored.forward_state(batch)
        assert torch.equal(
            original.pairs[("group_a", "group_b")].fake_bones,
            reloaded.pairs[("group_a", "group_b")].fake_bones,
        )

    def test_older_checkpoint_gets_a_clear_error(self):
        method = make_method()
        older_state = {key: value for key, value in method.state_dict().items() if key != "metadata"}
        with pytest.raises(ValueError, match="predates this PAN implementation"):
            method.load_state_dict(older_state)

    def test_incompatible_setup_is_rejected(self):
        method = make_method()
        state = copy.deepcopy(method.state_dict())
        state["metadata"]["mask_version"] = "repaired_masks"
        with pytest.raises(ValueError, match="mask_version"):
            method.load_state_dict(state)

        generator = torch.Generator().manual_seed(0)
        other_banks = make_specs()
        other_banks["group_a"] = make_spec(AMASS_PARENTS, AMASS_BODY_JOINTS, NUM_BANKS["group_a"] + 1, generator)
        with pytest.raises(ValueError, match="domains"):
            make_method(other_banks).load_state_dict(method.state_dict())

    def test_other_body_part_labels_are_rejected(self):
        method = make_method()
        relabeled = make_specs()
        spec = relabeled["group_a"]
        # The same topology, with the labels of the head and the left arm swapped.
        body_joints = {**spec.body_joints, 3: spec.body_joints[4], 4: spec.body_joints[3]}
        relabeled["group_a"] = DomainSpec(
            parents=spec.parents, offsets=spec.offsets, mean=spec.mean, std=spec.std, body_joints=body_joints
        )
        with pytest.raises(ValueError, match="domains"):
            make_method(relabeled).load_state_dict(method.state_dict())


class TestConfig:
    def test_yaml_settings(self):
        cfg = yaml.safe_load(PAN_CONFIG.read_text())
        assert cfg["trainer"]["batch_size"] == 128
        assert cfg["trainer"]["learning_rate"] == 0.001
        assert cfg["trainer"]["num_epochs"] == 1001
        assert "recipe" not in cfg["method"]["params"]
        assert "disc_logit_reg" not in cfg["method"]["params"]
        assert "grad_norm_clip" not in cfg["method"]["params"]
        for group in cfg["dataset"].values():
            assert group["params"]["sequence_length"] == 64
            assert group["params"]["stride"] == 32

        PANConfig(**cfg["method"]["params"])
        model_cfg = PANModelConfig.from_dict(cfg["method"]["model"])
        assert model_cfg == PANModelConfig()

    def test_adam_settings_have_no_schedule_or_decay(self):
        method = make_method()
        for optimizer in (method.generator_optimizer, method.discriminator_optimizer):
            group = optimizer.param_groups[0]
            assert group["betas"] == (0.9, 0.999)
            assert group["weight_decay"] == 0

    def test_there_is_one_pan_implementation(self):
        assert [path.name for path in PAN_CONFIG.parent.glob("pan*")] == ["pan.yaml"]
        with pytest.raises(TypeError):
            PANConfig(recipe="legacy")

        modules = (pan_config, pan_data, pan_losses, pan_method, pan_model, pan_utils, decoders, discriminators, encoders)
        names = [name for module in modules for name in vars(module)]
        assert not [name for name in names if "reference" in name.lower() or "legacy" in name.lower()]


def smooth_clip(num_joints: int, offset_scale: float, seed: int) -> MotionData:
    """ Near-rest motion, slow enough to hold contact frames. """
    generator = torch.Generator().manual_seed(seed)
    frames = 130

    rotations = torch.zeros(frames, num_joints, 4)
    rotations[..., 0] = 1.0
    phase = torch.linspace(0, 6.28, frames)[:, None, None]
    rotations = rotations + 0.01 * torch.randn(1, num_joints, 4, generator=generator) * phase.sin()
    rotations = torch.nn.functional.normalize(rotations, dim=-1)

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
    def test_pan_config_trains_and_checkpoints(self, tmp_path, monkeypatch):
        class Source:
            def __init__(self, name):
                parents = {"amass": AMASS_PARENTS, "cmu": CMU_PARENTS}[name]
                body_joints = {"amass": AMASS_BODY_JOINTS, "cmu": CMU_BODY_JOINTS}[name]
                self.collection = MotionCollection(
                    topology=SkeletonTopology(parents=torch.tensor(parents), body_joints=body_joints),
                    motions=[smooth_clip(len(parents), scale, seed) for seed, scale in enumerate((1.0, 1.0, 1.5))],
                )

            def load(self):
                return self.collection

        monkeypatch.setattr("motion_retargeting.methods.pan.data.build_adapter", Source)

        run = run_training(write_config(PAN_CONFIG, tmp_path), monkeypatch)

        assert type(run.method) is PANMethod
        assert run.method.domain_specs["group_a"].num_banks == 2
        assert run.method.domain_specs["group_a"].body_joints == AMASS_BODY_JOINTS
        assert run.method.domain_specs["group_b"].body_joints == CMU_BODY_JOINTS

        checkpoint = torch.load(run.trainer.checkpoint_dir / "final.pt", weights_only=True)
        assert checkpoint["method"]["metadata"]["official_commit"] == pan_method.OFFICIAL_COMMIT

        gifs = list(run.trainer.visualization_dir.rglob("*.gif"))
        assert {gif.name for gif in gifs} == {"group_a_to_group_b.gif", "group_b_to_group_a.gif"}

        run.trainer.load_checkpoint(run.trainer.checkpoint_dir / "final.pt")

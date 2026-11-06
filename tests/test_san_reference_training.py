"""
SAN reference recipe's routes, objective and update order on synthetic domains, with reference scalar
parity checks that run when SAN_REFERENCE_ROOT is set.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.nn.functional as F

from motion_retargeting.methods.pan.kinematics import END_EFFECTOR_VELOCITY_SCALE
from motion_retargeting.methods.san.config import SANModelConfig, SANReferenceConfig, SANReferenceDomainSpec
from motion_retargeting.methods.san.method import SANReferenceMethod
from motion_retargeting.methods.san.model import SANReferenceModel
from motion_retargeting.methods.san.utils import (
    SANReferenceGeneratorOutputs,
    SANReferencePairOutputs,
    select_banks,
)
from san_reference_support import copy_reference_weights, load_reference, reference_edges, requires_reference

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
                target_ids=torch.randint(specs[destination].num_banks, (BATCH_SIZE,), generator=generator),
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
    models, topologies = [], []
    for name in names:
        spec = method.domain_specs[name]
        topology = (0, *spec.parents.tolist()[1:])
        edges = reference_edges(reference, topology)
        auto_encoder = reference.AE(args, edges)
        static_encoder = reference.StaticEncoder(args, edges)
        discriminator = reference.Discriminator(args, edges)
        copy_reference_weights(method.model.domains[name], auto_encoder, static_encoder, discriminator)

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
        loss_recoder=SimpleNamespace(add_scalar=lambda name, value: recorded.__setitem__(name, float(value))),
    )

    draws = iter(target_ids)
    with mock.patch.object(torch, "randint", side_effect=lambda *a, **k: next(draws)):
        reference.GAN_model.forward(gan)
    reference.GAN_model.backward_G(gan)

    return SimpleNamespace(recorded=recorded, total=gan.loss_G_total.item())


@requires_reference
class TestAgainstReference:
    def test_every_scalar_equals_the_references(self):
        reference = load_reference()
        method = make_method()
        batch = make_batch(method.domain_specs)

        generator = torch.Generator().manual_seed(5)
        target_ids = [
            torch.randint(method.domain_specs[destination].num_banks, (BATCH_SIZE,), generator=generator)
            for _ in PARENTS
            for destination in PARENTS
        ]

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

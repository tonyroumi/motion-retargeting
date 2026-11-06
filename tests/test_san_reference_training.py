"""
SAN reference recipe's routes, objective and update order on synthetic domains, with reference scalar
parity checks that run when SAN_REFERENCE_ROOT is set.
"""

from __future__ import annotations

import torch

from motion_retargeting.methods.san.config import SANModelConfig, SANReferenceDomainSpec
from motion_retargeting.methods.san.model import SANReferenceModel
from motion_retargeting.methods.san.utils import select_banks

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

"""
SAN method smoke tests on synthetic domains, with no dataset (or data layer) involved.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import SkipTest
import importlib.util

import torch
import yaml

from motion_retargeting.methods.san.config import DomainSpec, SANConfig, SANModelConfig
from motion_retargeting.methods.san.losses import end_effector_terms
from motion_retargeting.methods.san.method import SANMethod
from motion_retargeting.methods.san.model import SANModel
from motion_retargeting.modules.skeletal import SkeletalConv, build_edge_list, construct_adj
from motion_retargeting.training.trainer import Trainer, TrainerConfig


# The AMASS and CMU skeletons of the dataset groups in configs/methods/san.yaml.
AMASS_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
CMU_PARENTS = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 13, 14, 11, 16, 17, 18, 11, 20, 21, 22]

BATCH_SIZE = 3
WINDOW = 32
NUM_BANKS = 2


def make_spec(parents):
    num_joints = len(parents)
    channels = 4 * (num_joints - 1) + 3
    offsets = 0.1 * torch.randn(NUM_BANKS, num_joints, 3)
    offsets[:, 0] = 0.0
    mean = torch.zeros(NUM_BANKS, channels)
    mean[:, ::4] = 1.0
    # Small enough that an untrained decoder's generated velocities include contact frames.
    std = 0.001 + 0.002 * torch.rand(NUM_BANKS, channels)
    return DomainSpec(parents=torch.tensor(parents), offsets=offsets, mean=mean, std=std)


def make_domain_batch(num_joints):
    """ The batch format of `SANMotionDataset` once collated. """
    return {
        "motion": torch.randn(BATCH_SIZE, WINDOW, 4 * (num_joints - 1) + 3),
        "bank_id": torch.randint(NUM_BANKS, (BATCH_SIZE,)),
    }


def make_batch():
    return {
        "group_a": make_domain_batch(len(AMASS_PARENTS)),
        "group_b": make_domain_batch(len(CMU_PARENTS)),
    }


def make_method(**cfg):
    specs = {"group_a": make_spec(AMASS_PARENTS), "group_b": make_spec(CMU_PARENTS)}
    return SANMethod(SANModelConfig(), specs, device="cpu", cfg=SANConfig(**cfg))


def test_yaml_matches_default_model_config():
    config = Path(__file__).parents[1] / "configs" / "methods" / "san.yaml"
    with open(config, "r") as f:
        cfg = yaml.safe_load(f)

    assert cfg["method"]["name"] == "san"
    assert SANModelConfig.from_dict(cfg["method"]["model"]) == SANModelConfig()
    assert SANConfig(**cfg["method"]["params"]).betas == [0.9, 0.999]


def test_motion_rows_map_to_graph_rows():
    edges = build_edge_list(AMASS_PARENTS)

    assert edges[0] == [0, 1]  # edge 0 ends at joint 1, and the root has no edge
    assert edges[3] == [1, 4]  # joint 4, child of joint 1
    assert len(construct_adj(edges)) == len(AMASS_PARENTS)  # E edges plus the root-position token


def test_skeletal_conv_only_connects_neighbors():
    adj = construct_adj(build_edge_list([-1, 0, 1, 2, 3, 4]), d=1)
    conv = SkeletalConv(adj, in_channels_per_joint=2, out_channels_per_joint=2, kernel_size=1, padding=0)

    x = torch.zeros(1, 2 * len(adj), 1, requires_grad=True)
    conv(x)[0, 0].backward()  # First channel of edge 0

    connected = x.grad[0, :, 0].abs().sum() > 0
    far_edge = x.grad[0, 2 * 3:2 * 4, 0]  # Edge 3 is out of reach of edge 0
    assert connected and far_edge.abs().sum() == 0


def test_generator_outputs_keep_the_batch_format():
    method = make_method()
    batch = make_batch()

    outputs = method.model.forward_generator(batch)

    for source, destination in (("group_a", "group_a"), ("group_b", "group_a"), ("group_a", "group_b"), ("group_b", "group_b")):
        assert outputs.pairs[(source, destination)].fake.shape == batch[destination]["motion"].shape
    for domain in ("group_a", "group_b"):
        assert outputs.reconstructions[domain].shape == batch[domain]["motion"].shape
    assert outputs.motion_codes["group_a"].shape == outputs.motion_codes["group_b"].shape
    assert outputs.motion_codes["group_a"].shape[-1] == WINDOW // 4


def test_train_step_updates_generators_and_discriminators():
    torch.manual_seed(0)
    method = make_method(loss_grad_norm_interval=1)
    method.train()

    before = {
        name: parameter.clone()
        for name, parameter in method.model.named_parameters()
    }

    metrics = method.train_step(make_batch())

    for domain, route in (("group_a", "group_a_to_group_b"), ("group_b", "group_b_to_group_a")):
        for term in (
            f"{domain}/reconstruction",
            f"{domain}/root_position",
            f"{domain}/joint_position",
            f"{route}/latent_consistency",
            f"{route}/end_effector",
            f"{route}/contact",
            f"{route}/adversarial",
            f"{domain}/discriminator",
        ):
            assert term in metrics
            assert f"loss_grad/{term}" in metrics

        for component in ("motion_encoder", "offset_encoder", "decoder", "discriminator"):
            assert metrics[f"grad/{domain}/{component}"] > 0

    assert "total/generator" in metrics and "total/discriminator" in metrics
    assert all(isinstance(value, float) and value == value for value in metrics.values())

    changed = {
        name
        for name, parameter in method.model.named_parameters()
        if not torch.equal(parameter, before[name])
    }
    for domain in ("group_a", "group_b"):
        for component in ("motion_encoder", "offset_encoder", "decoder", "discriminator"):
            assert any(name.startswith(f"domains.{domain}.{component}.") for name in changed)


def test_validation_step_leaves_parameters_alone():
    torch.manual_seed(0)
    method = make_method()
    method.eval()

    before = {name: parameter.clone() for name, parameter in method.model.named_parameters()}
    metrics = method.validation_step(make_batch())

    for domain in ("group_a", "group_b"):
        for name in (
            "retarget_latent_l1",
            "retarget_realism",
            "retarget_ee_velocity",
            "reconstruction_mse",
            "reconstruction_mpjpe_m",
            "reconstruction_squared_error_over_height2",
        ):
            assert isinstance(metrics[f"{domain}/val/{name}"], float)

    for name, parameter in method.model.named_parameters():
        assert torch.equal(parameter, before[name])


def test_end_effector_terms_select_contacts_on_the_generated_velocity():
    source = torch.ones(1, 2, 1, 3)
    fake = torch.ones(1, 2, 1, 3)

    # Half of the generated frames are slow. The contact term only sees those, whatever the source does.
    fake[0, 0] = 0.0
    transfer, contact = end_effector_terms(source, fake)
    assert torch.isclose(transfer, ((fake - source) ** 2).mean())
    assert torch.isclose(contact, torch.tensor(1.0))


def test_end_effector_terms_raise_without_a_generated_contact():
    # The source is in contact and the generated motion is not: the official selection is empty.
    try:
        end_effector_terms(torch.zeros(1, 2, 1, 3), torch.ones(1, 2, 1, 3))
    except RuntimeError as error:
        assert "contact threshold" in str(error)
    else:
        raise AssertionError("Expected a RuntimeError for an empty contact selection")


def test_incompatible_skeletons_are_rejected():
    # A second spine chain: not the six body parts every domain's latent space relies on.
    extra_chain = AMASS_PARENTS + [0, 22]

    try:
        make_spec(extra_chain)
    except ValueError as error:
        assert "6 body parts" in str(error)
    else:
        raise AssertionError("Expected a ValueError for an incompatible skeleton")


def test_exactly_two_domains_are_required():
    try:
        SANModel(SANModelConfig(), {"group_a": make_spec(AMASS_PARENTS)})
    except ValueError as error:
        assert "two domains" in str(error)
    else:
        raise AssertionError("Expected a ValueError for one domain")


def test_runs_under_the_trainer_and_checkpoints_round_trip():
    torch.manual_seed(0)
    method = make_method()

    with TemporaryDirectory() as log_dir:
        trainer = Trainer(
            method=method,
            train_loader=[make_batch(), make_batch()],
            val_loader=None,
            cfg=TrainerConfig(
                num_epochs=1,
                log_dir=log_dir,
                use_tensorboard=False,
                verbose=False,
                device="cpu",
            ),
        )
        trainer.train()

        restored = make_method()
        restored_trainer = Trainer(
            method=restored,
            train_loader=[],
            val_loader=None,
            cfg=TrainerConfig(log_dir=str(Path(log_dir) / "restored"), use_tensorboard=False, device="cpu"),
        )
        restored_trainer.load_checkpoint(trainer.checkpoint_dir / "final.pt")

    assert restored_trainer.global_step == 2
    for name, value in method.model.state_dict().items():
        assert torch.equal(value, restored.model.state_dict()[name])


def test_visualize_writes_one_animation_per_domain():
    if importlib.util.find_spec("motion_retargeting.data") is None:
        raise SkipTest("needs the data layer's visualization helpers")

    torch.manual_seed(0)
    method = make_method()
    method.eval()

    with TemporaryDirectory() as save_dir:
        method.visualize(make_batch(), save_dir)
        assert (Path(save_dir) / "group_a_to_group_b.gif").exists()
        assert (Path(save_dir) / "group_b_to_group_a.gif").exists()


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            try:
                test()
                print(f"ok   {name}")
            except SkipTest as reason:
                print(f"skip {name}: {reason}")

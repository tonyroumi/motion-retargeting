"""
SAN method smoke tests on synthetic domains, with no dataset (or data layer) involved.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import SkipTest
import importlib.util

import torch
import yaml

from motion_retargeting.methods.san.config import DomainSpec, SANConfig, SANModelConfig
from motion_retargeting.methods.san.losses import end_effector_loss
from motion_retargeting.methods.san.method import SANMethod
from motion_retargeting.methods.san.model import SANModel
from motion_retargeting.modules.skeletal import SkeletalConv, build_edge_list, construct_adj
from motion_retargeting.training.trainer import Trainer, TrainerConfig


# The AMASS and CMU skeletons of the dataset groups in configs/methods/san.yaml.
AMASS_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
CMU_PARENTS = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 13, 14, 11, 16, 17, 18, 11, 20, 21, 22]

BATCH_SIZE = 3
WINDOW = 32


def make_spec(parents):
    return DomainSpec(
        parents=torch.tensor(parents),
        num_joints=len(parents),
        vel_stats=SimpleNamespace(mean=torch.tensor([0.0, 0.0, 0.01]), std=torch.tensor([0.02, 0.02, 0.03])),
        ang_vel_stats=SimpleNamespace(mean=torch.tensor(0.0), std=torch.tensor(0.1)),
    )


def make_domain_batch(num_joints):
    """ The batch format of `MotionDataset` once collated: near-identity rotations and small offsets. """
    rotations = torch.zeros(BATCH_SIZE, WINDOW, num_joints + 1, 4)
    rotations[..., 0] = 1.0
    q_heading = torch.zeros(BATCH_SIZE, WINDOW, 4)
    q_heading[..., 0] = 1.0

    offsets = 0.1 * torch.randn(BATCH_SIZE, num_joints, 3)
    offsets[:, 0] = 0.0

    return {
        "motion": rotations + 0.05 * torch.randn_like(rotations),
        "offsets": offsets,
        "q_heading": q_heading,
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
    assert SANConfig(**cfg["method"]["params"]).betas == [0.09, 0.99]


def test_motion_rows_map_to_graph_rows():
    edges = build_edge_list(AMASS_PARENTS)

    assert edges[0] == [0, 1]  # the root's virtual edge
    assert edges[4] == [2, 5]  # joint 4, child of joint 1
    assert len(construct_adj(edges)) == len(AMASS_PARENTS) + 1


def test_skeletal_conv_only_connects_neighbors():
    adj = construct_adj(build_edge_list([-1, 0, 1, 2, 3, 4]), d=1)
    conv = SkeletalConv(adj, in_channels_per_joint=2, out_channels_per_joint=2, kernel_size=1, padding=0)

    x = torch.zeros(1, 2 * len(adj), 1, requires_grad=True)
    conv(x)[0, 0].backward()  # First channel of the root edge

    connected = x.grad[0, :, 0].abs().sum() > 0
    far_edge = x.grad[0, 2 * 4:2 * 5, 0]  # Joint 4 is out of reach of the root edge
    assert connected and far_edge.abs().sum() == 0


def test_generator_outputs_keep_the_batch_format():
    method = make_method()
    batch = make_batch()

    outputs = method.model.forward_generator(batch)

    assert outputs.group_aa.shape == batch["group_a"]["motion"].shape
    assert outputs.group_ba.shape == batch["group_a"]["motion"].shape
    assert outputs.group_bb.shape == batch["group_b"]["motion"].shape
    assert outputs.group_ab.shape == batch["group_b"]["motion"].shape
    assert outputs.motion_code_a.shape == outputs.motion_code_b.shape
    assert outputs.motion_code_a.shape[-1] == WINDOW // 4


def test_train_step_updates_generators_and_discriminators():
    torch.manual_seed(0)
    method = make_method(loss_grad_norm_interval=1)
    method.train()

    before = {
        name: parameter.clone()
        for name, parameter in method.model.named_parameters()
    }

    metrics = method.train_step(make_batch())

    for domain in ("group_a", "group_b"):
        for term in (
            "reconstruction",
            "root_position",
            "joint_position",
            "latent_consistency",
            "end_effector",
            "adversarial",
            "discriminator",
        ):
            assert f"{domain}/{term}" in metrics
            assert f"loss_grad/{domain}/{term}" in metrics

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
            "retarget_latent",
            "retarget_realism",
            "retarget_ee_velocity",
            "reconstruction_mse",
            "reconstruction_mpjpe",
        ):
            assert isinstance(metrics[f"{domain}/val/{name}"], float)

    for name, parameter in method.model.named_parameters():
        assert torch.equal(parameter, before[name])


def test_end_effector_loss_handles_batches_without_contact():
    predicted = torch.zeros(1, 2, 1, 3)
    target = torch.ones(1, 2, 1, 3)

    assert end_effector_loss(predicted, target).item() == 1.0

    # Half of the target frames are in contact: the extra term only sees those.
    target[0, 0] = 0.0
    predicted[0, 0] = 0.5
    expected = ((predicted - target) ** 2).mean() + 100 * 0.25
    assert torch.isclose(end_effector_loss(predicted, target), expected)


def test_incompatible_skeletons_are_rejected():
    # A second spine chain: the primal skeleton gets one region more than AMASS'.
    extra_chain = AMASS_PARENTS + [0, 22]

    try:
        SANModel(SANModelConfig(), {"group_a": make_spec(AMASS_PARENTS), "group_b": make_spec(extra_chain)})
    except ValueError as error:
        assert "primal skeleton" in str(error)
    else:
        raise AssertionError("Expected a ValueError for incompatible skeletons")


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

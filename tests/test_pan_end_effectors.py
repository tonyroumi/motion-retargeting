"""
PAN's end-effector terms: contacts selected on the source velocity, an empty source selection, and the
feet / head / hands and height path found from each supported topology's actual part labels, whose key
order is the labeler's (AMASS / CMU caches: arms, head, trunk, legs) or the reference's (Mixamo: legs,
spine, head, arms).
"""

from __future__ import annotations

from typing import Dict, List

import pytest
import torch

from motion_retargeting.data.adapters.mixamo import GROUP_A_PARTS, GROUP_B_PARTS
from motion_retargeting.methods.pan.config import DomainSpec
from motion_retargeting.methods.pan.losses import end_effector_terms

THRESHOLD = 0.008

# The cached interactive labels, as stored in data/processed/{amass,cmu}.pt.
AMASS_NAMES = [
    "pelvis", "l_hip", "r_hip", "spine1", "l_knee", "r_knee", "spine2", "l_ankle", "r_ankle", "spine3",
    "l_foot", "r_foot", "neck", "l_collar", "r_collar", "head", "l_shoulder", "r_shoulder",
    "l_elbow", "r_elbow", "l_wrist", "r_wrist",
]
AMASS_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
AMASS_PARTS = {3: [0, 3, 6, 9], 4: [1, 4, 7, 10], 5: [2, 5, 8, 11], 2: [12, 15], 0: [13, 16, 18, 20], 1: [14, 17, 19, 21]}

CMU_NAMES = [
    "root", "lfemur", "ltibia", "lfoot", "ltoes", "rfemur", "rtibia", "rfoot", "rtoes", "lowerback",
    "upperback", "thorax", "lowerneck", "upperneck", "head", "head_end", "lclavicle", "lhumerus",
    "lradius", "lwrist", "rclavicle", "rhumerus", "rradius", "rwrist",
]
CMU_PARENTS = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 13, 14, 11, 16, 17, 18, 11, 20, 21, 22]
CMU_PARTS = {3: [0, 9, 10, 11, 12, 16, 20], 4: [1, 2, 3, 4], 5: [5, 6, 7, 8], 2: [13, 14, 15], 0: [17, 18, 19], 1: [21, 22, 23]}

INTERACTIVE_ROLES = {0: "left_arm", 1: "right_arm", 2: "head", 3: "trunk", 4: "left_leg", 5: "right_leg"}
MIXAMO_ROLES = {0: "left_leg", 1: "right_leg", 2: "trunk", 3: "head", 4: "left_arm", 5: "right_arm"}

# A T-pose step per joint, by the anatomy of its part: legs down, arms out, trunk and head up.
ROLE_OFFSETS = {
    "left_leg": (0.02, -0.2, 0.0), "right_leg": (-0.02, -0.2, 0.0),
    "left_arm": (0.15, 0.0, 0.0), "right_arm": (-0.15, 0.0, 0.0),
    "trunk": (0.0, 0.1, 0.0), "head": (0.0, 0.1, 0.0),
}


def mixamo(parts: Dict[int, List[str]], anchors: Dict[str, str]) -> tuple[List[str], List[int], Dict[int, List[int]]]:
    """ The adapter's joint order; each part a chain, its first joint hung from `anchors` (or the root). """
    spine = parts[2]
    names = [spine[0]] + parts[0] + parts[1] + spine[1:] + parts[3] + parts[4] + parts[5]
    parent_names = {}
    for part in parts.values():
        for previous, joint in zip([anchors.get(part[0], spine[0])] + part, part):
            parent_names[joint] = previous
    parents = [-1] + [names.index(parent_names[name]) for name in names[1:]]
    return names, parents, {key: [names.index(name) for name in part] for key, part in parts.items()}


MIXAMO_A = mixamo(GROUP_A_PARTS, {"Neck": "Spine2", "LeftShoulder": "Spine2", "RightShoulder": "Spine2"})
MIXAMO_B = mixamo(GROUP_B_PARTS, {"Neck": "Spine2", "LeftShoulder": "Spine2", "RightShoulder": "Spine2"})

TOPOLOGIES = {
    "amass": (AMASS_NAMES, AMASS_PARENTS, AMASS_PARTS, INTERACTIVE_ROLES),
    "cmu": (CMU_NAMES, CMU_PARENTS, CMU_PARTS, INTERACTIVE_ROLES),
    "mixamo_a": (*MIXAMO_A, MIXAMO_ROLES),
    "mixamo_b": (*MIXAMO_B, MIXAMO_ROLES),
}
EXPECTED_END_EFFECTORS = {
    "amass": ["l_foot", "r_foot", "head", "l_wrist", "r_wrist"],
    "cmu": ["ltoes", "rtoes", "head_end", "lwrist", "rwrist"],
    "mixamo_a": ["LeftToeBase", "RightToeBase", "Head", "LeftHand", "RightHand"],
    "mixamo_b": ["LeftToe_End", "RightToe_End", "HeadTop_End", "LeftHand", "RightHand"],
}
# Joints whose bones the height sums: left foot tip to the root, then head tip to the root.
EXPECTED_HEIGHT_PATHS = {
    "amass": ["l_hip", "l_knee", "l_ankle", "l_foot", "spine1", "spine2", "spine3", "neck", "head"],
    "cmu": ["lfemur", "ltibia", "lfoot", "ltoes", "lowerback", "upperback", "thorax", "lowerneck", "upperneck", "head", "head_end"],
}


def domain_spec(topology: str, num_banks: int = 2) -> tuple[DomainSpec, List[str]]:
    names, parents, parts, roles = TOPOLOGIES[topology]
    role_of = {joint: roles[key] for key, part in parts.items() for joint in part}
    offsets = torch.tensor([ROLE_OFFSETS[role_of[joint]] for joint in range(len(names))])
    offsets[0] = 0.0
    scales = torch.linspace(0.9, 1.1, num_banks)[:, None, None]
    channels = 4 * len(names) + 4
    spec = DomainSpec(
        parents=torch.tensor(parents),
        offsets=offsets[None] * scales,
        mean=torch.zeros(num_banks, channels),
        std=torch.ones(num_banks, channels),
        body_joints=parts,
    )
    return spec, names


@pytest.mark.parametrize("topology", sorted(TOPOLOGIES))
def test_end_effectors_are_feet_head_and_hands(topology):
    spec, names = domain_spec(topology)
    assert [names[joint] for joint in spec.ee_ids] == EXPECTED_END_EFFECTORS[topology]


@pytest.mark.parametrize("topology", sorted(EXPECTED_HEIGHT_PATHS))
def test_height_is_the_left_leg_plus_head_path(topology):
    spec, names = domain_spec(topology)
    path = [names.index(name) for name in EXPECTED_HEIGHT_PATHS[topology]]
    expected = torch.linalg.vector_norm(spec.offsets[:, path], dim=-1).sum(dim=-1)
    assert torch.allclose(spec.heights, expected)


def velocities(planted_source: List[int], planted_fake: List[int], frames: int = 6) -> tuple[torch.Tensor, torch.Tensor]:
    """ [1, frames, 5, 3]: every end effector moving except those planted frames (of end effector 0). """
    torch.manual_seed(0)
    source = 0.5 + torch.rand(1, frames, 5, 3)
    fake = 0.5 + torch.rand(1, frames, 5, 3)
    source[0, planted_source, 0] = 0.0
    fake[0, planted_fake, 0] = 0.0
    return source, fake


def test_contacts_are_selected_on_the_source_velocity():
    """ Source planted at frames 1, 2 and the fake planted elsewhere: only frames 1, 2 count. """
    source, fake = velocities(planted_source=[1, 2], planted_fake=[4])
    transfer, contact = end_effector_terms(source, fake, THRESHOLD)
    assert torch.allclose(transfer, ((fake - source) ** 2).mean())
    assert torch.allclose(contact, (fake[0, [1, 2], 0] ** 2).mean())


def test_a_moving_fake_still_has_a_contact_term():
    """ The official, fake-selected mask is empty here; the source-selected one is not. """
    source, fake = velocities(planted_source=[3], planted_fake=[])
    _, contact = end_effector_terms(source, fake, THRESHOLD)
    assert torch.allclose(contact, (fake[0, 3, 0] ** 2).mean())


def test_a_source_without_contacts_gives_a_zero_contact_term():
    """ Even with the fake planted: nothing in the source to hold still. Zero, finite, and in the graph. """
    source, fake = velocities(planted_source=[], planted_fake=[0, 1, 2])
    fake.requires_grad_(True)
    transfer, contact = end_effector_terms(source, fake, THRESHOLD)
    assert contact.item() == 0.0
    assert torch.allclose(transfer, ((fake - source) ** 2).mean())
    (grad,) = torch.autograd.grad(contact, fake)
    assert torch.equal(grad, torch.zeros_like(fake))

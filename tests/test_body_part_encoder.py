"""
Body-part pairing: index k of the skeleton code must be body key k, the same body whose
token sits at position k of the motion code (see build_joint_mask).
"""

import pytest
import torch

from motion_retargeting.modules.encoders import BodyPartEncoder

# Iteration order of the real AMASS and CMU body maps, which is not sorted by key.
REAL_KEY_ORDER = [3, 4, 5, 2, 0, 1]
JOINTS_PER_BODY = 2


def make_body_joints(key_order):
    """ Body key k owns joints [2k, 2k + 1], inserted in `key_order`. """
    return {k: [JOINTS_PER_BODY * k + i for i in range(JOINTS_PER_BODY)] for k in key_order}


def test_skeleton_index_is_body_key():
    body_joints = make_body_joints(REAL_KEY_ORDER)
    assert list(body_joints) == REAL_KEY_ORDER

    torch.manual_seed(0)
    encoder = BodyPartEncoder(input_dim=3, embed_dim=8, body_joints=body_joints).eval()
    offsets = torch.randn(2, JOINTS_PER_BODY * len(body_joints), 3)
    z_skel = encoder(offsets)
    assert z_skel.shape == (2, len(body_joints), 8)

    for key, joints in body_joints.items():
        perturbed = offsets.clone()
        perturbed[:, joints] += 1.0
        changed = (encoder(perturbed) - z_skel).abs().amax(dim=(0, 2)) > 0
        assert changed.nonzero().flatten().tolist() == [key], f"body key {key} landed at the wrong index"


def test_body_keys_must_be_contiguous():
    with pytest.raises(ValueError):
        BodyPartEncoder(input_dim=3, embed_dim=8, body_joints={0: [0], 2: [1]})

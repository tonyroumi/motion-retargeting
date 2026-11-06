from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch

NUM_PARTS = 6
# Additive attention weight of a blocked token pair: negative infinity, so it gets exactly zero attention.
BLOCKED = float("-inf")


def labeled_parts(body_joints: Optional[Dict[int, List[int]]], num_joints: int) -> List[List[int]]:
    """
    The topology's manually labeled body parts as joint lists, ordered by key: left leg, right leg,
    spine, head, left arm, right arm. The labels must assign every joint to exactly one non-empty part.
    """
    if body_joints is None or sorted(body_joints) != list(range(NUM_PARTS)):
        raise ValueError(f"PAN needs {NUM_PARTS} labeled body parts keyed 0 to {NUM_PARTS - 1}, got {body_joints}")

    parts = [[int(joint) for joint in body_joints[key]] for key in range(NUM_PARTS)]
    if not all(parts) or sorted(joint for part in parts for joint in part) != list(range(num_joints)):
        raise ValueError(f"Labeled body parts must assign each of the {num_joints} joints to exactly one non-empty part: {parts}")
    return parts


def fresh_parts(parts: Sequence[Sequence[int]]) -> List[List[int]]:
    """ Joint parts as edge indices: drop a leading root joint, then edge = joint - 1. """
    return [[joint - 1 for joint in (part[1:] if part[0] == 0 else part)] for part in parts]


def attention_mask(joint_parts: Sequence[Sequence[int]], num_tokens: int) -> torch.Tensor:
    """
    Additive mask [P + num_tokens, P + num_tokens], rows querying columns. A part token reads its own
    members; a member reads the members of every part it belongs to, so a token shared by several parts
    relays across them; and every query reads the root token, which itself reads only itself. Member tokens
    never read part tokens.
    """
    parts = fresh_parts(joint_parts)
    num_parts = len(parts)
    size = num_parts + num_tokens

    allowed = torch.zeros(size, size, dtype=torch.bool)
    for index, part in enumerate(parts):
        members = [num_parts + edge for edge in part]
        allowed[index, members] = True
        for member in members:
            allowed[member, members] = True
    allowed[range(size), range(size)] = True
    allowed[:, -1] = True

    return torch.where(allowed, torch.tensor(0.0), torch.tensor(BLOCKED))


def part_token_mask(parts: Sequence[Sequence[int]], num_tokens: int) -> torch.Tensor:
    """ Bool [P, num_tokens]: which tokens each part owns, plus the always-on last column. """
    mask = torch.zeros(len(parts), num_tokens, dtype=torch.bool)
    for index, part in enumerate(parts):
        mask[index, part] = True
    mask[:, -1] = True
    return mask


def temporal_conv_mask(joint_parts: Sequence[Sequence[int]], num_tokens: int) -> torch.Tensor:
    """ `part_token_mask` of the encoder shortcut and final decoder layer, from the labeled body parts. """
    return part_token_mask(fresh_parts(joint_parts), num_tokens)


def skeleton_offset_mask(joint_parts: Sequence[Sequence[int]], num_joints: int) -> torch.Tensor:
    """ `part_token_mask` of the skeleton encoder's first layer, read against joint offsets. """
    return part_token_mask(fresh_parts(joint_parts), num_joints)


def tokens_to_parts(mask: torch.Tensor, channels_per_token: int, channels_per_part: int) -> torch.Tensor:
    """ Connectivity [P * out, tokens * in] of a layer from token channels to part blocks. """
    return mask.repeat_interleave(channels_per_part, dim=0).repeat_interleave(channels_per_token, dim=1)


def parts_to_parts(num_parts: int, in_per_part: int, out_per_part: int) -> torch.Tensor:
    """ Connectivity [P * out, P * in] between part blocks: each only reads itself. """
    return torch.eye(num_parts, dtype=torch.bool).repeat_interleave(out_per_part, dim=0).repeat_interleave(in_per_part, dim=1)


def parts_to_tokens(mask: torch.Tensor, channels_per_part: int, channels_per_token: int) -> torch.Tensor:
    """
    Connectivity [tokens * out, P * in] of the final decoder layer. An output token owned by several
    parts reads all of them; one owned by none reads everything (the official code's fallback for
    fresh training, not the zero row its published checkpoints hold).
    """
    token_mask = mask.t()  # [tokens, P]
    uncovered = ~token_mask.any(dim=1)
    token_mask = token_mask | uncovered[:, None]
    return token_mask.repeat_interleave(channels_per_token, dim=0).repeat_interleave(channels_per_part, dim=1)

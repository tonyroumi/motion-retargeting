from __future__ import annotations

from typing import List, Sequence

import torch

NUM_PARTS = 6
# Additive attention weight of a blocked token pair, as in the official code.
BLOCKED = -1e20


def topology_chains(parents: Sequence[int]) -> List[List[int]]:
    """
    Joint chains in the official code's traversal order: depth-first from the root, children ascending,
    starting a new chain at each branching non-root joint. Parents come before children.
    """
    # parents = [int(parent) for parent in parents]
    children: List[List[int]] = [[] for _ in parents]
    for child, parent in enumerate(parents[1:], start=1):
        children[parent].append(child)

    chains: List[List[int]] = []

    def walk(joint: int, chain: List[int]) -> None:
        if joint != 0 and len(children[joint]) > 1:
            chains.append(chain)
            chain = [joint]
        if not children[joint]:
            chains.append(chain)
        for child in children[joint]:
            walk(child, chain + [child])

    walk(0, [0])
    return chains


def humanoid_chains(parents: Sequence[int]) -> List[List[int]]:
    """
    `topology_chains`, validated to be left leg, right leg, spine, head, left arm, right arm: two legs
    and a spine leaving the root, then three limbs leaving the spine's end. This checks structure only;
    that the first leg and arm are the left ones comes from the adapters' child order.
    """
    parents = [int(parent) for parent in parents]
    if parents[0] != -1 or any(not 0 <= parent < child for child, parent in enumerate(parents[1:], start=1)):
        raise ValueError(f"Expected the root first and parents before children, got {parents}")

    chains = topology_chains(parents)
    if len(chains) != NUM_PARTS:
        raise ValueError(f"PAN needs {NUM_PARTS} body parts, this topology has {len(chains)}: {chains}")

    leaves = set(range(len(parents))) - set(parents)
    left_leg, right_leg, spine, head, left_arm, right_arm = chains
    if (
        any(chain[0] != 0 for chain in (left_leg, right_leg, spine))
        or any(chain[0] != spine[-1] for chain in (head, left_arm, right_arm))
        or any(chain[-1] not in leaves for chain in (left_leg, right_leg, head, left_arm, right_arm))
    ):
        raise ValueError(f"Chains do not match legs, spine, head and arms in order: {chains}")
    return chains


def fresh_parts(chains: Sequence[Sequence[int]]) -> List[List[int]]:
    """ Chains as edge indices: drop a leading root joint, then edge = joint - 1. """
    return [[joint - 1 for joint in (chain[1:] if chain[0] == 0 else chain)] for chain in chains]


def double_converted_parts(chains: Sequence[Sequence[int]]) -> List[List[int]]:
    """ The official code's second conversion, applied to `fresh_parts` as if they were joints. """
    return fresh_parts(fresh_parts(chains))


def attention_mask(chains: Sequence[Sequence[int]], num_tokens: int) -> torch.Tensor:
    """
    Additive mask [P + num_tokens, P + num_tokens], rows querying columns. A part token reads its own
    members; a member reads the members of every part it belongs to, so a shared branch token relays
    across parts; and every query reads the root token, which itself reads only itself. Member tokens
    never read part tokens.
    """
    parts = fresh_parts(chains)
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


def temporal_conv_mask(chains: Sequence[Sequence[int]], num_tokens: int) -> torch.Tensor:
    """ `part_token_mask` of the encoder shortcut and final decoder layer (double converted). """
    return part_token_mask(double_converted_parts(chains), num_tokens)


def skeleton_offset_mask(chains: Sequence[Sequence[int]], num_joints: int) -> torch.Tensor:
    """ `part_token_mask` of the skeleton encoder's first layer, read against joint offsets. """
    return part_token_mask(fresh_parts(chains), num_joints)


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

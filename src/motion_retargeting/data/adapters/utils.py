"""
Adapter helpers: interactive labeling of which joints form each body part.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

import matplotlib.pyplot as plt
import torch

from motion_retargeting.data.representations.skeleton import SkeletonTopology
from motion_retargeting.data.visualization.visualize import visualize_offsets


def label_joint_bodies(
    topology: SkeletonTopology,
    offsets: torch.Tensor,
) -> Dict[int, List[int]]:
    """
    Interactively assign each joint of `topology` to a body part, shown in the
    rest pose given by one clip's `offsets` ([J, 3]).

    All joints, including the root joint at index 0, can be grouped.
    """
    num_joints = topology.num_joints
    selectable_joints = set(range(num_joints))

    _, fig, ax, scatter = visualize_offsets(
        offsets.detach().cpu().numpy(),
        parent_indices=topology.parents.tolist(),
        joint_names=topology.joint_names,
        save_path=None,
        show=False,
        close=False,
    )

    ax.set_title(
        "Click joints, <enter> to commit a body, "
        "'q' once all joints are assigned"
    )

    joint_group: dict[int, int] = {}
    selected: set[int] = set()
    next_group_id = 0

    body_colors = plt.get_cmap("tab20")

    def colors():
        colors = []

        for idx in range(num_joints):
            if idx in selected:
                colors.append("red")
            elif idx in joint_group:
                colors.append(body_colors(joint_group[idx] % 20))
            else:
                colors.append("gray")

        return colors

    def redraw():
        scatter.set_color(colors())
        fig.canvas.draw_idle()

    def on_pick(event):
        for idx in event.ind:
            idx = int(idx)

            if idx in joint_group:
                continue

            selected.symmetric_difference_update({idx})

        redraw()

    def on_key(event):
        nonlocal next_group_id

        if event.key == "enter":
            if not selected:
                return

            for idx in selected:
                joint_group[idx] = next_group_id

            next_group_id += 1
            selected.clear()
            redraw()

        elif event.key == "q":
            if not selectable_joints.issubset(joint_group):
                return

            plt.close(fig)

    fig.canvas.mpl_connect("pick_event", on_pick)
    fig.canvas.mpl_connect("key_press_event", on_key)

    redraw()
    plt.show()

    # Any joint never explicitly grouped becomes its own singleton body.
    for idx in range(num_joints):
        if idx not in joint_group:
            joint_group[idx] = next_group_id
            next_group_id += 1

    body_joints: Dict[int, List[int]] = defaultdict(list)

    for idx in range(num_joints):
        body_joints[joint_group[idx]].append(idx)

    return dict(body_joints)

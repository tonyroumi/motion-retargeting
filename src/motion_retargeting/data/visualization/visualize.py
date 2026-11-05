"""
Matplotlib rendering of skeleton rest poses and motion animations.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def visualize_offsets(
    offsets,
    parent_indices=None,
    joint_names=None,
    save_path=None,
    show=False,
    close=True,
):
    """
    Plot a skeleton in its rest pose from per-joint parent-relative offsets.

    Returns (positions, fig, ax, scatter) so callers can keep annotating or
    attaching interaction callbacks to the plot before it is shown, saved,
    or closed.
    """
    num_joints = len(offsets)

    # Compute global positions by accumulating offsets
    positions = np.zeros((num_joints, 3))

    if parent_indices is None:
        # Assume simple chain: each joint's parent is the previous joint
        parent_indices = [-1] + list(range(num_joints - 1))

    # Forward pass: accumulate offsets to get global positions
    for i in range(num_joints):
        if parent_indices[i] == -1:
            # Root joint
            positions[i] = offsets[i]
        else:
            # Child joint: parent position + offset
            positions[i] = positions[parent_indices[i]] + offsets[i]

    # Create 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot joints as points
    scatter = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
            c='red', s=50, marker='o', picker=True, label='Joints')

    # Plot bones (connections between parent and child)
    for i in range(num_joints):
        if parent_indices[i] != -1:
            parent_pos = positions[parent_indices[i]]
            child_pos = positions[i]
            ax.plot([parent_pos[0], child_pos[0]],
                [parent_pos[1], child_pos[1]],
                [parent_pos[2], child_pos[2]], 'b-', linewidth=2)

    # Label joints
    labels = [f'{i}: {name}' for i, name in enumerate(joint_names)] if joint_names is not None else [f'J{i}' for i in range(num_joints)]
    for i in range(num_joints):
        ax.text(positions[i, 0], positions[i, 1], positions[i, 2],
                labels[i], fontsize=8)

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Skeleton Structure (T-Pose from Offsets)')
    ax.legend()

    # Equal aspect ratio
    max_range = np.array([positions[:, 0].max() - positions[:, 0].min(),
                        positions[:, 1].max() - positions[:, 1].min(),
                        positions[:, 2].max() - positions[:, 2].min()]).max() / 2.0
    mid_x = (positions[:, 0].max() + positions[:, 0].min()) * 0.5
    mid_y = (positions[:, 1].max() + positions[:, 1].min()) * 0.5
    mid_z = (positions[:, 2].max() + positions[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Skeleton visualization saved to: {save_path}")

    if show:
        plt.show()

    if close:
        plt.close(fig)

    return positions, fig, ax, scatter


def visualize_motion(
    positions,
    parent_indices,
    joint_names=None,
    fps=30.0,
    save_path=None,
    show=False,
    close=True,
):
    """
    Animate a skeleton from per-frame global joint positions [T, J, 3]
    (e.g. the output of ForwardKinematics).

    `save_path` may end in .gif (Pillow) or .mp4 (ffmpeg).

    Returns (anim, fig, ax) so callers can keep customizing the animation
    before it is shown, saved, or closed.
    """
    if hasattr(positions, "detach"):
        positions = positions.detach().cpu().numpy()
    positions = np.asarray(positions)
    num_frames, num_joints, _ = positions.shape

    bones = [(parent_indices[i], i) for i in range(num_joints) if parent_indices[i] != -1]

    # Create 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot joints as points
    scatter = ax.scatter(positions[0, :, 0], positions[0, :, 1], positions[0, :, 2],
            c='red', s=50, marker='o', label='Joints')

    # Plot bones (connections between parent and child)
    lines = [ax.plot([], [], [], 'b-', linewidth=2)[0] for _ in bones]

    # Label joints
    texts = []
    if joint_names is not None:
        texts = [ax.text(*positions[0, i], joint_names[i], fontsize=8) for i in range(num_joints)]

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    # Equal aspect ratio, fixed over the whole sequence
    flat = positions.reshape(-1, 3)
    max_range = (flat.max(axis=0) - flat.min(axis=0)).max() / 2.0
    mid = (flat.max(axis=0) + flat.min(axis=0)) * 0.5
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    def update(frame):
        pose = positions[frame]
        scatter._offsets3d = (pose[:, 0], pose[:, 1], pose[:, 2])
        for line, (parent, child) in zip(lines, bones):
            line.set_data_3d(*pose[[parent, child]].T)
        for i, text in enumerate(texts):
            text.set_position_3d(pose[i])
        ax.set_title(f'Skeleton Motion (frame {frame + 1}/{num_frames})')
        return [scatter, *lines, *texts]

    anim = FuncAnimation(fig, update, frames=num_frames, interval=1000.0 / fps)

    if save_path:
        anim.save(str(save_path), fps=fps, dpi=100)
        print(f"Motion visualization saved to: {save_path}")

    if show:
        plt.show()

    if close:
        plt.close(fig)

    return anim, fig, ax

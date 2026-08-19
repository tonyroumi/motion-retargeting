from .kinematics import ForwardKinematics

from mpl_toolkits.mplot3d import Axes3D
from numpy.typing import ArrayLike
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import numpy as np

class SkeletonVisualizer:
    """Skeleton and motion data visualization utilities"""
    @staticmethod
    def _set_axes_equal(ax):
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        y_range = abs(y_limits[1] - y_limits[0])
        z_range = abs(z_limits[1] - z_limits[0])

        x_middle = np.mean(x_limits)
        y_middle = np.mean(y_limits)
        z_middle = np.mean(z_limits)

        plot_radius = 0.5 * max(x_range, y_range, z_range)

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    @staticmethod
    def visualize_skeleton(
      global_position: ArrayLike, 
      height: float,
      foot_idx: int, 
      head_idx: int, 
      save_path: str
    ):
        """ Visualize a single skeleton frame with height measurement. """        
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")

        xs = global_position[:, 0]
        ys = global_position[:, 1]
        zs = global_position[:, 2]

        ax.scatter(xs, ys, zs, s=40, c="black")

        # Label each joint by its index
        for idx, (x, y, z) in enumerate(global_position):
            ax.text(x, y, z, str(idx), color="red", fontsize=8)

        # Get foot and head positions
        foot_pos = global_position[foot_idx]
        head_pos = global_position[head_idx]
        
        # Calculate the direction from foot to head
        direction = head_pos - foot_pos
        direction_normalized = direction / np.linalg.norm(direction)
        
        expected_head_pos = foot_pos + direction_normalized * height
        
        ax.plot([foot_pos[0], expected_head_pos[0]], 
                [foot_pos[1], expected_head_pos[1]], 
                [foot_pos[2], expected_head_pos[2]], 
                'b-', linewidth=2, label=f'Height: {height:.3f}')
        
        ax.legend()

        # Make axes equal
        SkeletonVisualizer._set_axes_equal(ax)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close(fig)

    @staticmethod
    def visualize_motion(global_positions: ArrayLike, save_path: str, fps: int = 30):
        """ Visualize motion sequence as a video. """     
        if global_positions.ndim == 4:
            num_windows, window_size, num_joints, _ = global_positions.shape
            # Stitch all windows together along the time dimension
            global_positions = global_positions.reshape(num_windows * window_size, num_joints, 3)
        
        T = global_positions.shape[0]

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()

        out = cv2.VideoWriter(
            save_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h)
        )

        frame_iter = tqdm(range(T), desc=f"Rendering frames", leave=False, position=1)
        for t in frame_iter:
            ax.cla()

            frame_pos = global_positions[t]  # [J, 3]
            xs = frame_pos[:, 0]
            ys = frame_pos[:, 1]
            zs = frame_pos[:, 2]

            ax.scatter(xs, ys, zs, s=40, c="black")

            for idx, (x, y, z) in enumerate(frame_pos):
                ax.text(x, y, z, str(idx), color="red", fontsize=8)

            SkeletonVisualizer._set_axes_equal(ax)

            fig.canvas.draw()
            buf = np.asarray(fig.canvas.buffer_rgba())
            rgb = cv2.cvtColor(buf, cv2.COLOR_RGBA2RGB)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            out.write(bgr)

        out.release()
        plt.close(fig)

    @staticmethod
    def visualize_dataset(cross_motion_dataloader: DataLoader, save_path: str):
        (Path(save_path) / "domain_a").mkdir(parents=True, exist_ok=True)
        (Path(save_path) / "domain_b").mkdir(parents=True, exist_ok=True)

        motion_datasets = cross_motion_dataloader.dataset.domains
        for i, batch in enumerate(cross_motion_dataloader):
            B, _, T = batch.motions[0].shape

            positions_A = ForwardKinematics.forward_batched(
                batch.rotations[0][:, :-3]
                  .reshape(B, -1, 4, T)
                  .permute(0, 3, 1, 2), 
                batch.offsets[0].reshape(B, -1, 3),
                batch.rotations[0][:, -3:],
                motion_datasets[0].topology
            )
            positions_B = ForwardKinematics.forward_batched(
                batch.rotations[1][:, :-3]
                  .reshape(B, -1, 4, T)
                  .permute(0, 3, 1, 2), 
                batch.offsets[1].reshape(B, -1, 3),
                batch.rotations[1][:, -3:],
                motion_datasets[1].topology
            )
            
            print("Visualizing gt positions...")
            SkeletonVisualizer.visualize_motion(batch.gt_positions[0], f"{save_path}/domain_a/gt_position_batch_{i}.mp4")
            SkeletonVisualizer.visualize_motion(batch.gt_positions[1], f"{save_path}/domain_b/gt_position_batch_{i}.mp4")

            print("Visualizing positions from fk...")
            SkeletonVisualizer.visualize_motion(positions_A, f"{save_path}/domain_a/rotations_batch_{i}.mp4")
            SkeletonVisualizer.visualize_motion(positions_B, f"{save_path}/domain_b/rotations_batch_{i}.mp4")

    @staticmethod
    def visualize_offsets(offsets, parent_indices=None, save_path='skeleton_offsets.png'):
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
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                c='red', s=50, marker='o', label='Joints')
        
        # Plot bones (connections between parent and child)
        for i in range(num_joints):
            if parent_indices[i] != -1:
                parent_pos = positions[parent_indices[i]]
                child_pos = positions[i]
                ax.plot([parent_pos[0], child_pos[0]],
                    [parent_pos[1], child_pos[1]],
                    [parent_pos[2], child_pos[2]], 'b-', linewidth=2)
        
        # Label joints
        for i in range(num_joints):
            ax.text(positions[i, 0], positions[i, 1], positions[i, 2], 
                    f'J{i}', fontsize=8)
        
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
        
        # Save the plot
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)  # Close the figure to free memory
        print(f"Skeleton visualization saved to: {save_path}")
        
        return positions
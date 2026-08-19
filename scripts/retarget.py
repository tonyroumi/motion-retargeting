"""
Motion Retargeting Script

Retargets motion from a source skeleton to a target skeleton using a trained SkeletalGAN model.
"""

import argparse
from pathlib import Path
import numpy as np
import torch

from src.data import SkeletonMetadata, MotionSequence
from src.models.networks import SkeletalGAN
from src.utils import SkeletonVisualizer, ForwardKinematics, ArrayUtils

class MotionRetargeter:  
    def __init__(self, model_path: str, window_size: int = 64):
        self.model = SkeletalGAN.load_for_inference(model_path)
        self.window_size = window_size
        self.window_step = window_size // 2
    
    def create_motion_windows(self, motion_data: np.ndarray) -> np.ndarray:
        T = motion_data.shape[0]
        num_windows = T // self.window_step - 1
        windows = []
        
        for i in range(num_windows):
            start = i * self.window_step
            end = start + self.window_size
            if end > T:
                break
            windows.append(motion_data[start:end])
        
        return np.stack(windows) if windows else np.array([])
    
    def prepare_motion(self, motion_seq: MotionSequence) -> torch.Tensor:
        # Combine rotations and root position
        rots = ArrayUtils.to_torch(motion_seq.rotations).flatten(-2)
        root_pos = ArrayUtils.to_torch(motion_seq.positions[:, 0])
        full_motion = torch.hstack([rots, root_pos])
        
        full_motion = full_motion.unsqueeze(0).transpose(1, 2)
        return self.model.domains[0].norm(full_motion)
    
    def retarget(
        self,
        motion_seq: MotionSequence,
        source_skeleton: SkeletonMetadata,
        target_skeleton: SkeletonMetadata
    ) -> np.ndarray:
        """
        Retarget motion from source to target skeleton.
        
        Args:
            motion_seq: Source motion sequence
            source_skeleton: Source skeleton metadata
            target_skeleton: Target skeleton metadata
            
        Returns:
            Retargeted global positions [B, T, J, 3]
        """
        # Prepare inputs
        full_motion = self.prepare_motion(motion_seq)
        source_offsets = ArrayUtils.to_torch(
            source_skeleton.offsets.reshape(-1)
        ).unsqueeze(0)
        target_offsets = ArrayUtils.to_torch(
            target_skeleton.offsets.reshape(-1)
        ).unsqueeze(0)
        
        # Translate motion
        translated_motion = self.model.translate(
            full_motion, source_offsets, target_offsets
        )
        
        # Compute forward kinematics
        B, _, T = translated_motion.shape
        rotations = translated_motion[:, :-3].reshape(B, -1, 4, T).permute(0, 3, 1, 2)
        root_positions = translated_motion[:, -3:]
        
        global_positions = ForwardKinematics.forward_batched(
            rotations,
            ArrayUtils.to_torch(target_skeleton.offsets.reshape(-1, 3)).unsqueeze(0),
            root_positions,
            target_skeleton
        )
        
        return global_positions


def setup_output_dirs(output_dir: Path) -> dict[str, Path]:
    """Create output directory structure."""
    dirs = {
        'root': output_dir,
        'videos': output_dir / 'videos',
        'skeletons': output_dir / 'skeletons'
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Retarget motion between different skeletal structures'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--source-skeleton',
        type=str,
        required=True,
        help='Path to source skeleton metadata (.npz)'
    )
    parser.add_argument(
        '--target-skeleton',
        type=str,
        required=True,
        help='Path to target skeleton metadata (.npz)'
    )
    parser.add_argument(
        '--motion',
        type=str,
        required=True,
        help='Path to source motion sequence (.npz)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/retargeting',
        help='Output directory for visualizations (default: outputs/retargeting)'
    )

    return parser.parse_args()


def main():
    import debugpy
    print("[DEBUG] Waiting for debugger to attach on 0.0.0.0:5678 ...")
    debugpy.listen(("0.0.0.0", 5678))
    debugpy.wait_for_client()
    print("[DEBUG] Debugger attached.")

    """Main retargeting pipeline."""
    args = parse_args()
    
    # Setup output directories
    output_dirs = setup_output_dirs(Path(args.output_dir))
    
    # Extract names for file naming
    source_name = Path(args.source_skeleton).stem
    target_name = Path(args.target_skeleton).stem
    motion_name = Path(args.motion).stem
    
    print(f"Loading model from: {args.model}")
    print(f"Source skeleton: {source_name}")
    print(f"Target skeleton: {target_name}")
    print(f"Motion: {motion_name}")
    print("-" * 100)
    
    # Load data
    source_skeleton = SkeletonMetadata.load(args.source_skeleton)
    target_skeleton = SkeletonMetadata.load(args.target_skeleton)
    source_motion = MotionSequence.load(args.motion)
    
    # Perform retargeting
    retargeter = MotionRetargeter(args.model)
    retargeted_positions = retargeter.retarget(
        source_motion,
        source_skeleton,
        target_skeleton
    )
    
   
    # Visualize offsets
    source_skeleton_path = output_dirs['skeletons'] / f"{source_name}_offsets.png"
    SkeletonVisualizer.visualize_offsets(
        offsets=source_skeleton.offsets,
        parent_indices=source_skeleton.kintree,
        save_path=source_skeleton_path
    )

    target_skeleton_path = output_dirs['skeletons'] / f"{target_name}_offsets.png"
    SkeletonVisualizer.visualize_offsets(
        offsets=target_skeleton.offsets,
        parent_indices=target_skeleton.kintree[0],
        save_path=target_skeleton_path
    )
   
    # Source motion video
    source_motion_path = output_dirs['videos'] / f"{source_name}_{motion_name}_original.mp4"
    SkeletonVisualizer.visualize_motion(
        source_motion.positions,
        str(source_motion_path)
    )

    # Retargeted motion video
    retargeted_path = output_dirs['videos'] / f"{source_name}_to_{target_name}_{motion_name}.mp4"
    SkeletonVisualizer.visualize_motion(
        retargeted_positions,
        str(retargeted_path)
    )
    
    print(f"All outputs saved to: {output_dirs['root']}")


if __name__ == "__main__":
    main()
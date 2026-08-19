"""
Skeletal GAN for unpaired motion translation from one motion domain to another. 
"""

from .autoencoder import SkeletalAutoEncoder
from .discriminator import SkeletalDiscriminator
from .encoder import SkeletalEncoder

from omegaconf import DictConfig
from typing import Dict, Any, Tuple, Literal, List, Callable
import torch
import torch.nn as nn

from src.core.normalization import NormalizationStats
from src.core.types import MotionOutput, PairedSample, SkeletonTopology
from src.utils.kinematics import ForwardKinematics
from src.utils.skeleton import SkeletonUtils
from src.utils import ImagePool, Logger

class SkeletalDomainModule(nn.Module):
    def __init__(
        self,
        topology: SkeletonTopology,
        normalization_stats: NormalizationStats,
        offset_encoder_params: DictConfig,
        auto_encoder_params: DictConfig,
        discriminator_params: DictConfig,
    ):
        super().__init__()

        self.topology = topology
        self.num_joints = len(topology.edge_adjacency)
        # Generator
        self.offset_encoder = SkeletalEncoder(adj_init=topology.edge_adjacency,
                                              edge_init=topology.edge_topology,
                                              params=offset_encoder_params,
                                              return_all=True)
        self.auto_encoder = SkeletalAutoEncoder(adj_init=topology.edge_adjacency,
                                                edge_init=topology.edge_topology,
                                                params=auto_encoder_params)
        # Discriminator
        self.discriminator = SkeletalDiscriminator(pooled_info=self.offset_encoder.pooling_hierarchy,
                                                   discriminator_params=discriminator_params)    
        
        self.norm_stats = normalization_stats
    
    def forward(self, motions: torch.Tensor, offsets: List[torch.Tensor]) -> Tuple[torch.Tensor]:
        return self.auto_encoder(motions, offsets)

    def encode(self, reconstructed: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        return self.auto_encoder.encoder(reconstructed, offset=offset, pad_global=True)
    
    def decode(self, latent_representation: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        return self.auto_encoder.decoder(latent_representation, offset=offset)

    def denorm(self, motions: torch.Tensor) -> torch.Tensor:
        return self.norm_stats.denorm(motions)

    def norm(self, motions: torch.Tensor) -> torch.Tensor:
        return self.norm_stats.norm(motions)
        
    def generator_parameters(self):
        return list(self.auto_encoder.parameters()) + list(self.offset_encoder.parameters())

    def discriminator_parameters(self):
        return self.discriminator.parameters()
    
    def discriminators_requires_grad_(self, requires_grad: bool = True):
        for para in self.discriminator_parameters():
            para.requires_grad = requires_grad

class SkeletalGAN(nn.Module):   
    def __init__(
        self,
        topologies: Tuple[SkeletonTopology, SkeletonTopology],
        normalization_stats: Tuple[NormalizationStats],
        offset_encoder_params: DictConfig,
        auto_encoder_params: DictConfig,
        discriminator_params: DictConfig
    ):
        super().__init__()     

        self.topologies = topologies
        
        self.domains = nn.ModuleList([
            SkeletalDomainModule(
                topologies[0], 
                normalization_stats[0], 
                offset_encoder_params=offset_encoder_params, 
                auto_encoder_params=auto_encoder_params, 
                discriminator_params=discriminator_params
            ),
            SkeletalDomainModule(
                topologies[1],
                normalization_stats[1], 
                offset_encoder_params=offset_encoder_params, 
                auto_encoder_params=auto_encoder_params, 
                discriminator_params=discriminator_params
            ),
        ])

        self.offset_encoder_params = offset_encoder_params
        self.auto_encoder_params = auto_encoder_params
        self.discriminator_params = discriminator_params
    
    def _encode_offsets(
        self, 
        offsets: Tuple[torch.Tensor, torch.Tensor], 
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        offset_features = []
        for i, domain in enumerate(self.domains):
            offset_features.append(domain.offset_encoder(offsets[i].unsqueeze(-1)))
        return offset_features
    
    def forward_discriminator(self, motions: torch.Tensor, idx: int):
        return self.domains[idx].discriminator(motions)
    
    def generator_parameters(self):
        return [p for d in self.domains for p in d.generator_parameters()]

    def discriminator_parameters(self):
        return [p for d in self.domains for p in d.discriminator_parameters()]
      
    def discriminators_requires_grad_(self, requires_grad: bool = True) -> None:
        for domain in self.domains:
            domain.discriminators_requires_grad_(requires_grad)
    
    def forward(self, batch: PairedSample):
        B, _, T = batch.motions[0].shape

        offset_features = self._encode_offsets(offsets=batch.offsets)

        # -------------------------
        # Reconstruction phase
        # -------------------------
        reconstruction_out: Dict[int, MotionOutput] = {}
        for i, domain in enumerate(self.domains):
            latents, reconstructed = domain(batch.motions[i], offset_features[i])
            
            reconstructed_rot = domain.denorm(reconstructed)
            reconstructed_pos = ForwardKinematics.forward_batched(
                quaternions=reconstructed_rot[:, :-3]
                    .reshape(B, -1, 4, T)
                    .permute(0, 3, 1, 2),
                offsets=batch.offsets[i].reshape(B, -1, 3), 
                root_pos=reconstructed_rot[:, -3:],
                topology=domain.topology,
                world=True
            )

            reconstruction_out[i] = MotionOutput(
                latents=latents,
                motion=reconstructed,
                rotations=reconstructed_rot,
                positions=reconstructed_pos,
            )

        # -------------------------
        # Retargeting phase
        # -------------------------
        retarget_out: Dict[Tuple[int, int], MotionOutput] = {}
        for i, src in enumerate(self.domains):
            for j, dst in enumerate(self.domains):  
                retargetted_motion = dst.decode(reconstruction_out[i].latents, offset_features[j])
                retargetted_latents = dst.encode(retargetted_motion, offset_features[j])

                retargetted_rots = dst.denorm(retargetted_motion)

                retargetted_pos = ForwardKinematics.forward_batched(
                    quaternions=retargetted_rots[:, :-3]
                        .reshape(B, -1, 4, T)
                        .permute(0, 3, 1, 2),
                    offsets=offset_features[j][0].reshape(B, -1, 3),
                    root_pos=retargetted_rots[:, -3:],
                    topology=dst.topology,
                    world=True
                ) 

                retargetted_ee_vels = SkeletonUtils.get_ee_velocity(
                    retargetted_pos, 
                    topology=dst.topology
                )  / batch.heights[j][:, None, None, None]

                retarget_out[(i,j)] = MotionOutput(
                    motion=retargetted_motion,
                    latents=retargetted_latents,
                    rotations=retargetted_rots,
                    positions=retargetted_pos,
                    ee_vels=retargetted_ee_vels,
                )
        return reconstruction_out, retarget_out

    @torch.no_grad
    def translate(
        self, 
        source_motion: torch.Tensor,
        source_offset: torch.Tensor,
        target_offset: torch.Tensor,
        source_domain: int = 0,
        target_domain: int = 1,
    ) -> MotionOutput: 
        """
        Translate motion from source domain to target domain.
        
        Args:
            source_motion: Source motion tensor [C, T] (normalized)
            source_offset: Source skeleton offsets [J*3] 
            target_offset: Target skeleton offsets [J*3]
            source_domain: Source domain index (0 or 1)
            target_domain: Target domain index (0 or 1)
            denormalize: Whether to denormalize the output rotations
            return_positions: Whether to compute forward kinematics
        
        Returns:
            MotionOutput with translated motion
        """
        self.eval()
        
        # Encode offsets
        src_offset_features, tgt_offset_features = self._encode_offsets((source_offset, target_offset))
        
        # Encode source motion to latent space
        latents = self.domains[source_domain].encode(
            source_motion, 
            offset=src_offset_features
        )
        
        # Decode to target domain
        translated_motion = self.domains[target_domain].decode(
            latents, 
            offset=tgt_offset_features
        )
        
        translated_rots = self.domains[target_domain].denorm(translated_motion)
        
        return translated_rots

    @classmethod
    def load_for_inference(cls, checkpoint_path: str, device: str = 'cpu') -> 'SkeletalGAN':
        """ Load a trained model for inference. """        
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        model = cls(
            topologies=checkpoint['topologies'],
            normalization_stats=checkpoint['normalization_stats'],
            offset_encoder_params=checkpoint['config']['offset_encoder'],
            auto_encoder_params=checkpoint['config']['auto_encoder'],
            discriminator_params=checkpoint['config']['discriminator']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        return model
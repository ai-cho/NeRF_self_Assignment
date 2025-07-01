"""
Pytorch implementation of MLP used in NeRF (ECCV 2020).
"""

from typing import Tuple
from typeguard import typechecked

from jaxtyping import Float, jaxtyped
import torch
import torch.nn as nn


class NeRF(nn.Module):
    """
    A multi-layer perceptron (MLP) used for learning neural radiance fields.

    For architecture details, please refer to 'NeRF: Representing Scenes as
    Neural Radiance Fields for View Synthesis (ECCV 2020, Best paper honorable mention)'.

    Attributes:
        pos_dim (int): Dimensionality of coordinate vectors of sample points.
        view_dir_dim (int): Dimensionality of view direction vectors.
        feat_dim (int): Dimensionality of feature vector within forward propagation.
    """

    def __init__(
        self,
        pos_dim: int,
        view_dir_dim: int,
        feat_dim: int = 256,
    ) -> None:
        """
        Constructor of class 'NeRF'.
        """
        super().__init__()

        # TODO

        self.pos_dim = pos_dim
        self.view_dir_dim = view_dir_dim
        self.feat_dim = feat_dim

        self.mlp_1 = nn.Sequential(
            nn.Linear(pos_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU()
        )

        # before orange arrow
        self.mlp_2 = nn.Sequential(
            nn.Linear(feat_dim + pos_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU()
        )

        self.mlp_3 = nn.Sequential(
            nn.Linear(feat_dim, feat_dim)
        )

        self.mlp_4 = nn.Sequential(
            nn.Linear(feat_dim + view_dir_dim, 128),
            nn.ReLU()
        )
    
        self.sigma_layer = nn.Sequential(
            nn.Linear(feat_dim, 1),
            nn.ReLU()
        )

        self.color_layer = nn.Sequential(
            nn.Linear(128, 3),
            nn.Sigmoid()
        )

        # raise NotImplementedError("Task 1")

    @jaxtyped
    @typechecked
    def forward(
        self,
        pos: Float[torch.Tensor, "num_sample pos_dim"],
        view_dir: Float[torch.Tensor, "num_sample view_dir_dim"],
    ) -> Tuple[Float[torch.Tensor, "num_sample 1"], Float[torch.Tensor, "num_sample 3"]]:
        """
        Predicts color and density.

        Given sample point coordinates and view directions,
        predict the corresponding radiance (RGB) and density (sigma).

        Args:
            pos: The positional encodings of sample points coordinates on rays.
            view_dir: The positional encodings of ray directions.

        Returns:
            sigma: The density predictions evaluated at the given sample points.
            radiance: The radiance predictions evaluated at the given sample points.
        """

        # TODO

        z = torch.cat([self.mlp_1(pos), pos], dim=-1)
        z = self.mlp_2(z)
        sigma = self.sigma_layer(z)

        z = torch.cat([self.mlp_3(z), view_dir], dim=-1)
        color = self.color_layer(self.mlp_4(z))
        
        return sigma, color
        # raise NotImplementedError("Task 1")

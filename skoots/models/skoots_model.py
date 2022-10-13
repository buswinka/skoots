import torch
from torch import Tensor
import torch.nn as nn
from bism.models.unext import UNeXT_3D as unext
from bism.models.spatial_embedding import SpatialEmbedding
from typing import Optional, Dict, List, Tuple

from skoots.lib.vector_to_embedding import vector_to_embedding
from skoots.lib.embedding_to_prob import baked_embed_to_prob

from skoots.train.loss import tversky


class SkootsModel(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 vector_scale_factor: Optional[Tuple[float, float, float]] = (60., 60., 6.),
                 *,
                 backend: nn.Module = unext,
                 _model_kwargs: Optional[Dict[str, ...]] = None):
        """
        Generates a Skoots Model

        :param in_channels:
        :param backend:
        :param _model_kwargs:
        """
        super(SkootsModel, self).__init__()

        # Lets pass model kwargs for future flexibility...
        if _model_kwargs is None:
            _model_kwargs = {'in_channels', in_channels}
        model = backend(**_model_kwargs)

        self.scale = nn.Parameter(torch.tensor(vector_scale_factor), requires_grad=False)
        self.backend = SpatialEmbedding(model)  # returns a 5 dim tensor always

        self.loss_embed =

    def forward(self, x: Dict[str, Tensor]):
        if self.training:
            out = self.backend(x['image'])
            probability_map: Tensor = out[:, [-1], ...]
            vector: Tensor = out[:, 0:3:1, ...]
            predicted_skeleton: Tensor = out[:, [-2], ...]

            embedding: Tensor = vector_to_embedding(self.scale, vector)
            out: Tensor = baked_embed_to_prob(embedding, x['baked'], x['sigma'])

            _loss_embed = loss_embed(out, masks.gt(0).float())  # out = [B, 2/3, X, Y, Z?]
            _loss_prob = loss_prob(probability_map, masks.gt(0).float())
            _loss_skeleton = loss_skele(predicted_skeleton, skele_masks.gt(0).float())
            loss = _loss_embed + (1 * _loss_prob) + (1 * _loss_skeleton)








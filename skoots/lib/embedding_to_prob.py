from typing import Tuple, Dict, Optional, List
import torch
from torch import Tensor
from torch.autograd import Function
import torch.nn as nn
from typing import Tuple, Union

import triton
import triton.language as tl


@torch.jit.script
def baked_embed_to_prob(embedding: Tensor, baked: Tensor, sigma: Tensor, eps: float = 1e-16) -> Tensor:
    """
    N Dimmensional embedding to probability with a baked skeleton array

    :param embedding: 4/5D embedding tenosr
    :param baked:  a 4/5D baked skeleton tensor
    :param sigma:
    :return:
    """


    sigma = sigma + torch.tensor(eps, device=embedding.device)  # when sigma goes to zero, things tend to break

    # Common operation. Done outside of loop for speed.
    sigma = sigma.pow(2).mul(2).mul(-1)

    out = torch.exp((embedding - baked)
                    .pow(2)
                    .transpose(1,-1) # make this work for 2D and 3D by following pytorch broadcasting rules (channels last dim)
                    .div(sigma)
                    .transpose(1,-1)
                    .sum(dim=1, keepdim=True))

    return out



if __name__ == '__main__':
   pass

import torch
from torch import Tensor


@torch.jit.script
def baked_embed_to_prob(embedding: Tensor, baked: Tensor, sigma: Tensor, eps: float = 1e-16) -> Tensor:
    r"""
    N Dimensional embedding to probability with a baked skeleton array

    Calculates a gauyssian based on a euclidean distance from a spatial embedding and a baked skeleton pixel.

    :math: \frac{1}{2}

    .. math::
        (a + b)^2 = a^2 + 2ab + b^2 \\
        \Phi(x, p) = exp(\frac{(x - p)^2}{-2\sigma^2})

    :param embedding: 4/5D embedding tenosr
    :param baked:  a 4/5D baked skeleton tensor
    :param eps: small float for numerical stability
    :param sigma:
    :return:
    """

    sigma = sigma + torch.tensor(eps, device=embedding.device)  # when sigma goes to zero, things tend to break
    sigma = sigma.pow(2).mul(2).mul(-1)

    out = torch.exp((embedding - baked)
                    .pow(2)
                    .transpose(1, -1)  # work for 2D and 3D by following pytorch broadcasting rules (channels last dim)
                    .div(sigma)
                    .transpose(1, -1)
                    .sum(dim=1, keepdim=True))

    return out


if __name__ == '__main__':
    pass

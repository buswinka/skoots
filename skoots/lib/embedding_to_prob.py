import torch
from torch import Tensor


@torch.jit.script
def baked_embed_to_prob(
    embedding: Tensor, baked_skeletons: Tensor, sigma: Tensor, eps: float = 1e-16
) -> Tensor:
    r"""
    N Dimensional embedding to probability with a baked skeleton array

    Calculates a probability :math:`\phi` based on a euclidean distance between a spatial
    embedding :math:`E_i` and a baked skeleton pixel :math:`S_i`.

    .. math::
        \phi(E_i, S_i) =exp\left(\sum_{k \in [x,y,z]} \frac{(E_{ki} - S_{ki})^2}{-2\sigma^2_k} \right)

    In three spatial dimmensions, this expands to

    .. math::
        \phi(E_i, S_i) =exp\left(\frac{(E_{xi} - S_{xi})^2}{-2\sigma^2_x} + \frac{(E_{yi} - S_{yi})^2}{-2\sigma_y^2} + \frac{(E_{zi} - S_{zi})^2}{-2\sigma^2_z}\right)

    Shapes:
        - embedding: :math:`(B_{in}, 2, X_{in}, Y_{in})` or :math:`(B_{in}, 3, X_{in}, Y_{in}, Z_{in})`
        - baked_skeletons: :math:`(B_{in}, 2, X_in, Y_{in})` or :math:`(B_{in}, 3, X_{in}, Y_in, Z_{in})`
        - sigma: :math:`(2)` or :math:`(3)`

        - returns: :math:`(B_{in}, 1, X_{in}, Y_{in})` or :math:`(B_{in}, 1, X_{in}, Y_{in}, Z_{in})`

    :param embedding: embedding tensor
    :param baked_skeletons: a baked skeleton tensor
    :param sigma: Standard deviation of the gaussian. Larger values give higher probability further away.
    :param eps: small value for numerical stability
    :return: Probability matrix
    """

    sigma = sigma + eps  # when sigma goes to zero, things tend to break
    sigma = sigma.pow(2).mul(2).mul(-1)

    out = torch.exp(
        (embedding - baked_skeletons)
        .pow(2)
        .transpose(
            1, -1
        )  # work for 2D and 3D by following pytorch broadcasting rules (channels last dim)
        .div(sigma)
        .transpose(1, -1)
        .sum(dim=1, keepdim=True)
    )

    return out


if __name__ == "__main__":
    pass

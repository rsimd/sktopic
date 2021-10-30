from typing import Optional
import torch
from torch.distributions import constraints, Independent
from torch.distributions.normal import Normal
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import SoftmaxTransform, StickBreakingTransform
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.logistic_normal import LogisticNormal as StickBreakingLogisticNormal


__all__ = [
    "SoftmaxLogisticNormal",
    "StickBreakingLogisticNormal",
]


class SoftmaxLogisticNormal(TransformedDistribution):
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.simplex
    has_rsample = True

    def __init__(self, loc, scale, validate_args=None):
        base_dist = Normal(loc, scale)
        super().__init__(base_dist, 
                        SoftmaxTransform(), 
                        validate_args=validate_args)

    @staticmethod
    def from_alpha(alpha:torch.Tensor, validate_args=None):
        h_dim = alpha.squeeze().size(0)
        loc = (alpha.log().T - alpha.log().mean(dim=1) ).T
        scale = ( ( (1.0/alpha)*( 1 - (2.0/h_dim) ) ).T + ( 1.0/(h_dim*h_dim) )*torch.sum(1.0/alpha,1) ).T
        return SoftmaxLogisticNormal(loc,scale, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(SoftmaxLogisticNormal, _instance)
        return super(SoftmaxLogisticNormal).expand(batch_shape, _instance=new)

    @property
    def loc(self):
        return self.base_dist.base_dist.loc

    @property
    def scale(self):
        return self.base_dist.base_dist.scale

    @property
    def mean(self):
        return F.softmax(self.loc, dim=-1)

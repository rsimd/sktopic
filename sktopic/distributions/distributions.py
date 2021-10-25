from typing import Optional
import torch
from torch.distributions import constraints, Independent
from torch.distributions.normal import Normal
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import SoftmaxTransform, StickBreakingTransform
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.rnn import RNNCell
from torch.distributions.logistic_normal import LogisticNormal as StickBreakingLogisticNormal


__all__ = [
    "SoftmaxLogisticNormal",
    "StickBreakingLogisticNormal",
    "GaussianSoftmaxConstruction",
    "GaussianStickBreakingConstruction",
    "RecurrentStickBreakingConstruction"
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


class GaussianSoftmaxConstruction(nn.Sequential):
    def __init__(self, n_topics, device=None, dtype=None):
        super().__init__(
            nn.Linear(n_topics,n_topics,True,device,dtype),
            nn.Softmax(dim=1),
            )


class GaussianStickBreakingConstruction(nn.Module):
    def __init__(self, n_topics, device=None,dtype=None):
        super().__init__()
        self.linear = nn.Linear(n_topics,n_topics,True,device,dtype)
        self.transform = StickBreakingTransform()

    def forward(self,z):
        z = self.linear(z)
        return self.transform(z[:,:-1])


class RecurrentStickBreakingConstruction(nn.Module):
    def __init__(self, n_topics, limit_topic:Optional[int]=None,device=None,dtype=None):
        super().__init__()
        self.n_topics = n_topics
        if limit_topic is None:
            self.limit_topic = n_topics
        else:
            self.limit_topic = limit_topic
        self.device = device
        self.dtype = dtype
        self.rnncell = nn.RNNCell(self.n_topics,self.n_topics, device=device,dtype=dtype)
        self.sigmoid = nn.Sigmoid()
        self.transform = StickBreakingTransform()
        """
        zを十分に大きい次元数（self.limit_topic)生成して、self.transform(eta[:,self.n_topics-1])すれば無限次元NTMになりそう。
        """

    def forward(self,z):
        eta = []
        for i in range(self.n_topics-1):
            if i == 0:
                h0 = torch.rand(size=(1,self.n_topics),dtype=self.dtype).to(z.device) # (1,K)
                hi = self.rnncell(h0) # (1,K)
            else:
                hi= self.rnncell(hi,hi) # (1,K),(1,K)
            eta_i = self.sigmoid(torch.matmul(hi,z.T)) # (1,K) (M,K).T
            eta.append(eta_i)
        eta = torch.stack(eta).squeeze().T # (M,K-1)
        return self.transform(eta) # (M,K)


class TakgTopicGenerator(nn.Sequential):
    def __init__(self, n_topics, n_layers=1, activation=nn.Tanh, use_bias=True, device=None,dtype=None):
        seq = []
        for _ in range(n_layers):
            seq += [nn.Linear(n_topics,n_topics,use_bias,device,dtype),activation()]
        self.seq = nn.Sequential(*seq)

    def forward(self,x):
        h = self.seq(x)
        return h + x

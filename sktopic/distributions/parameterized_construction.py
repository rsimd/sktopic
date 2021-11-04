from typing import Optional
import torch
from torch.distributions import constraints, Independent
from torch.distributions.normal import Normal
from torch.distributions.transforms import StickBreakingTransform
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.rnn import RNNCell
from sparsemax import Sparsemax

__all__ = [
    "GaussianSoftmaxConstruction",
    "GaussianStickBreakingConstruction",
    "RecurrentStickBreakingConstruction"
]
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
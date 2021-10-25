from typing import (
    Sequence, Optional, Any,
    Union,
)
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.distributions import Normal

from ..ntm import (
    NTM,
    Compressor, 
    H2GaussParams, 
    Decoder,
    ELBO
)


class ClusterLoss(ELBO):
    def __init__(self, n_components, n_cluster,mu=0, lv=1):
        super().__init__(mu=mu, lv=lv)
        self.linear = nn.Linear(n_components,n_cluster)
        self.cce = nn.CrossEntropyLoss()
        
    def forward(self, lnpx:torch.Tensor, x:torch.Tensor, posterior, model=None,cluster_labels=None, **kwargs):
        AUX = super().forward(lnpx, x, posterior, model, **kwargs)
        logit = self.linear(kwargs["topic_proportion"])
        AUX["cce"] = self.cce(logit,cluster_labels)
        AUX["loss"] += AUX["cce"]
        return AUX

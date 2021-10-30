from typing import Sequence,Optional,Any
from collections import OrderedDict
import torch
import torch.nn as nn
from .base import (
    NTM, Compressor, H2GaussParams, Decoder)
from torch.distributions import (
    Normal,
    Independent,
    MultivariateNormal,
    )
from sktopic.distributions import (
    SoftmaxLogisticNormal,
    GaussianSoftmaxConstruction,
    GaussianStickBreakingConstruction,
    RecurrentStickBreakingConstruction,
    )
from torch.distributions.kl import kl_divergence

__all__ = [
    "NVDM",
    "NVLDA",
    "ProdLDA",
    "ETM",
    "DeltaNVDM",
    "GaussianSoftmaxModel",
    "GaussianStickBreakingModel",
    "RecurrentStickBreakingModel",
    ]


class NVDM(NTM):
    def __init__(self, dims:Sequence[int],embed_dim:Optional[int]=None, 
            activation_hidden:str="Softplus",
            dropout_rate_hidden:float=0.2, dropout_rate_theta:float=0.2, 
            device:Any=None, dtype:Any=None,n_sampling=1):
        super().__init__(dims,embed_dim,activation_hidden,
                        dropout_rate_hidden,dropout_rate_theta,
                        device,dtype,topic_model=False,n_sampling=n_sampling)

class NVLDA(NTM):
    def __init__(self, dims:Sequence[int],embed_dim:Optional[int]=None, 
            activation_hidden:str="Softplus",
            dropout_rate_hidden:float=0.2, dropout_rate_theta:float=0.2, 
            device:Any=None, dtype:Any=None, alpha:Optional[float]=None,n_sampling=1):
        super().__init__(dims,embed_dim,activation_hidden,
                        dropout_rate_hidden,dropout_rate_theta,
                        device,dtype,topic_model=True, n_sampling=n_sampling)
        self.alpha_pz = alpha if alpha is not None else 1/self.n_components

    def _build(self)->None:
        return super()._build(rt = SoftmaxLogisticNormal)

    def get_loss(self, x, lnpx, q_z):
        nll = -torch.sum(x * lnpx, dim=1).mean()
        p_z = self.rt.from_alpha(torch.ones_like(q_z.loc)*self.alpha_pz)
        kld = kl_divergence(q_z, p_z)
        if kld.ndim == 2:
            kld = kld.sum(1)
        kld = kld.mean()
        return dict(nll=nll,kld=kld, elbo=nll+kld)

class ProdLDA(NTM):
    def __init__(self, dims:Sequence[int],embed_dim:Optional[int]=None, 
            activation_hidden:str="Softplus",
            dropout_rate_hidden:float=0.2, dropout_rate_theta:float=0.2, 
            device:Any=None, dtype:Any=None,alpha:Optional[float]=None,n_sampling=1):
        super().__init__(dims,embed_dim,activation_hidden,
                        dropout_rate_hidden,dropout_rate_theta,
                        device,dtype,topic_model=False,n_sampling=n_sampling)
        self.alpha_pz = alpha if alpha is not None else 1/self.n_components

    def _build(self)->None:
        return super()._build(rt = SoftmaxLogisticNormal)

    def get_loss(self, x, lnpx, q_z):
        nll = - torch.sum(x*lnpx, dim=1).mean()
        p_z = self.rt.from_alpha(torch.ones_like(q_z.loc)*self.alpha_pz)
        kld = kl_divergence(q_z, p_z)
        if kld.ndim == 2:
            kld = kld.sum(1)
        kld = kld.mean()
        return dict(nll=nll,kld=kld, elbo=nll+kld)

class DeltaNVDM(NTM):
    def __init__(self, dims:Sequence[int],embed_dim:Optional[int]=None, 
            activation_hidden:str="Softplus",
            dropout_rate_hidden:float=0.2, dropout_rate_theta:float=0.2, 
            device:Any=None, dtype:Any=None,n_sampling=1):
        super().__init__(dims,embed_dim,activation_hidden,
                        dropout_rate_hidden,dropout_rate_theta,
                        device,dtype,topic_model=False,n_sampling=n_sampling)

    def _build(self)->None:
        return super()._build(map_theta = nn.Softmax(dim=-1))


class ETM(NTM):
    def __init__(self, dims:Sequence[int],embed_dim:int, 
            activation_hidden:str="Softplus",
            dropout_rate_hidden:float=0.2, dropout_rate_theta:float=0.2, 
            device:Any=None, dtype:Any=None,n_sampling=1):
        super().__init__(dims,embed_dim,activation_hidden,
                        dropout_rate_hidden,dropout_rate_theta,
                        device,dtype,topic_model=True,n_sampling=n_sampling)

    def _build(self)->None:
        super()._build(map_theta = nn.Softmax(dim=-1))


class GaussianSoftmaxModel(NTM):
    def _build(self)->None:
        super()._build(
            map_theta = GaussianSoftmaxConstruction(
            self.n_components,self.device,self.dtype
            )
        )


class GaussianStickBreakingModel(NTM):
    def _build(self)->None:
        super()._build(
            map_theta = GaussianStickBreakingConstruction(
            self.n_components,self.device,self.dtype
            )
        )


class RecurrentStickBreakingModel(NTM):
    def _build(self) -> None:
        super()._build(
            map_theta = RecurrentStickBreakingConstruction(
            self.n_components,device=self.device,dtype=self.dtype
            )
        )
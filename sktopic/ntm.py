from operator import pos
from typing import Callable, Optional, Any, OrderedDict, Sequence, Tuple, Union, TypeVar
from more_itertools import windowed
from numpy.core.fromnumeric import compress
import torch
from torch.distributions.kl import kl_divergence
from torch.distributions.normal import Normal 
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

from .distributions import SoftmaxLogisticNormal
from collections import OrderedDict
from .utils import normalizel2

__all__ = [
    "Compressor",
    "H2GaussParams",
    "Decoder",
    "NTM"
]


class Compressor(nn.Sequential):
    def __init__(self, dims:Sequence[int], activation:Callable=nn.Softplus, 
            dropout_rate:float=0.2, device=None, dtype=None)->None:
        assert len(dims)>=2, "dims must have len(dims)=>2."
        layers = []
        for in_dim, out_dim in windowed(dims,2):
            layers += [nn.Linear(in_dim,out_dim,True,device,dtype), activation()]
        super().__init__(
            *layers,
            #nn.BatchNorm1d(dims[-1]),
            nn.Dropout(p=dropout_rate),
        )


class H2GaussParams(nn.Module):
    def __init__(self, dims:Sequence[int],device=None,dtype=None, output_std=True):
        super().__init__()
        assert len(dims)==2, "dims must have len(dims)=>2."
        self.n_components = dims[1]
        self.mu_q_theta = nn.Sequential(
            nn.Linear(dims[0],dims[1],True,device,dtype),
            #nn.BatchNorm1d(dims[1]),
            )
        self.lv_q_theta = nn.Sequential(
            nn.Linear(dims[0],dims[1],True,device,dtype),
            #nn.BatchNorm1d(dims[1]),
            )
        self.output_std=output_std
    
    def forward(self,h:torch.Tensor)->Tuple[torch.Tensor,torch.Tensor]:
        loc = self.mu_q_theta(h)
        lv = self.lv_q_theta(h)
        #loc,lv = torch.split(h, split_size_or_sections=self.n_components, dim=-1)
        scale = (0.5*lv).exp()
        return loc,scale


class Decoder(nn.Module):
    def __init__(self, dims: Sequence[int], embed_dim:Optional[int]=None,
            dropout_rate:float=0.2, use_bias: bool=True, 
            device=None, dtype=None, topic_model:bool=False):
        super().__init__()
        assert len(dims)==2, "dims must have len(dims)==2"
        
        self.n_components = dims[0]
        self.n_features = dims[-1]
        self.embed_dim = embed_dim
        self.dtype = dtype
        self.device = device
        self.dropout = nn.Dropout(p=dropout_rate)
        self.use_bias = use_bias
        self.topic_model = topic_model
        self.eps = 1e-7

        self._set_beta()
        if self.topic_model:
            self.softmax = nn.Softmax(dim=-1)
        else:
            self.log_softmax = nn.LogSoftmax(dim=-1)
            if self.use_bias:
                self.bias = nn.Parameter(
                    torch.zeros(size=(self.n_features,), device=device, dtype=dtype), 
                    requires_grad=True, )

    def forward(self,theta:torch.Tensor, )->torch.Tensor:
        theta = self.dropout(theta)
        beta = self.get_beta()
        if self.topic_model:
            phi = self.softmax(beta)
            px = torch.matmul(theta, phi)
            return torch.log(px + self.eps)
        else:
            logit = torch.matmul(theta, beta)
            if self.use_bias:
                logit += self.bias
            return self.log_softmax(logit)

    def _set_beta(self)->None:
        if isinstance(self.embed_dim, int):
            self.topic_embeddings = nn.Embedding(self.n_components, self.embed_dim,device=self.device, dtype=self.dtype)
            self.word_embeddings = nn.Embedding(self.n_features, self.embed_dim,device=self.device, dtype=self.dtype)
        else:
            self.beta = nn.Embedding(self.n_features, self.n_components,device=self.device, dtype=self.dtype)

    def get_beta(self)->torch.Tensor:
        if self.embed_dim:
            t = self.topic_embeddings.weight
            e = self.word_embeddings.weight
            # normalize
            t = F.normalize(t)
            e = F.normalize(e)
            return torch.matmul(t, e.T)
        return self.beta.weight.T


class NTM(nn.Module):
    def __init__(self, dims:Sequence[int],embed_dim:Optional[int]=None, 
            activation_hidden:str="Softplus",
            dropout_rate_hidden:float=0.2, dropout_rate_theta:float=0.2, 
            device:Any=None, dtype:Any=None, topic_model:bool=False, n_sampling:int=1):
        super().__init__()
        assert len(dims)>=3, "dims must have len(dims)>=3." 
        self.vocab_size = dims[0]
        self.n_hiddens = dims[1:-1]
        self.n_components = dims[-1]
        self.dims = dims
        self.embed_dim = embed_dim
        self.activation_hidden = activation_hidden
        self.dropout_rate_hidden = dropout_rate_hidden
        self.dropout_rate_theta = dropout_rate_theta
        self.device = device
        self.dtype = dtype
        self.topic_model = topic_model
        self.n_sampling = n_sampling
        self._build()

    def _build(self,compressor=None,
                h2params=None, map_theta=nn.Identity(), 
                decoder=None,rt=Normal)->None:
        if compressor is None:
            compressor = Compressor(
                    self.dims[:-1],
                    activation=eval(f"nn.{self.activation_hidden}"), 
                    dropout_rate=self.dropout_rate_hidden, 
                    device=self.device, 
                    dtype=self.dtype,
                )
        if h2params is None:
            h2params = H2GaussParams(
                    self.dims[-2:],
                    device=self.device, 
                    dtype=self.dtype,
                )
        if decoder is None:
            decoder=Decoder(
                [self.n_components, self.vocab_size,],
                embed_dim=self.embed_dim,
                dropout_rate=self.dropout_rate_theta,
                use_bias=True,
                device=self.device,
                dtype=self.dtype,
                topic_model=self.topic_model
                )
        
        self.rt = rt
        self.encoder = nn.Sequential(compressor, h2params)
        self.decoder = nn.ModuleDict(
            OrderedDict(map_theta=map_theta,decoder=decoder)
            )
    
    def forward(self,x:torch.Tensor, deterministic:Optional[bool]=None)->Tuple[torch.Tensor,torch.Tensor]:
        params = self.encoder(x)
        posterior = self.rt(*params)
        if deterministic is None:
            deterministic = not self.training

        if not deterministic:
            z = posterior.rsample().to(x.device)
            # z = torch.zeros_like(posterior.loc)
            # for _ in range(self.n_sampling):
            #     z += posterior.rsample().to(x.device)
            # z /= self.n_sampling
        else:
            z = posterior.mean.to(x.device)

        θ = self.decoder["map_theta"](z)
        lnpx = self.decoder["decoder"](θ)

        return dict(
            topic_proportion=θ,
            posterior = posterior,
            lnpx=lnpx,
            topic_dist = self.get_beta(),
        )

    def transform(self, x:torch.Tensor, deterministic:Optional[bool]=None)->torch.Tensor:
        params = self.encoder(x)
        posterior = self.rt(*params)
        
        if deterministic is None:
            deterministic = not self.training

        if not deterministic:
            z = posterior.rsample().to(x.device)
        else:
            z = posterior.mean.to(x.device)

        θ = self.decoder["map_theta"](z)
        return θ

    def get_beta(self):
        return self.decoder["decoder"].get_beta()

    def get_loss(self, y_pred, input, posterior, **kwargs):
        nll = - torch.sum(input*y_pred, dim=1).mean()

        p_z = self.rt(
            torch.zeros((1,self.n_components)).to(input.device),
            torch.ones((1,self.n_components)).to(input.device)
        )
        kld = kl_divergence(p_z, posterior)
        if kld.ndim == 2:
            kld = kld.mean(1)
        kld = kld.sum()
        return dict(nll=nll,kld=kld, elbo=nll+kld)


# TorchDistribution= TypeVar(
#     name="TorchDistribution", 
#     bound=torch.distributions.Distribution)

class ELBO(nn.Module):
    def __init__(self, mu=0,lv=1):
        super().__init__()
        self.mu = mu
        self.lv = lv
        self._initilized = False

    def forward(self, lnpx:torch.Tensor, x:torch.Tensor, posterior:Any, model=None, **kwargs):
        AUX = {}
        if not self._initilized:
            self.n_components = posterior.loc.size(1)
            self.dist = posterior.__class__
            self._initilized = True

        nll = - torch.sum(x*lnpx, dim=1).mean()        
        _device = lnpx.device
        p_z = self.dist(
            torch.ones((1,self.n_components)).to(_device) * self.mu,
            torch.ones((1,self.n_components)).to(_device) * self.lv,
        )
        kld = kl_divergence(p_z, posterior)
        if kld.ndim == 2:
            kld = kld.mean(1)
        kld = kld.sum()
        
        AUX["elbo"]=nll+kld
        AUX["nll"]=nll
        AUX["kld"]=kld
        
        AUX["loss"]=AUX["elbo"]
        
        AUX.update(self.culc_metrics(lnpx,x,posterior,model=model, **kwargs))
        return AUX

    def culc_metrics(self, lnpx, x, posterior, model=None, **kwargs):
        AUX = {}
        #AUX["ppl"] = (-(lnpx * x).sum() / x.sum()).exp()

        return AUX


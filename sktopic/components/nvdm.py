from typing import Callable, ForwardRef, Optional, Any, OrderedDict, Sequence, Tuple, Union, TypeVar
from more_itertools import windowed
import torch
#from ..distributions.rt import kl_divergence 
from torch.distributions.kl import kl_divergence
#from ..distributions.rt import Normal 
from torch.distributions.normal import Normal
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from ..components.regularizers import RegularizerUsingTopicEmbeddingsDiversity

__all__ = [
    "Compressor",
    "H2GaussParams",
    "Decoder",
    "NTM",
    "ELBO",
]


class Compressor(nn.Sequential):
    def __init__(self, dims:Sequence[int], activation:Callable=nn.Softplus, 
            dropout_rate:float=0.2, device=None, dtype=None, input_normalize=False)->None:
        assert len(dims)>=2, "dims must have len(dims)=>2."
        layers = []
        for in_dim, out_dim in windowed(dims,2):
            layers += [
                nn.Linear(in_dim,out_dim,True,device,dtype), 
                nn.BatchNorm1d(out_dim), 
                nn.AlphaDropout(p=dropout_rate),
                activation(),
                ]
        super().__init__(
            *layers,
        )
        self.input_normalize = input_normalize

    def forward(self,x):
        if self.input_normalize:
            x = F.normalize(x,p=2,dim=1)
        return super().forward(x)


class H2GaussParams(nn.Module):
    def __init__(self, dims:Sequence[int],device=None,dtype=None, output_std=True):
        super().__init__()
        assert len(dims)==2, "dims must have len(dims)=>2."
        self.n_components = dims[1]
        self.mu_q_theta = nn.Sequential(
            nn.Linear(dims[0],dims[1],True,device,dtype),
            nn.Softplus(),
            nn.BatchNorm1d(dims[1]),
            )
        self.lv_q_theta = nn.Sequential(
            nn.Linear(dims[0],dims[1],True,device,dtype),
            nn.Softplus(),
            nn.BatchNorm1d(dims[1]),
            #nn.Tanh()
            )
        self.output_std=output_std
    
    def forward(self,h:torch.Tensor)->Tuple[torch.Tensor,torch.Tensor]:
        loc = self.mu_q_theta(h)
        lv = self.lv_q_theta(h)
        #loc,lv = torch.split(h, split_size_or_sections=self.n_components, dim=-1)
        #return loc,lv
        #lv = F.tanh(lv)
        scale = (0.5*lv).exp()
        #scale = torch.clamp(scale, min=1e-6,max=1.0) # std 1e-7, 88.7
        return loc,scale


class Decoder(nn.Module):
    def __init__(self, dims: Sequence[int], embed_dim:Optional[int]=None,
            dropout_rate:float=0.0, use_bias: bool=True, 
            device=None, dtype=None, topic_model:bool=False):
        super().__init__()
        assert len(dims)==2, "dims must have len(dims)==2"
        
        self.n_components = dims[0]
        self.n_features = dims[-1]
        self.embed_dim = embed_dim
        self.dtype = dtype
        self.device = device
        self.dropout = nn.AlphaDropout(p=dropout_rate)
        self.use_bias = use_bias
        self.topic_model = topic_model
        self.eps = 1e-6

        self._set_beta()
        if self.topic_model:
            self.softmax = nn.Softmax(dim=-1)
        else:
            self.log_softmax = nn.LogSoftmax(dim=-1)
        if self.use_bias:
            self.bias = nn.Parameter(
                torch.zeros(size=(self.n_features,), device=device, dtype=dtype), 
                requires_grad=True, )

    def _set_beta(self)->None:
        if isinstance(self.embed_dim, int):
            self.topic_embeddings = nn.Embedding(self.n_components, self.embed_dim,device=self.device, dtype=self.dtype,max_norm=1)
            #self.topic_embeddings.weight = F.normalize(self.topic_embeddings.weight,p=2,dim=1)
            self.word_embeddings = nn.Embedding(self.n_features, self.embed_dim,device=self.device, dtype=self.dtype,max_norm=1)
            #self.word_embeddings.weight = F.normalize(self.word_embeddings.weight,p=2,dim=1)
            
            #self.topic_bias = nn.Parameter(torch.zeros(size=(self.n_components,1), device=self.device, dtype=self.dtype), requires_grad=True, )
            #self.word_bias = nn.Parameter(torch.zeros(size=(self.n_features,1), device=self.device, dtype=self.dtype), requires_grad=True, )
        else:
            self.beta = nn.Embedding(self.n_features, self.n_components,device=self.device, dtype=self.dtype,)

    def get_beta(self)->torch.Tensor:
        if self.embed_dim:
            t = self.topic_embeddings.weight # + self.topic_bias
            e = self.word_embeddings.weight # + self.word_bias
            return F.linear(
                t,#F.normalize(t,p=2,dim=1),
                e #F.normalize(e,p=2,dim=1).T,
            )
            
        return self.beta.weight.T

    def forward(self,theta:torch.Tensor)->torch.Tensor:
        theta = self.dropout(theta)
        
        #assert theta.sum() == theta.size(0),theta.sum()
        beta = self.get_beta()
        if self.topic_model:
            if self.use_bias:
                beta = beta+self.bias
            phi = self.softmax(beta)
            px = torch.matmul(theta, phi)
            log_softmax= torch.log(px  +1e-7)

            assert not theta.isnan().any(), "■■■■theta has NaN!■■■■"
            assert not phi.isnan().any(), "■■■■phi has NaN!■■■■"
            assert not px.isnan().any(), "■■■■px has NaN!■■■■"
            assert not log_softmax.isnan().any(), "■■■■Forward Prop has NaN!■■■■"
            
            return log_softmax
        else:
            logit = torch.matmul(theta, beta)
            if self.use_bias:
                logit =logit+self.bias #+ self.eps
            return self.log_softmax(logit)

class NTM(nn.Module):
    def __init__(self, dims:Sequence[int],embed_dim:Optional[int]=None, 
            activation_hidden:str="Tanh",
            dropout_rate_hidden:float=0.2, dropout_rate_theta:float=0.0, 
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
    
    def forward(self,x:torch.Tensor, deterministic:Optional[bool]=None)->dict[str,Any]:
        if deterministic is None:
            deterministic = not self.training
        
        params = self.encoder(x)
        posterior = self.rt(*params)
        if deterministic:
            z = posterior.mean.to(x.device)
            θ = self.decoder["map_theta"](z)
            lnpx = self.decoder["decoder"](θ)
            return dict(
                topic_proportion=θ,
                posterior = posterior,
                lnpx=lnpx,
                topic_dist = self.get_beta(),
            )
        # not deterministic
        for n in range(self.n_sampling):
            z = posterior.rsample().to(x.device)
            θ = self.decoder["map_theta"](z)
            assert not torch.isnan(θ).sum(), "θ has NaN."
            lnpx = self.decoder["decoder"](θ)
            assert not torch.isnan(lnpx).sum(), "lnpx has NaN."
            if n==0:
                dump = lnpx
            else:
                dump = lnpx + dump
        return dict(
            topic_proportion=θ,
            posterior = posterior,
            lnpx= dump/ self.n_sampling,
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
        

class ELBO(nn.Module):
    def __init__(self, 
    mu:float=0.0,lv:float=1.0, 
    arccos_lambda:float=5.0, 
    l1_lambda:float=0.001,
    l2_lambda:float=0.001,
    ):
        super().__init__()
        self.mu = mu
        self.lv = lv
        self._initilized = False
        self.regularizer = RegularizerUsingTopicEmbeddingsDiversity()
        self.arccos_lambda = arccos_lambda
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

    def forward(self, lnpx:torch.Tensor, x:torch.Tensor, posterior:Any, model=None, **kwargs):
        AUX = {}
        if not self._initilized:
            self.n_components = posterior.loc.size(1)
            self.dist = posterior.__class__
            self._initilized = True

        nll = -(x*lnpx).sum(1).mean()     
        _device = lnpx.device
        p_z = self.dist(
            torch.ones((1,self.n_components)).to(_device) * self.mu,
            torch.ones((1,self.n_components)).to(_device) * self.lv,
        )
        kld = kl_divergence(p_z, posterior)
        if kld.ndim == 2:
            kld = kld.mean(1)
        kld = kld.mean()
        AUX["elbo"]=nll+kld
        AUX["nll"]=nll
        AUX["kld"]=kld
        AUX["loss"]=AUX["elbo"]
        
        norm_reg = torch.tensor(0.).to(_device)
        for param in model.encoder.parameters():
            if param.requires_grad:
                norm_reg += torch.norm(param,p=1)
        AUX["l1norm_reg"]=norm_reg
        AUX["loss"] += self.l1_lambda * norm_reg

        norm_reg = torch.tensor(0.).to(_device)
        for param in model.decoder.parameters():
            if param.requires_grad:
                norm_reg += torch.norm(param,p=2)
        AUX["l2norm_reg"]=norm_reg
        AUX["loss"] += self.l2_lambda * norm_reg

        if model.embed_dim is not None:
            arccos = self.regularizer(model.decoder["decoder"].topic_embeddings.weight) #(K,L)
            AUX["topic_arccos"] = arccos
            AUX["loss"] = AUX["loss"] + self.arccos_lambda * arccos
        AUX.update(self.culc_metrics(lnpx,x,posterior,model=model, **kwargs))
        assert not torch.isnan(AUX["loss"]).sum(), "Loss has NaN."
        return AUX

    def culc_metrics(self, lnpx, x, posterior, model=None, **kwargs):
        AUX = {}
        #AUX["ppl"] = (-(lnpx * x).sum() / x.sum()).exp()
        return AUX
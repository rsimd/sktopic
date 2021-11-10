"""
original: https://github.com/zll17/Neural_Topic_Models/blob/master/models/wae.py#L89
"""
from typing import Optional,Any
from numpy.core.fromnumeric import size
import torch 
import torch.nn as nn
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture


__all__ = [
    "mmd_loss_diffusion",
    "mmd_loss_tv",
    "prior_sample",
    "MMD",
    "BoWCrossEntropy",
    ]


def diffusion_kernel(a:torch.Tensor, tmpt:float)->torch.Tensor:
    return -torch.acos(a).pow(2).exp() / tmpt

def mmd_loss_diffusion(pred_theta:torch.Tensor, target_theta:torch.Tensor,t:float=0.1, eps:float=1e-6)->torch.Tensor:
    """computes the mmd loss with information diffusion kernel

    Parameters
    ----------
    pred_theta : torch.Tensor
        Topic proportion vectors from neuralnet. (batch_size, num_topics)
    target_theta : torch.Tensor
        Topic proportion vectors from prior. (batch_size, num_topics)
    t : float, optional
        temp value, by default 0.1
    eps : float, optional
        Smoothing factor, by default 1e-6

    Returns
    -------
    torch.Tensor
        mmd loss value, it is scalar.
    """

    M,K = pred_theta.size()
    device = pred_theta.device
    qx = torch.clamp(pred_theta, eps, 1) **0.5
    qy = torch.clamp(target_theta, eps, 1) **0.5
    xx = qx @ qx.T
    yy = qy @ qy.T
    xy = qx @ qy.T

    off_diag = 1 - torch.eye(K).to(device)
    k_xx = diffusion_kernel(torch.clamp(xx,0,1-eps),t)
    k_yy = diffusion_kernel(torch.clamp(yy,0,1-eps),t)
    k_xy = diffusion_kernel(torch.clamp(xy,0,1-eps),t)
    sum_xx = (k_xx * off_diag).sum() / (M * (M-1))
    sum_yy = (k_yy * off_diag).sum() / (M * (M-1))
    sum_xy = 2* k_xy.sum() / M**2
    return sum_xx + sum_yy - sum_xy

def mmd_loss_tv(pred_theta:torch.Tensor,target_theta:torch.Tensor)->torch.Tensor:
    """computes the mmd loss with tv kernel(?)

    Parameters
    ----------
    pred_theta : torch.Tensor
        Topic proportion vectors from neuralnet. (batch_size, num_topics)
    target_theta : torch.Tensor
        Topic proportion vectors from prior. (batch_size, num_topics)
    
    Returns
    -------
    torch.Tensor
        mmd loss value, it is scalar.
    """
    M,K = pred_theta.size()
    device = pred_theta.device
    sum_xx = torch.zeros(1).to(device)
    for i in range(M):
        for j in range(i+1, M):
            sum_xx = sum_xx + torch.norm(pred_theta[i]-pred_theta[j], p=1).to(device)
    sum_xx = sum_xx / (M * (M-1))

    sum_yy = torch.zeros(1).to(device)
    for i in range(target_theta.shape[0]):
        for j in range(i+1, target_theta.shape[0]):
            sum_yy = sum_yy + torch.norm(target_theta[i]-target_theta[j], p=1).to(device)
    sum_yy = sum_yy / (target_theta.shape[0] * (target_theta.shape[0]-1))

    sum_xy = torch.zeros(1).to(device)
    for i in range(M):
        for j in range(target_theta.shape[0]):
            sum_xy = sum_xy + torch.norm(pred_theta[i]-target_theta[j], p=1).to(device)
    sum_yy = sum_yy / (M * target_theta.shape[0])
    return sum_xx + sum_yy - sum_xy

def prior_sample(
    shape:tuple[int,int], 
    dirichlet_alpha:float=0.1, 
    encoded:Optional[torch.Tensor]=None, 
    dtype:Any=torch.float32, 
    dist:str='dirichlet',
    )->torch.Tensor:
    """sample from prior distribution

    Parameters
    ----------
    shape : Tuple[int,int]
        (batch_size, n_topics)
    dirichlet_alpha : float, optional
        dirichlet prior's hyper-parameter alpha, by default 0.1
    encoded : Optional[torch.Tensor], optional
        encoded vectors from encoder networks, by default None
    dtype : Any, optional
        data type on torch, by default torch.float32
    dist : str, optional
        name of prior distribution, by default 'dirichlet'

    Returns
    -------
    torch.Tensor
        [description]
    """
    M,K = shape
    if dist == 'dirichlet':
        concentration= torch.ones(shape,dtype=dtype)*dirichlet_alpha
        z_true = torch.distributions.Dirichlet(concentration=concentration)
        return z_true
    
    elif dist == 'gaussian':
        logits = torch.randn(shape,dtype=dtype)
        z_true = F.softmax(logits, dim=1)
        return z_true
    
    elif dist == 'gmm_std':
        odes = torch.eye(K,dtype=dtype)*20
        ides = torch.randint(low=0,high=K, size=M)
        mus = odes[ides]
        sigmas = torch.ones(shape)*0.2*20
        z_true = torch.randn(mus, sigmas)
        z_true = F.softmax(z_true, dim=1)
        return z_true
    
    elif dist=='gmm_ctm' and encoded!=None:
        with torch.no_grad():
            gmm = GaussianMixture(n_components=K,covariance_type='full',max_iter=200)
            gmm.fit(encoded.detach().cpu().numpy())
            #hid_vecs = torch.from_numpy(hid_vecs).to(self.device)
            gmm_spls, _spl_lbls = gmm.sample(n_samples=len(encoded))
            theta_prior = torch.from_numpy(gmm_spls).float()
            theta_prior = F.softmax(theta_prior,dim=1)
            return theta_prior
    else:
        return prior_sample(shape,dist='dirichlet')


class MMD(nn.Module):
    def __init__(self, prior_name:str="dirichlet", kernel="diffusion", t:float=0.1, eps:float=1e-6, dirichlet_alpha:float=0.1) -> None:
        super().__init__()
        self.prior_name = prior_name
        self.t = t 
        self.eps = eps
        self.dirichlet_alpha = dirichlet_alpha
        self.kernel = kernel

    def forward(self, pred_theta:torch.Tensor)->torch.Tensor:
        shape = pred_theta.size()
        target_theta = prior_sample(shape,dirichlet_alpha=self.dirichlet_alpha, dtype=pred_theta.dtype, dist=self.prior_name)
        
        if self.kernel == "diffusion":
            loss = mmd_loss_diffusion(pred_theta, target_theta,self.t,self.eps)
            return loss
        loss = mmd_loss_tv(pred_theta,target_theta)
        return loss 


class BoWCrossEntropy(nn.Module):
    def __init__(self, pred_mode="logsoftmax") -> None:
        super().__init__()
        self.pred_mode = pred_mode

    def forward(self, pred:torch.Tensor, target:torch.Tensor)->torch.Tensor:
        if self.pred_mode == "logsoftmax":
            return -(pred * target).sum(-1).mean()
        if self.pred_mode == "softmax":
            return -(pred.log() * target).sum(-1).mean()
        if self.pred_mode == "logit":
            return -(F.log_softmax(pred,-1)* target).sum(-1).mean()

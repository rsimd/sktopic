"""
original: https://github.com/zll17/Neural_Topic_Models/blob/master/models/wae.py#L89
"""
from typing import Optional,Any
from numpy.core.fromnumeric import size
import torch 
import torch.nn as nn
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture


__all__ = ["mmd_loss_diffusion","mmd_loss_tv","prior_sample"]


def diffusion_kernel(a:torch.Tensor, tmpt:float)->torch.Tensor:
    return -torch.acos(a).pow(2).exp() / tmpt

def mmd_loss_diffusion(x:torch.Tensor,y:torch.Tensor,t:float=0.1, eps:float=1e-6)->torch.Tensor:
    """computes the mmd loss with information diffusion kernel

    Parameters
    ----------
    x : torch.Tensor
        Topic proportion vectors from neuralnet. (batch_size, num_topics)
    y : torch.Tensor
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

    M,K = x.size()
    device = x.device
    qx = torch.clamp(x, eps, 1) **0.5
    qy = torch.clamp(y, eps, 1) **0.5
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

def mmd_loss_tv(x:torch.Tensor,y:torch.Tensor)->torch.Tensor:
    """computes the mmd loss with tv kernel(?)

    Parameters
    ----------
    x : torch.Tensor
        Topic proportion vectors from neuralnet. (batch_size, num_topics)
    y : torch.Tensor
        Topic proportion vectors from prior. (batch_size, num_topics)
    
    Returns
    -------
    torch.Tensor
        mmd loss value, it is scalar.
    """
    M,K = x.size()
    device = x.device
    sum_xx = torch.zeros(1).to(device)
    for i in range(M):
        for j in range(i+1, M):
            sum_xx = sum_xx + torch.norm(x[i]-x[j], p=1).to(device)
    sum_xx = sum_xx / (M * (M-1))

    sum_yy = torch.zeros(1).to(device)
    for i in range(y.shape[0]):
        for j in range(i+1, y.shape[0]):
            sum_yy = sum_yy + torch.norm(y[i]-y[j], p=1).to(device)
    sum_yy = sum_yy / (y.shape[0] * (y.shape[0]-1))

    sum_xy = torch.zeros(1).to(device)
    for i in range(M):
        for j in range(y.shape[0]):
            sum_xy = sum_xy + torch.norm(x[i]-y[j], p=1).to(device)
    sum_yy = sum_yy / (M * y.shape[0])
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

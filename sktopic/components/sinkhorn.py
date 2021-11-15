from typing import Optional,Any
import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
    
def sinkhorn(beta:torch.Tensor,theta:torch.Tensor, bow:torch.Tensor, alpha:int=20, max_iter:int=5000, threshold:float=.5e-2,check_step:int=20)->torch.Tensor:
    """Neural Sinkhorn algorithm

    Parameters
    ----------
    beta : torch.Tensor
        Topic-Word Distribution (n_components, vocab_size), from normalize(beta,dim=1) or softmax(beta,dim=1)
    theta : torch.Tensor
        Topic Proportion vectors (mibibatch_size,n_components)
    bow : torch.Tensor
        BoW matrix (minibatch_size, vocab_size)
    alpha : int, optional
        constant value, by default 20
    max_iter : int, optional
        max iteration, by default 5000
    threshold : float, optional
        Threshold for stop iteration, by default .5e-2
    check_step : int, optional
        step of checking threshold, by default 20

    Returns
    -------
    torch.Tensor
        sinkhorn_divergences
    """
    M = 1 - beta # K,V
    theta_t = theta.T # M,K ->K,M
    #bow_t = (bow / bow.sum(1,keepdim=True)).T #M,V->V,M
    bow_t = F.softmax(bow, dim=1).T
    device = beta.device

    n_topics = theta.size(1)
    vocab_size = bow.size(1)

    u = (torch.ones_like(theta_t) / n_topics).to(device) # Psi_1
    v = (torch.ones_like(bow_t) / vocab_size).to(device) # Psi_2

    K = torch.exp(-M * alpha)

    for i in range(1, max_iter+1):
        # K(K,V), bow_t(V,M), u(K,M)
        _v = bow_t / (K.T @ u) # Psi_2  # diag(v).T K diag(u)1n = b
        u = theta_t / (K @ _v) # Psi_1  # diag(u) K diag(v)1_m = a

        if i % check_step == 0:
            v = bow_t   / (K.T @ u) # Psi_2 
            u = theta_t / (K   @ v) # Psi_1 
            bb = v * (K.T @ u) # (V,M) *(K,V)@(K,M)
            err = (bb - bow_t).abs().sum(dim=0).norm(p=float('inf'))
            #print(i,">>>", float(v.sum()),float(u.sum()),float(bb.sum()), float(err),)
            if err <= threshold:
                #print("break",err)
                break
    """
    u(K,M) H(K,V), M(K,V), v(V,M) -> (K,M).sum(0) -> M
    y=K*M, (K,V)
    y=y@v, (K,M)
    y=u*y, (K,M)
    y.sum(0), (M,)
    """
    #print(err)
    return torch.mul(u, K*M @ v).sum(dim=0)

class Sinkhorn(nn.Module):
    def __init__(self,alpha:int=20, max_iter:int=5000, threshold:float=.5e-2,check_step:int=20) -> None:
        super().__init__()
        self.alpha = alpha
        self.max_iter = max_iter 
        self.threshold = threshold
        self.check_step = check_step

    def forward(self, beta:torch.Tensor,theta:torch.Tensor, bow:torch.Tensor, model:Optional[Any]=None, **kwargs):
        divergence = sinkhorn(beta,theta,bow,self.alpha,self.max_iter,self.threshold,self.check_step)
        return divergence.mean()


'''
def sinkhorn_torch(unnormalized_beta:torch.Tensor,theta:torch.Tensor, bow:torch.Tensor, alpha:int=20, max_iter:int=5000, threshold:float=.5e-2)->torch.Tensor:
    """Neural Sinkhorn algorithm

    Parameters
    ----------
    unnormalized_beta : torch.Tensor
        Topic-Word Distribution (n_components, vocab_size)
    theta : torch.Tensor
        Topic Proportion vectors (mibibatch_size,n_components)
    bow : torch.Tensor
        BoW matrix (minibatch_size, vocab_size)
    alpha : int, optional
        constant value, by default 20
    max_iter : int, optional
        max iteration, by default 5000
    threshold : float, optional
        Threshold for stop iteration, by default .5e-2

    Returns
    -------
    torch.Tensor
        sinkhorn_divergences
    """
    M = 1 - unnormalized_beta # K,V
    theta_t = theta.T # M,K ->K,M
    bow_t = F.softmax(bow).T #M,V->V,M
    device = unnormalized_beta.device

    _,n_topics = theta.size()

    u = (torch.ones_like(theta_t) / n_topics).to(device) # Psi_1
    v = (torch.ones_like(bow_t)).to(device) # Psi_2

    K = torch.exp(-M * alpha)

    for i in range(1, max_iter+1):
        # K(K,V), bow_t(V,M), u(K,M)
        v = bow_t / (K.T @ u)
        u = theta_t / (K @ v) # Psi_1

        if i % 20 == 0:
            v = bow_t   / K.T @ u # Psi_2 # diag(v).T K diag(u)1n = b
            u = theta_t / K   @ v # Psi_1 # diag(u) K diag(v)1_m = a
            bb = v * (K.T @ u)
            err = (bb - bow_t).abs().sum(dim=0).norm(p=float('inf'))
            
            if err <= threshold:
                break
    """
    u(K,M) H(K,V), M(K,V), v(V,M) -> (K,M).sum(0) -> M
    y=K*M, (K,V)
    y=y@v, (K,M)
    y=u*y, (K,M)
    y.sum(0), (M,)
    """
    return torch.mul(u, K*M @ v).sum(dim=0)
'''

"""
import numpy as np 
import torch 
import torch.nn.functional as F
from sktopic.components.sinkhorn import sinkhorn

V = 10000; M = 200; K = 100
theta = torch.randn((M,K))
theta = F.softmax(theta,1).cuda()
bow = torch.randn((M,V)).mul(10).abs().cuda()
beta = torch.randn((K,V)).cuda()
beta = F.softmax(beta,1)
sinkhorn(beta,theta,bow).mean()
"""
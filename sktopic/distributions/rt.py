from typing import Optional,Any 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 


def kl_divergence(prior, posterior):
    if prior is None:
        kl = -0.5 * torch.sum(1 + posterior.logvar - posterior.loc.pow(2) - posterior.logvar.exp(), dim=-1).mean()
    else:
        num_topics = posterior.loc.size(1)
        posterior_variance = posterior.logvar.exp()
        prior_variance = prior.logvar.exp()
        
        var_division = torch.sum(posterior_variance / prior_variance, dim=1)
        # diff means term
        diff_means = prior.loc - posterior.loc
        diff_term = torch.sum(
            (diff_means * diff_means) / prior_variance, dim=1)
        # logvar det division term
        logvar_det_division = \
            prior_variance.log().sum() - posterior.logvar.sum(dim=1)
        # combine terms
        kl = 0.5 * (var_division + diff_term - num_topics + logvar_det_division)
    return kl


class Normal(nn.Module):
    name = "Normal"
    def __init__(self, loc, logvar):
        super().__init__()
        self.loc = loc 
        self.logvar = logvar

    @property
    def mean(self):
        return self.loc 
    
    @property
    def scale(self):
        return torch.exp(0.5 * self.logvar)

    def reparameterize(self, mu, logvar):
        """Returns a sample from a Gaussian distribution via reparameterization.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul_(std).add_(mu)
        else:
            return mu

    def rsample(self,):
        return self.reparameterize(self.loc,self.logvar)

class SoftmaxLogisticNormal(Normal):
    name = "SoftmaxLogisticNormal"

    @property
    def mean(self):
        return F.softmax(self.loc, dim=-1)

    @staticmethod
    def from_alpha(alpha:torch.Tensor):
        h_dim = alpha.squeeze().size(0)
        loc = (alpha.log().T - alpha.log().mean(dim=1) ).T
        scale = ( ( (1.0/alpha)*( 1 - (2.0/h_dim) ) ).T + ( 1.0/(h_dim*h_dim) )*torch.sum(1.0/alpha,1) ).T
        return SoftmaxLogisticNormal(loc,scale.log())

    def reparameterize(self, mu, logvar):
        z = super().reparameterize(mu,logvar)
        return F.softmax(z, dim=-1)

    
"""
self._loss(x, word_dists, prior_mean, prior_var, posterior_mean, posterior_var, posterior_log_var)


def _loss(self, inputs, word_dists, prior_mean, prior_variance, posterior_mean, posterior_variance, posterior_log_variance):

"""
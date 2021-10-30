from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F
from .base import Compressor, H2GaussParams
from ..distributions import SoftmaxLogisticNormal


class DETM(nn.Module):
    def __init__(self, n_components:int, n_times:int,vocab_size:int, t_hidden_dim:int,
    eta_hidden_dim:int, embed_dim:int, encoder_dropout_rate:float=0.0,eta_nlayers:int=3,
    eta_dropout_rate:float=0.0, delta:float=5e-3, theta_act:str="Tanh",device=None,dtype=None):
        super().__init__()

        ## define hyperparameters
        self.n_components = n_components
        self.n_times = n_times
        self.vocab_size = vocab_size
        self.t_hidden_dim = t_hidden_dim
        self.eta_hidden_dim = eta_hidden_dim
        self.embed_dim = embed_dim
        self.encoder_dropout_rate = encoder_dropout_rate
        self.eta_dropout_rate = eta_dropout_rate
        self.eta_nlayers = eta_nlayers
        self.delta = delta
        self.theta_act = theta_act
        self.device=device
        self.dtype=dtype
                
        ## define variational distribution for \eta via amortizartion... eta is K x T
        self.q_eta_map = nn.Linear(self.vocab_size, self.eta_hidden_dim)
        self.q_eta = nn.LSTM(
            self.eta_hidden_dim, self.eta_hidden_dim, self.eta_nlayers, 
            dropout=self.eta_dropout_rate,
            )
        self.params_q_eta = H2GaussParams(
            [self.eta_hidden_dim+self.n_components, self.n_components],
            device=self.device, 
            dtype=self.dtype,
            )

        ## define variational distribution for \theta_{1:D} via amortizartion... theta is K x D
        self.q_theta = Compressor(
            [self.vocab_size+self.n_components, self.t_hidden_dim, self.t_hidden_dim],
            activation=eval(f"nn.{theta_act}"), 
            dropout_rate=encoder_dropout_rate,
            device=self.device, 
            dtype=self.dtype,
            )
        self.params_q_theta = H2GaussParams(
            [self.t_hidden_dim, self.n_components], 
            device=self.device, 
            dtype=self.dtype,
            )

        ## define the variational parameters for the topic embeddings over time (alpha) ... alpha is K x T x L
        self.mu_q_alpha = nn.Parameter(torch.randn(self.n_components, self.n_times, self.embed_dim))
        self.logsigma_q_alpha = nn.Parameter(torch.randn(self.n_components, self.n_times, self.embed_dim))

        ## define the word embedding matrix \rho
        self.word_embeddings  = nn.Embedding(self.vocab_size,self.embed_dim)
        self._losses = {}

    def add_loss(self, **kv_pair)->None:
        for k,v in kv_pair.items():
            self._losses[k] = v

    def reparameterize(self, mu, logvar, input_std=False):
        """Returns a sample from a Gaussian distribution via reparameterization.
        """
        if not input_std:
            std = torch.exp(0.5 * logvar)
        else:
            std = logvar

        if self.training:
            eps = torch.randn_like(std)
            return eps.mul_(std).add_(mu)
        else:
            return mu

    def get_kl(self, q_mu, q_logsigma, p_mu=None, p_logsigma=None):
        """Returns KL( N(q_mu, q_logsigma) || N(p_mu, p_logsigma) ).
        """
        if p_mu is not None and p_logsigma is not None:
            sigma_q_sq = torch.exp(q_logsigma)
            sigma_p_sq = torch.exp(p_logsigma)
            kl = ( sigma_q_sq + (q_mu - p_mu)**2 ) / ( sigma_p_sq + 1e-6 )
            kl = kl - 1 + p_logsigma - q_logsigma
            kl = 0.5 * torch.sum(kl, dim=-1)
        else:
            kl = -0.5 * torch.sum(1 + q_logsigma - q_mu.pow(2) - q_logsigma.exp(), dim=-1)
        return kl

    def get_word_embeddings(self):
        return self.word_embeddings.weight

    def get_eta(self, rnn_inp): ## structured amortized inference
        device = rnn_inp.device
        inp = self.q_eta_map(rnn_inp).unsqueeze(1)
        hidden = self.init_hidden()
        output, _ = self.q_eta(inp, hidden)
        output = output.squeeze()

        etas = torch.zeros(self.n_times, self.n_components).to(device)
        kl_eta = []

        inp_0 = torch.cat([output[0], torch.zeros(self.n_components,).to(device)], dim=0)
        mu_0,logsigma_0 = self.params_q_eta(inp_0)
        etas[0] = self.reparameterize(mu_0, logsigma_0, input_std=True)

        p_mu_0 = torch.zeros(self.n_components,).to(device)
        logsigma_p_0 = torch.zeros(self.n_components,).to(device)
        kl_0 = self.get_kl(mu_0, logsigma_0, p_mu_0, logsigma_p_0)
        kl_eta.append(kl_0)
        for t in range(1, self.n_times):
            inp_t = torch.cat([output[t], etas[t-1]], dim=0)
            mu_t, logsigma_t = self.params_q_eta(inp_t)
            etas[t] = self.reparameterize(mu_t, logsigma_t, input_std=True)

            p_mu_t = etas[t-1]
            logsigma_p_t = torch.log(self.delta * torch.ones(self.n_components,).to(device))
            kl_t = self.get_kl(mu_t, logsigma_t, p_mu_t, logsigma_p_t)
            kl_eta.append(kl_t)
        kl_eta = torch.stack(kl_eta).sum()
        return etas, kl_eta

    def get_theta(self, eta, bows, times): ## amortized inference
        """Returns the topic proportions.
        """
        device = eta.device
        eta_td = eta[times.type('torch.LongTensor')]
        inp = torch.cat([bows, eta_td], dim=1)
        q_theta = self.q_theta(inp)
        mu_theta,logsigma_theta = self.params_q_theta(q_theta)
        z = self.reparameterize(mu_theta, logsigma_theta, input_std=True)
        theta = F.softmax(z, dim=-1)
        kl_theta = self.get_kl(mu_theta, logsigma_theta, eta_td, torch.zeros(self.n_components).to(device))
        return theta, kl_theta

    def get_alpha(self): ## mean field
        device = self.mu_q_alpha.device
        alphas = torch.zeros(self.n_times, self.n_components, self.embed_dim).to(device)
        kl_alpha = []

        alphas[0] = self.reparameterize(self.mu_q_alpha[:, 0, :], self.logsigma_q_alpha[:, 0, :])

        p_mu_0 = torch.zeros(self.n_components, self.embed_dim).to(device)
        logsigma_p_0 = torch.zeros(self.n_components, self.embed_dim).to(device)
        kl_0 = self.get_kl(self.mu_q_alpha[:, 0, :], self.logsigma_q_alpha[:, 0, :], p_mu_0, logsigma_p_0)
        kl_alpha.append(kl_0)
        for t in range(1, self.n_times):
            alphas[t] = self.reparameterize(self.mu_q_alpha[:, t, :], self.logsigma_q_alpha[:, t, :]) 
            p_mu_t = alphas[t-1]
            logsigma_p_t = torch.log(self.delta * torch.ones(self.n_components, self.embed_dim).to(device))
            kl_t = self.get_kl(self.mu_q_alpha[:, t, :], self.logsigma_q_alpha[:, t, :], p_mu_t, logsigma_p_t)
            kl_alpha.append(kl_t)
        kl_alpha = torch.stack(kl_alpha).sum()
        return alphas, kl_alpha.sum()

    def get_beta(self, alpha=None):
        """Returns the topic matrix \beta of shape K x V
        """
        if alpha is None:
            alpha, _ = self.get_alpha()
        tmp = alpha.view(-1, self.embed_dim) # (K*T, L)
        logit = torch.mm(tmp, self.word_embeddings.weight.permute(1, 0)) # (K*T,L)@(V,L).T=(K*T,V)
        logit = logit.view(alpha.size(0), alpha.size(1), -1) # (K,T,V)
        beta = F.softmax(logit, dim=-1)
        return beta 

    def get_lnpx(self, theta, beta):
        theta = theta.unsqueeze(1) # (M,1,K)
        px = torch.bmm(theta, beta).squeeze(1) # (M,1,K)@(M,K,V)
        lnpx = torch.log(px+1e-6) # (M,V)
        return lnpx

    def get_nll(self, lnpx, bows):
        nll = -lnpx * bows
        nll = nll.sum(-1)
        return nll

    def forward(self, X:torch.Tensor, timestamp:torch.Tensor, 
    X_mean:torch.Tensor, normalized_bows:Optional[torch.Tensor]=None):
        
        if normalized_bows is None:
            normalized_bows = X/ X.sum(1).unsqueeze(1)
            assert not torch.isnan(normalized_bows.sum()), "単語数0の文書が含まれているかも？"
        eta, kl_eta = self.get_eta(X_mean)
        θ, kl_theta = self.get_theta(eta, bows=normalized_bows, times=timestamp)
        kl_theta = kl_theta.sum()
        
        α, kl_alpha = self.get_alpha()
        β = self.get_beta(α)
        _β = β[timestamp.type('torch.LongTensor')]
        lnpx = self.get_lnpx(θ,_β)
        nll = self.get_nll(lnpx, X)
        nll = nll.mean()
        #self._losses()
        output = dict(
            topic_proportion=θ,
            lnpx = lnpx,
            topic_dist = β,
        )
        losses = dict(
            nll=nll,
            kld_alpha=kl_alpha,
            kld_eta=kl_eta,
            kld_theta=kl_theta
        )
        output["losses"] = losses
        return output
        
    def init_hidden(self):
        """Initializes the first hidden state of the RNN used as inference network for \eta.
        """
        weight = next(self.parameters())
        nlayers = self.eta_nlayers
        nhid = self.eta_hidden_dim
        return (weight.new_zeros(nlayers, 1, nhid), weight.new_zeros(nlayers, 1, nhid))
    
    def transform(self, x, timestamps, mean_x):
        eta, kl_eta = self.get_eta(mean_x)
        theta, kl_theta = self.get_theta(eta, x, timestamps)
        return theta


class ScalingLoss(nn.Module):
    def __init__(self,num_docs:Optional[int]=None):
        super().__init__()
        self.num_docs = num_docs

    def forward(self, output, y_true, X, **kwargs):
        losses = output["losses"]
        if self.num_docs is None:
            coeff = 1.0
        else:
            coeff = self.num_docs / X.size(0)
        
        nelbo = losses["nll"] + losses["kld_theta"]
        loss = nelbo * coeff + losses["kld_alpha"] + losses["kld_eta"]

        losses["nelbo"] = nelbo
        losses["loss"] = loss
        losses.update(self.culc_metrics(losses, **kwargs))
        return losses

    def culc_metrics(self, output, model=None, **kwargs):
        aux = {}
        return aux


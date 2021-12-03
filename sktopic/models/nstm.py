from typing import Sequence,Callable,Any,Optional
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import dropout
from sktopic.components.sinkhorn import Sinkhorn, sinkhorn

from sktopic.trainers.vae import Trainer,Dataset
from ..components import Compressor
from ..components.mmd_loss import BoWCrossEntropy
from torch.utils.data import DataLoader
from skorch.dataset import CVSplit
from skorch.utils import to_device,to_numpy,to_tensor
from sktopic.callbacks import PerplexityScoring
from skorch.callbacks import PassthroughScoring,EpochTimer,PrintLog

class GumbelSoftmax(nn.Module):
    def __init__(self,hard=False,eps=1e-10,dim=-1):
        super().__init__()
        self.hard = hard 
        self.eps = eps
        self.dim = dim

    def forward(self,logits,tau=1):
        return F.gumbel_softmax(logits,tau, self.hard, self.eps, self.dim)
class NSTM(nn.Module):
    def __init__(self, vocab_size:int, n_components:int, hidden_dims:Sequence[int]=None, embed_dim:int=None, activation:str="Softplus", dropout_rate:float=0.5, device="cpu", dtype:Any=torch.float32) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [n_components*3,n_components*2]
        self.n_components = n_components
        self.vocab_size = vocab_size 
        self.embed_dim = embed_dim
        self.hidden_dims = hidden_dims
        dims = [vocab_size] + hidden_dims + [n_components]
        self.device = device 
        self.dtype = dtype
        _activation = eval(f"nn.{activation}")
        self.encoder = nn.Sequential(
            Compressor(dims, _activation, dropout_rate=dropout_rate,device=self.device, dtype=self.dtype),
            nn.BatchNorm1d(n_components),
            nn.Softmax(dim=1),
            #GumbelSoftmax(),
        )
        self._set_beta()

    def _set_beta(self)->None:
        if isinstance(self.embed_dim, int):
            self.topic_embeddings = nn.Embedding(self.n_components, self.embed_dim,device=self.device, dtype=self.dtype)
            self.word_embeddings = nn.Embedding(self.vocab_size, self.embed_dim,device=self.device, dtype=self.dtype)
        else:
            self.beta = nn.Embedding(self.vocab_size, self.n_components,device=self.device, dtype=self.dtype)

    def get_beta(self)->torch.Tensor:
        if self.embed_dim:
            t = self.topic_embeddings.weight
            e = self.word_embeddings.weight
            # normalize
            t = F.normalize(t)
            e = F.normalize(e)
            return torch.matmul(t, e.T)
        return F.normalize(self.beta.weight.T)

    def decode(self, theta):
        beta = self.get_beta() # (K,V)
        lnpx = F.log_softmax(theta @ (1 + beta) , dim=1)
        return lnpx

    def forward(self, x):
        θ = self.encoder(x)
        beta = self.get_beta() # (K,V)
        lnpx = F.log_softmax(θ @ (1 + beta) , dim=1)

        return dict(
            topic_proportion=θ,
            lnpx=lnpx,
            topic_dist = beta,
        )

    def transform(self, x):
        return self.encoder(x)


class SinkhornLoss(Sinkhorn):
    def forward(self, beta:torch.Tensor,theta:torch.Tensor, bow:torch.Tensor, model:Optional[Any]=None, **kwargs):
        divergence = super().forward(beta,theta,bow,)
        cce = -(F.log_softmax(theta @ beta,1) * bow).sum(1).mean()
        mean_length = bow.sum(1).mean()
        V = bow.size(1)
        cce_scaling = 1/(mean_length * torch.log(torch.tensor(V)))
        return dict(
            sinkhorn_divergence= divergence,
            nll = cce,
            loss = cce*cce_scaling + divergence
        )

class NeuralSinkhornTopicModel(Trainer):
    def __init__(self,
            n_features:int, 
            n_components:int, 
            hidden_dims:Sequence[int]=None, 
            embed_dim:int=None, 
            activation_hidden:str="Softplus", 
            dropout_rate_hidden:float=0.5,
            criterion:Callable=SinkhornLoss,
            optimizer:Any=torch.optim.Adam,
            lr:float=0.01,
            max_epochs:int=10,
            batch_size:int=128,
            iterator_train:Any=DataLoader,
            iterator_valid:Any=DataLoader,
            dataset:Any=Dataset,
            train_split:Callable[[int], Any]=CVSplit(5),
            callbacks:Optional[Any]=None,
            warm_start:bool=False,
            verbose:int=1,
            device:str='cpu',
            use_amp:bool = False,
            **kwargs,
            ):
        super().__init__(
            NSTM(n_features,n_components,hidden_dims,embed_dim,activation_hidden,dropout_rate_hidden),
            criterion,
            optimizer=optimizer,
            lr=lr,
            max_epochs=max_epochs,
            batch_size=batch_size,
            iterator_train=iterator_train,
            iterator_valid=iterator_valid,
            dataset=dataset,
            train_split=train_split,
            callbacks=callbacks,
            warm_start=warm_start,
            verbose=verbose,
            device=device,
            **kwargs
            )
    @property
    def _default_callbacks(self)->None:
        return [
            ('epoch_timer', EpochTimer()),
            ("valid_ppl", PerplexityScoring(on_train=False)),
            ("train_ppl", PerplexityScoring(on_train=True)),
            ('logging_valid_ppl', PassthroughScoring(name='valid_ppl')),
            ('logging_train_ppl', PassthroughScoring(name='train_ppl',on_train=True)),
            ('train_loss', PassthroughScoring(name='train_loss',on_train=True)),
            ('sk_divergence', PassthroughScoring(name='train_sinkhorn_divergence',on_train=True)),
            ('valid_loss', PassthroughScoring(name='valid_loss')),
            ('print_log', PrintLog()),

        ]
    @torch.no_grad()
    def get_topic_word_distributions(self, decode=False, safety=True, numpy=True):
        self.module_.eval()
        if decode:
            logbeta = self.module_.decode(torch.eye(self.module_.n_components).to(self.device))
            beta = logbeta.exp()
            if safety:
                beta = F.normalize(beta, p=1,dim=1)            
            return to_numpy(beta) if numpy else beta
        else:
            beta = self.module_.get_beta()
            return to_numpy(beta) if numpy else beta
    
    def get_loss(self, x, topic_proportion, training=False, **kwargs):
        x = to_tensor(x, device=topic_proportion.device)
        if isinstance(self.criterion_, torch.nn.Module):
            self.criterion_.train(training)
        # (self, lnpx, x, posterior, model=None, **kwargs)
        if self._pseudo_labels is None:
            cluster_labels = None
        else:
            cluster_labels = to_device(self._pseudo_labels, x.device) 
            
        outputs = self.criterion_(
            self.module_.get_beta(),
            topic_proportion,
            x,
            cluster_labels=cluster_labels,
            model=self.module_,
            **kwargs
            )

        return outputs

    def train_step(self, batch, **fit_params):
        Xi, yi= batch
        self.module_.train()
        #print(self._pseudo_labels)
        self.optimizer_.zero_grad()
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            y_preds = self.infer(Xi,**fit_params)
            outputs = self.get_loss(Xi, topic_proportion=y_preds["topic_proportion"],training=True)
        self.scaler.scale(outputs["loss"]).backward()
        self.scaler.step(self.optimizer_)
        self.scaler.update()
        self.record_to_history(outputs, prefix="train")
        return dict(y_pred=y_preds["lnpx"], **outputs )

    @torch.no_grad()
    def validation_step(self, batch, **fit_params):
        Xi, yi = batch
        self.module_.eval()
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            y_preds = self.infer(Xi,**fit_params)
            outputs = self.get_loss(Xi,topic_proportion=y_preds["topic_proportion"])
        self.record_to_history(outputs, prefix="valid")
        return dict(y_pred=y_preds["lnpx"], **outputs )

    @torch.no_grad()
    def evaluation_step(self, batch, **fit_params):
        Xi,yi = batch
        self.module_.eval()
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            y_preds = self.infer(Xi,**fit_params)
            outputs = self.get_loss(Xi,topic_proportion=y_preds["topic_proportion"])
        #self.record_to_history(aux, prefix="valid")
        return dict(y_pred=y_preds["lnpx"], **outputs )

"""
from sktopic.models.nstm import NeuralSinkhornTopicModel as NSTM
from sktopic.datasets import fetch_20NewsGroups
from sktopic.callbacks import WECoherenceScoring,TopicDiversityScoring
dataset = fetch_20NewsGroups()
V = dataset["X"].shape[1]
m = NSTM(V,20,embed_dim=50,
callbacks = [
    WECoherenceScoring(dataset["id2word"]),
    TopicDiversityScoring(dataset["id2word"]),],
batch_size=2000,
device="cuda",
)
m.fit(dataset["X"])
from octis.evaluation_metrics.coherence_metrics import WECoherencePairwise
output = m.get_model_outputs(X_tr=dataset["X"], id2word=dataset["id2word"])
we = WECoherencePairwise()
print("WE=", we.score(output))
from sktopic.metrics.npmi import NormalizedPointwiseMutualInformation as NPMI
npmi = NPMI(dataset["X"], id2word=dataset["id2word"])
print("NPMI=", npmi(output["topics"]))
"""
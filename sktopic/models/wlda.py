from typing import ForwardRef, Sequence,Callable,Any,Optional
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.linear import Identity
from ..components import NTM
from ..components import MMDLoss

from sktopic.trainers.vae import Trainer,Dataset
from torch.utils.data import DataLoader
from skorch.dataset import CVSplit
from skorch.utils import to_device,to_numpy,to_tensor
from sktopic.callbacks import PerplexityScoring
from skorch.callbacks import PassthroughScoring,EpochTimer,PrintLog

class LatentNoise(nn.Module):
    def __init__(self,dirichlet_alpha:float=0.1, noise_rate:float=0.1) -> None:
        super().__init__()
        self.dirichlet_alpha = dirichlet_alpha
        self.noise_rate = noise_rate
        assert 0.0 <= self.noise_rate < 1.0, "please set as 0 <= noise_rate < 1"

    def forward(self,theta:torch.Tensor)->torch.Tensor:
        if self.training:
            concentration= torch.ones_like(theta)*self.dirichlet_alpha
            noise = torch.distributions.Dirichlet(concentration=concentration).sample()
            return (1-self.noise_rate) * theta + self.noise_rate * noise
        else:
            return theta

class WLDA(NTM):
    def __init__(self, dims: Sequence[int], embed_dim: Optional[int] = None, activation_hidden: str = "Tanh", dropout_rate_hidden: float = 0.2, dropout_rate_theta: float = 0.2, device: Any = None, dtype: Any = None, topic_model: bool = False, dirichlet_alpha:float=0.1, noise_rate:float=0.1):
        self.dirichlet_alpha=dirichlet_alpha
        self.noise_rate=noise_rate
        super().__init__(dims, embed_dim=embed_dim, activation_hidden=activation_hidden, dropout_rate_hidden=dropout_rate_hidden, dropout_rate_theta=dropout_rate_theta, device=device, dtype=dtype, topic_model=topic_model)
        

    def _build(self)->None:
        super()._build(
            h2params=nn.Sequential(
                nn.Linear(self.n_hiddens[-1], self.n_components),
                nn.BatchNorm1d(self.n_components),
                nn.Softmax(dim=1),
                ),
            rt = Identity(),
            map_theta = LatentNoise(self.dirichlet_alpha,self.noise_rate),
        )

    def forward(self,x:torch.Tensor, **kwargs)->dict[str,Any]:
        h = self.encoder(x)
        θ = self.decoder["map_theta"](h)
        lnpx = self.decoder["decoder"](θ)

        return dict(
            topic_proportion=θ,
            posterior = None,
            lnpx=lnpx,
            topic_dist = self.get_beta(),
        )
    
    def transform(self, x:torch.Tensor, **kwargs)->torch.Tensor:
        h = self.encoder(x)
        θ = self.decoder["map_theta"](h)
        return θ
    
class WassersteinLatentDirichletAllocation(Trainer):
    def __init__(self,
            vocab_size:int,
            n_components:int, 
            hidden_dims:Optional[Sequence[int]]=None,
            embed_dim:Optional[int]=None,
            activation_hidden: str = "Softplus",
            dropout_rate_hidden: float = 0.2, 
            dropout_rate_theta: float = 0.2,
            topic_model: bool = False,
            optimizer:Any=torch.optim.Adam,
            dirichlet_alpha = 0.1,
            noise_rate = 0.1,
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
            criterion:Callable=MMDLoss,
            **kwargs,
            ):
        """ Sklearn like trainer for GaussianSoftmaxModel

        Parameters
        ----------
        vocab_size : int
            Number of Unique tokens
        n_components : int
            Number of Topics
        hidden_dims : Optional[Sequence[int]], optional
            List of number of units of hidden layers, by default None
        embed_dim : Optional[int], optional
            Dimension of word embeddings, by default None
        activation_hidden : str, optional
            Activation function of hidden layers, by default "Softplus"
        dropout_rate_hidden : float, optional
            Dropout rate of output of hidden layers, by default 0.2
        dropout_rate_theta : float, optional
            Dropout rate of Topic-proportion vectors, by default 0.2
        topic_model: bool, optional
            if True, Topic Model style decoder, esle DocumentModel stype, by default False
        optimizer : Any, optional
            class object of torch.optim, by default torch.optim.Adam
        lr : float, optional
            learning rate, by default 0.01
        max_epochs : int, optional
            max epochs of training, by default 10
        batch_size : int, optional
            minibatch size, by default 128
        iterator_train : Any, optional
            ..., by default DataLoader
        iterator_valid : Any, optional
            ..., by default DataLoader
        dataset : Any, optional
            ..., by default Dataset
        train_split : Callable[[int], Any], optional
            train test split option, by default CVSplit(5)
        callbacks : Optional[Any], optional
            skorch callbacks, by default None
        warm_start : bool, optional
            [description], by default False
        verbose : int, optional
            flag of monitoring on training step, by default 1
        device : str, optional
            cpu/gpu, by default 'cpu'
        use_amp : bool, optional
            AMP tranining flag, by default False
        """
        if "criterion__dirichlet_alpha" in kwargs:
            dirichlet_alpha = kwargs["criterion__dirichlet_alpha"]
            print("------WARNING-------")
            print(f"you overwrote `dirichlet_alpha` with `criterion__dirichlet_alpha`={kwargs['criterion__dirichlet_alpha']}")
            print("--------------------")

        if hidden_dims is None:
            hidden_dims = [n_components*3,n_components*2]
        _dims = [vocab_size]+hidden_dims+[n_components]
        _module = WLDA(
            _dims,embed_dim,activation_hidden,
            dropout_rate_hidden,dropout_rate_theta,
            topic_model=topic_model, device=device,
            dirichlet_alpha=dirichlet_alpha,noise_rate=noise_rate,)
         
        super().__init__(
            module=_module,
            criterion=criterion,
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
            criterion__dirichlet_alpha = dirichlet_alpha,
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
            ('valid_loss', PassthroughScoring(name='valid_loss')),
            ('train_MMD', PassthroughScoring(name='train_mmd',on_train=True)),
            ('valid_MMD', PassthroughScoring(name='valid_mmd')),
            ('print_log', PrintLog()),
        ]
    def get_loss(self, lnpx, x, posterior=None, training=False, topic_proportion=None,**kwargs):
        x = to_tensor(x, device=lnpx.device)

        if isinstance(self.criterion_, torch.nn.Module):
            self.criterion_.train(training)
        
        return self.criterion_(
            lnpx, x,
            posterior=posterior,
            theta = topic_proportion,
            model=self.module_,
            **kwargs
            )
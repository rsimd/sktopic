from typing import Sequence,Optional,Any, Callable
import torch
from sktopic.trainers.vae import Trainer
from ..components import NTM, ELBO
from ..trainers.vae import Dataset, Trainer
from torch.utils.data import DataLoader
from skorch.dataset import CVSplit
from ..distributions import RecurrentStickBreakingConstruction


class RSB(NTM):
    def _build(self) -> None:
        super()._build(
            map_theta = RecurrentStickBreakingConstruction(
            self.n_components,device=self.device,dtype=self.dtype
            )
        )


class RecurrentStickBreakingModel(Trainer):
    def __init__(self,
            vocab_size:int,
            n_components:int, 
            hidden_dims:Optional[Sequence[int]]=None,
            embed_dim:Optional[int]=None,
            activation_hidden: str = "Softplus",
            dropout_rate_hidden: float = 0.2, 
            dropout_rate_theta: float = 0.2,
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
            criterion:Callable=ELBO,
            **kwargs,
            ):
        """ Sklearn like trainer for RecurrentStickBreakingModel

        Parameters
        ----------
        vocab_size : int
            [description]
        n_components : int
            [description]
        hidden_dims : Optional[Sequence[int]], optional
            [description], by default None
        embed_dim : Optional[int], optional
            [description], by default None
        activation_hidden : str, optional
            [description], by default "Softplus"
        dropout_rate_hidden : float, optional
            [description], by default 0.2
        dropout_rate_theta : float, optional
            [description], by default 0.2
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
        if hidden_dims is None:
            hidden_dims = [n_components*3,n_components*2]
        _dims = [vocab_size]+hidden_dims+[n_components]
        _module = RSB(_dims,embed_dim,activation_hidden,dropout_rate_hidden,dropout_rate_theta)
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
            **kwargs
            )
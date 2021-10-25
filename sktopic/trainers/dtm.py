"""Neural net base class

This is the most flexible class, not making assumptions on the kind of
task being peformed. Subclass this to create more specialized and
sklearn-conforming classes like NeuralNetClassifier.

"""
from typing import Any,Callable,Sequence,Tuple,Optional

import numpy as np
from scipy.sparse.csr import csr_matrix
from sklearn.base import TransformerMixin
import torch
import pandas as pd
from torch.utils.data import DataLoader

from skorch.callbacks import EpochTimer
from skorch.callbacks import PrintLog
from skorch.callbacks import PassthroughScoring
from skorch.dataset import Dataset
from skorch.dataset import CVSplit
from skorch.dataset import get_len
from skorch.utils import TeeGenerator
from skorch.utils import is_dataset
from skorch.utils import to_device
from skorch.utils import to_numpy
from skorch.utils import to_tensor
from skorch.net import NeuralNet
from scipy import sparse
from sktopic.callbacks import PerplexityScoring
from ..utils.get_results import get_similar_words, get_topic_top_words


class TimestampDataset(Dataset):
    def transform(self, X, timestamp, y=None):
        if y is None:
            y = torch.Tensor([0]) if y is None else y

        if sparse.issparse(X):
            X = X.toarray().squeeze(0)
        
        return torch.from_numpy(X), timestamp, y


# pylint: disable=too-many-instance-attributes
class DynamicTrainer(NeuralNet, TransformerMixin):
    def __init__(self,module:Callable,criterion:Callable,
            optimizer:Any=torch.optim.Adam,
            lr:float=0.01,
            max_epochs:int=10,
            batch_size:int=128,
            iterator_train:Any=DataLoader,
            iterator_valid:Any=DataLoader,
            dataset:Any=TimestampDataset,
            train_split:Callable[[int], Any]=CVSplit(5),
            callbacks:Optional[Any]=None,
            warm_start:bool=False,
            verbose:int=1,
            device:str='cpu',
            use_amp:bool = False,
            **kwargs,
            ):
        super().__init__(
            module,
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
        self.use_amp = use_amp

    def initialize(self):
        return super().initialize()

    def get_dataset(self, X, timestamp, y=None): ##
        if is_dataset(X):
            return X

        dataset = self.dataset
        is_initialized = not callable(dataset)

        kwargs = self.get_params_for('dataset')
        if kwargs and is_initialized:
            raise TypeError("Trying to pass an initialized Dataset while "
                            "passing Dataset arguments ({}) is not "
                            "allowed.".format(kwargs))

        if is_initialized:
            return dataset

        return dataset(X, timestamp, y, **kwargs)

    def get_split_datasets(self, X, timestamp, y=None, **fit_params): ##
        dataset = self.get_dataset(X, timestamp, y) 
        if not self.train_split:
            return dataset, None

        # After a change in (#646),
        # `y` is no longer passed to `self.train_split` if it is `None`.
        # To revert to the previous behavior, remove the following two lines:
        if y is None:
            return self.train_split(dataset, **fit_params)
        return self.train_split(dataset, y, **fit_params)

    def validation_step(self, batch, X_mean, **fit_params): ##
        self._set_training(False)
        Xi, timestamp, yi = batch
        with torch.no_grad():
            y_pred = self.infer(X=Xi, timestamp=timestamp,X_mean=X_mean, **fit_params)
            losses = self.get_loss(y_pred, yi, X=Xi, training=False)
            #loss = losses["loss"]
        self.record_to_history(losses, prefix="valid")
        return {
            #'loss': loss,
            'y_pred': y_pred,
            **losses,
        }

    def train_step_single(self, batch, X_mean, **fit_params): ##
        self._set_training(True)
        Xi, timestamp, yi = batch
        y_pred = self.infer(Xi,timestamp,X_mean, **fit_params)
        losses = self.get_loss(y_pred, yi, X=Xi, training=True)
        loss = losses["loss"]
        loss.backward()
        self.record_to_history(losses, prefix="train")
        return {
            #'loss': loss,
            'y_pred': y_pred,
            **losses,
        }

    def train_step(self, batch, X_mean, **fit_params): ##
        step_accumulator = self.get_train_step_accumulator()

        def step_fn():
            self._zero_grad_optimizer()
            step = self.train_step_single(batch, X_mean, **fit_params) #wip
            step_accumulator.store_step(step)
            self.notify(
                'on_grad_computed',
                named_parameters=TeeGenerator(self.get_all_learnable_params()),
                batch=batch,
            )
            return step['loss']

        self._step_optimizer(step_fn) # 
        return step_accumulator.get_step()

    def evaluation_step(self, batch, X_mean, training=False):##
        self.check_is_fitted()
        Xi,timestamp,_ = batch
        with torch.set_grad_enabled(training):
            self._set_training(training)
            return self.infer(Xi, timestamp, X_mean)

    def fit_loop(self, X, timestamp, X_mean, y=None, epochs=None, **fit_params): ##
        self.check_data(X, y)
        epochs = epochs if epochs is not None else self.max_epochs

        dataset_train, dataset_valid = self.get_split_datasets(
            X, timestamp, y, **fit_params) #wip
        on_epoch_kwargs = {
            'dataset_train': dataset_train,
            'dataset_valid': dataset_valid,
        }

        for _ in range(epochs):
            self.notify('on_epoch_begin', **on_epoch_kwargs)

            self.run_single_epoch(dataset_train, X_mean, training=True, prefix="train",
                                  step_fn=self.train_step, **fit_params)

            self.run_single_epoch(dataset_valid, X_mean, training=False, prefix="valid",
                                  step_fn=self.validation_step, **fit_params)

            self.notify("on_epoch_end", **on_epoch_kwargs)
        return self

    def run_single_epoch(self, dataset, X_mean, training, prefix, step_fn, **fit_params):
        if dataset is None:
            return

        batch_count = 0
        for batch in self.get_iterator(dataset, training=training):
            self.notify("on_batch_begin", batch=batch, training=training)
            step = step_fn(batch, X_mean, **fit_params)
            self.history.record_batch(prefix + "_loss", step["loss"].item())
            batch_size = (get_len(batch[0]) if isinstance(batch, (tuple, list))
                          else get_len(batch))
            self.history.record_batch(prefix + "_batch_size", batch_size)
            self.notify("on_batch_end", batch=batch, training=training, **step)
            batch_count += 1

        self.history.record(prefix + "_batch_count", batch_count)

    # pylint: disable=unused-argument
    def partial_fit(self, X, timestamp, X_mean, y=None, classes=None, **fit_params): ##
        if not self.initialized_:
            self.initialize()

        self.notify('on_train_begin', X=X, y=y)
        try:
            self.fit_loop(X,timestamp, X_mean,y, **fit_params) #wip
        except KeyboardInterrupt:
            pass
        self.notify('on_train_end', X=X, y=y)
        return self

    def fit(self, X, timestamp, X_mean, y=None, **fit_params): ##
        if not self.warm_start or not self.initialized_:
            self.initialize()

        self.partial_fit(X,timestamp, X_mean, y, **fit_params) #wip
        return self

    def forward_iter(self, X,timestamp, X_mean, training=False, device='cpu'): ##
        dataset = self.get_dataset(X, timestamp)
        iterator = self.get_iterator(dataset, training=training)
        for batch in iterator:
            yp = self.evaluation_step(batch, X_mean, training=training)
            yield to_device(yp, device=device)

    def forward(self, X, timestamp, X_mean, training=False, device='cpu'): ##
        y_infer = list(self.forward_iter(X,timestamp, X_mean, training=training, device=device))

        is_multioutput = len(y_infer) > 0 and isinstance(y_infer[0], tuple)
        if is_multioutput:
            return tuple(map(torch.cat, zip(*y_infer)))
        return torch.cat(y_infer)

    def infer(self, X, timestamp, X_mean, **fit_params):
        X = to_tensor(X, device=self.device)
        if isinstance(X, dict):
            x_dict = self._merge_x_and_fit_params(X,timestamp, X_mean, fit_params)
            return self.module_(**x_dict)
        return self.module_(X, timestamp, X_mean, **fit_params)

    def predict_proba(self, X, timestamp, X_mean):
        nonlin = self._get_predict_nonlinearity()
        y_probas = []
        for yp in self.forward_iter(X, timestamp, X_mean, training=False):
            if isinstance(yp, tuple):
                yp = yp[0]
            elif isinstance(yp, dict):
                yp = yp["lnpx"]
            yp = nonlin(yp)
            y_probas.append(to_numpy(yp))
        y_proba = np.concatenate(y_probas, 0)
        return y_proba

    # pylint: disable=unused-argument
    def get_loss(self, y_pred, y_true, X=None, training=False): ##
        y_true = to_tensor(y_true, device=self.device)
        return self.criterion_(y_pred, y_true, X)

    def record_to_history(self, logger_dict:dict[str, torch.Tensor], prefix:str)->None:
        """record evaluation score to histroy object

        Parameters
        ----------
        logger_dict : Dict[str, torch.Tensor]
            key-score pair
        prefix : str
            prefix of key to store
        """
        for key in logger_dict:
            if logger_dict[key].ndim != 0:
                continue
            self.history.record_batch(prefix + f"_{key}", logger_dict[key].item())

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
            ('print_log', PrintLog()),
        ]

    def transform(self, X:csr_matrix, timestamp:np.ndarray, X_mean:np.ndarray, training:bool=False)->np.ndarray:
        self._set_training(training)
        dataset = self.get_dataset(X, timestamp)
        iterator = self.get_iterator(dataset, training=training)
        X_mean = X_mean if isinstance(X_mean, torch.Tensor) else torch.from_numpy(X_mean)
        X_mean = X_mean.to(self.device)
        θ = []
        for batch in iterator:
            yp = self.evaluation_step(batch, X_mean, training=False)
            tmp = to_numpy(yp["topic_proportion"])
            θ.append(tmp)
        return np.vstack(θ)
    
    @property
    def components_(self):
        self._set_training(False)
        betas = self.module_.get_beta() # (T,K,V)
        (T,K,V) = betas.shape
        return to_numpy(betas)
    
    @torch.no_grad()
    def get_topic_top_words(self,id2word,topn=10)->list[pd.DataFrame]:
        self._set_training(False)
        beta = self.module_.get_beta() # T,K,V
        (T,_,_) = beta.size()
        results = []
        for t in range(T):
            df = get_topic_top_words(beta[t],id2word,topn)
            results.append(df)
        
        return results

    @torch.no_grad()
    def get_similar_words(self,queries:list[str],id2word:dict[int,str],topn:int=10)->list[pd.DataFrame]:
        self._set_training(False)
        beta = self.module_.get_beta() # T,K,V
        (T,_,_) = beta.size()
        results = []
        for t in range(T):
            df = get_similar_words(beta[t],queries,id2word,topn)
            results.append(df)
        return results

    def set_pretrained_embeddings(self, embeddings:np.ndarray, trainable=False)->None:
        """Set pretrained embeddings for NTM module

        Parameters
        ----------
        embeddings : np.ndarray
            pretrained np.ndarray matrix
        trainable : bool, optional
            flag of trainable, by default False
        """
        if self.initialized_:
            module = self.module_
        else:
            module = self.module

        sample = list(module.parameters())[0]

        module.word_embeddings.load_state_dict(
            {'weight': torch.from_numpy(embeddings).to(sample.dtype).to(sample.device)},
            )
        module.word_embeddings.weight.requires_grad = trainable
        
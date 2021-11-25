from re import I
from typing import Optional, Any, Callable, Dict
from sklearn.utils import shuffle
from skorch.callbacks import training
from tqdm.notebook import tqdm_notebook as tqdm
from more_itertools import chunked,windowed
from itertools import chain

import numpy as np
import scipy as sp
import pandas as pd
from sklearn.base import TransformerMixin
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import skorch
from skorch.dataset import CVSplit
from skorch.dataset import get_len
from skorch.utils import to_numpy, to_device
from skorch.setter import optimizer_setter
from scipy import sparse 
from sklearn.cluster import KMeans
from sktopic.utils.math import split_seq, seq2bow, bow2seq
from sktopic.callbacks import PerplexityScoring
from skorch.callbacks import PassthroughScoring,EpochTimer,PrintLog
from ..utils.get_results import get_similar_words, get_topwords, get_topic_top_words


class Dataset(skorch.dataset.Dataset):
    def transform(self, X,y=None):
        if sparse.issparse(X):
            X = X.toarray().squeeze(0)

        if y is None:
            return X, 0.0 # Noneにしたけどホントに良いのか？

        if sparse.issparse(y):
            y = y.toarray().squeeze(0)
        return X, y


class Trainer(skorch.NeuralNet, TransformerMixin):
    def __init__(self,module:Callable,criterion:Callable,
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
        """Sklearn like NTM traienr for NTM based on basic VAE

        Parameters
        ----------
        module : Callable
            instance or subclass of torch.nn.Module
        criterion : Callable
            Criterion class
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

    @property
    def components_(self)->torch.Tensor:
        """property for get Topic-Word distributions

        Returns
        -------
        torch.Tensor
            Size([n_topics, vocab_size]), 
            if module is TM, return softmax(beta),
            else if module is DM, return (unnormalized) beta. 
        """
        
        if self.initialized_:
            return to_numpy(self.module_.get_beta())
        else:
            return to_numpy(self.module.get_beta())
    
    def initialize(self)->"Trainer":
        """Initializes all components on the :class:`.NeuralNet`.

        Returns
        -------
        Trainer
            return (initialized) self
        """
        super().initialize()
        self.transform_args = None
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self._pseudo_labels = None
        return self

    @torch.no_grad()
    def predict_proba(self, X:torch.Tensor, logscale=False)->np.ndarray:
        """prodict probabirity of p(X; params)

        Parameters
        ----------
        X : torch.Tensor
            Input BoW (Test set)

        Returns
        -------
        np.ndarray
            p(X; params)
        """
        self.module_.train(False)
        outputs = []
        datasets = self.get_dataset(X)
        for data in self.get_iterator(datasets,):
            xi = data[0].to(self.device)
            output = self.infer(xi)
            lnpx = output["lnpx"]
            if logscale:
                z = lnpx
            else:
                z = lnpx.exp() # from log_softmax
            z /= z.sum(1, keepdim=True)
            outputs.append(to_numpy(z))
        proba = np.vstack(outputs)
        return proba
        
    def transform(self, X: torch.Tensor, training: bool=False)->np.ndarray:
        """transform to topic proportion vectors from Input BoW matrix

        Parameters
        ----------
        X : torch.Tensor
            Input BoW matrix
        training : bool, optional
            flag of tranining or evaluation, by default False

        Returns
        -------
        np.ndarray
            topic proportion vectors (a.k.a Theta)
        """
        self.module_.train(training)
        outputs = []
        datasets = self.get_dataset(X)
        for data in self.get_iterator(datasets):
            xi = data[0].to(self.device)
            theta = self.module_.transform(xi)
            outputs.append(theta.detach().cpu().numpy())
        topic_proportions = np.vstack(outputs)
        return topic_proportions

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

    def record_to_history(self, logger_dict:Dict[str, torch.Tensor], prefix:str)->None:
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

    def get_loss(self, lnpx, x, posterior, training=False, **kwargs):
        x = skorch.utils.to_tensor(x, device=lnpx.device)

        if isinstance(self.criterion_, torch.nn.Module):
            self.criterion_.train(training)
        # (self, lnpx, x, posterior, model=None, **kwargs)
        if self._pseudo_labels is None:
            cluster_labels = None
        else:
            cluster_labels = to_device(self._pseudo_labels, x.device) 
            
        outputs = self.criterion_(
            lnpx, x,
            posterior,
            cluster_labels=cluster_labels,
            model=self.module_,
            **kwargs
            )

        return outputs

    def train_step(self, batch, **fit_params):
        self.module_.train()
        self.optimizer_.zero_grad()
        Xi, yi= batch
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            y_preds = self.infer(Xi,**fit_params)
            outputs = self.get_loss(y_preds["lnpx"], Xi, posterior=y_preds["posterior"], training=True, topic_proportion=y_preds["topic_proportion"])
        assert not outputs["loss"].isnan()
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
            outputs = self.get_loss(y_preds["lnpx"],Xi, posterior=y_preds["posterior"],training=False, topic_proportion=y_preds["topic_proportion"])
        self.record_to_history(outputs, prefix="valid")
        return dict(y_pred=y_preds["lnpx"], **outputs )

    @torch.no_grad()
    def evaluation_step(self, batch, **fit_params):
        Xi,yi = batch
        self.module_.eval()
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            y_preds = self.infer(Xi,**fit_params)
            outputs = self.get_loss(y_preds["lnpx"],Xi, posterior=y_preds["posterior"],training=False, topic_proportion=y_preds["topic_proportion"])
        #self.record_to_history(aux, prefix="valid")
        return dict(y_pred=y_preds["lnpx"], **outputs )

    def infer(self, x, **fit_params):
        #print(type(x), x)
        x = skorch.utils.to_tensor(x, device=self.device)
        #yi = skorch.utils.to_tensor(yi, device=self.device) if yi is not None else None
        if isinstance(x, dict):
            x_dict = self._merge_x_and_fit_params(x, fit_params)
            return self.module_(**x_dict)
        return self.module_(x,**fit_params)

    @torch.no_grad()
    def get_similar_words(self, queries:list[str], id2word:dict[int,str], topn:int=10)->pd.DataFrame:
        """Get DataFrame fo Query-Anser pairs

        Parameters
        ----------
        queries : list[str]
            list of query
        id2word : dict[int,str]
            id-word pairs
        topn : int, optional
            number of words, by default 10

        Returns
        -------
        pd.DataFrame
            Topic-Word distribution matrix's shape needs be (topic_dim, vocab_size)
        """
        self.module_.eval()
        return get_similar_words(self.module_.get_beta(), queries, id2word, topn)

    @torch.no_grad()
    def get_topic_top_words(self, id2word:dict[int,str], topn:int=10, decode=True)->pd.DataFrame:
        """Get DataFrame of Topic representation words

        Parameters
        ----------
        id2word : dict[int,str]
            id-word pairs
        topn : int, optional
            number of words, by default 10

        Returns
        -------
        pd.DataFrame
            Table of Topic representation words

        Raises
        ------
        ValueError
            Topic-Word distribution matrix's shape needs be (topic_dim, vocab_size)
        """
        self.module_.eval()
        beta = self.get_topic_word_distributions(decode=decode, numpy=False)
        return get_topic_top_words(beta, id2word,topn=topn)

    @torch.no_grad()
    def get_topic_word_distributions(self, safety=True, decode=True, numpy=True):
        self.module_.eval()
        if decode:
            logbeta = self.module_.decoder["decoder"](torch.eye(self.module_.n_components).to(self.device))
            if safety:
                beta = F.normalize(logbeta.exp(), p=1,dim=1)
                return to_numpy(beta) if numpy else beta
            return to_numpy(logbeta.exp()) if numpy else logbeta.exp()
        else:
            beta = self.module_.get_beta()
            return to_numpy(beta) if numpy else beta

    def get_model_outputs(self, X_tr:Any, X_te:Optional[Any]=None, id2word:dict[int,str]=None)->dict[str, Any]:
        """get output for octis.evaluation_metrics

        Parameters
        ----------
        X_tr : Any
            Train dataset or (Sparse) matrix
        X_te : Any
            Test dataset or (Sparse) matrix
        id2word : dict[int,str]
            dictionary of word id and word surface pair.

        Returns
        -------
        dict[str, Any]
            topics: the list of the most significative words foreach topic (list of lists of strings).
            topic-word-matrix: an NxV matrix of weights where N is the number of topics and V is the vocabulary length.
            topic-document-matrix: an NxD matrix of weights where N is the number of topics and D is the number of documents in the corpus.
            if your model supports the training/test partitioning it should also return:
                test-topic-document-matrix: the document topic matrix of the test set.
        """
        outputs = dict()
        outputs["topics"] = self.get_topic_top_words(id2word,topn=50).values.tolist()
        outputs["topic-word-matrix"] = self.get_topic_word_distributions()#to_numpy(self.module_.get_beta())
        outputs["topic-document-matrix"] = self.transform(X_tr, training=False).T
        if X_te is not None:
            outputs["test-topic-document-matrix"] = self.transform(X_te, training=False).T
        return outputs

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

        module.decoder["decoder"].word_embeddings.load_state_dict(
            {'weight': torch.from_numpy(embeddings).to(sample.dtype).to(sample.device)},
            )
        module.decoder["decoder"].word_embeddings.weight.requires_grad = trainable

    def perplexity(self, X:Any,X_target=None, n_samples:int=-1, exp_fn:Callable=torch.exp, )->float:
        """Perplexity

        Parameters
        ----------
        X : Any
            Input data
        n_samples : int, optional
            number of sub-sample data, by default -1
        exp_fn : Callable, optional
            exp_fn(nll.sum()/X.sum()), by default torch.exp

        Returns
        -------
        float
            Perplexity score
        """
        self.module_.train(False)
        if X_target is None:
            if n_samples >=1 and isinstance(X, sparse.csr_matrix):
                X = shuffle(X, n_samples=n_samples)
            reduced_nll = 0.0
            n_tokens = X.sum()

            datasets = self.get_dataset(X)
            for data in self.get_iterator(datasets,):
                xi = data[0].to(self.device)
                output = self.infer(xi)
                lnpx = output["lnpx"]
                reduced_nll += (lnpx * xi).sum() # torch.Tensor
            
            ppl = exp_fn(-reduced_nll / n_tokens)
            return float(to_numpy(ppl))
        else:
            if n_samples >=1 and isinstance(X, sparse.csr_matrix):
                X,X_target = shuffle(X,X_target, n_samples=n_samples)
            reduced_nll = 0.0
            n_tokens = X_target.sum()

            datasets = self.get_dataset(X)
            datasets2 = self.get_dataset(X_target)
            for data,target in zip(self.get_iterator(datasets,),self.get_iterator(datasets2,)):
                xi = data[0].to(self.device)
                yi = target[0].to(self.device)
                output = self.infer(xi)
                lnpx = output["lnpx"]
                reduced_nll += (lnpx * yi).sum() # torch.Tensor
            
            ppl = exp_fn(-reduced_nll / n_tokens)
            return float(to_numpy(ppl))

    def perplexity_from_missing_bow(self, X, n_samples:int=-1, n_experiments=10, exp_fn:Callable=torch.exp, split_rate:float=0.5, seed=None)->float:
        """Perplexity (Reconstruct from missing information)

        Parameters
        ----------
        X : scipy.sparse.csr_matrix
            Input data
        n_samples : int, optional
            number of subsample data, by default -1
        exp_fn : Callable, optional
            exp_fn(nll.sum()/ X.sum()), by default torch.exp
        split_rate: float, optional
            Ratio of the size of the missing document to the length of the document, by default 0.5
        
        Returns
        -------
        float
            Perplexity score
        """
        self.module_.train(False)
        if n_samples >=1 and isinstance(X, sparse.csr_matrix):
            X = shuffle(X, n_samples=n_samples)
        
        seq = bow2seq(X)
        seq1, seq2 = split_seq(seq, rate=split_rate, seed=seed)
        X_input = seq2bow(seq1, shape=X.shape)
        X_target = seq2bow(seq2, shape=X.shape)
        del seq, seq1, seq2
        
        n_tokens = X_target.sum()
        input_datasets = self.get_dataset(X_input)
        target_datasets = self.get_dataset(X_target)
        iters = zip(self.get_iterator(input_datasets),self.get_iterator(target_datasets),)
        
        reduced_nll = 0.0
        for input_data, target_data in iters:
            x_input = input_data[0].to(self.device)
            x_target = target_data[0].to(self.device)
            output = self.infer(x_input)
            lnpx = output["lnpx"]
            reduced_nll += (lnpx * x_target).sum() # torch.Tensor
        
        ppl = exp_fn(-reduced_nll / n_tokens)
        return float(to_numpy(ppl))

    #####
    def fit_loop(self, X,y=None, epochs=None, **fit_params): ##
        self.check_data(X, y)
        epochs = epochs if epochs is not None else self.max_epochs

        dataset_train, dataset_valid = self.get_split_datasets(
            X,y, **fit_params) #wip
        on_epoch_kwargs = {
            'dataset_train': dataset_train,
            'dataset_valid': dataset_valid,
        }

        for _ in range(epochs):
            self.notify('on_epoch_begin', **on_epoch_kwargs)

            self.run_single_epoch(dataset_train, training=True, prefix="train",
                                  step_fn=self.train_step, **fit_params)

            self.run_single_epoch(dataset_valid, training=False, prefix="valid",
                                  step_fn=self.validation_step, **fit_params)

            self.notify("on_epoch_end", **on_epoch_kwargs)
        return self

    def run_single_epoch(self, dataset, training, prefix, step_fn, **fit_params):
        if dataset is None:
            return

        batch_count = 0
        for batch in self.get_iterator(dataset, training=training):
            self.notify("on_batch_begin", batch=batch, training=training)
            step = step_fn(batch, **fit_params)
            self.history.record_batch(prefix + "_loss", step["loss"].item())
            batch_size = (get_len(batch[0]) if isinstance(batch, (tuple, list))
                          else get_len(batch))
            self.history.record_batch(prefix + "_batch_size", batch_size)
            self.notify("on_batch_end", batch=batch, training=training, **step)
            batch_count += 1

        self.history.record(prefix + "_batch_count", batch_count)

    # pylint: disable=unused-argument
    def partial_fit(self, X,y=None, classes=None, **fit_params): ##
        if not self.initialized_:
            self.initialize()

        self.notify('on_train_begin', X=X, y=y)
        try:
            self.fit_loop(X,y, **fit_params) #wip
        except KeyboardInterrupt:
            pass
        self.notify('on_train_end', X=X, y=y)
        return self

    def fit(self, X,y=None, **fit_params): ##
        if not self.warm_start or not self.initialized_:
            self.initialize()

        self.partial_fit(X,y, **fit_params) #wip
        return self
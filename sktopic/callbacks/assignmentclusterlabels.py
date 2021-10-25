from typing import Optional, Any
import numpy as np
import torch 
import torch.nn as nn
from skorch.callbacks import Callback
import sklearn.cluster
import warnings
from skorch.exceptions import SkorchWarning
warnings.filterwarnings('ignore', category=SkorchWarning)

class AssignmentClusterlabelsByMyself(Callback):
    def initialize(self, n_cluster:Optional[int]=None, method:str="SpectralClustering", 
    warm_start:int=0, random_state:int=np.random.randint(0,10000),
    ):
        self.n_cluster = n_cluster
        self.method = method
        self.warm_start = warm_start
        self._current_epoch = 0
        self.random_state = random_state
        return super().initialize()

    def on_train_begin(self, net:Any, X:torch.Tensor=None, y:torch.Tensor=None, **kwargs):
        """callback on_train_begin

        Parameters
        ----------
        net : Any
            skorch Trainer
        X : torch.Tensor, optional
            Input data, by default None
        y : torch.Tensor, optional
            Input label, by default None
        """
        if self.n_cluster is None:
            self.n_cluster = net.module_.n_components
        # initialize clustering method
        #self._method = eval(f"sklearn.cluster.{self.method}")(self.n_cluster, n_jobs=-1)
        self._method = sklearn.cluster.MiniBatchKMeans(self.n_cluster, random_state=self.random_state)
        # override Loss Class Object
        ...

    @torch.no_grad()
    def on_epoch_begin(self, net, dataset_train=None, dataset_valid=None, **kwargs)->None:
        #return super().on_epoch_begin(net, dataset_train=dataset_train, dataset_valid=dataset_valid, **kwargs)
        """(re-)set pseudo-labels for training and evaluation as deepcluster

        Parameters
        ----------
        net : Trainer object
            Skorch Trainer object
        dataset_train : Dataset, optional
            Training Dataset, by default None
        dataset_valid : Dataset, optional
            Evaluation dataset, by default None
        """
        theta = net.transform(dataset_train)
        self._method.fit(theta)
        self._current_epoch +=1

    @torch.no_grad()
    def on_batch_begin(self, net, batch, training=None, **kwargs):
        X,y=batch
        if self.warm_start > self._current_epoch:
            return 
        theta = net.transform(X)
        net._pseudo_labels = torch.from_numpy(self._method.predict(theta)).type(torch.long)

    def on_train_end(self, net, X=None, y=None, **kwargs):
        """Reset labels"""
        net._pseudo_labels = None


class AssignmentClusterlabelsByBert(Callback):
    NotImplemented
    # dataset_trainの形を変えないとBertを使うのは難しいかもしれない

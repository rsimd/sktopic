import numpy as np 
from scipy import sparse
from dataclasses import dataclass
from sklearn.model_selection import train_test_split,StratifiedKFold,KFold
from typing import Optional

class CorpusContainer:
    def __init__(self,
    id2word:dict[int,str],
    word2id:dict[str,int],
    X:sparse.csr_matrix,
    labels: np.ndarray = None,
    corpus: list[str] = None,
    label_names: list[str] = None,
    data_name:Optional[str]=None):
    
        self.id2word = id2word 
        self.word2id = word2id
        self.X = X 
        self.labels = np.array(labels)
        self.corpus = corpus 
        self._splitted = False
        self.label_names = label_names
        if label_names is not None and isinstance(self.labels[0], str):
            self.labels = [self.label_names.index(key) for key in self.labels]
        self.V = self.X.shape[1]
        self.D = self.X.shape[0]
        self.name = data_name

    @property
    def vocab_size(self):
        return self.X.shape[1]
    @property
    def num_docs(self):
        return self.X.shape[0]

    def train_test_split(self, seed=np.random.randint(0,2**10), test_size=0.2):
        index = np.arange(len(self.labels))
        self._seed = seed
        self.X_tr, self.X_te, self.y_tr, self.y_te, self.index_tr, self.index_te= train_test_split(self.X, self.labels, index, random_state=seed)
        self._splitted = True
        return self.X_tr, self.X_te, self.y_tr, self.y_te

    def train_test_split_stratifiedkfold(self,target_fold=0,n_splits=5, seed=111, shuffle=True):
        assert target_fold < n_splits
        assert self.labels is not None

        k_fold = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
        folds = k_fold.split(self.X,self.labels)
        self.index_tr, self.index_te = list(folds)[target_fold]
        self.X_tr, self.X_te = self.X[self.index_tr],self.X[self.index_te]
        self.y_tr, self.y_te = self.labels[self.index_tr],self.labels[self.index_te]
        self._splitted = True
        return self.X_tr, self.X_te, self.y_tr, self.y_te
    
    def train_test_split_kfold(self,target_fold=0,n_splits=5, seed=111, shuffle=True):
        assert target_fold < n_splits
        k_fold = KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
        folds = k_fold.split(self.X,)
        self.index_tr, self.index_te = list(folds)[target_fold]
        self.X_tr, self.X_te = self.X[self.index_tr],self.X[self.index_te]
        if self.labels is not None:
            self.y_tr, self.y_te = self.labels[self.index_tr],self.labels[self.index_te]
            self._splitted = True
            return self.X_tr, self.X_te, self.y_tr, self.y_te
        else:
            self._splitted = True
            return self.X_tr, self.X_te
    
    def get_labels(self):
        if self.labels is None:
            return None

        if self._splitted:
            return np.concatenate([self.y_tr, self.y_te])
        else:
            return self.labels

    @property
    def num_labels(self):
        if self.labels is None:
            return 0
        l = self.get_labels()
        return len(set(l.tolist()))

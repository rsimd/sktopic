import numpy as np 
from scipy import sparse
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

@dataclass 
class CorpusContainer:
    id2word:dict[int,str]
    word2id:dict[str,int]
    X:sparse.csr_matrix 
    labels: np.ndarray = None
    corpus: list[str] = None
    label_names: list[str] = None
    __splitted:bool = False
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
        self.__splitted = True
        return self.X_tr, self.X_te, self.y_tr, self.y_te
    
    def get_labels(self):
        if self.__splitted:
            return np.concatenate([self.y_tr, self.y_te])
        else:
            return self.labels

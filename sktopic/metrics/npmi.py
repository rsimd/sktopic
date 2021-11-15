import numpy as np 
from typing import Optional,Dict, Sequence, Any, List
from tqdm import tqdm 
from scipy import sparse
from scipy.sparse import csr_matrix

class NormalizedPointwiseMutualInformation:
    """Normalized (Pointwise) Mutual Information (NPMI)
    topic_wordsの参照コーパス内での共起回数を元に算出するTopic Coherence
    $NPMI ∈ [-1,1]$, higher is better.
    """
    arg_type = "words"
    name = "NPMI"
    def __init__(
        self, 
        corpus:csr_matrix, 
        word2id:Optional[Dict[str,int]]=None,
        id2word:Optional[Dict[int,str]]=None,
        N:int=10,
        ):
        self.corpus = corpus if isinstance(corpus, csr_matrix) \
            else corpus.tocsr()
        
        self.word2id = word2id
        if word2id is None:
            self.word2id = {v:k for k,v in id2word.items()}

        self.D, self.V = corpus.shape
        self.N = N
        self.lnD = np.log(self.D)
        
        self.mask = corpus > 0 # or 1 ? 
        self.Dw = np.asarray(self.mask.sum(axis=0))[0]
        self.co_occurence = {}

        self.score_average:float
        self.scores:np.ndarray

    def reflesh(self):
        self.Dw = np.asarray(self.mask.sum(axis=0))[0]
        self.co_occurence = {}
        self.score_average = None
        self.scores = None

    def __call__(
        self,
        topic_words:Sequence[Sequence[str]], 
        verbose:bool=True,
        use_cache=True
        ) -> float:
        if not use_cache:
            self.reflesh()
            
        topic_words = [topic[:self.N] for topic in topic_words]
        topic_wordids:List[List[int]] = [[self.word2id[w] for w in line] for line in topic_words]
        del topic_words
        
        K = len(topic_wordids)
        N = self.N
        
        assert sum(N == len(topic) for topic in topic_wordids) == K,\
            "all topic_word list needs to have same number of words."
        
        topic_coherences = []
        num_pair:int = np.triu(np.ones((N,N)),k=1).sum()

        if verbose:
            pbar = tqdm(total=num_pair*K)
        
        for k in range(K):
            wordids = topic_wordids[k]
            tc = 0

            for i, i_id in enumerate(wordids):
                Dwi:int = self.Dw[i_id] # iの出る文の数
                j = i+1
                tmp_tc = 0

                while j < N and j > i:
                    j_id = wordids[j]
                    Dwj = self.Dw[j_id]
                    id_pair = (i_id, j_id)
                    
                    if id_pair in self.co_occurence:
                        Dwiwj = self.co_occurence[id_pair]
                    else:
                        Dwiwj = self.mask[:,i_id].multiply(self.mask[:,j_id]).sum()
                        self.co_occurence[id_pair] = self.co_occurence[(j_id, i_id)] = Dwiwj 
                    
                    if Dwiwj == 0:
                        Fwiwj = -1
                    else:
                        lnp_wi = np.log(Dwi) - self.lnD
                        lnp_wj = np.log(Dwj) - self.lnD
                        lnp_wij = np.log(Dwiwj) - self.lnD
                        Fwiwj = -1
                        Fwiwj += (lnp_wi + lnp_wj) / lnp_wij
                    
                    # update tmp
                    tmp_tc += Fwiwj
                    j+=1
                    verbose and pbar.update(1) # type:ignore
                # end while
                tc += tmp_tc
            # end for - in enumerate(words)
            topic_coherences.append(tc/num_pair)
        # end for - in range(K)    
        score_average = np.mean(topic_coherences)
        verbose and pbar.close() # type:ignore
        
        # Temporarily save the score
        self.score_average = score_average
        self.scores = np.array(topic_coherences)
        return self.score_average
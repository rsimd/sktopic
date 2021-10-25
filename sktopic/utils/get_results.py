from typing import Dict, Any
import numpy as np
from multipledispatch import dispatch
import torch 
import pandas as pd
from skorch.utils import to_numpy

@np.vectorize
def id2word_vectorized(id:int, id2word:Dict[int,str])->str:
    return id2word[id]

def get_topwords(beta:np.ndarray, N:int, id2word:Dict[int,str]) ->np.ndarray:
    """
    φ (K,|V|)
    """
    ids = np.asarray(beta).argsort(axis=-1)[:,-N:][:,::-1]
    return id2word_vectorized(ids, id2word)

# def get_topwords(beta:torch.Tensor, N:int, id2word:dict[int,str])->list[list[str]]:
#     ...

def get_word_similarity(beta:torch.Tensor, query:str, word2id:dict[str,int])->torch.Tensor:
    """Get word similarity vector

    Parameters
    ----------
    beta : torch.Tensor
        topic-word distribution
    query : str
        query word
    word2id : dict[str,int]
        word-id pairs

    Returns
    -------
    torch.Tensor
        similarity vector

    Raises
    ------
    ValueError
        Topic-Word distribution matrix's shape needs be (topic_dim, vocab_size)
    """
    cossim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    if query in word2id:
        q_id = word2id[query]
    else:
        print(f"{query} not in vocaburary")
        return None

    q_vec = beta[q_id]
    results_val = cossim(q_vec[None,:], beta)
    return results_val

def get_similar_words(beta:torch.Tensor, queries:list[str], id2word:dict[int,str], topn:int=10)->pd.DataFrame:
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
    word2id = {v:k for k,v in id2word.items()}
    df = {}
    for query in queries:
        results_val = get_word_similarity(beta, query, word2id)
        if results_val is None:
            continue
        results_score, results_arg = torch.sort(results_val)
        results_arg = to_numpy(results_arg)[::-1][1:topn+1]
        results_score = to_numpy(results_score)[::-1][1:topn+1]
        # results_arg = torch.argsort(results_val).cpu().numpy()[::-1][1:topn+1]
        # results_score = torch.sort(results_val).cpu().numpy()[::-1][1:topn+1]
        results_token = [id2word[key.item()] for key in results_arg]
        # significant_digits:str=".3g"
        df[query] = [f"{token} ({score:.3g})" for (token,score) in zip(results_token,results_score.tolist())]
    return pd.DataFrame(df)

def get_topic_top_words(beta:torch.Tensor, id2word:dict[int,str], topn:int=10)->pd.DataFrame:
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
    res = {}
    for k, beta_k in enumerate(torch.argsort(beta, axis=1)):
        ids = beta_k.cpu().numpy()[::-1][:topn]
        words = [id2word[id] for id in ids]
        res[f"Topic_{k}"] = words
    return pd.DataFrame(res)
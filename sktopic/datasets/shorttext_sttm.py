import numpy as np 
from scipy import sparse
from typing import Any, Sequence, Optional
import pickle 
from omegaconf import OmegaConf
import rootpath
import os 
import pandas as pd 
from pandarallel import pandarallel
from functools import partial

__all__ = [
    "fetch_shortext",
    "fetch_SearchSnippets",
    "fetch_StackOverflow",
    "fetch_Biomedical",
    "fetch_TrecTweet",
    "fetch_GoogleNews",
    "fetch_PascalFlicker",
]

MODULE_PATH = rootpath.detect() + "/sktopic/datasets"

def load_cache(data_home:str)->dict[str,Any]:
    X = sparse.load_npz("/".join(data_home,"corpus.npz"))
    labels = pd.read_csv("/".join(data_home,"labels.txt"), header=None).values.T.squeeze()
    with open("/".join(data_home,"vocab.txt")) as f:
        vocabs = f.readlines()
    id2word = {k:v for k,v in enumerate(vocabs)}
    word2id = {v:k for k,v in id2word.items()}
    outputs = dict(
        X=X, 
        id2word=id2word,
        word2id=word2id, 
        labels=labels,
    )
    return outputs

def load_from_uri(config:dict[str,str], data_home:Optional[str]=None)->dict[str,Any]:
    pandarallel.initialize()
    # download txt
    url = config.corpus
    token_url = config.labels

    df=pd.read_table(url,header=None)
    df.columns = ["doc"]
    labels = pd.read_csv(token_url, header=None).values.T.squeeze()
    
    # split doc
    df["splited"] =  df.doc.parallel_apply(lambda line:line.split())
    D = df.shape[0]
    assert D == len(labels), "corpus and lables needs to be same length."
    
    tokens = [token for line in df.splited for token in line]
    vocabs = set(tokens)
    id2word = {k:v for k,v in enumerate(vocabs)}
    word2id = {v:k for k,v in id2word.items()}
    V = len(id2word)
    del tokens, vocabs

    def tolil(line):
        placeholder = sparse.lil_matrix((1, V), dtype=np.float32)
        for token in line:
            id = word2id[token]
            placeholder[0, id] += 1.0
        return placeholder

    X = sparse.vstack(df["splited"].parallel_apply(tolil))
    X = X.tocsr()
    
    outputs = dict(
        X=X,
        id2word=id2word,
        word2id=word2id, 
        labels=labels,
    )
    if data_home is not None:
        sparse.save_npz(f"{data_home}/corpus",X)
        # ファイルを保存

    return outputs

def fetch_shortext(data_name:str,
    *,
    data_home:str=None,
    subset:str="all",
    download_if_missing:bool=True,
    random_state:Optional[int]=None,
    shuffle:bool=False,
    use_cache:bool=True,
    )->dict[str,Any]:
    """fetch shorttext from https://github.com/qiang2100/STTM

    Parameters
    ----------
    data_name : str
        dataset name
    data_home : str, optional
        Specify another download and cache folder for the datasets. By default all datasets is stored in 'sktopic.__path__/data' subfolders., by default None
    subset : str, optional
        select from [train, test, all] , by default "all"
    download_if_missing : bool, optional
        If False, raise a IOError if the data is not locally available instead of trying to download the data from the source site., by default True
    random_state : Optional[int], optional
        Determines random number generation for dataset shuffling. Pass an int for reproducible output across multiple function calls., by default None
    shuffle : bool, optional
        Whether to shuffle dataset, by default False
    use_cache : bool, optional
        if true, try to load cache from local, else download form web, by default True

    Returns
    -------
    dict[str,Any]
        dataset dictonary

    Raises
    ------
    IOError
        If you try to use a cache file and it can't be loaded, return an error.
    """
    if data_home is None:
        data_home = rootpath.detect() + "/data"
    try:
        os.makedirs(data_home)
    except:
        print(f"File exists: {data_home}")
    if use_cache:
        files = os.listdir(data_home)
        flag = [
            "corpus.npz" in files,
            "labels.txt" in files,
            "vocab.txt" in files,
        ]
        if np.sum(flag) == True:
            outputs = load_cache(data_home)
            return outputs
        elif not download_if_missing:
            raise IOError("The cache files does not exist.")
        else:
            pass
    config = OmegaConf.load(f"{MODULE_PATH}/sourse.yaml")[data_name]
    outputs = load_from_uri(config)
    return outputs


__docstrings = """This function is that made the data_name of the following function unchangeable.
----------------------
""" + fetch_shortext.__doc__

fetch_SearchSnippets = partial(fetch_shortext,data_name="SearchSnippets")
fetch_SearchSnippets.__doc__ = __docstrings

fetch_StackOverflow = partial(fetch_shortext,data_name="StackOverflow")
fetch_StackOverflow.__doc__ = __docstrings

fetch_Biomedical = partial(fetch_shortext,data_name="Biomedical")
fetch_Biomedical.__doc__ = __docstrings

fetch_TrecTweet = partial(fetch_shortext,data_name="TrecTweet")
fetch_TrecTweet.__doc__ = __docstrings

fetch_GoogleNews = partial(fetch_shortext,data_name="GoogleNews")
fetch_GoogleNews.__doc__ = __docstrings

fetch_PascalFlicker = partial(fetch_shortext,data_name="PascalFlicker")
fetch_PascalFlicker.__doc__ = __docstrings

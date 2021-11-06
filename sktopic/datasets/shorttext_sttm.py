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
import os
import pprint
import time
import urllib.error
import urllib.request

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

def download_file(url:str,dst_path:str,mode="w")->None:
    try:
        with urllib.request.urlopen(url) as web_file:
            data = web_file.read()
            if "w" == mode:
                data = data.decode('utf-8')
            with open(dst_path, mode=mode) as local_file:
                local_file.write(data)
    except urllib.error.URLError as e:
        print(e)

def load_cache(data_dir:str)->dict[str,Any]:
    X = sparse.load_npz("/".join([data_dir,"corpus.npz"]))
    labels = pd.read_csv("/".join([data_dir,"labels.txt"]), header=None).values.T.squeeze()
    with open("/".join([data_dir,"vocabs.txt"])) as f:
        vocabs = f.readlines()
    id2word = {k:v for k,v in enumerate(vocabs)}
    word2id = {v:k for k,v in id2word.items()}
    df=pd.read_table("/".join([data_dir,"corpus.txt"]),header=None)
    df.columns = ["doc"]
    outputs = dict(
        X=X, 
        id2word=id2word,
        word2id=word2id, 
        labels=labels,
        corpus = df.doc.to_list()
    )
    return outputs

def load_from_uri(config:dict[str,str], data_home:Optional[str]=None)->dict[str,Any]:
    pandarallel.initialize()
    # download txt
    df=pd.read_table(config.corpus,header=None)
    df.columns = ["doc"]
    labels = pd.read_csv(config.labels, header=None).values.T.squeeze()
    
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
        corpus = df.doc.to_list()
    )
    if data_home is not None:
        # ファイルを保存
        data_path = f"{data_home}/{config.name}"
        try:
            os.makedirs(data_path)
        except:
            pass
        sparse.save_npz(f"{data_path}/corpus",X)
        download_file(config.labels, f"{data_path}/labels.txt")
        download_file(config.corpus, f"{data_path}/corpus.txt")
        with open(f"{data_path}/vocabs.txt", "w") as f:
            vocabs = "\n".join([word for word in word2id.keys()])
            f.write(vocabs)
    return outputs

def fetch_shortext(data_name:str,
    *,
    data_home:str=None,
    download_if_missing:bool=True,
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
    data_dir = "/".join([data_home, data_name])
    if use_cache:
        try:
            outputs = load_cache(data_dir)
            return outputs
        except:
            msg = "The cache files does not exist."
            if download_if_missing:
                print(msg)
            else:
                raise IOError(msg)
    print("Download corpus...")
    config = OmegaConf.load(f"{MODULE_PATH}/sourse.yaml")[data_name]
    outputs = load_from_uri(config, data_home)
    return outputs


__docstrings = \
"""This function is that made the data_name of the following function unchangeable.
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

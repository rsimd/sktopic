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

from scipy.sparse import data
from .shorttext_sttm import download_file
from .shorttext_sttm import load_cache as _load_cache
from .corpus_container import CorpusContainer
__all__ = [
    "fetch_20NewsGroups",
    "fetch_BBCNews", 
    "fetch_DBLP", 
    "fetch_M10",
]

MODULE_PATH = rootpath.detect() + "/sktopic/datasets"


def load_cache(data_dir:str)->dict[str,Any]:
    outputs = _load_cache(data_dir)
    if isinstance(outputs["labels"][0], np.int64):
        return outputs
    outputs["label_names"] = list(set(outputs["labels"]))
    outputs["labels"] = [outputs["label_names"].index(label) for label in outputs["labels"]]
    return outputs

def load_from_uri(data_name,config:dict[str,str], data_home:Optional[str]=None)->dict[str,Any]:
    pandarallel.initialize()
    # download txt
    #files = ["corpus.txt", "labels.txt", "vocabulary.txt", "metadata.json", "corpus.tsv"]
    corpus_path = "/".join([config["data_root"],data_name,"corpus.txt"])
    labels_path = "/".join([config["data_root"],data_name,"labels.txt"])
    #vocab_path = "/".join([config["data_root"],data_name, "vocabulary.txt"])
    meta_path = "/".join([config["data_root"], data_name,"metadata.json"])
    tsv_path = "/".join([config["data_root"], data_name,"corpus.tsv"])

    df=pd.read_table(corpus_path,header=None)
    df.columns = ["doc"]
    labels = pd.read_csv(labels_path, header=None).values.T.squeeze()
    
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
        data_path = f"{data_home}/{data_name}"
        try:
            os.makedirs(data_path)
        except:
            pass
        sparse.save_npz(f"{data_path}/corpus",X)
        download_file(labels_path, f"{data_path}/labels.txt")
        download_file(corpus_path, f"{data_path}/corpus.txt")
        download_file(tsv_path, f"{data_path}/corpus.tsv")
        download_file(meta_path, f"{data_path}/metadata.json")
        
        with open(f"{data_path}/vocabulary.txt", "w") as f:
            vocabs = "\n".join([word for word in word2id.keys()])
            f.write(vocabs)
        if not isinstance(outputs["labels"][0], np.int64):
            with open(f"{data_path}/label_names.txt", "w") as f:
                label_names = list(set(outputs["labels"]))
                f.write("\n".join(label_names))
    if isinstance(outputs["labels"][0], np.int64):
        return outputs

    outputs["label_names"] = label_names
    outputs["labels"] = [label_names.index(label) for label in outputs["labels"]]
    return outputs


def fetch_shortext(data_name:str,
    *,
    data_home:str=None,
    download_if_missing:bool=True,
    use_cache:bool=True,
    )->CorpusContainer:
    """fetch shorttext from https://github.com/MIND-Lab/OCTIS

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
        data_home = rootpath.detect() + "/datasets"
    data_dir = "/".join([data_home, data_name])
    if use_cache:
        try:
            outputs = load_cache(data_dir)
            return CorpusContainer(**outputs)
        except:
            msg = "The cache files does not exist."
            if download_if_missing:
                print(msg)
            else:
                raise IOError(msg)
    print("Download corpus...")
    config = OmegaConf.load(f"{MODULE_PATH}/sourse.yaml")["Octis"]
    outputs = load_from_uri(data_name, config, data_home)

    return CorpusContainer(**outputs)

#TODO labelsをstrからintへ、label_namesを別途用意.

__docstrings = \
"""This function is that made the data_name of the following function unchangeable.
----------------------
""" + fetch_shortext.__doc__

fetch_20NewsGroups = partial(fetch_shortext,data_name="20NewsGroup")
fetch_20NewsGroups.__doc__ = __docstrings

fetch_BBCNews = partial(fetch_shortext,data_name="BBC_news")
fetch_BBCNews.__doc__ = __docstrings

fetch_DBLP = partial(fetch_shortext,data_name="DBLP")
fetch_DBLP.__doc__ = __docstrings

fetch_M10 = partial(fetch_shortext,data_name="M10")
fetch_M10.__doc__ = __docstrings

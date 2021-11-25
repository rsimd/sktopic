from typing import List, Any, Tuple
import numpy as np
import torch
from tqdm import tqdm 
from scipy.sparse import csr_matrix, lil_matrix
import random 

TokenSequence = List[List[int]]
Shape = Tuple[int,int]

def force_flatten(somematrix:np.ndarray)->np.ndarray:
    return np.squeeze(np.asarray(somematrix))

def normalizel2(x:torch.Tensor)->torch.Tensor:
    return x/torch.norm(x, p=2, dim=1, keepdim=True)


def seq2bow(seq:TokenSequence, shape:Shape, dtype=np.float32, verbose:bool=False)->csr_matrix:
    bow = lil_matrix(shape)
    D = shape[0]

    for index, line in (tqdm(enumerate(seq)) if verbose else enumerate(seq)):
        for w in line:
            bow[index, w] +=1

    if dtype is not None:
        bow = bow.astype(dtype)
        
    verbose and pbar.close() # type:ignore
    return bow.tocsr()


def bow2seq(bow:csr_matrix, verbose:bool=False)->TokenSequence:
    """
    bow: scipy.sparse.*_matrix, BoW stype matrix
    """
    bow = bow.tolil()
    D,_ = bow.shape

    seq = []
    for d in (tqdm(range(D)) if verbose else range(D)):
        data = bow.data[d]
        row = bow.rows[d]
        seq.append([])
        for cnt,w in zip(data, row):
            for _ in range(int(cnt)):
                seq[-1].append(w)

    #verbose and pbar.close() # type:ignore
    return seq

def split_seq(seq:TokenSequence, rate:float=0.5, seed:int=None)->Tuple[TokenSequence,TokenSequence]:
    random.seed(seed)
    part1=[]
    part2=[]
    for index in range(len(seq)):
        line = seq[index]
        random.shuffle(line)
        split_point = int(len(line) *rate)
        part1.append(line[:split_point])
        part2.append(line[split_point:])
        
    return part1, part2

def tree_map(func, dict_obj):
    tmp = {}
    for key in dict_obj:
        val = dict_obj[key]
        tmp[key] = func(val)
    return tmp 
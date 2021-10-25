from typing import Sequence, Any,Tuple
import numpy as np
import torch
import torch.nn as nn


def mean_squared_cosine_deviation_among_topics(phi):
    """
    ## Mean Squared Cosine Deviation among Topics
    0に近いほど違うトピックが出ている
    phi: shape(K, V)
    phi: softmax(beta, axis=1)
    """
    CosSim= nn.CosineSimilarity(dim=1, eps=1e-7)
    K = phi.shape[0]
    #tmp = []
    summarized = 0.0
    for i in range(K):
        for j in range(K):
            if j <= i:
                continue
            c = CosSim(phi[i][None,:],phi[j][None,:])
            #tmp.append(c)
            summarized += c**2
    #return torch.sum(torch.stack(tmp)) * (2.0/(K**2.0-K))
    return (summarized * (2.0 / (K**2.0-K)))**0.5

def check_sparsity(para, sparsity_threshold=1e-3):
    num_weights = para.shape[0] * para.shape[1]
    num_zero = (para.abs() < sparsity_threshold).sum().float()
    return num_zero / float(num_weights)

def l1_penalty(para):
    return nn.L1Loss()(para, torch.zeros_like(para))

def update_l1(cur_l1, cur_sparsity, sparsity_target):
    diff = sparsity_target - cur_sparsity
    cur_l1.mul_(2.0 ** diff)


#class TopicDiversity():
def topic_diversity(topic_words:Sequence[Sequence[str]])->Tuple[float, np.ndarray]:
    K = len(topic_words)
    topn = len(topic_words[0])
    td_list = []
    
    for k in range(K):
        target = set(topic_words[k])
        others = set(w for _k, words in enumerate(topic_words) for w in words if _k != k)
        num_unique = len(target-others)
        td = num_unique / topn
        td_list.append(td)
    model_average_score = np.mean(td_list)
    return model_average_score, np.array(td_list)
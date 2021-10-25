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

class MeanSquaredCosineDeviatoinAmongTopics:
    def __init__(self) -> None:
        """Mean Squared Cosine Deviation among Topics.
        The closer the value is to 0, the more diverse the topic distribution can be considered.
        """
        self.similarity_mesure = nn.CosineSimilarity(dim=1, eps=1e-7)
    
    def __call__(self, topic_word_dist:torch.Tensor)->torch.Tensor:
        """ Return the average MSCD score.

        Parameters
        ----------
        topic_word_dist : torch.Tensor
            Size([n_topics, vocab_size])
            Topic-Word Distribution, it doesn't have to be a probability distribution.
        
        Returns
        -------
        torch.Tensor
            Size([]) ==scalar
            Average of the scores of all topics
        """
        
        K = topic_word_dist.shape[0]
        #tmp = []
        summarized = 0.0
        for i in range(K):
            for j in range(K):
                if j <= i:
                    continue
                c = self.similarity_mesure(
                    topic_word_dist[i][None,:],
                    topic_word_dist[j][None,:],
                    )
                #tmp.append(c)
                summarized += c**2
        #return torch.sum(torch.stack(tmp)) * (2.0/(K**2.0-K))
        return (summarized * (2.0 / (K**2.0-K)))**0.5



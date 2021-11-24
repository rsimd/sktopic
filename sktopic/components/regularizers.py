from typing import Any, Sequence, Optional
import torch 
import torch.nn as nn

def cosine_similarity_matrix(matrix:torch.Tensor) -> torch.Tensor:
    """get cosine similarity matrix(N,N) from input matrix(N,M).
    ※ The code of this function was obtained from:
    https://qiita.com/fam_taro/items/dac3b1bcfc01461a0120
    """
    #d = matrix @ matrix.T
    #norm = (matrix**2).sum(axis=1, keepdims=True) ** 0.5
    #s = d / norm / norm.T # (N, N)
    matrix = matrix.T
    norm = (matrix * matrix).sum(0, keepdim=True) ** .5
    m_norm = matrix/norm
    S = m_norm.T @ m_norm
    return S

def topic_embeddings_diversity(topic_embeddings:torch.Tensor)->torch.Tensor:
    """Regularizer based on Topic embeddings diversity

    Parameters
    ----------
    topic_embeddings : torch.Tensor
        Topic Embedding Vectors, (n_topics, embed_dim)

    Returns
    -------
    torch.Tensor
        score
    """
    ρ = topic_embeddings
    K = topic_embeddings.size(0)

    θ = cosine_similarity_matrix(ρ).clip(-1,1).arccos()
    ζ = θ.sum() / K**2
    ν =  ((θ - ζ)**2).sum() / K**2
    diversity = ζ - ν
    return -diversity

class RegularizerUsingTopicEmbeddingsDiversity(nn.Module):
    def forward(self, topic_embeddings:torch.Tensor)->torch.Tensor:
        return topic_embeddings_diversity(topic_embeddings)
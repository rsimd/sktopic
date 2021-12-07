import os 
import numpy as np 
import torch 
from typing import Any, Optional
from ..datasets.corpus_container import CorpusContainer
from gensim.models import KeyedVectors
from tqdm import tqdm 
from octis.evaluation_metrics import (
    topic_significance_metrics,classification_metrics,
    similarity_metrics,diversity_metrics,coherence_metrics)
from sktopic.metrics import MeanSquaredCosineDeviatoinAmongTopics
from sktopic.metrics.purity import Purity, NormalizedMutualInformation
import yaml 
import pickle
import torch.nn.functional as F
from sktopic.metrics.diversity import WETopicCentroidDistance
from sktopic.metrics.wecoherence import WECoherenceCentroid,WECoherencePairwise


def get_kv(cfg_path="/workdir/datasets.yaml"):
    outputs = {}
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    for key,val in cfg["embeddings"].items():
        save_path =os.path.join(cfg["env"]["data_root"], val["file"])
        with open(save_path, "br") as f:
            wv = pickle.load(f)
        outputs[key] = wv
    return outputs

def override_word_embeddings(trainer:Any, dataset:CorpusContainer, keyedvectors:Optional[KeyedVectors]=None,train_embed:Optional[bool]=None,normalize=False)->Any:
    self = trainer
    def get_emb(dataset, kv=keyedvectors,sample_token="test"):
        embed_dim = len(kv[sample_token])
        id2word = dataset.id2word
        word_embeddings = []
        num_unk = 0
        for key in id2word:
            if key in kv:
                word_embeddings.append(kv[key])
            else:
                word_embeddings.append(np.zeros(embed_dim,dtype=np.float32))
                num_unk += 1
        word_embeddings = np.vstack(word_embeddings)
        print("Number of Unknown words:", num_unk)
        return torch.from_numpy(word_embeddings).type(torch.float32)
    E = get_emb(dataset)
    if normalize:
        E = F.normalize(E,p=2,dim=1)
    try:
        print("module has word_embeddings")
        assert self.module.word_embeddings.weight.data.shape == E.shape
        self.module.word_embeddings.weight.data = E
        if train_embed is not None:
            self.module.word_embeddings.weight.requires_grad_(train_embed)
    except:
        print("module.decoder['decoder'] has word_embeddings")
        assert self.module.decoder["decoder"].word_embeddings.weight.data.shape == E.shape
        self.module.decoder["decoder"].word_embeddings.weight.data = E
        if train_embed is not None:
            self.module.decoder["decoder"].word_embeddings.weight.requires_grad_(train_embed)
    return self

def eval_all(trainer:Any, dataset: CorpusContainer)->dict[str,float]:
    self = trainer
    def get_all_metrics(dataset,texts_external=None, 
        kv_path_dict:dict[str,str]={},
        dummy_kv_path:str="/workdir/datasets/dummy_kv.txt"):
        print("(1/3) Loading defaults------------------------------------")
        texts = [line.split() for line in dataset.corpus]
        
        #similarity_metrics.WordEmbeddingsCentroidSimilarity.binary = False # because https://github.com/MIND-Lab/OCTIS/issues/41 closed
        #similarity_metrics.WordEmbeddingsPairwiseSimilarity.binary = False # because https://github.com/MIND-Lab/OCTIS/issues/41 closed
        outputs = dict(
            F1_lsvm=classification_metrics.F1Score(dataset, average="macro"),
            Precision_lsvm=classification_metrics.PrecisionScore(dataset, average="macro"),
            Recall_lsvm=classification_metrics.RecallScore(dataset, average="macro"),
            ACC_lsvm=classification_metrics.AccuracyScore(dataset, average="macro"),
            F1_rbfsvm=classification_metrics.F1Score(dataset, average="macro",kernel="rbf"),
            Precision_rbfsvm=classification_metrics.PrecisionScore(dataset, average="macro",kernel="rbf"),
            Recall_rbfsvm=classification_metrics.RecallScore(dataset, average="macro",kernel="rbf"),
            ACC_rbfsvm=classification_metrics.AccuracyScore(dataset, average="macro",kernel="rbf"),
            Cocherence_umass=coherence_metrics.Coherence(measure='u_mass',texts=texts),
            Cocherence_npmi=coherence_metrics.Coherence(measure='c_npmi',texts=texts),
            TopicDiversity=diversity_metrics.TopicDiversity(),
            External_InvertedRBO=diversity_metrics.InvertedRBO(),
            #LogOddsRatio=diversity_metrics.LogOddsRatio(),
            #TopicDistributions_KLDivergence=diversity_metrics.KLDivergence(),
            #Internal_RBO=similarity_metrics.RBO(), # InvertedRBOの逆。ほんとにそのまんまで意味無し
            #KL_uniform=topic_significance_metrics.KL_uniform(),
            #KL_vacuous=topic_significance_metrics.KL_vacuous(),
            KL_background=topic_significance_metrics.KL_background(),
            MeanSquaredCosineDeviatoinAmongTopics=MeanSquaredCosineDeviatoinAmongTopics(),
            Purity=Purity(dataset),
            NMI=NormalizedMutualInformation(dataset),
            )
            
        if texts_external is not None:
            print("(2/3) Adding Wiki based COH-------------------------------")
            for key,_texts in texts_external:
                tmp = {
                    f"Cocherence_cv_{key}":coherence_metrics.Coherence(measure='c_v',texts=_texts),
                    f"Cocherence_uci_{key}":coherence_metrics.Coherence(measure='c_uci',texts=_texts),
                    f"Cocherence_npmi_{key}":coherence_metrics.Coherence(measure='c_npmi',texts=_texts),
                }
            outputs.update(tmp)
        
        print("(3/3) Adding WETC-----------------------------------------")
        def get_wc_metrics_(suffix=None, wv=None):
            tmp = dict(
            WECoherencePairwise=coherence_metrics.WECoherencePairwise(word2vec_path=dummy_kv_path),
            WECoherenceCentroid=WECoherenceCentroid(word2vec_path=dummy_kv_path),               
            #External_WordEmbeddingsInvertedRBO=diversity_metrics.WordEmbeddingsInvertedRBO(word2vec_path=dummy_kv_path),
            #External_WordEmbeddingsInvertedRBOCentroid=diversity_metrics.WordEmbeddingsInvertedRBOCentroid(word2vec_path=dummy_kv_path),
            #Internal_WordEmbeddingsRBOMatch=similarity_metrics.WordEmbeddingsRBOMatch(word2vec_path=dummy_kv_path),
            #Internal_WordEmbeddingsRBOCentroid=similarity_metrics.WordEmbeddingsRBOCentroid(word2vec_path=dummy_kv_path),
            #Internal_WordEmbeddingsPairwiseSimilarity=similarity_metrics.WordEmbeddingsPairwiseSimilarity(word2vec_path=dummy_kv_path),
            #nternal_WordEmbeddingsCentroidSimilarity=similarity_metrics.WordEmbeddingsCentroidSimilarity(word2vec_path=dummy_kv_path),
            WETopicCentroidDistance=WETopicCentroidDistance(word2vec_path=dummy_kv_path),
            )
            if suffix==None:
                a = coherence_metrics.WECoherencePairwise()
                wv = a._wv
                for k,v in tmp.items():
                    tmp[k]._wv = wv
                outputs.update(tmp)
            else:
                # only for Loading file of glove format embeddings 
                #wv = KeyedVectors.load_word2vec_format(kv_path, binary=False, no_header=True) # kv_path
                tmp2 = {}
                for k,v in tmp.items():
                    tmp[k]._wv = wv
                    tmp2[k + "_" + suffix] = tmp[k]
                outputs.update(tmp2)
                
        pbar = tqdm(total=1 + len(kv_path_dict))
        #get_wc_metrics_()
        pbar.update()
        for (suffix, wv) in kv_path_dict.items():
            get_wc_metrics_(suffix,wv)
            pbar.update()
        return outputs

    def eval(metrics_dict, output):
        results = {}
        for key,method in tqdm(metrics_dict.items()):
            try:
                results[key] = method.score(output)
            except Exception as e:
                print(e)
                results[key] = np.nan
        return results

    a = get_all_metrics(dataset,None,get_kv())
    output = self.get_model_outputs(X_tr=dataset.X_tr, X_te=dataset.X_te, id2word=dataset.id2word)
    r = eval(a,output)

    r["Perplexity"] = ppl = self.perplexity(dataset.X_te)
    r["NormalizedPerplexity"] = ppl / len(dataset.id2word)
    r["PerplexityRecons"] = recons_ppl = self.perplexity_from_missing_bow(dataset.X_te)
    r["NormalizedPerplexityRecons"] = recons_ppl / len(dataset.id2word)
    return r 
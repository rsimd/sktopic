from typing import Sequence,Tuple
import numpy as np 

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

class TopicDiversity:
    def __init__(self, topn=25):
        self.topn = topn
        self._each_topic_score = []

    def __call__(self,topic_words:Sequence[Sequence[str]])->float:
        """Return average score

        Parameters
        ----------
        topic_words : Sequence[Sequence[str]]
            List of Topic represented words. 
            ex. [[str,...],[str,...],...]
            
        Returns
        -------
        float
            score average
        """

        K = len(topic_words)
        N = len(topic_words[0])
        assert N>=self.topn, f"number of topic words>={self.topn}"
        _topic_words = topic_words[:self.topn]
        td_list = []
        
        for k in range(K):
            target = set(_topic_words[k])
            others = set(w for _k, words in enumerate(_topic_words) for w in words if _k != k)
            num_unique = len(target-others)
            td = num_unique / self.topn
            td_list.append(td)
        
        self._each_topic_score = np.array(td_list)
        model_average_score = np.mean(td_list)
        return model_average_score


from octis.evaluation_metrics.coherence_metrics import AbstractMetric
from gensim.models import KeyedVectors
import gensim.downloader as api
import octis.configuration.citations as citations
from scipy import spatial

class WETopicCentroidDistance(AbstractMetric):
    def __init__(self, topk=10, word2vec_path=None, binary=True):
        """
        Initialize metric

        Parameters
        ----------
        topk : how many most likely words to consider
        w2v_model_path : a word2vector model path, if not provided, google news 300 will be used instead
        """
        super().__init__()

        self.topk = topk
        self.binary = binary
        self.word2vec_path = word2vec_path
        if self.word2vec_path is None:
            self._wv = api.load('word2vec-google-news-300')
        else:
            self._wv = KeyedVectors.load_word2vec_format(
                self.word2vec_path, binary=self.binary)

    @staticmethod
    def info():
        return {
            "citation": citations.em_word_embeddings_pc,
            "name": "Coherence word embeddings centroid"
        }

    def score(self, model_output):
        """
        Retrieve the score of the metric

        :param model_output: dictionary, output of the model. key 'topics' required.
        :return topic coherence computed on the word embeddings

        """
        topics = model_output["topics"]
        if self.topk > len(topics[0]):
            raise Exception('Words in topics are less than topk')
        else:
            T = []
            for topic in topics:
                E = []
                # average vector of the words in topic (centroid)
                t = np.zeros(self._wv.vector_size)

                # Create matrix E (normalize word embeddings of
                # words represented as vectors in wv) and
                # average vector of the words in topic
                for word in topic[0:self.topk]:
                    if word in self._wv.key_to_index.keys():
                        word_embedding = self._wv.__getitem__(word)
                        normalized_we = word_embedding/np.sum(word_embedding)
                        E.append(normalized_we)
                        t = list(map(np.add, t, word_embedding))
                t = np.array(t)
                t = t/(len(t)*np.sum(t))
                T.append(t)

            result = 0
            K = len(topics)
            for i in range(K):
                distance = 0
                for j in range(K):
                    if i == j:
                        continue
                    distance += spatial.distance.cosine(T[i], T[j])
                    #result += distance
                result += distance / (K-1) 
            result /= K
            return result
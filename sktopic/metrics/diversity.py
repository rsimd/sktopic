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
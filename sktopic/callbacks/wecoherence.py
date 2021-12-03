from typing import Callable, Optional
import numpy as np
from numpy.core.numeric import binary_repr
from numpy.lib.twodim_base import triu
from skorch.callbacks import Callback, EpochScoring
from skorch.callbacks.scoring import _cache_net_forward_iter, convert_sklearn_metric_function
from torch.optim.optimizer import Optimizer
from octis.evaluation_metrics import coherence_metrics
from octis.evaluation_metrics import diversity_metrics
from sktopic.metrics import wecoherence as coherence_metrics2

"""
>>> def my_score(net, X=None, y=None):
...     losses = net.history[-1, 'batches', :, 'my_score']
...     batch_sizes = net.history[-1, 'batches', :, 'valid_batch_size']
...     return np.average(losses, weights=batch_sizes)
>>> net = MyNet(callbacks=[
...     ('my_score', Scoring(my_score, name='my_score'))
"""

__all__ = ["TopicScoring", "WECoherenceScoring","TopicDiversityScoring","TopicQualityScoring"]

class TopicScoring(EpochScoring):
    def __init__(self, scoring_fn, id2word:dict[int,str], topn:int=10, lower_is_better=False, name=None):
        self.topn = topn
        self.id2word = id2word
        self.scoring_fn = scoring_fn
        #self.diversitymodel = diversity_metrics.TopicDiversity(topk=self.topn)
        super().__init__(self._scoring, lower_is_better=lower_is_better, on_train=False, name=name, target_extractor=..., use_caching=True)

    def _get_output(self, net)-> float:
        topic_words = net.get_topic_top_words(self.id2word,topn=self.topn, decode=False)
        topic_words = topic_words.values.tolist()
        
        model_output = {"topics": topic_words}
        return model_output

    def _scoring(self, net, X=None, y=None)-> float:
        model_output = self._get_output(net)
        score = self.scoring_fn(model_output)
        return score
    
    # pylint: disable=unused-argument,arguments-differ
    def on_batch_end(self,net=None,X=None,y=None,**kwargs):
        return

    # pylint: disable=unused-argument,arguments-differ
    def on_epoch_end(self,net, dataset_train=None,dataset_valid=None,**kwargs):
        current_score = self._scoring(net)
        self._record_score(net.history, current_score)


class WECoherenceScoring(TopicScoring):
    def __init__(self, id2word:dict[int,str], method:str="pairwise", kv_path:Optional[str]=None, topn:int=10, binary: bool=True, coherence_object=None):
        self.kv_path = kv_path
        self.method = method
        self.binary = binary
             
        if coherence_object is None:
            if self.method == "centroid":
                self.coherencemodel = coherence_metrics2.WECoherenceCentroid(word2vec_path=self.kv_path,binary=self.binary,topk=topn)
                _name = "wetc_c"
            elif self.method == "pairwise":
                self.coherencemodel = coherence_metrics.WECoherencePairwise(word2vec_path=self.kv_path,binary=self.binary,topk=topn)
                _name = "wetc_pw"
            else:
                NotImplementedError
        else:
            self.coherencemodel = coherence_object
            if self.method == "centroid":
                _name = "wetc_c"
            elif self.method == "pairwise":
                _name = "wetc_pw"
        super().__init__(self.coherencemodel.score,id2word,topn,lower_is_better=False, name=_name)

class TopicDiversityScoring(TopicScoring):
    def __init__(self, id2word: dict[int, str], topn: int = 10):
        name = "td"
        lower_is_better=False
        self.model = diversity_metrics.TopicDiversity(topn)
        scoring_fn = self.model.score
        super().__init__(scoring_fn, id2word, topn=topn, lower_is_better=lower_is_better, name=name)

class TopicQualityScoring(TopicScoring):
    def __init__(self, id2word:dict[int,str], method:str="pairwise", kv_path:Optional[str]=None, topn:int=10, binary: bool=True, coherence_object=None):
        self.coherence = WECoherenceScoring(id2word,method,kv_path,topn,binary,coherence_object)
        self.diversity = TopicDiversityScoring(id2word,topn)
        self.scoring_fn = lambda x:self.coherence.scoring(x)*self.diversity.scoring(x)
        super().__init__(self.scoring_fn, id2word, topn=topn, lower_is_better=False, name="tq")

    def on_epoch_end(self,net, dataset_train=None,dataset_valid=None,**kwargs):
        current_score = self._scoring(net)
        self._record_score(net.history, current_score)



# class WECoherenceScoring(EpochScoring):
#     def __init__(self, id2word:dict[int,str], method:str="pairwise", kv_path:Optional[str]=None, topn:int=10, binary: bool=True):
#         self.kv_path = kv_path
#         self.method = method
#         self.binary = binary
#         self.topn = topn
#         self.id2word = id2word

#         if self.method == "centroid":
#             self.coherencemodel = coherence_metrics.WECoherenceCentroid(word2vec_path=self.kv_path,binary=binary,topk=self.topn)
#             _name = "wetc_c"
#         elif self.method == "pairwise":
#             self.coherencemodel = coherence_metrics.WECoherencePairwise(word2vec_path=self.kv_path,binary=self.binary,topk=self.topn)
#             _name = "wetc_pw"
#         else:
#             NotImplementedError
#         #self.diversitymodel = diversity_metrics.TopicDiversity(topk=self.topn)
#         super().__init__(self._scoring, lower_is_better=False, on_train=False, name=_name, target_extractor=..., use_caching=True)

#     def _get_output(self, net)-> float:
#         topic_words = net.get_topic_top_words(self.id2word,topn=self.topn)
#         topic_words = topic_words.values.tolist()
        
#         model_output = {"topics": topic_words}
#         return model_output

#     def _scoring(self, net, X=None, y=None)-> float:
#         model_output = self._get_output(net)
#         tc = self.coherencemodel.score(model_output)
#         return tc
    
#     # pylint: disable=unused-argument,arguments-differ
#     def on_batch_end(self,net,X=None,y=None,**kwargs):
#         return

#     # pylint: disable=unused-argument,arguments-differ
#     def on_epoch_end(self,net,dataset_train=None,dataset_valid=None,**kwargs):
#         current_score = self._scoring(net)
#         self._record_score(net.history, current_score)

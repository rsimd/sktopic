from typing import Sequence
import numpy as np 
from sklearn.metrics import accuracy_score, cluster
from sklearn.svm import LinearSVC
from sklearn.metrics import normalized_mutual_info_score


__all__ = ["Purity", "NormalizedMutualInformation"]
# ------------------------------- #
"""
The MIT License (MIT)
Copyright (c) 2017 David Mugisha
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""    

def _purity_score(y_true, y_pred):
    """Purity score
    To compute purity, each cluster is assigned to the class which is most frequent 
    in the cluster [1], and then the accuracy of this assignment is measured by counting 
    the number of correctly assigned documents and dividing by the number of documents.
    We suppose here that the ground truth labels are integers, the same with the predicted clusters i.e
    the clusters index.
    Args:
        y_true(np.ndarray): n*1 matrix Ground truth labels
        y_pred(np.ndarray): n*1 matrix Predicted clusters
    
    Returns:
        float: Purity score
    
    References:
        [1] https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html
    """
    # matrix which will hold the majority-voted labels
    y_voted_labels = np.zeros(y_true.shape)
    # Ordering labels
    ## Labels might be missing e.g with set like 0,2 where 1 is missing
    ## First find the unique labels, then map the labels to an ordered set
    ## 0,2 should become 0,1
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true==labels[k]] = ordered_labels[k]
    # Update unique labels
    labels = np.unique(y_true)
    # We set the number of bins to be n_classes+2 so that 
    # we count the actual occurence of classes between two consecutive bin
    # the bigger being excluded [bin_i, bin_i+1[
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred==cluster], bins=bins)
        # Find the most present label in the cluster
        winner = np.argmax(hist)
        y_voted_labels[y_pred==cluster] = winner
    
    return accuracy_score(y_true, y_voted_labels)


# add 2021/04/02
def purity(labels_true:Sequence[int], labels_pred:Sequence[int])->float:
    """wrapper of `purity_score`
    original: https://gist.github.com/jhumigas/010473a456462106a3720ca953b2c4e2
    
    y_true(np.ndarray): n*1 matrix Ground truth labels
    y_pred(np.ndarray): n*1 matrix Predicted clusters
    example:
    >>>cluster = [0,0,0,0,0,0, 1,1,1,1,1,1, 2,2,2,2,2]
    >>>label =   [3,3,3,3,3,4, 3,4,4,4,4,5, 3,3,5,5,5]
    >>>purity(label, cluster)
    #0.7058823529411765
    """ 
    return _purity_score(
        np.asarray(labels_true),
        np.asarray(labels_pred),
        )


class Purity():
    def __init__(self,dataset, clustering=None):
        self.dataset = dataset
        self.clustering = clustering

    def __call__(self, topic_proportion:Sequence[int],true_labels:Sequence[int])->float:
        """wrapper of `purity_score`
        original: https://gist.github.com/jhumigas/010473a456462106a3720ca953b2c4e2
        
        y_true(np.ndarray): n*1 matrix Ground truth labels
        y_pred(np.ndarray): n*1 matrix Predicted clusters
        example:
        >>>cluster = [0,0,0,0,0,0, 1,1,1,1,1,1, 2,2,2,2,2]
        >>>label =   [3,3,3,3,3,4, 3,4,4,4,4,5, 3,3,5,5,5]
        >>>purity(label, cluster)
        #0.7058823529411765
        """ 
        top_topics = np.argmax(topic_proportion, axis=-1)
        return _purity_score(
            y_true = np.asarray(true_labels),
            y_pred = top_topics,
        )

    def score(self,model_output):
        if self.clustering is None:
            return self.__call__(model_output["topic-document-matrix"].T, self.dataset.y_tr)
        
        if self.clustering == "kmeans":
            K = model_output["topic-document-matrix"].shape[0]
            self._clustering = KMeans(K)
            self._clustering.fit(model_output["topic-document-matrix"].T)
            pred_labels = self._clustering.labels_
            return _purity_score(
                y_true = np.asarray(self.dataset.y_tr),
                y_pred = pred_labels,
                )
        NotImplementedError

from sklearn.cluster import KMeans
class NormalizedMutualInformation():
    def __init__(self,dataset, clustering=None):
        self.dataset = dataset
        self.clustering = clustering

    def __call__(self, topic_proportion:np.ndarray,true_labels:Sequence[int])->float:
        top_topics = np.argmax(topic_proportion, axis=-1)
        return normalized_mutual_info_score(
            labels_true = np.asarray(true_labels),
            labels_pred = top_topics,
        )
    def score(self,model_output):
        if self.clustering is None:
            return self.__call__(model_output["topic-document-matrix"].T, self.dataset.y_tr)

        if self.clustering == "kmeans":
            K = model_output["topic-document-matrix"].shape[0]
            self._clustering = KMeans(K)
            self._clustering.fit(model_output["topic-document-matrix"].T)
            pred_labels = self._clustering.labels_
            return normalized_mutual_info_score(
                labels_true = np.asarray(self.dataset.y_tr),
                labels_pred = pred_labels,
                )
        NotImplementedError

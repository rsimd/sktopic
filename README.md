# sktopic
Implementation of Neural Topic Models that can be used like sklearn based on pytorch/skorch. 

(Work In Progress)

## Installation
 
```bash
pip install git+https://github.com/rsimd/sktopic.git
```

## Dataset
```python
from sktopics import datasets
datasets.fetch_20NewsGroups
datasets.fetch_BBCNews
datasets.fetch_Biomedical
datasets.fetch_DBLP
datasets.fetch_GoogleNews
datasets.fetch_M10
datasets.fetch_PascalFlicker
datasets.fetch_SearchSnippets
datasets.fetch_StackOverflow
datasets.fetch_TrecTweet
```

## Models
```python
from sktopics import models

models.GaussianSoftmaxModel
models.GaussianStickBreakingModel
models.RecurrentStickBreakingModel
models.NeuralVariationalLatentDirichletAllocation
models.ProductOfExpertsLatentDirichletAllocation
models.NeuralSinkhornTopicModel
models.WassersteinLatentDichchletAllocation
```


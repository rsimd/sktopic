import sys
if ".." not in sys.path:
    sys.path.append("/workdir/")
from sktopic import datasets
import torch 
import skorch 
import sktopic 
from tqdm import tqdm 
from sktopic.utils.model_utils import get_kv, eval_all, override_word_embeddings
import wandb 
dataset = datasets.fetch_M10()
#dataset.train_test_split_stratifiedkfold()
dataset.train_test_split()
_WECoherenceScoring = sktopic.callbacks.WECoherenceScoring(dataset.id2word,kv_path="/workdir/datasets/dummy_kv.txt",binary=False)
_kvs = get_kv()
_WECoherenceScoring.coherencemodel._wv = _kvs["glove.42B.300d"]

import time
from sktopic import models

sktopic.utils.manual_seed(int(time.time())) #1637809708)#
#sktopic.utils.manual_seed(1637805045) #e44

V = dataset.vocab_size
print("shape=",dataset.X.shape, "labels=",dataset.num_labels)
K = 50
L = 300
M = 2000

use_pwe= True
train_embed=True
lr     = 0.001
lr_dec = 0.0001

callbacks = [
        _WECoherenceScoring, #WECoherenceScoring(dataset.id2word),
        sktopic.callbacks.TopicDiversityScoring(dataset.id2word),
        sktopic.callbacks.NaNChecker(),
        #EarlyStopping(patience=10,threshold=1e-6),
        #LRScheduler(),
        skorch.callbacks.GradientNormClipping(gradient_clip_value=0.25),
        #Freezer(lambda x:x.endswith("word_embeddings.weight"))
        ]

m = models.ProductOfExpertsLatentDirichletAllocation(
    V,K,
    embed_dim=L,
    hidden_dims=[500]*2,
    dropout_rate_theta=0.2,
    dropout_rate_hidden=0.2,
    callbacks=callbacks,
    batch_size=M,
    max_epochs= 5,
    n_sampling= 3,
    activation_hidden="SiLU",
    device="cuda",
    lr=lr,
    optimizer=torch.optim.ASGD,
    optimizer__param_groups = [
        ("*.decoder.*",  {"lr":lr_dec}),
        ("*.topic_embeddings.*", {"lr":lr_dec/3, "weight_decay":1e-5}),
        ("*.word_embeddings.*", {"lr":lr_dec/2})
    ],
    criterion__arccos_lambda = 10.0,
    criterion__l2_lambda = 0.001,
    #criterion__prior_name="dirichlet",#"gmm_ctm",
    #topic_model=False,
    )
if use_pwe:
    from sktopic.utils.model_utils import override_word_embeddings
    m = override_word_embeddings(m,dataset,_kvs[f"glove.6B.{L}d"],normalize=True,train_embed=train_embed)

# print(m.module.get_beta().sum())
m.fit(dataset.X_tr)
# print(m.module_.get_beta().sum())
# m.partial_fit(dataset.X_tr[:32])
# m.module_.decoder["decoder"].word_embeddings.weight.requires_grad
# results = eval_all(m,dataset)
# m.module.get_beta().sum()
# alpha = m.module.decoder["decoder"].topic_embeddings.weight
# alpha
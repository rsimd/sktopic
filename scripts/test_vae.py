import sys
if ".." not in sys.path:
    sys.path.append("/workdir/")

from sktopic import datasets
from sktopic import models 
import sktopic
import wandb
from sktopic.callbacks import WECoherenceScoring,TopicDiversityScoring
from skorch.callbacks import WandbLogger
from sktopic.utils.model_utils import get_kv,eval_all

def main(K,L, dataset,MODEL,
        batch_size=2000,
        max_epochs=1,
        dropout_rate_hidden=0.5,
        device="cpu",
        #criterion__prior_name="dirichlet",#"gmm_ctm",
        lr=0.05,
        use_pwe=False, train_pwe=False, exp_seed=0,data_seed=0):
    #from sklearn.model_selection import train_test_split
    #dataset = DATASET()
    dataset.train_test_split_stratifiedkfold(data_seed)
    _WECoherenceScoring = WECoherenceScoring(dataset.id2word,kv_path="/workdir/datasets/dummy_kv.txt",binary=False)
    _kvs = get_kv()
    _WECoherenceScoring.coherencemodel._wv = _kvs["glove.6B.50d"]
    seed = sktopic.utils.manual_seed(exp_seed)
    # wandb logger
    wandb_run = wandb.init(project="sktopic", entity="rsimd")
    wandb_run.config.update(
        dict(K=K,L=L,model=MODEL.__name__, dataset=dataset.name,
        batch_size=batch_size,max_epochs=max_epochs,
        dropout_rate_hidden=dropout_rate_hidden,
        device=device,lr=lr,use_pwe=use_pwe, train_pwe=train_pwe,
        pwe="glove.42B.300d" if use_pwe else None,
        exp_seed=exp_seed, data_seed=data_seed,
    ))
    callbacks = [
            _WECoherenceScoring, #WECoherenceScoring(dataset.id2word),
            TopicDiversityScoring(dataset.id2word),
            WandbLogger(wandb_run),
            #EarlyStopping(patience=10,threshold=1e-6),
            #LRScheduler(),
            #GradientNormClipping(gradient_clip_value=1.0)
            ]
    V = dataset.vocab_size
    m = MODEL(V,K,embed_dim=L,hidden_dims=[500,500],
        #dropout_rate_theta=0.5,
        dropout_rate_hidden=dropout_rate_hidden,
        callbacks=callbacks,
        batch_size=batch_size,
        max_epochs=max_epochs,
        device=device,
        #criterion__prior_name="dirichlet",#"gmm_ctm",
        lr=lr,
        #topic_model=False,
        verbose=False,
        )

    if use_pwe:
        from sktopic.utils.model_utils import override_word_embeddings
        m = override_word_embeddings(m,dataset,_kvs["glove.42B.300d"],train_embed=train_pwe)

    m.fit(dataset.X_tr)
    results = eval_all(m,dataset)
    
    wandb_run.log(results)
    wandb_run.finish()
    return (m, m.history_, results)
    
DATASETs = [
datasets.fetch_20NewsGroups,datasets.fetch_BBCNews,datasets.fetch_Biomedical,
datasets.fetch_DBLP,datasets.fetch_GoogleNews,datasets.fetch_M10, 
datasets.fetch_PascalFlicker,datasets.fetch_SearchSnippets, datasets.fetch_StackOverflow,
datasets.fetch_TrecTweet,
]
MODELs = [
models.NeuralVariationalDocumentModel, 
models.ProductOfExpertsLatentDirichletAllocation,
models.GaussianSoftmaxModel, 
models.GaussianStickBreakingModel,
models.RecurrentStickBreakingModel,
models.NeuralSinkhornTopicModel,
models.WassersteinLatentDirichletAllocation,
]
Ks = [10,20,50,100,200]
Ls = [None,300]
for L in Ls:
    for use_pwe in [(False,False),(True,False),(True,True)]:
        for K in Ks:
            for DATASET in DATASETs:
                for MODEL in MODELs:
                    dataset = DATASET()
                    m, history,results = main(K,L,dataset,MODEL, use_pwe=use_pwe[0],train_pwe=use_pwe[1])
                    model_outputs = m.get_model_outputs(dataset.X_tr,dataset.X_te, dataset.id2word)
                    # save ouptuts
                    # reflesh memory
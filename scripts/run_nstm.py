import os, sys
if ".." not in sys.path:
    sys.path.append("/workdir/") # remove
    from typing import Any
from sktopic import datasets
import torch 
import skorch 
import sktopic 
from tqdm import tqdm 
from sktopic.utils.model_utils import get_kv, eval_all, override_word_embeddings
import wandb 
import time
from sktopic import models
import hydra 
from omegaconf import DictConfig
import matplotlib.pyplot as plt 

# from sktopic.callbacks.wecoherence import TopicScoring
# from octis.evaluation_metrics import coherence_metrics

# class NPMIScoring(TopicScoring):
#     def __init__(self,texts,id2word):
#         name="npmi"
#         lower_is_better=False
#         self.model=coherence_metrics.Coherence(measure='c_npmi',texts=texts)
#         scoring_fn = self.model.score
#         super().__init__(scoring_fn, id2word, topn=10, lower_is_better=lower_is_better, name=name)

def save_topics(topics:list[list[str]], fpath="topics.txt")->None:
    with open(fpath, "w") as f:
        tmp = ""
        for line in topics:
            tmp += " ".join(line)
            tmp += "\n"
        f.write(tmp)

def get_config(cfg:DictConfig)->dict[str,Any]:
    wandb_config = dict(cfg)
    model_config = wandb_config.pop("model")
    for key in model_config:
        wandb_config["model."+key] = model_config[key]
    return wandb_config

@hydra.main(config_name='run_nstm.yaml')
def main(cfg:DictConfig)->None:
    print("start.........")
    dataset = eval(f"datasets.fetch_{cfg.corpus_name}()")
    dataset.train_test_split_stratifiedkfold()
    #_WETC_pw= sktopic.callbacks.WECoherenceScoring(dataset.id2word,kv_path="/workdir/datasets/dummy_kv.txt",binary=False)
    _WETC_c = sktopic.callbacks.WECoherenceScoring(dataset.id2word,kv_path="/workdir/datasets/dummy_kv.txt",binary=False,method="centroid")
    _kvs = get_kv()
    #_WETC_pw.coherencemodel._wv = _kvs["glove.42B.300d"]
    _WETC_c.coherencemodel._wv = _kvs["glove.42B.300d"]
    SEED = int(time.time())
    sktopic.utils.manual_seed(SEED) #(1637809708)#(1637805045) #e44
    V = dataset.vocab_size
    # ----------------------------------------------------
    print("shape=",dataset.X.shape, "labels=",dataset.num_labels)    
    wandb_run = wandb.init(entity="rsimd",reinit=True) #wandb.init(project="sktopic", entity="rsimd")
    wandb_run.config.update(get_config(cfg))
    # ----------------------------------------------------
    callbacks = [
            #_WETC_pw, #WECoherenceScoring(dataset.id2word),
            _WETC_c,
            sktopic.callbacks.TopicDiversityScoring(dataset.id2word),
            sktopic.callbacks.NaNChecker(),
            #skorch.callbacks.EarlyStopping(patience=10,threshold=1e-6),
            #LRScheduler(),
            skorch.callbacks.GradientNormClipping(gradient_clip_value=0.25),
            skorch.callbacks.WandbLogger(wandb_run),
            #NPMIScoring([line.split() for line in dataset.corpus], dataset.id2word)
            ]

    model_cls = models.NeuralSinkhornTopicModel
    m = model_cls(
        V,cfg.model.n_components,
        embed_dim=cfg.model.embed_dim,
        hidden_dims=[500]*2,
        dropout_rate_hidden=cfg.model.dropout_rate_hidden,
        callbacks=callbacks,
        batch_size=cfg.model.batch_size,
        max_epochs=cfg.model.max_epochs,
        activation_hidden=cfg.model.activation_hidden,
        device=cfg.model.device,
        lr=cfg.model.lr,
        optimizer=torch.optim.Adam,
        optimizer__param_groups=[
            #("*.decoder.*",  {"lr":lr_dec}),
            ("*.topic_embeddings.*", {"lr":cfg.model.lr_dec}),
            ("*.word_embeddings.*", {"lr":cfg.model.lr_dec})
        ],
        )
    import torch.nn.functional as F 
    
    if cfg.use_pwe:
        m = override_word_embeddings(m,dataset,_kvs[f"glove.6B.{cfg.model.embed_dim}d"],normalize=True,train_embed=cfg.train_pwe)

    print(f"Run {m.__class__.__name__} num_topics={cfg.model.n_components}, embed_dim={cfg.model.embed_dim}",)
    try:
        m.fit(dataset.X_tr)
        results = eval_all(m,dataset)
        wandb_run.log(results)

        model_outputs = m.get_model_outputs(dataset.X_tr, dataset.X_te, dataset.id2word)
        # model_outputをbinaryとしてファイルに保存する。
        fpath = os.path.join(wandb_run.dir, "topics.txt")
        save_topics(model_outputs["topics"],fpath)
        wandb_run.save(fpath)
        wandb_run.finish()
    except:
        print("An error has occurred")

if __name__ == "__main__":
    main()

"""
python run_vae.py -m model_name=NeuralVariationalDocumentModel,ProductOfExpertsLatentDirichletAllocation,NeuralVariationalLatentDirichletAllocation,GaussianSoftmaxModel,GaussianStickBreakingModel,RecurrentStickBreakingModel corpus_name=SearchSnippets,StackOverflow,Biomedical,TrecTweet,GoogleNews,PascalFlicker use_pwe=True,False train_pwe=True,False model.lr=0.001 model.lr_dec=0.001 model.arccos_lambda=0.0 model.batch_size=64
"""

import os 
import sys
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
    
@hydra.main(config_name='run_vae.yaml')
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
    wandb_run = wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        job_type=cfg.wandb.job_type,
        reinit=True,) #wandb.init(project="sktopic", entity="rsimd")
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
            skorch.callbacks.WandbLogger(wandb_run)
            ]

    model_cls = eval(f"models.{cfg.model_name}")
    m = model_cls(
        V,cfg.model.n_components,
        embed_dim=cfg.model.embed_dim,
        hidden_dims=[500]*2,
        dropout_rate_theta=cfg.model.dropout_rate_theta,
        dropout_rate_hidden=cfg.model.dropout_rate_hidden,
        callbacks=callbacks,
        batch_size=cfg.model.batch_size,
        max_epochs=cfg.model.max_epochs,
        n_sampling=cfg.model.n_sampling,
        activation_hidden=cfg.model.activation_hidden,
        device=cfg.model.device,
        lr=cfg.model.lr,
        optimizer=torch.optim.Adam,
        optimizer__param_groups=[
            #("*.decoder.*",  {"lr":lr_dec}),
            ("*.topic_embeddings.*", {"lr":cfg.model.lr_dec}),
            ("*.word_embeddings.*", {"lr":cfg.model.lr_dec})
        ],
        criterion__arccos_lambda=cfg.model.arccos_lambda,
        criterion__l2_lambda=cfg.model.l2_lambda,
        criterion__l1_lambda=cfg.model.l1_lambda,
        )
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
    except:
        ...
    wandb_run.finish()

if __name__ == "__main__":
    main()

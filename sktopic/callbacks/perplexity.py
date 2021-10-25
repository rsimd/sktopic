import torch
from skorch.dataset import unpack_data
from skorch.callbacks import BatchScoring
from skorch.callbacks.scoring import _cache_net_forward_iter
import warnings
from skorch.exceptions import SkorchWarning
warnings.filterwarnings('ignore', category=SkorchWarning)

class PerplexityScoring(BatchScoring):
    def __init__(
            self,
            on_train=False,
            name=None,
            ): 
        if name is None:
            if on_train:
                name = "train_ppl"
            else:
                name = "valid_ppl"
                
        scoring = lambda net, X,nll: (nll* X.shape[0] / X.sum()).exp()
        super().__init__(scoring, True, on_train,name)
        
    def on_batch_end(self, net, batch, training, **kwargs):
        if training != self.on_train:
            return
        #batch = X
        #X, y = unpack_data(batch)
        y_preds = [kwargs['y_pred']]
        if isinstance(batch, torch.Tensor):
            X = batch
        else:
            X = batch[0]
        with _cache_net_forward_iter(net, self.use_caching, y_preds) as cached_net:
            # In case of y=None we will not have gathered any samples.
            # We expect the scoring function to deal with y=None.
            try:
                score = self._scoring(net, X, kwargs["nll"])
                score = float(score)
                cached_net.history.record_batch(self.name_, score)
            except KeyError:
                print(f"kwargsにnllがないからPerplexityが計算できない。training={training}")
                pass
from skorch.callbacks import Callback
import torch 

class NaNChecker(Callback):
    def check_nan(self,net):
        reduced_params = torch.stack([p.sum() for p in net.module_.parameters()])
        reduced_params = reduced_params.sum()
        return torch.isnan(reduced_params)
    
    def on_train_begin(self, net,**kwargs):
        """Called at the beginning of training."""
        assert not self.check_nan(net),\
            f"NaN appeared in the parameters. @on_train_begin"
    
    def on_batch_end(self,net,**kwargs):
        assert not self.check_nan(net),\
            f"NaN appeared in the parameters. @epoch:{len(net.history_)}, batch:{len(net.history_[-1,'batches'])}"

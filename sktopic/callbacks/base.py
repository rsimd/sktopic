from tqdm import tqdm
import skorch
from skorch.callbacks import EpochScoring
from ..metrics import check_sparsity, mean_squared_cosine_deviation_among_topics as mscd

__all__ = [
    "TrainingProgress",
    "SparsityChecker",
    "MSCDChecker",
]


class TrainingProgress(skorch.callbacks.Callback):
    def __init__(self, max_epochs):
        super().__init__()
        self.max_epochs = max_epochs

    def on_train_begin(self, net, **kwargs):
        self.pbar = tqdm(total=self.max_epochs, leave=False)

    def on_epoch_end(self, net, **kwargs):
        self.pbar.update(1)

    def on_train_end(self, net, **kwargs):
        del self.pbar

SparsityChecker = EpochScoring(
    scoring =lambda trainer,X=None,y=None: check_sparsity(trainer.module_.get_beta()).item(),
    lower_is_better=False,
    name ="Sparsity_of_beta",
    )

MSCDChecker = EpochScoring(
    scoring=lambda trainer,X=None,y=None: mscd(trainer.module_.get_beta()).item(),
    lower_is_better=True,
    name ="MSCD",
    )


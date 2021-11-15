#from .nvdm import *
from .clusterawarentm import ClusterLoss
from .dntm import DETM, ScalingLoss
#from . import base 

from .nvdm import NeuralVariationalDocumentModel
from .prodlda import ProductOfExpertsLatentDirichletAllocation
from .nvlda import NeuralVariationalLatentDirichletAllocation
from .gsm import GaussianSoftmaxModel
from .gsb import GaussianStickBreakingModel
from .rsb import RecurrentStickBreakingModel

# OT
from .nstm import NeuralSinkhornTopicModel
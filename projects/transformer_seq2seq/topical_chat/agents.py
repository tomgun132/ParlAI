import torch
import numpy as np

from parlai.utils.torch import padded_tensor
from parlai.utils.misc import round_sigfigs

from parlai.agents.transformer.transformer import TransformerGeneratorAgent, TransformerRankerAgent

from .modules import SoonToBeCreatedModel


class TopicSelectionAgent(TransformerRankerAgent):
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
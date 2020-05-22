from tsct import util

import random
import math
from argparse import ArgumentParser

import torch
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule

################################################################################
#
# Let's get the simple modules out of the way
#


class MLP(LightningModule):
    """ Multilayer perceptron
    """

    @staticmethod
    def add_model_specific_args(parent_parser):
        """ Add a configuration parameter for the layer: the number of units of the hidden layers is a multiple of the input units.
        """
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('--MLPMultiplier', type=int, default=4)
        return parser

    def __init(self, runParam):
        super().__init__()

        layerSize = runParam.nHeads * runParam.MLPMultiplier

        self.layer1 = nn.Linear(runParam.nHeads, layerSize)
        self.layer2 = nn.Linear(layerSize, runParam.nHeads)
        self.layerNorm = nn.layerNorm(runParam.segmentLength)

    def forward(self, input):
        batch_size, channels, width, height = input.size()

        input = input.view(batch_size, -1)

        # layer 1
        y = self.layer1(input)
        y = torch.relu(y)

        # layer 2
        y = self.layer2(y)

        # Final normalisation (note the formulation as residual)
        # Normalisation includes the training of an affine transformation
        output = self.layerNorm(input + y)

        return output

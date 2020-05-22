
class RunParam:
    """Class containing global parameters.

    :param segmentLength: the length of a training segment. Default = 60
    :param nLayers: number of transformer blocks. Default = 8
    :param nHeads: number of self-attention heads per block. Default = 8
    :param nMemory: number of states kept from training previous segments. Default = 30
    :param MLPMultiplier: within a transformer block, the multi-layer perceptron receives has a number of neurons being multiple 

    """

    def __init__(self, segmentLength=60, nLayers=8, nHeads=8, nMemory=30, MLPMultiplier=4):
        self.segmentLength = segmentLength
        self.nLayers = nLayers
        self.nHeads = nHeads
        self.nMemory = nMemory
        self.MLPMultiplier = MLPMultiplier

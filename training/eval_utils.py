# Copyright (c) Facebook, Inc. and its affiliates.

class DatasetEvaluator:
    """
    Base class for a dataset evaluator.

    This class will accumulate information of the inputs/outputs (by:meth:`process`),
    add produce evaluation results in the end (by :meth:`evaluate`).
    """

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        pass

    def process(self, inputs, outputs):
        """
        Process the pair of inputs and outputs.
        """
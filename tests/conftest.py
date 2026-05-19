"""Session-wide patches needed to run on CPU-only machines."""

import pytest
import torch
from detectron2.engine.defaults import DefaultPredictor

_orig_torch_load = torch.load
_orig_predictor_call = DefaultPredictor.__call__


def _cpu_load(*args, **kwargs):
    kwargs.setdefault("map_location", "cpu")
    return _orig_torch_load(*args, **kwargs)


def _cpu_predictor_call(self, original_image):
    # The predictor is deserialized from pickle with CUDA config; fix at call time.
    self.cfg.MODEL.DEVICE = "cpu"
    if hasattr(self, "model"):
        self.model = self.model.cpu()
    return _orig_predictor_call(self, original_image)


@pytest.fixture(autouse=True, scope="session")
def remap_cuda_to_cpu():
    """BRAILS model weights are serialized on CUDA; remap to CPU at load time."""
    torch.load = _cpu_load
    DefaultPredictor.__call__ = _cpu_predictor_call
    yield
    torch.load = _orig_torch_load
    DefaultPredictor.__call__ = _orig_predictor_call

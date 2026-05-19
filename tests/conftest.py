"""Session-wide patches needed to run on CPU-only machines."""

from pathlib import Path
import pytest
import torch
from detectron2.engine.defaults import DefaultPredictor


def pytest_addoption(parser):
    parser.addoption(
        "--image-dir",
        default=None,
        help="Override the image directory used by integration tests (default: data/gsv/)",
    )


@pytest.fixture(scope="session")
def image_dir(request):
    override = request.config.getoption("--image-dir")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[1] / "data" / "gsv"

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

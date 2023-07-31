from .network import *
from .common import *
from .m2tr_transform import *

MODELS = {
    "BRCNet": BRCNet,
}


def load_model(name="BRCNet"):
    assert name in MODELS.keys(), f"Model name can only be one of {MODELS.keys()}."
    print(f"Using model: '{name}'")
    return MODELS[name]

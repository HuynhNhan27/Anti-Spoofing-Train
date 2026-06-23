from .minifasv2.model import MultiFTNet
from .detnet.BasicModule import MydetNet59
from .feathernet.FeatherNet import FeatherNetA, FeatherNetB
from .aenet.AENet import AENet
from .MN3.MN3 import mobilenetv3_large, mobilenetv3_small

__all__ = [
    "MultiFTNet",
    "MydetNet59",
    "FeatherNetA",
    "FeatherNetB",
    "AENet",
    "mobilenetv3_large",
    "mobilenetv3_small",
]

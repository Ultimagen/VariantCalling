from enum import Enum

SPAN_DEL = "*"
IGNORE = -1
MISS = -2


class GtType(Enum):
    APPROXIMATE = 1
    EXACT = 2


class VcfType(Enum):
    SINGLE_SAMPLE = 1
    DEEP_VARIANT = 4
    JOINT = 2

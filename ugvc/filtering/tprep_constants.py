from enum import Enum

SPAN_DEL = "*"
IGNORE = -1
MISS = -2


class GtType(Enum):
    APPROXIMATE = "approximate"
    EXACT = "exact"

    def __str__(self):
        return self.value


class VcfType(Enum):
    SINGLE_SAMPLE = "single_sample"
    DEEP_VARIANT = "deep_variant"
    DEEP_VARIANT_WITH_SOFTCLIP_COUNTS = "deep_variant_extended"
    JOINT = "joint_callset"

    def __str__(self):
        return self.value

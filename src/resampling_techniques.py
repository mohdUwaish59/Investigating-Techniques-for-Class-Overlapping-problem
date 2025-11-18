"""
Resampling Techniques Module
Contains various resampling techniques for imbalanced learning

This module now imports from the modular techniques package.
For better organization, each technique is in its own file under src/techniques/
"""

# Import all techniques from the modular package
from techniques import (
    BaseSampler,
    EHSO,
    RFCL,
    SVDDWSMOTE,
    NBUS,
    NBBasic,
    NBTomek,
    NBComm,
    NBRec,
    RandomOverSampler,
    RandomUnderSampler
)

# For backward compatibility
__all__ = [
    'BaseSampler',
    'EHSO',
    'RFCL',
    'SVDDWSMOTE',
    'NBUS',
    'NBBasic',
    'NBTomek',
    'NBComm',
    'NBRec',
    'RandomOverSampler',
    'RandomUnderSampler'
]

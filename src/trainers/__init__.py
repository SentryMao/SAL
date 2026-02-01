# Terminology note:
# - SPL (Segment Positional Labeling): 2-loss method using Positional labels
# - CSM (Cross-Segment Mixing): mixing-based data augmentation method
# - SAL: uses both SPL and CSM together

from src.trainers.label_generators import SPLLabelGenerator, TransitionLabelGenerator
from src.trainers.baseline_trainer import BaselineTrainer
from src.trainers.transition_trainer import TransitionTrainer
from src.trainers.spl_trainer import SPLTrainer
from src.trainers.sal_trainer import SALTrainer
from src.trainers.csm_trainer import CSMTrainer

__all__ = [
    "SPLLabelGenerator",
    "TransitionLabelGenerator",
    "BaselineTrainer",
    "TransitionTrainer",
    "SPLTrainer",
    "SALTrainer",
    "CSMTrainer",
]


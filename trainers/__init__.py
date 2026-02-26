# trainers/__init__.py (updated)
from .source_trainer import SourceTrainer
from .target_trainer_fewshot import FewShotTargetTrainer

__all__ = ['SourceTrainer', 'FewShotTargetTrainer']

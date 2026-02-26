# datasets/__init__.py (updated)
from .office31 import Office31Dataset, get_office31_loaders
from .office31_fewshot import get_office31_fewshot_loaders

__all__ = ['Office31Dataset', 'get_office31_loaders', 'get_office31_fewshot_loaders']

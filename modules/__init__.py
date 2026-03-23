# Пакет modules для программы распознавания
from . import data_loader
from . import data_saver
from . import training
from . import recognition
from . import quality_evaluation
from . import help_viewer
from . import exit_handler

__all__ = [
    'data_loader',
    'data_saver',
    'training',
    'recognition',
    'quality_evaluation',
    'help_viewer',
    'exit_handler'
]
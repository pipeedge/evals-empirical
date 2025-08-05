"""
MLR System Modules
Multivocal Literature Review System for AI/LLM Evaluation Framework Research
"""

__version__ = "1.0.0"
__author__ = "MLR System"
__description__ = "Automated research system for practitioner-generated content analysis"

from .data_acquisition import DataAcquisitionOrchestrator
from .filtering_preprocessing import FilteringPreprocessingOrchestrator
from .thematic_analysis import ThematicAnalysisOrchestrator

__all__ = [
    "DataAcquisitionOrchestrator",
    "FilteringPreprocessingOrchestrator", 
    "ThematicAnalysisOrchestrator"
] 
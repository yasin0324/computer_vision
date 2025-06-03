"""
Path configuration management for the project.
"""

from pathlib import Path
from typing import Dict, Union

class Paths:
    """Path configuration class."""
    
    # Project root
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    
    # Source code paths
    SRC_DIR = PROJECT_ROOT / "src"
    SCRIPTS_DIR = PROJECT_ROOT / "scripts"
    NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
    TESTS_DIR = PROJECT_ROOT / "tests"
    DOCS_DIR = PROJECT_ROOT / "docs"
    CONFIGS_DIR = PROJECT_ROOT / "configs"
    
    # Data paths
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    EXTERNAL_DATA_DIR = DATA_DIR / "external"
    
    # Processed data subdirectories
    SPLITS_DIR = PROCESSED_DATA_DIR / "splits"
    METADATA_DIR = PROCESSED_DATA_DIR / "metadata"
    CACHE_DIR = PROCESSED_DATA_DIR / "cache"
    
    # Output paths
    OUTPUTS_DIR = PROJECT_ROOT / "outputs"
    MODELS_DIR = OUTPUTS_DIR / "models"
    LOGS_DIR = OUTPUTS_DIR / "logs"
    RESULTS_DIR = OUTPUTS_DIR / "results"
    FIGURES_DIR = OUTPUTS_DIR / "figures"
    
    # Model subdirectories
    BASELINE_MODELS_DIR = MODELS_DIR / "baseline"
    ATTENTION_MODELS_DIR = MODELS_DIR / "attention"
    CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
    
    # Logs subdirectories
    TENSORBOARD_LOGS_DIR = LOGS_DIR / "tensorboard"
    TRAINING_LOGS_DIR = LOGS_DIR / "training"
    
    # Results subdirectories
    METRICS_DIR = RESULTS_DIR / "metrics"
    PREDICTIONS_DIR = RESULTS_DIR / "predictions"
    ANALYSIS_DIR = RESULTS_DIR / "analysis"
    
    # Figures subdirectories
    DATA_ANALYSIS_FIGURES_DIR = FIGURES_DIR / "data_analysis"
    TRAINING_CURVES_FIGURES_DIR = FIGURES_DIR / "training_curves"
    EVALUATION_FIGURES_DIR = FIGURES_DIR / "evaluation"
    
    # PlantVillage dataset path
    PLANTVILLAGE_DIR = RAW_DATA_DIR / "PlantVillage"
    
    @classmethod
    def create_all_directories(cls) -> None:
        """Create all necessary directories."""
        directories = [
            cls.SRC_DIR, cls.SCRIPTS_DIR, cls.NOTEBOOKS_DIR, cls.TESTS_DIR,
            cls.DOCS_DIR, cls.CONFIGS_DIR, cls.DATA_DIR, cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR, cls.EXTERNAL_DATA_DIR, cls.SPLITS_DIR,
            cls.METADATA_DIR, cls.CACHE_DIR, cls.OUTPUTS_DIR, cls.MODELS_DIR,
            cls.LOGS_DIR, cls.RESULTS_DIR, cls.FIGURES_DIR,
            cls.BASELINE_MODELS_DIR, cls.ATTENTION_MODELS_DIR, cls.CHECKPOINTS_DIR,
            cls.TENSORBOARD_LOGS_DIR, cls.TRAINING_LOGS_DIR, cls.METRICS_DIR,
            cls.PREDICTIONS_DIR, cls.ANALYSIS_DIR, cls.DATA_ANALYSIS_FIGURES_DIR,
            cls.TRAINING_CURVES_FIGURES_DIR, cls.EVALUATION_FIGURES_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_data_split_path(cls, split_name: str) -> Path:
        """Get path for data split file."""
        return cls.SPLITS_DIR / f"{split_name}_split.csv"
    
    @classmethod
    def get_model_path(cls, model_name: str, model_type: str = "best") -> Path:
        """Get path for saved model."""
        if model_type == "checkpoint":
            return cls.CHECKPOINTS_DIR / f"{model_name}.pth"
        elif "attention" in model_name.lower():
            return cls.ATTENTION_MODELS_DIR / f"{model_name}_best.pth"
        else:
            return cls.BASELINE_MODELS_DIR / f"{model_name}_best.pth"
    
    @classmethod
    def get_log_path(cls, experiment_name: str, log_type: str = "training") -> Path:
        """Get path for log files."""
        if log_type == "tensorboard":
            return cls.TENSORBOARD_LOGS_DIR / experiment_name
        else:
            return cls.TRAINING_LOGS_DIR / f"{experiment_name}.log"
    
    @classmethod
    def get_results_path(cls, experiment_name: str, result_type: str = "metrics") -> Path:
        """Get path for results files."""
        if result_type == "metrics":
            return cls.METRICS_DIR / f"{experiment_name}_metrics.json"
        elif result_type == "predictions":
            return cls.PREDICTIONS_DIR / f"{experiment_name}_predictions.csv"
        else:
            return cls.ANALYSIS_DIR / f"{experiment_name}_analysis.json"
    
    @classmethod
    def get_figure_path(cls, figure_name: str, figure_type: str = "evaluation") -> Path:
        """Get path for figure files."""
        if figure_type == "data_analysis":
            return cls.DATA_ANALYSIS_FIGURES_DIR / f"{figure_name}.png"
        elif figure_type == "training_curves":
            return cls.TRAINING_CURVES_FIGURES_DIR / f"{figure_name}.png"
        else:
            return cls.EVALUATION_FIGURES_DIR / f"{figure_name}.png"
    
    @classmethod
    def to_dict(cls) -> Dict[str, str]:
        """Convert paths to dictionary."""
        return {
            'project_root': str(cls.PROJECT_ROOT),
            'src_dir': str(cls.SRC_DIR),
            'data_dir': str(cls.DATA_DIR),
            'outputs_dir': str(cls.OUTPUTS_DIR),
            'plantvillage_dir': str(cls.PLANTVILLAGE_DIR),
            'splits_dir': str(cls.SPLITS_DIR),
            'models_dir': str(cls.MODELS_DIR),
            'logs_dir': str(cls.LOGS_DIR),
            'results_dir': str(cls.RESULTS_DIR),
            'figures_dir': str(cls.FIGURES_DIR)
        }

# Create global paths instance
paths = Paths() 
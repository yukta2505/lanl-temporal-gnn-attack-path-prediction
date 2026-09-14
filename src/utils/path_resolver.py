"""
GODLY Path Resolver - NO HARDCODED PATHS!
Automatically finds project root and resolves all paths relative to it.
"""
import os
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class PathResolver:
    """
    Resolves all project paths dynamically - NO HARDCODING!
    
    Automatically detects project root and provides paths to:
    - data/
    - checkpoints/
    - config/
    - logs/
    - outputs/
    """
    
    _instance = None
    _project_root: Optional[Path] = None
    
    def __new__(cls):
        """Singleton pattern - one instance for entire application"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize path resolver"""
        if self._project_root is None:
            self._project_root = self._find_project_root()
            logger.info(f"🎯 Project root detected: {self._project_root}")
    
    @staticmethod
    def _find_project_root() -> Path:
        """
        Find project root by looking for marker files.
        
        Searches upward from current file until finding:
        - requirements.txt
        - config/ directory
        - src/ directory
        
        Returns
        -------
        Path
            Absolute path to project root
        """
        # Start from this file's location
        current = Path(__file__).resolve().parent
        
        # Search upward for project markers
        for _ in range(10):  # Max 10 levels up
            markers = [
                current / "requirements.txt",
                current / "config",
                current / "src",
                current / "app"
            ]
            
            if any(marker.exists() for marker in markers):
                return current
            
            current = current.parent
        
        # Fallback: use current working directory
        logger.warning("⚠️ Could not find project root, using current directory")
        return Path.cwd()
    
    @property
    def project_root(self) -> Path:
        """Get project root directory"""
        return self._project_root
    
    @property
    def data_dir(self) -> Path:
        """Get data directory (data/)"""
        return self._project_root / "data"
    
    @property
    def graphs_dir(self) -> Path:
        """Get graphs directory (data/graphs/)"""
        path = self.data_dir / "graphs"
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @property
    def metadata_dir(self) -> Path:
        """Get metadata directory (data/metadata/)"""
        path = self.data_dir / "metadata"
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @property
    def raw_data_dir(self) -> Path:
        """Get raw data directory (data/raw/)"""
        path = self.data_dir / "raw"
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @property
    def checkpoints_dir(self) -> Path:
        """Get checkpoints directory (checkpoints/)"""
        path = self._project_root / "checkpoints"
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @property
    def config_dir(self) -> Path:
        """Get config directory (config/)"""
        return self._project_root / "config"
    
    @property
    def logs_dir(self) -> Path:
        """Get logs directory (logs/)"""
        path = self._project_root / "logs"
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @property
    def outputs_dir(self) -> Path:
        """Get outputs directory (outputs/)"""
        path = self._project_root / "outputs"
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @property
    def temp_dir(self) -> Path:
        """Get temp directory (temp/)"""
        path = self._project_root / "temp"
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_entity_mappings_path(self) -> Path:
        """Get entity mappings file path"""
        return self.metadata_dir / "entity_mappings.pkl"
    
    def get_graph_metadata_path(self) -> Path:
        """Get graph generation metadata path"""
        return self.metadata_dir / "graph_generation_metadata.json"
    
    def get_graph_snapshots_metadata_path(self) -> Path:
        """Get graph snapshots metadata path"""
        return self.graphs_dir / "metadata.json"
    
    def get_best_model_path(self) -> Path:
        """
        Get best trained model path.
        
        Searches in order:
        1. checkpoints/best_thgnn_model.pth
        2. best_thgnn_model.pth (root)
        3. Any best_*.pth in checkpoints/
        
        Returns
        -------
        Path
            Path to best model checkpoint
        
        Raises
        ------
        FileNotFoundError
            If no model checkpoint found
        """
        candidates = [
            self.checkpoints_dir / "best_thgnn_model.pth",
            self._project_root / "best_thgnn_model.pth",
            self._project_root / "best_trained_model.pth"
        ]
        
        # Check candidates
        for path in candidates:
            if path.exists():
                logger.info(f"✅ Found model: {path}")
                return path
        
        # Search for any best_*.pth
        import glob
        pattern = str(self.checkpoints_dir / "best_*.pth")
        matches = glob.glob(pattern)
        if matches:
            path = Path(matches[0])
            logger.info(f"✅ Found model: {path}")
            return path
        
        # Not found
        raise FileNotFoundError(
            f"❌ No model checkpoint found! Searched:\n" +
            "\n".join(f"  - {p}" for p in candidates)
        )
    
    def validate_structure(self) -> dict:
        """
        Validate project directory structure.
        
        Returns
        -------
        dict
            Status of each required directory/file
        """
        checks = {
            "project_root": self.project_root.exists(),
            "data_dir": self.data_dir.exists(),
            "graphs_dir": self.graphs_dir.exists(),
            "metadata_dir": self.metadata_dir.exists(),
            "checkpoints_dir": self.checkpoints_dir.exists(),
            "config_dir": self.config_dir.exists(),
            "entity_mappings": self.get_entity_mappings_path().exists(),
            "graph_metadata": self.get_graph_metadata_path().exists(),
        }
        
        try:
            checks["model_checkpoint"] = self.get_best_model_path().exists()
        except FileNotFoundError:
            checks["model_checkpoint"] = False
        
        return checks
    
    def print_status(self):
        """Print project structure status"""
        print("\n" + "="*60)
        print("🎯 THGNN PROJECT STRUCTURE STATUS")
        print("="*60)
        print(f"📁 Project Root: {self.project_root}")
        print()
        
        checks = self.validate_structure()
        for name, exists in checks.items():
            status = "✅" if exists else "❌"
            print(f"  {status} {name.replace('_', ' ').title()}")
        
        print("="*60 + "\n")
        
        # Return True if all critical paths exist
        critical = ["project_root", "data_dir", "checkpoints_dir", "config_dir"]
        return all(checks.get(k, False) for k in critical)


# Global singleton instance
_resolver = PathResolver()


def get_project_root() -> Path:
    """Get project root directory"""
    return _resolver.project_root


def get_data_dir() -> Path:
    """Get data directory"""
    return _resolver.data_dir


def get_graphs_dir() -> Path:
    """Get graphs directory"""
    return _resolver.graphs_dir


def get_metadata_dir() -> Path:
    """Get metadata directory"""
    return _resolver.metadata_dir


def get_checkpoints_dir() -> Path:
    """Get checkpoints directory"""
    return _resolver.checkpoints_dir


def get_config_dir() -> Path:
    """Get config directory"""
    return _resolver.config_dir


def get_logs_dir() -> Path:
    """Get logs directory"""
    return _resolver.logs_dir


def get_best_model_path() -> Path:
    """Get best model checkpoint path"""
    return _resolver.get_best_model_path()


def get_entity_mappings_path() -> Path:
    """Get entity mappings file path"""
    return _resolver.get_entity_mappings_path()


def validate_project_structure() -> bool:
    """Validate and print project structure"""
    return _resolver.print_status()


if __name__ == "__main__":
    # Test the resolver
    validate_project_structure()


def get_graph_metadata_path() -> Path:
    """Get graph generation metadata path"""
    return _resolver.get_graph_metadata_path()

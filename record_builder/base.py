from abc import ABC, abstractmethod
from typing import List, Dict, Any
import pandas as pd

class BaseRecordBuilder(ABC):
    """
    Abstract base class for note chunking.
    """

    @abstractmethod
    def build_records(self, df: pd.DataFrame, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split a DataFrame with notes into a list of dictionaries ready to be saved in JSONL.
        
        Args:
            df (pd.DataFrame): DataFrame containing notes (must have columns pitch, start, end, velocity).
            metadata (Dict): Metadata of the piece (e.g., composer, source).
        Returns:
            List[Dict]: List of dictionaries. Each dictionary is one training record:
                        {
                            "notes_first": [...], 
                            "notes_second": [...], 
                            "metadata": {...}
                        }
        """
        pass
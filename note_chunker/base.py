from abc import ABC, abstractmethod
from typing import List, Dict, Any
import pandas as pd

class NoteChunker(ABC):
    """
    Abstrakcyjna klasa bazowa dla mechanizmów dzielenia nut na fragmenty.
    """

    @abstractmethod
    def chunk_notes(self, df: pd.DataFrame, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Dzieli DataFrame z nutami na listę słowników gotowych do zapisu w JSONL.
        
        Args:
            df (pd.DataFrame): DataFrame zawierający nuty (musi mieć kolumny pitch, start, end, velocity).
            metadata (Dict): Metadane utworu (np. kompozytor, źródło).

        Returns:
            List[Dict]: Lista słowników. Każdy słownik to jeden rekord treningowy:
                        {
                            "notes_first": [...], 
                            "notes_second": [...], 
                            "metadata": {...}
                        }
        """
        pass
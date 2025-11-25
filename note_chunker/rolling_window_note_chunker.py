from typing import List, Dict, Any
import pandas as pd
from .base import NoteChunker

class RollingWindowNoteChunker(NoteChunker):
    def __init__(self, window_size: int = 100, predict_size: int = 20, stride: int = 50):
        """
        Args:
            window_size (int): Łączna liczba nut w jednym chunku (input + target).
            predict_size (int): Ile ostatnich nut ma być w 'notes_second' (to co przewidujemy).
            stride (int): O ile nut przesuwamy okno dla kolejnej próbki.
        """
        self.window_size = window_size
        self.predict_size = predict_size
        self.stride = stride

        if predict_size >= window_size:
            raise ValueError("predict_size musi być mniejsze niż window_size")

    def chunk_notes(self, df: pd.DataFrame, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        chunks = []
        total_notes = len(df)

        for start_idx in range(0, total_notes - self.window_size + 1, self.stride):
            end_idx = start_idx + self.window_size
            
            window_df = df.iloc[start_idx:end_idx].copy()
            
            split_point = self.window_size - self.predict_size
            
            df_first = window_df.iloc[:split_point]
            df_second = window_df.iloc[split_point:]

            record = {
                "notes_first": df_first.to_dict(orient='records'),
                "notes_second": df_second.to_dict(orient='records'),
                "metadata": metadata.copy()
            }
            
            record['metadata']['chunk_start_index'] = start_idx
            
            chunks.append(record)

        return chunks
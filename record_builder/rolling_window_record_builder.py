from typing import List, Dict, Any
import pandas as pd
from .base import BaseRecordBuilder

class RollingWindowRecordBuilder(BaseRecordBuilder):
    """
    Implementation of RecordBuilder using a sliding (rolling) window approach.
    Splits the track into overlapping sequences of fixed length.
    """
    def __init__(self, window_size: int = 100, predict_size: int = 20, stride: int = 50):
        """
        Args:
            window_size (int): Total number of notes in one chunk (input + target).
            predict_size (int): How many of the last notes should be in 'notes_second' (what we predict).
            stride (int): Step size for moving the windows (controls overlap).
        """
        self.window_size = window_size
        self.predict_size = predict_size
        self.stride = stride

        if predict_size >= window_size:
            raise ValueError("predict_size must be lower then window_size")

    def build_records(self, df: pd.DataFrame, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Applies the rolling window logic to create input/target pairs.
        """
        chunks = []
        total_notes = len(df)

        for start_idx in range(0, total_notes - self.window_size + 1, self.stride):
            end_idx = start_idx + self.window_size
            
            window_df = df.iloc[start_idx:end_idx].copy()
            
            split_point = self.window_size - self.predict_size
            
            df_input = window_df.iloc[:split_point]
            df_output = window_df.iloc[split_point:]

            record = {
                "notes_first": df_input.to_dict(orient='records'),
                "notes_second": df_output.to_dict(orient='records'),
                "metadata": metadata.copy()
            }
            
            record['metadata']['chunk_start_index'] = start_idx
            
            chunks.append(record)

        return chunks
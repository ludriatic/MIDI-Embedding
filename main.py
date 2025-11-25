import pandas as pd
import json
from datasets import load_dataset
from note_chunker.rolling_window_note_chunker import RollingWindowNoteChunker
from tqdm import tqdm

OUTPUT_FILES = {
    "train": "train.jsonl",
    "validation": "validation.jsonl",
    "test": "test.jsonl"
}

def main():
    dataset = load_dataset("epr-labs/maestro-sustain-v2", streaming=True)
    
    chunker = RollingWindowNoteChunker(
        window_size=100,
        predict_size=20,
        stride=50
    )

    total_chunks_created = 0
    
    for split_name, output_file in OUTPUT_FILES.items():
        try:
            data_split = dataset[split_name]
        except Exception:
            print(f"Pominięto {split_name} - brak w datasecie.")
            continue

        print(f"Przetwarzanie: {split_name} -> {output_file}")
        
        songs_processed = 0
        
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for sample in tqdm(data_split, desc=f"Przetwarzanie {split_name}"):
                try:
                    notes_data = sample['notes']
                    df_notes = pd.DataFrame(notes_data)

                    if 'start' in df_notes.columns:
                        df_notes = df_notes.sort_values('start').reset_index(drop=True)
                    
                    source_info = sample.get('source', {})
                    if isinstance(source_info, str):
                        try:
                            source_info = json.loads(source_info.replace("'", '"'))
                        except:
                            pass

                    composer = "unknown"
                    title = "unknown"
                    
                    if isinstance(source_info, dict):
                        composer = source_info.get('composer', 'unknown')
                        title = source_info.get('title', 'unknown')

                    metadata = {
                        "source_dataset": "maestro-sustain-v2",
                        "composer": composer,
                        "title": title,
                    }

                    chunks = chunker.chunk_notes(df_notes, metadata)

                    if chunks:
                        for chunk in chunks:
                            f_out.write(json.dumps(chunk) + '\n')
                            total_chunks_created += 1
                    
                    songs_processed += 1
                    
                except KeyError as e:
                    if songs_processed == 0: 
                        print(f"\n [Pominięto próbkę z powodu braku klucza]: {e}")
                    continue
                except Exception as e:
                    if songs_processed < 2:
                        print(f"\n [Błąd podczas przetwarzania próbki]: {e}")
                    continue
        print(f"Zakończono przetwarzanie {split_name}. Utworzono {songs_processed} utworów.")

if __name__ == "__main__":
    main()
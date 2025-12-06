import pandas as pd
import json
from datasets import load_dataset
from record_builder.rolling_window_record_builder import RollingWindowRecordBuilder
from tqdm import tqdm

OUTPUT_FILES = {
    "train": "train.jsonl",
    "validation": "validation.jsonl",
    "test": "test.jsonl"
}

def main():
    dataset = load_dataset("epr-labs/maestro-sustain-v2", streaming=True)
    
    builder = RollingWindowRecordBuilder(
        window_size=100,
        predict_size=20,
        stride=50
    )

    total_records_created = 0
    
    for split_name, output_file in OUTPUT_FILES.items():
        try:
            data_split = dataset[split_name]
        except Exception:
            print(f"Skipped {split_name} - not found in dataset.")
            continue

        print(f"Processing: {split_name} -> {output_file}")
        
        songs_processed = 0
        
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for sample in tqdm(data_split, desc=f"Processing {split_name}"):
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

                    records = builder.build_records(df_notes, metadata)

                    if records:
                        for record in records:
                            f_out.write(json.dumps(record) + '\n')
                            total_records_created += 1
                    
                    songs_processed += 1
                    
                except KeyError as e:
                    if songs_processed == 0: 
                        print(f"\n [Skipped sample due to missing key]: {e}")
                    continue
                except Exception as e:
                    if songs_processed < 2:
                        print(f"\n [Error while processing sample]: {e}")
                    continue
        print(f"Finished processing {split_name}. Created {songs_processed} songs.")

if __name__ == "__main__":
    main()
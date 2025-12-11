import streamlit as st
import json
import pandas as pd
import linecache
import altair as alt

st.set_page_config(page_title="Music LLM Inspector", layout="wide")

class DatasetReader:
    """Reads a JSONL file line by line."""
    def __init__(self, file_path):
        self.file_path = file_path
        self._count_lines()

    def _count_lines(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.total_lines = sum(1 for _ in f)

    def get_sample(self, index):
        line = linecache.getline(self.file_path, index + 1)
        if not line:
            return None
        return json.loads(line)

def plot_custom_pianoroll(notes_first, notes_second):

    df1 = pd.DataFrame(notes_first)
    df1['type'] = 'Input (Context)'
    
    df2 = pd.DataFrame(notes_second)
    df2['type'] = 'Target (Prediction)'
    
    df = pd.concat([df1, df2], ignore_index=True)
    
    if df.empty:
        st.warning("No notes to display.")
        return


    min_start = df['start'].min()
    df['start_norm'] = df['start'] - min_start
    df['end_norm'] = df['end'] - min_start
    
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('start_norm', title='Time (seconds)'),
        x2='end_norm',
        y=alt.Y('pitch', title='Pitch', scale=alt.Scale(domain=[df['pitch'].min()-2, df['pitch'].max()+2])),
        color=alt.Color('type', title='Data Type', scale=alt.Scale(domain=['Input (Context)', 'Target (Prediction)'], range=['#1f77b4', '#d62728'])),
        tooltip=['pitch', 'velocity', 'start', 'type']
    ).properties(
        height=400,
        title="Piano Roll Visualization"
    ).interactive()

    st.altair_chart(chart, width='stretch')


def main():
    st.title("Music LLM Dataset Explorer")
    
    st.sidebar.header("Configuration")
    split_name = st.sidebar.selectbox("Dataset:", ["train", "validation", "test"])
    
    file_map = {"train": "train.jsonl", "validation": "validation.jsonl", "test": "test.jsonl"}
    selected_file = file_map[split_name]

    try:
        reader = DatasetReader(selected_file)
        st.sidebar.info(f"File: `{selected_file}`")
        st.sidebar.write(f"Records: **{reader.total_lines}**")

        if 'record_idx' not in st.session_state:
            st.session_state.record_idx = 0

        col_prev, col_curr, col_next = st.sidebar.columns([1, 2, 1])
        
        if col_prev.button("◀ Prev"):
            st.session_state.record_idx = max(0, st.session_state.record_idx - 1)
        if col_next.button("Next ▶"):
            st.session_state.record_idx = min(reader.total_lines - 1, st.session_state.record_idx + 1)
            

        record_idx = st.sidebar.number_input(
            "Go to index:", 
            min_value=0, 
            max_value=reader.total_lines - 1,
            value=st.session_state.record_idx
        )
        st.session_state.record_idx = record_idx

        sample = reader.get_sample(st.session_state.record_idx)
        
        if sample:
            st.subheader(f"Record #{st.session_state.record_idx}")
            
            m = sample['metadata']
            c1, c2, c3 = st.columns(3)
            c1.metric("Composer", m.get('composer', 'Unknown'))
            c2.metric("Title", m.get('title', 'Unknown'))
            c3.metric("Year", m.get('year', '-'))

            st.divider()
            plot_custom_pianoroll(sample['notes_first'], sample['notes_second'])
            
            with st.expander("Raw JSON Data"):
                st.json(sample)

    except FileNotFoundError:
        st.error(f"File not found: {selected_file}. Run main.py!")
    except Exception as e:
        st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
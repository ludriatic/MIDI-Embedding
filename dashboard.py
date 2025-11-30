import streamlit as st
import json
import pandas as pd
import linecache
import altair as alt

st.set_page_config(page_title="Music LLM Inspector", layout="wide")

class DatasetReader:
    """Odczytuje plik JSONL linia po linii."""
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
        st.warning("Brak nut do wyświetlenia.")
        return

    min_start = df['start'].min()
    df['start_norm'] = df['start'] - min_start
    df['end_norm'] = df['end'] - min_start
    
    # 5. Rysowanie wykresu (Paski)
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('start_norm', title='Czas (sekundy)'),
        x2='end_norm',
        y=alt.Y('pitch', title='Wysokość (Pitch)', scale=alt.Scale(domain=[df['pitch'].min()-2, df['pitch'].max()+2])),
        color=alt.Color('type', title='Rodzaj danych', scale=alt.Scale(domain=['Input (Context)', 'Target (Prediction)'], range=['#1f77b4', '#d62728'])),
        tooltip=['pitch', 'velocity', 'start', 'type']
    ).properties(
        height=400,
        title="Wizualizacja Piano Roll"
    ).interactive()

    st.altair_chart(chart, use_container_width=True)


def main():
    st.title("Music LLM Dataset Explorer")
    
    st.sidebar.header("Konfiguracja")
    split_name = st.sidebar.selectbox("Zbiór danych:", ["train", "validation", "test"])
    
    file_map = {"train": "train.jsonl", "validation": "validation.jsonl", "test": "test.jsonl"}
    selected_file = file_map[split_name]

    try:
        reader = DatasetReader(selected_file)
        st.sidebar.info(f"Plik: `{selected_file}`")
        st.sidebar.write(f"Rekordy: **{reader.total_lines}**")

        if 'chunk_idx' not in st.session_state:
            st.session_state.chunk_idx = 0

        col_prev, col_curr, col_next = st.sidebar.columns([1, 2, 1])
        
        if col_prev.button("◀ Prev"):
            st.session_state.chunk_idx = max(0, st.session_state.chunk_idx - 1)
        if col_next.button("Next ▶"):
            st.session_state.chunk_idx = min(reader.total_lines - 1, st.session_state.chunk_idx + 1)
            

        chunk_idx = st.sidebar.number_input(
            "Idź do indeksu:", 
            min_value=0, 
            max_value=reader.total_lines - 1,
            value=st.session_state.chunk_idx
        )
        st.session_state.chunk_idx = chunk_idx

        sample = reader.get_sample(st.session_state.chunk_idx)
        
        if sample:
            st.subheader(f"Chunk #{st.session_state.chunk_idx}")
            
            m = sample['metadata']
            c1, c2, c3 = st.columns(3)
            c1.metric("Kompozytor", m.get('composer', 'Unknown'))
            c2.metric("Tytuł", m.get('title', 'Unknown'))
            c3.metric("Rok", m.get('year', '-'))

            st.divider()
            plot_custom_pianoroll(sample['notes_first'], sample['notes_second'])
            
            with st.expander("Surowe dane JSON"):
                st.json(sample)

    except FileNotFoundError:
        st.error(f"Brak pliku {selected_file}. Uruchom main.py!")
    except Exception as e:
        st.error(f"Błąd: {e}")

if __name__ == "__main__":
    main()
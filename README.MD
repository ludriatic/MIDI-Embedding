# Music LLM Preprocessing

Projekt przygotowujący dane muzyczne (dataset Maestro) do trenowania modelu LLM.

## Opis
Skrypt pobiera dataset `epr-labs/maestro-sustain-v2` (Hugging Face) w trybie streaming, przetwarza nuty z formatu kolumnowego na DataFrame, a następnie dzieli utwory na fragmenty (chunks) metodą Rolling Window.

## Struktura
* `main.py` - Główny skrypt wykonawczy (ETL).
* `note_chunker/` - Moduł zawierający logikę dzielenia nut.
  * `base.py` - Interfejs abstrakcyjny.
  * `rolling_window_note_chunker.py` - Implementacja przesuwanego okna.

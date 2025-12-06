# 🎹 MIDI Embedding Project

> **Next-generation music generation pipeline using Large Language Models (LLM).**

This repository contains the **Data Preprocessing & Visualization** module for the MIDI Embedding project. Its primary goal is to transform raw MIDI performances into tokenized training records suitable for "Next Token Prediction" tasks.

## 🚀 Project Overview

The core objective is to train a generative model to compose expressive piano music. To achieve this, we treat music as a sequence of events.

## 🛠️ Architecture

The project follows Object-Oriented Programming (OOP) principles to ensure scalability and maintainability.

### Core Components

* **`RecordBuilder` (Interface):** An abstract base class defining the contract for data processing strategies.
* **`RollingWindowRecordBuilder`:** Implementation of the sliding window algorithm. It slices continuous musical performances into overlapping chunks (e.g., 100 notes), splitting them into:
    * **Input Context:** Sequence of notes the model sees (e.g., 80 notes).
    * **Prediction Target:** The subsequent notes the model attempts to predict (e.g., 20 notes).
* **`main.py`:** The execution script that orchestrates the streaming, transformation, and saving of `train`, `validation`, and `test` splits.

### Visualization Dashboard

A **Streamlit** dashboard is included to inspect the quality of the generated data. It features:
* Interactive navigation through generated JSONL files.
* **Piano Roll Visualization** (using Altair) to visually compare the Input Context vs. Prediction Target.
* Metadata inspection (Composer, Title, Year).

## 📦 Installation

Prerequisites: Python 3.10+ (Recommended: 3.11).

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd midi-embedding
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv .venv
    # Windows:
    .venv\Scripts\activate
    # Linux/Mac:
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## ⚙️ Usage

### 1. Data Processing
Run the main ETL script to generate the dataset files (`train.jsonl`, `validation.jsonl`, `test.jsonl`):

```bash
python main.py
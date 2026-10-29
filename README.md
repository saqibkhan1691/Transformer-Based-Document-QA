# StudyDoc AI — Transformer-Based Document Question Answering Assistant

Upload a text-based PDF, ask a question, and receive an extractive answer with a source page and supporting passage. This project combines a Transformer QA model with a lightweight TF-IDF retrieval pipeline.

## Architecture

`Streamlit UI → PyMuPDF page extraction → overlapping text chunks → TF-IDF retrieval → DistilBERT QA → answer + page evidence`

## Folder structure

```text
StudyDoc-AI/
├── app.py
├── requirements.txt
├── README.md
├── ARCHITECTURE.md
├── src/
│   ├── pdf_processor.py
│   ├── retriever.py
│   └── qa_model.py
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   └── 02_finetune_distilbert.ipynb
├── data/
├── models/
└── results/
```

## Run in VS Code (Windows)

Python 3.10 or 3.11 is recommended. Open this folder in VS Code, then use PowerShell: 

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, use Command Prompt: `.venv\Scripts\activate.bat`. The first app run downloads `distilbert/distilbert-base-cased-distilled-squad` from Hugging Face. Internet is required for the first download.

## Fine-tune in Jupyter

Open `notebooks/02_finetune_distilbert.ipynb` and run the cells in order. It downloads SQuAD v1.1, fine-tunes DistilBERT on a small subset, prints approximate evaluation metrics, and saves the model to `models/studydoc-qa`. Restart Streamlit after training to use the locally saved model. CPU training can be slow; use a GPU if available.

## Dataset and evaluation

The training notebook uses SQuAD v1.1, a general reading-comprehension dataset with context paragraphs, questions, and answer spans. It is not specifically made from college notes. The notebook prints approximate feature-level Exact Match and token F1 scores; these are educational approximations, not official SQuAD benchmark results. Report only the values you actually get when you run it.

## Limitations

- Scanned/image-only PDFs are not supported (no OCR).
- Tables, columns, and long answers can reduce accuracy.
- TF-IDF is a baseline retriever.
- The model may produce incorrect answers; always verify the displayed passage.


## Troubleshooting: blank or weak answers

- The app works without running notebooks, using a pretrained QA model downloaded from Hugging Face.
- Run `notebooks/02_finetune_distilbert.ipynb` only if you need to demonstrate fine-tuning for the assignment; restart the app after training.
- Ask questions whose answers are explicitly present in the PDF and inspect the supporting passage.
- Scanned PDFs and complex table layouts need OCR/layout-aware processing, which this prototype does not include.
- This model can still make mistakes; verify its answer against the cited text.

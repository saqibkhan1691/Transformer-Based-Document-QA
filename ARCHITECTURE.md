# System Architecture

```text
[Streamlit UI: Upload PDF + Question]
                 |
                 v
[PyMuPDF: Extract text page by page]
                 |
                 v
[Chunking: Text passages + page numbers]
                 |
                 v
[TF-IDF: Retrieve relevant passages]
                 |
                 v
[DistilBERT QA Transformer: Predict answer span]
                 |
                 v
[Display answer + source page + evidence passage]
```

## Training pipeline

SQuAD v1.1 → tokenize question/context → map answer character offsets to token start/end labels → fine-tune DistilBERT → evaluate approximate EM/F1 → save model to `models/studydoc-qa`.

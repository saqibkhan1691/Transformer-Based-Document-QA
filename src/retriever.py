
import re
from typing import List, Dict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def split_into_chunks(
    pages: List[Dict],
    chunk_words: int = 100,
    overlap_words: int = 20,
) -> List[Dict]:
    chunks = []

    for page_item in pages:
        text = page_item.get("text", "").strip()
        if not text:
            continue

        # Preserve paragraph boundaries where PDF extraction provides them.
        paragraphs = re.split(r"\n\s*\n", text)

        units = []
        for paragraph in paragraphs:
            paragraph = re.sub(r"[ \t]+", " ", paragraph).strip()
            if not paragraph:
                continue

            sentences = re.split(r"(?<=[.!?])\s+", paragraph)
            units.extend(s.strip() for s in sentences if s.strip())

        current_words = []

        for unit in units:
            words = unit.split()
            if not words:
                continue

            # Keep each chunk bounded in size.
            while words:
                available = chunk_words - len(current_words)

                if len(words) <= available:
                    current_words.extend(words)
                    words = []
                else:
                    current_words.extend(words[:available])

                    chunks.append({
                        "page": page_item["page"],
                        "text": " ".join(current_words).strip(),
                    })

                    current_words = current_words[-overlap_words:]
                    words = words[available:]

            # Finish a chunk after a sentence when it is already large enough.
            if len(current_words) >= chunk_words:
                chunks.append({
                    "page": page_item["page"],
                    "text": " ".join(current_words).strip(),
                })
                current_words = current_words[-overlap_words:]

        if current_words:
            chunks.append({
                "page": page_item["page"],
                "text": " ".join(current_words).strip(),
            })

    # Remove duplicates and fragments that are too short.
    unique_chunks = []
    seen = set()

    for chunk in chunks:
        content = chunk["text"].strip()
        key = re.sub(r"\W+", " ", content.lower()).strip()

        if len(content.split()) < 8 or key in seen:
            continue

        seen.add(key)
        unique_chunks.append({
            "page": chunk["page"],
            "text": content,
        })

    return unique_chunks


def retrieve_passages(
    question: str,
    chunks: List[Dict],
    top_k: int = 3,
) -> List[Dict]:
    if not question.strip() or not chunks:
        return []

    texts = [chunk["text"] for chunk in chunks]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    try:
        matrix = vectorizer.fit_transform(texts + [question])
    except ValueError:
        return []

    scores = cosine_similarity(
        matrix[-1],
        matrix[:-1]
    ).flatten()

    indices = scores.argsort()[::-1][:min(top_k, len(chunks))]

    results = []
    for index in indices:
        if scores[index] <= 0:
            continue

        results.append({
            **chunks[int(index)],
            "score": float(scores[index]),
        })

    return results

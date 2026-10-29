
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# Project root ke andar saved fine-tuned model
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_MODEL_PATH = PROJECT_ROOT / "models" / "studydoc-qa"

# Fallback model, only if local model is unavailable
FALLBACK_MODEL = "distilbert/distilbert-base-cased-distilled-squad"


class QuestionAnswerer:
    def __init__(self):
        if (
            LOCAL_MODEL_PATH.exists()
            and (LOCAL_MODEL_PATH / "config.json").exists()
            and (LOCAL_MODEL_PATH / "model.safetensors").exists()
        ):
            model_source = str(LOCAL_MODEL_PATH)
            self.model_source = "Your fine-tuned local model"
        else:
            model_source = FALLBACK_MODEL
            self.model_source = "Pretrained fallback model"

        self.tokenizer = AutoTokenizer.from_pretrained(model_source)
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            model_source
        )

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def answer(self, question: str, context: str):
        if not question.strip() or not context.strip():
            return {"answer": "", "score": 0.0}

        encoded = self.tokenizer(
            question.strip(),
            context,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation="only_second",
            max_length=384,
        )

        sequence_ids = encoded.sequence_ids(0)
        encoded.pop("offset_mapping")

        inputs = {
            key: value.to(self.device)
            for key, value in encoded.items()
        }

        outputs = self.model(**inputs)
        start_probs = torch.softmax(outputs.start_logits[0], dim=-1)
        end_probs = torch.softmax(outputs.end_logits[0], dim=-1)

        context_indices = [
            i for i, sid in enumerate(sequence_ids) if sid == 1
        ]

        if not context_indices:
            return {"answer": "", "score": 0.0}

        best_score = -1.0
        best_start = best_end = None

        for start in context_indices:
            max_end = min(context_indices[-1], start + 24)

            for end in range(start, max_end + 1):
                if sequence_ids[end] != 1:
                    continue

                score = (
                    start_probs[start] * end_probs[end]
                ).item()

                if score > best_score:
                    best_score = score
                    best_start, best_end = start, end

        if best_start is None:
            return {"answer": "", "score": 0.0}

        answer_ids = inputs["input_ids"][0][best_start:best_end + 1]
        answer = self.tokenizer.decode(
            answer_ids,
            skip_special_tokens=True,
        ).strip(" .,;:\n\t")

        return {
            "answer": answer,
            "score": float(best_score),
        }

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models/gemma-3-270m")
ADAPTER_BASE = os.path.join(PROJECT_ROOT, "models/obelisk-lora-adapter")
HEAD_PATH = os.path.join(PROJECT_ROOT, "models/obelisk_classifier_head.pt")

OBELISK_LABELS = [
    "nominal",
    "lethal_mutation",
    "thermal_spike",
    "regression_to_rlhf",
    "memory_pressure",
    "sensor_drift",
]

CLASSIFICATION_PROMPT = """Classify system telemetry into exactly one category.

Categories: {labels}

Telemetry: {telemetry}

Category:"""


LABEL_TOKENS = {
    "nominal": 35069,
    "lethal_mutation": 57493,
    "thermal_spike": 11775,
    "regression_to_rlhf": 21055,
    "memory_pressure": 6571,
    "sensor_drift": 13064,
}

LABEL_IDS = list(LABEL_TOKENS.values())


class LogitCalibrator(nn.Module):
    def __init__(self, num_labels: int = 6):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(num_labels) * 2.0)
        self.bias = nn.Parameter(torch.zeros(num_labels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight + self.bias


class ObeliskModel:
    def __init__(self, model_dir: str = MODEL_PATH):
        self.model_dir = model_dir
        self.model = None
        self.tokenizer = None
        self._calibrator = None
        self._device = torch.device("cpu")
        self._labels_str = ", ".join(OBELISK_LABELS)

    def load(self) -> bool:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, trust_remote_code=True)
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_dir,
                torch_dtype=torch.bfloat16,
                device_map="cpu",
                trust_remote_code=True,
            ).eval()

            self._calibrator = LogitCalibrator(num_labels=len(OBELISK_LABELS))

            if os.path.exists(HEAD_PATH):
                try:
                    self._calibrator.load_state_dict(torch.load(HEAD_PATH, map_location=self._device, weights_only=True))
                except Exception:
                    pass

            return True
        except ImportError:
            return False
        except Exception:
            return False

    def _label_logits(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        next_logits = outputs.logits[0, -1, LABEL_IDS]
        return next_logits.unsqueeze(0)

    def classify(self, telemetry: dict, adapter_label: str | None = None) -> str:
        label, _ = self.classify_with_confidence(telemetry, adapter_label)
        return label

    def classify_with_confidence(
        self, telemetry: dict, adapter_label: str | None = None
    ) -> tuple[str, float]:
        if self.model is None:
            return ("nominal", 1.0)

        prompt = CLASSIFICATION_PROMPT.format(
            labels=self._labels_str,
            telemetry=json.dumps(telemetry),
        )
        raw_logits = self._label_logits(prompt)
        cal_logits = self._calibrator(raw_logits)
        probs = F.softmax(cal_logits, dim=-1)
        confidence, pred = probs.max(dim=1)
        return (OBELISK_LABELS[pred.item()], confidence.item())

    def train_head(self, records: list[dict]) -> dict:
        if self.model is None or not records:
            return {}

        from torch.utils.data import DataLoader, TensorDataset

        logits_list = []
        targets_list = []
        for r in records:
            label = r.get("label", "nominal")
            tel = r.get("telemetry", {})
            prompt = CLASSIFICATION_PROMPT.format(
                labels=self._labels_str,
                telemetry=json.dumps(tel),
            )
            raw_logits = self._label_logits(prompt)
            logits_list.append(raw_logits)
            targets_list.append(OBELISK_LABELS.index(label) if label in OBELISK_LABELS else 0)

        logits_tensor = torch.cat(logits_list, dim=0)
        targets = torch.tensor(targets_list, dtype=torch.long)

        dataset = TensorDataset(logits_tensor, targets)
        loader = DataLoader(dataset, batch_size=8, shuffle=True)

        self._calibrator.train()
        optimizer = torch.optim.AdamW(self._calibrator.parameters(), lr=1e-3)

        losses = []
        for epoch in range(100):
            total_loss = 0.0
            for logits, lbl in loader:
                out = self._calibrator(logits)
                loss = F.cross_entropy(out, lbl)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            losses.append(total_loss)
            if epoch > 5 and losses[-1] > losses[-2]:
                break

        self._calibrator.eval()
        with torch.inference_mode():
            out = self._calibrator(logits_tensor)
            preds = out.argmax(dim=1)
            accuracy = (preds == targets).float().mean().item()

        torch.save(self._calibrator.state_dict(), HEAD_PATH)
        return {"accuracy": accuracy, "epochs": len(losses), "loss": losses[-1]}

    def unload(self):
        self.model = None
        self.tokenizer = None
        self._calibrator = None
        import gc
        gc.collect()

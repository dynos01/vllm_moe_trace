import threading
import json
import torch
import datetime

from typing import List

class TraceContext(threading.local):
    def __init__(self) -> None:
        self.clear()

    def add(self, model_name: str, router_logits: torch.Tensor, k: int) -> None:
        self.model_name = model_name
        self.k = k
        self.records.append(router_logits)

    def dump(self, path: str, phase: str) -> None:
        if self.model_name is None:
            return

        try:
            with open(path, "r") as f:
                old_records = json.load(f)
        except:
            old_records = {
                "records": []
            }

        data = {
            "model_name": self.model_name,
            "time": str(datetime.datetime.now()),
            "phase": phase,
            "experts": [_transform(i, self.k) for i in self.records],
        }

        old_records["records"].append(data)

        with open(path, "w") as f:
            json.dump(old_records, f)

        self.clear()

    def clear(self) -> None:
        self.model_name = None
        self.k = 1
        self.records = []

def _transform(router_logits: torch.Tensor, k: int) -> List[int]:
    router_weights = torch.nn.functional.softmax(router_logits, dim=-1)
    _, idx = torch.topk(
        router_weights,
        k,
        dim=-1
    )

    return idx.tolist()

expert_tracer = TraceContext()

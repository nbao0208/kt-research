import json
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
from sklearn.preprocessing import LabelEncoder


class LabelEncoderWithUNK:
    def __init__(self, unk_value: int = -1):
        self.encoder = LabelEncoder()
        self.fitted = False
        self.unk_value = unk_value
        self.classes_: Optional[np.ndarray] = None

    def fit(self, values: List[Any]) -> "LabelEncoderWithUNK":
        self.encoder = self.encoder.fit(values)
        self.classes_ = self.encoder.classes_
        self.fitted = True
        return self

    def transform(self, values: List[Any]) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Encoder not fitted yet")
        result = []
        for v in values:
            if v in self.encoder.classes_:
                result.append(int(self.encoder.transform([v])[0]))
            else:
                result.append(self.unk_value)
        return np.array(result, dtype=np.int64)

    def fit_transform(self, values: List[Any]) -> np.ndarray:
        self.fit(values)
        return self.transform(values)

    def inverse_transform(self, encoded: List[int]) -> List[Any]:
        known = []
        for e in encoded:
            if e == self.unk_value:
                known.append("__UNK__")
            else:
                known.append(self.encoder.inverse_transform([e])[0])
        return known

    def get_mapping(self) -> dict:
        if not self.fitted:
            return {}
        return {str(k): int(v) for k, v in zip(self.encoder.classes_, self.encoder.transform(self.encoder.classes_))}


def save_encoder(encoder: LabelEncoderWithUNK, path: Path) -> None:
    mapping = encoder.get_mapping()
    mapping["__UNK__"] = encoder.unk_value
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(mapping, f, indent=2)


def load_encoder(path: Path) -> LabelEncoderWithUNK:
    with open(path) as f:
        mapping = json.load(f)
    unk_value = mapping.pop("__UNK__", -1)
    encoder = LabelEncoderWithUNK(unk_value=unk_value)
    classes = sorted([k for k in mapping.keys() if k != "__UNK__"], key=lambda x: mapping[x])
    encoder.encoder.classes_ = np.array(classes)
    encoder.classes_ = encoder.encoder.classes_
    encoder.fitted = True
    return encoder


def save_mapping(mapping: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(mapping, f, indent=2, default=str)

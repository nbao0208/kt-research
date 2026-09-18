import logging
from typing import Any, Dict, Optional

import torch

from src.models.attention_kt import AttentiveContextualKT
from src.models.bkt import BKT, BKTTrainer
from src.models.dbkt import DBKTTrainer, DynamicBayesianKnowledgeTracing
from src.models.dkt import DeepKnowledgeTracing
from src.models.lfa import LFAAux, LFACore

logger = logging.getLogger(__name__)


def build_model(
    cfg: Dict[str, Any],
    num_skills: int,
    num_students: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> object:
    model_name = cfg.get("name", "")

    if model_name == "bkt":
        return BKT(
            p_init=cfg.get("p_init", 0.3),
            p_learn=cfg.get("p_learn", 0.1),
            p_slip=cfg.get("p_slip", 0.1),
            p_guess=cfg.get("p_guess", 0.2),
            bounds_min=cfg.get("bounds", {}).get("min", 0.0001),
            bounds_max=cfg.get("bounds", {}).get("max", 0.9999),
        )

    elif model_name == "lfa":
        variant = cfg.get("lfa_variant", "core")
        if variant == "aux":
            n_aux = cfg.get("aux_features", {}).get("n_aux", 0)
            model = LFAAux(
                n_students=num_students or 1,
                n_skills=num_skills,
                n_aux_features=n_aux,
                embedding_dim=cfg.get("embedding_dim", 16),
                dropout=cfg.get("dropout", 0.1),
            )
        else:
            model = LFACore(
                n_students=num_students or 1,
                n_skills=num_skills,
                embedding_dim=cfg.get("embedding_dim", 16),
                dropout=cfg.get("dropout", 0.1),
            )
        return model

    elif model_name == "dbkt":
        return DynamicBayesianKnowledgeTracing(
            num_skills=num_skills,
            feature_dim=len(cfg.get("dynamic_transition", {}).get("features", ["attempts_before", "failure_before"])),
            use_forgetting=cfg.get("dynamic_transition", {}).get("use_forgetting", True),
            constrain_slip_guess=cfg.get("parameters", {}).get("constrain_slip_guess", True),
            slip_guess_penalty=cfg.get("parameters", {}).get("slip_guess_penalty", 1.0),
        )

    elif model_name == "dkt":
        return DeepKnowledgeTracing(
            num_skills=num_skills,
            embedding_dim=cfg.get("embedding_dim", 64),
            hidden_size=cfg.get("hidden_size", 128),
            num_layers=cfg.get("num_layers", 1),
            dropout=cfg.get("dropout", 0.2),
            rnn_type=cfg.get("rnn_type", "lstm"),
            input_mode=cfg.get("input_mode", "embedding"),
        )

    elif model_name == "attention_kt":
        return AttentiveContextualKT(
            num_skills=num_skills,
            embedding_dim=cfg.get("embedding_dim", 64),
            num_heads=cfg.get("num_heads", 2),
            num_layers=cfg.get("num_layers", 1),
            dropout=cfg.get("dropout", 0.2),
            max_seq_len=cfg.get("max_seq_len", 200),
            use_position=cfg.get("use_position", True),
            use_decay=cfg.get("use_decay", True),
            decay_type=cfg.get("decay_type", "position"),
        )

    else:
        raise ValueError(f"Unknown model name: {model_name}")
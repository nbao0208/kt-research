import pytest

from src.models.factory import build_model


class TestModelFactory:
    def test_build_bkt(self):
        cfg = {"name": "bkt", "p_init": 0.3, "p_learn": 0.1, "p_slip": 0.1, "p_guess": 0.2}
        model = build_model(cfg, num_skills=5)
        from src.models.bkt import BKT
        assert isinstance(model, BKT)
        assert model.p_init == 0.3

    def test_build_lfa_core(self):
        cfg = {"name": "lfa", "lfa_variant": "core", "embedding_dim": 8}
        model = build_model(cfg, num_skills=5, num_students=10)
        from src.models.lfa import LFACore
        assert isinstance(model, LFACore)

    def test_build_lfa_aux(self):
        cfg = {"name": "lfa", "lfa_variant": "aux", "embedding_dim": 8, "aux_features": {"n_aux": 3}}
        model = build_model(cfg, num_skills=5, num_students=10)
        from src.models.lfa import LFAAux
        assert isinstance(model, LFAAux)

    def test_build_dbkt(self):
        cfg = {
            "name": "dbkt",
            "dynamic_transition": {"features": ["attempts_before", "failure_before"], "use_forgetting": True},
            "parameters": {"constrain_slip_guess": True, "slip_guess_penalty": 1.0},
        }
        model = build_model(cfg, num_skills=5)
        from src.models.dbkt import DynamicBayesianKnowledgeTracing
        assert isinstance(model, DynamicBayesianKnowledgeTracing)

    def test_build_dkt(self):
        cfg = {"name": "dkt", "embedding_dim": 16, "hidden_size": 32, "rnn_type": "lstm", "input_mode": "embedding"}
        model = build_model(cfg, num_skills=5)
        from src.models.dkt import DeepKnowledgeTracing
        assert isinstance(model, DeepKnowledgeTracing)

    def test_build_attention_kt(self):
        cfg = {"name": "attention_kt", "embedding_dim": 16, "num_heads": 2}
        model = build_model(cfg, num_skills=5)
        from src.models.attention_kt import AttentiveContextualKT
        assert isinstance(model, AttentiveContextualKT)

    def test_unknown_model_raises(self):
        cfg = {"name": "unknown_model"}
        with pytest.raises(ValueError, match="Unknown model name"):
            build_model(cfg, num_skills=5)
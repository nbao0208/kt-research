from pathlib import Path

from omegaconf import OmegaConf

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


class TestConfigs:
    def _load(self, relative_path: str):
        path = CONFIGS_DIR / relative_path
        assert path.exists(), f"Config not found: {path}"
        return OmegaConf.load(path)

    def test_data_config_loads(self):
        cfg = self._load("data/assistments2017.yaml")
        assert cfg.raw_path is not None
        assert cfg.processed_dir is not None
        assert cfg.student_handling.min_student_interactions >= 1
        assert cfg.skill_handling.min_skill_interactions >= 1
        assert cfg.split.train_frac > 0.0

    def test_model_bkt_config_loads(self):
        cfg = self._load("model/bkt.yaml")
        assert 0.0 <= cfg.p_init <= 1.0
        assert cfg.optimize.n_restarts >= 1

    def test_model_lfa_config_loads(self):
        cfg = self._load("model/lfa.yaml")
        assert cfg.embedding_dim >= 1
        assert cfg.learning_rate > 0.0
        assert cfg.batch_size >= 1

    def test_model_dbkt_config_loads(self):
        cfg = self._load("model/dbkt.yaml")
        assert cfg.name == "dbkt"
        assert cfg.dynamic_transition.enabled is True
        assert cfg.optimizer.learning_rate > 0.0

    def test_model_dkt_config_loads(self):
        cfg = self._load("model/dkt.yaml")
        assert cfg.name == "dkt"
        assert cfg.rnn_type in ("lstm", "gru")
        assert cfg.embedding_dim >= 1
        assert cfg.hidden_size >= 1

    def test_model_attention_kt_config_loads(self):
        cfg = self._load("model/attention_kt.yaml")
        assert cfg.name == "attention_kt"
        assert cfg.num_heads >= 1
        assert cfg.embedding_dim >= 1

    def test_trainer_base_config_loads(self):
        cfg = self._load("trainer/base.yaml")
        assert cfg.device == "cpu"

    def test_experiment_bkt_composes(self):
        data_cfg = self._load("data/assistments2017.yaml")
        model_cfg = self._load("model/bkt.yaml")
        trainer_cfg = self._load("trainer/base.yaml")
        exp_cfg = self._load("experiment/bkt_assistments.yaml")
        assert exp_cfg.experiment_name is not None
        assert data_cfg.raw_path is not None
        assert model_cfg.p_init is not None
        assert trainer_cfg.device is not None

    def test_experiment_lfa_core_composes(self):
        exp_cfg = self._load("experiment/lfa_core_assistments.yaml")
        assert exp_cfg.experiment_name is not None
        assert exp_cfg.model.lfa_variant == "core"

    def test_experiment_lfa_aux_composes(self):
        exp_cfg = self._load("experiment/lfa_aux_assistments.yaml")
        assert exp_cfg.experiment_name is not None
        assert exp_cfg.model.lfa_variant == "aux"

    def test_experiment_baseline_composes(self):
        exp_cfg = self._load("experiment/baseline_skill_average.yaml")
        assert exp_cfg.experiment_name is not None
        assert exp_cfg.baseline_type == "skill_average"

    def test_experiment_dbkt_core_composes(self):
        exp_cfg = self._load("experiment/dbkt_core_assistments.yaml")
        assert exp_cfg.experiment_name is not None
        assert "dbkt" in exp_cfg.experiment_name

    def test_experiment_dkt_core_composes(self):
        exp_cfg = self._load("experiment/dkt_core_assistments.yaml")
        assert exp_cfg.experiment_name is not None
        assert "dkt" in exp_cfg.experiment_name

    def test_experiment_attention_core_composes(self):
        exp_cfg = self._load("experiment/attention_core_assistments.yaml")
        assert exp_cfg.experiment_name is not None
        assert "attention" in exp_cfg.experiment_name

    def test_all_experiments_have_unique_names(self):
        names = []
        for exp in [
            "bkt_assistments", "lfa_core_assistments", "lfa_aux_assistments",
            "baseline_skill_average", "dbkt_core_assistments",
            "dkt_core_assistments", "attention_core_assistments",
        ]:
            cfg = self._load(f"experiment/{exp}.yaml")
            names.append(cfg.experiment_name)
        assert len(names) == len(set(names)), "Experiment names should be unique"

import numpy as np
import pandas as pd

from src.models.bkt import BKT, BKTOnlineEvaluator


class TestBKT:
    def test_predict_probability_bounds(self):
        model = BKT(p_init=0.3, p_learn=0.1, p_slip=0.1, p_guess=0.2)
        for p_known in np.linspace(0, 1, 11):
            prob = model.predict(p_known)
            assert 0.0 <= prob <= 1.0, f"p_known={p_known} gave prob={prob}"

    def test_update_returns_valid_probability(self):
        model = BKT(p_init=0.3, p_learn=0.1, p_slip=0.1, p_guess=0.2)
        p_known = model.p_init
        for correct in [1, 0, 1, 0]:
            p_known = model.update(p_known, correct)
            assert 0.0 <= p_known <= 1.0, f"correct={correct} gave p_known={p_known}"

    def test_sequence_log_likelihood_improves_with_more_likely_params(self):
        sequence = [1, 1, 1, 0, 1, 1, 0, 1]
        good_model = BKT(p_init=0.6, p_learn=0.3, p_slip=0.1, p_guess=0.2)
        bad_model = BKT(p_init=0.1, p_learn=0.01, p_slip=0.4, p_guess=0.4)
        good_ll = good_model.sequence_log_likelihood(sequence)
        bad_ll = bad_model.sequence_log_likelihood(sequence)
        assert good_ll > bad_ll, f"Good model LL ({good_ll}) should be > bad model LL ({bad_ll})"

    def test_high_learner_low_slip(self):
        sequence = [1, 1, 1, 1, 1]
        model = BKT(p_init=0.5, p_learn=0.3, p_slip=0.05, p_guess=0.1)
        ll = model.sequence_log_likelihood(sequence)
        assert ll < 0, "Log-likelihood should be negative"

    def test_low_learner_high_slip(self):
        sequence = [0, 0, 0, 0, 0]
        model = BKT(p_init=0.1, p_learn=0.01, p_slip=0.4, p_guess=0.3)
        ll = model.sequence_log_likelihood(sequence)
        assert ll < 0, "Log-likelihood should be negative"

    def test_get_params(self):
        model = BKT(p_init=0.3, p_learn=0.1, p_slip=0.05, p_guess=0.2)
        params = model.get_params()
        assert params["p_init"] == 0.3
        assert params["p_learn"] == 0.1
        assert params["p_slip"] == 0.05
        assert params["p_guess"] == 0.2

    def test_bkt_evaluator_on_synthetic_data(self):
        skill_models = {
            "A": BKT(p_init=0.4, p_learn=0.2, p_slip=0.1, p_guess=0.15),
        }
        evaluator = BKTOnlineEvaluator(skill_models)
        df = pd.DataFrame(
            {
                "studentId": [1, 1, 1, 1, 1],
                "skill": ["A", "A", "A", "A", "A"],
                "correct": [1, 0, 1, 1, 0],
                "startTime": [1, 2, 3, 4, 5],
                "action_num": [1, 1, 1, 1, 1],
            }
        )
        metrics = evaluator.evaluate(df)
        assert "global" in metrics
        assert "auc" in metrics["global"]
        assert "logloss" in metrics["global"]
        assert "n" in metrics["global"]

    def test_multiple_students_evaluation(self):
        skill_models = {
            "A": BKT(p_init=0.3, p_learn=0.1, p_slip=0.1, p_guess=0.2),
        }
        evaluator = BKTOnlineEvaluator(skill_models)
        df = pd.DataFrame(
            {
                "studentId": [1, 1, 2, 2],
                "skill": ["A", "A", "A", "A"],
                "correct": [1, 0, 0, 1],
                "startTime": [1, 2, 1, 2],
                "action_num": [1, 1, 1, 1],
            }
        )
        metrics = evaluator.evaluate(df)
        assert metrics["global"]["n"] == 4

    def test_evaluator_handles_unknown_skill(self):
        evaluator = BKTOnlineEvaluator(skill_models={})
        df = pd.DataFrame(
            {
                "studentId": [1],
                "skill": ["UNKNOWN"],
                "correct": [1],
                "startTime": [1],
                "action_num": [1],
            }
        )
        metrics = evaluator.evaluate(df)
        assert metrics["global"]["n"] == 1


    def test_save_and_load_bkt_params_with_global(self, tmp_path):
        from src.models.bkt import load_bkt_params, save_bkt_params

        skill_models = {
            "math_add": BKT(p_init=0.4, p_learn=0.15, p_slip=0.08, p_guess=0.25),
            "math_sub": BKT(p_init=0.35, p_learn=0.12, p_slip=0.05, p_guess=0.2),
        }
        global_model = BKT(p_init=0.3, p_learn=0.1, p_slip=0.1, p_guess=0.2)

        file_path = tmp_path / "test_bkt_params.csv"
        save_bkt_params(skill_models, global_model, file_path)

        loaded_skills, loaded_global = load_bkt_params(file_path)

        assert len(loaded_skills) == 2
        assert "math_add" in loaded_skills
        assert "math_sub" in loaded_skills
        assert np.isclose(loaded_skills["math_add"].p_init, 0.4)
        assert np.isclose(loaded_skills["math_add"].p_learn, 0.15)
        assert loaded_global is not None
        assert np.isclose(loaded_global.p_init, 0.3)
        assert np.isclose(loaded_global.p_learn, 0.1)

    def test_bkt_trainer_satisfies_cognitive_bounds(self):
        from src.models.bkt import BKTTrainer
        trainer = BKTTrainer({"optimize": {"n_restarts": 3, "max_iter": 50}, "n_jobs": 1})
        sequences = [
            {"skill": "algebra", "sequence": [1, 0, 1, 1, 1, 0, 1]},
            {"skill": "algebra", "sequence": [0, 1, 1, 1, 1, 1]},
        ]
        skill_models, global_model = trainer.fit(sequences)
        assert "algebra" in skill_models
        model = skill_models["algebra"]
        params = model.get_params()
        assert params["p_slip"] <= 0.30
        assert params["p_guess"] <= 0.35
        assert params["p_slip"] + params["p_guess"] < 0.75


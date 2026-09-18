import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import minimize
from tqdm import tqdm

from src.evaluation.metrics import compute_metrics

logger = logging.getLogger(__name__)


class BKT:
    def __init__(
        self,
        p_init: float = 0.3,
        p_learn: float = 0.1,
        p_slip: float = 0.1,
        p_guess: float = 0.2,
        bounds_min: float = 0.0001,
        bounds_max: float = 0.9999,
    ):
        self.p_init = np.clip(p_init, bounds_min, bounds_max)
        self.p_learn = np.clip(p_learn, bounds_min, bounds_max)
        self.p_slip = np.clip(p_slip, bounds_min, bounds_max)
        self.p_guess = np.clip(p_guess, bounds_min, bounds_max)
        self.bounds_min = bounds_min
        self.bounds_max = bounds_max

    def predict(self, p_known: float) -> float:
        return p_known * (1.0 - self.p_slip) + (1.0 - p_known) * self.p_guess

    def update(self, p_known: float, correct: int) -> float:
        p_correct = self.predict(p_known)
        if p_correct == 0.0:
            p_known_given_obs = 0.0
        else:
            if correct == 1:
                p_known_given_obs = (p_known * (1.0 - self.p_slip)) / p_correct
            else:
                p_known_given_obs = (p_known * self.p_slip) / (1.0 - p_correct)

        p_known_given_obs = np.clip(p_known_given_obs, 0.0, 1.0)
        p_known_next = p_known_given_obs + (1.0 - p_known_given_obs) * self.p_learn
        return np.clip(p_known_next, 0.0, 1.0)

    def sequence_log_likelihood(self, sequence: List[int]) -> float:
        p_known = self.p_init
        log_lik = 0.0
        for correct in sequence:
            p_correct = self.predict(p_known)
            p_correct = np.clip(p_correct, self.bounds_min, self.bounds_max)
            log_lik += np.log(p_correct) if correct == 1 else np.log(1.0 - p_correct)
            p_known = self.update(p_known, correct)
        return log_lik

    def get_params(self) -> Dict[str, float]:
        return {
            "p_init": float(self.p_init),
            "p_learn": float(self.p_learn),
            "p_slip": float(self.p_slip),
            "p_guess": float(self.p_guess),
        }


def _bkt_negative_log_likelihood(params: np.ndarray, sequences: List[List[int]]) -> float:
    p_init, p_learn, p_slip, p_guess = params
    model = BKT(p_init=p_init, p_learn=p_learn, p_slip=p_slip, p_guess=p_guess)
    total_nll = 0.0
    for seq in sequences:
        total_nll -= model.sequence_log_likelihood(seq)
    return total_nll


def _fit_restart_worker(
    args: Tuple[List[float], List[List[int]], str, List[Tuple[float, float]], int, float, float]
) -> Tuple[float, Optional[Dict[str, float]]]:
    x0, sequences, method, bounds, max_iter, bounds_min, bounds_max = args
    try:
        result = minimize(
            _bkt_negative_log_likelihood,
            x0=x0,
            args=(sequences,),
            method=method,
            bounds=bounds,
            options={"maxiter": max_iter, "disp": False},
        )
        if result.fun < float("inf"):
            p_init = float(np.clip(result.x[0], bounds[0][0], bounds[0][1]))
            p_learn = float(np.clip(result.x[1], bounds[1][0], bounds[1][1]))
            p_slip = float(np.clip(result.x[2], bounds[2][0], bounds[2][1]))
            p_guess = float(np.clip(result.x[3], bounds[3][0], bounds[3][1]))

            # Degeneracy check: reject solutions where knowing the skill does not meaningfully predict success
            if p_slip + p_guess >= 0.75:
                return float("inf"), None

            params = {
                "p_init": p_init,
                "p_learn": p_learn,
                "p_slip": p_slip,
                "p_guess": p_guess,
            }
            return result.fun, params
    except Exception as e:
        logger.warning("BKT restart optimization failed: %s", e)
    return float("inf"), None


def _fit_skill_worker(args: Tuple[str, List[List[int]], Dict[str, Any]]) -> Tuple[str, Optional[Dict[str, float]], List[List[int]]]:
    skill, skill_seqs, config = args
    single_config = dict(config)
    single_config["n_jobs"] = 1
    trainer = BKTTrainer(single_config)
    model = trainer._fit_single_skill(skill_seqs)
    params = model.get_params() if model is not None else None
    return skill, params, skill_seqs if model is not None else []


class BKTTrainer:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.method = self.config.get("optimize", {}).get("method", "l-bfgs-b")
        self.n_restarts = self.config.get("optimize", {}).get("n_restarts", 5)
        self.max_iter = self.config.get("optimize", {}).get("max_iter", 1000)
        self.bounds_min = self.config.get("bounds", {}).get("min", 0.0001)
        self.bounds_max = self.config.get("bounds", {}).get("max", 0.9999)
        self.fallback_to_global = self.config.get("fallback_to_global", True)
        self.n_jobs = self.config.get("n_jobs", -1)

    def _fit_single_skill(self, sequences: List[List[int]]) -> Optional[BKT]:
        if not sequences:
            return None

        # Cognitive bounds prevent semantic inversion: p_slip <= 0.30, p_guess <= 0.35
        bounds = [
            (self.bounds_min, min(0.999, self.bounds_max)),
            (self.bounds_min, min(0.500, self.bounds_max)),
            (self.bounds_min, min(0.300, self.bounds_max)),
            (self.bounds_min, min(0.350, self.bounds_max)),
        ]
        initial_guesses = [
            [0.3, 0.1, 0.1, 0.2],
            [0.5, 0.05, 0.15, 0.15],
            [0.1, 0.2, 0.05, 0.1],
            [0.7, 0.1, 0.15, 0.1],
            [0.2, 0.25, 0.1, 0.05],
        ]
        guesses = initial_guesses[: min(self.n_restarts, len(initial_guesses))]
        worker_args = [(x0, sequences, self.method, bounds, self.max_iter, self.bounds_min, self.bounds_max) for x0 in guesses]

        if self.n_jobs != 1 and len(guesses) > 1:
            n_jobs_restarts = min(self.n_jobs if self.n_jobs > 0 else 8, len(guesses))
            results = Parallel(n_jobs=n_jobs_restarts)(
                delayed(_fit_restart_worker)(arg) for arg in worker_args
            )
        else:
            results = [_fit_restart_worker(arg) for arg in worker_args]

        best_nll = float("inf")
        best_params = None

        for nll, params in results:
            if params is not None and nll < best_nll:
                best_nll = nll
                best_params = params

        if best_params is not None:
            return BKT(
                p_init=best_params["p_init"],
                p_learn=best_params["p_learn"],
                p_slip=best_params["p_slip"],
                p_guess=best_params["p_guess"],
                bounds_min=self.bounds_min,
                bounds_max=self.bounds_max,
            )
        return None

    def fit(self, sequences: List[Dict[str, Any]], skill_col: str = "skill") -> Tuple[Dict[str, BKT], Optional[BKT]]:
        skill_groups: Dict[str, List[List[int]]] = {}
        for seq in sequences:
            skill = seq[skill_col]
            if skill not in skill_groups:
                skill_groups[skill] = []
            skill_groups[skill].append(seq["sequence"])

        skill_models: Dict[str, BKT] = {}
        global_sequences: List[List[int]] = []

        logger.info("Fitting BKT parameters for %d skills in parallel (n_jobs=%d)...", len(skill_groups), self.n_jobs)
        worker_args = [(skill, skill_seqs, self.config) for skill, skill_seqs in skill_groups.items()]

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_skill_worker)(arg)
            for arg in tqdm(worker_args, desc=f"Fitting BKT Parallel [n_jobs={self.n_jobs}]", unit="skill")
        )

        for skill, params, skill_seqs in results:
            if params is not None:
                skill_models[skill] = BKT(
                    p_init=params["p_init"],
                    p_learn=params["p_learn"],
                    p_slip=params["p_slip"],
                    p_guess=params["p_guess"],
                    bounds_min=self.bounds_min,
                    bounds_max=self.bounds_max,
                )
                global_sequences.extend(skill_seqs)
            else:
                logger.warning("BKT training failed for skill '%s', will use global fallback", skill)

        global_model = None
        if self.fallback_to_global and global_sequences:
            logger.info("Fitting global BKT fallback model on %d sequences in parallel...", len(global_sequences))
            global_model = self._fit_single_skill(global_sequences)
            if global_model:
                logger.info("Trained global BKT model: %s", global_model.get_params())

        return skill_models, global_model


class BKTOnlineEvaluator:
    def __init__(self, skill_models: Dict[str, BKT], global_model: Optional[BKT] = None):
        self.skill_models = skill_models
        self.global_model = global_model
        self.state: Dict[Tuple[str, str], float] = {}

    def reset_state(self) -> None:
        self.state = {}

    def predict(
        self,
        df: pd.DataFrame,
        student_col: str = "studentId",
        skill_col: str = "skill",
        correct_col: str = "correct",
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        sort_cols = [student_col, "startTime"]
        if "action_num" in df.columns:
            sort_cols.append("action_num")
        df_sorted = df.sort_values(sort_cols).reset_index(drop=True).copy()

        n_rows = len(df_sorted)
        if n_rows == 0:
            return np.array([]), np.array([]), df_sorted

        students = df_sorted[student_col].to_numpy()
        skills = df_sorted[skill_col].to_numpy()
        corrects = df_sorted[correct_col].to_numpy(dtype=np.int64)

        y_pred = np.zeros(n_rows, dtype=np.float64)

        for i in range(n_rows):
            student = str(students[i])
            skill = skills[i]
            correct = int(corrects[i])
            key = (student, str(skill))

            p_known = self.state.get(key, None)
            model = self.skill_models.get(skill, self.global_model)
            if model is None:
                y_pred[i] = 0.5
                continue

            if p_known is None:
                p_known = model.p_init
                self.state[key] = p_known

            p_correct = model.predict(p_known)
            y_pred[i] = p_correct
            self.state[key] = model.update(p_known, correct)

        df_sorted["prediction"] = y_pred
        return corrects, y_pred, df_sorted

    def evaluate(
        self,
        df: pd.DataFrame,
        student_col: str = "studentId",
        skill_col: str = "skill",
        correct_col: str = "correct",
    ) -> Dict[str, Any]:
        self.reset_state()
        y_true, y_pred, df_eval = self.predict(df, student_col, skill_col, correct_col)
        from src.evaluation.metrics import compute_comprehensive_metrics
        return compute_comprehensive_metrics(df_eval, y_true_col=correct_col, y_pred_col="prediction", skill_col=skill_col, student_col=student_col)



GLOBAL_SKILL_KEY = "__GLOBAL__"


def save_bkt_params(
    skill_models: Dict[str, BKT],
    global_model: Optional[BKT] = None,
    filepath: Union[str, Path] = "bkt_params.csv",
) -> None:
    """Saves per-skill and global BKT model parameters to a CSV file."""
    bkt_params = []
    for skill, model in skill_models.items():
        params = model.get_params()
        params["skill"] = skill
        bkt_params.append(params)

    if global_model is not None:
        g_params = global_model.get_params()
        g_params["skill"] = GLOBAL_SKILL_KEY
        bkt_params.append(g_params)

    params_df = pd.DataFrame(bkt_params)
    params_df.to_csv(filepath, index=False)
    logger.info("Saved %d skill models + global model (%s) to %s", len(skill_models), global_model is not None, filepath)


def load_bkt_params(filepath: Union[str, Path]) -> Tuple[Dict[str, BKT], Optional[BKT]]:
    """Loads per-skill and global BKT models from a CSV file."""
    params_df = pd.read_csv(filepath)
    skill_models: Dict[str, BKT] = {}
    global_model: Optional[BKT] = None

    for _, row in params_df.iterrows():
        skill_name = str(row["skill"])
        model = BKT(
            p_init=float(row["p_init"]),
            p_learn=float(row["p_learn"]),
            p_slip=float(row["p_slip"]),
            p_guess=float(row["p_guess"]),
        )
        if skill_name == GLOBAL_SKILL_KEY:
            global_model = model
        else:
            skill_models[skill_name] = model

    return skill_models, global_model


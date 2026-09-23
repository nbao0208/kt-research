import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluation.metrics import compute_metrics
from src.models.llm_reasoner import BaseLLMReasoner, PedagogicalPromptBuilder
from src.models.sfn_kt import (
    CognitiveAnomalyRegulator,
    CognitiveWeightedBCELoss,
    SFNKTModel,
    SoftECELoss,
)
from src.utils.io import ensure_dir, save_json
from src.utils.logging import WandbLogger

logger = logging.getLogger(__name__)


class SFNKTTrainer:
    """
    3-Stage Decoupled Trainer for SFN-KT:
      - Stage 1: Fast Backbone independent optimization & base sensor freezing.
      - Stage 2: SCDT cognitive anomaly scan, Q-Former alignment, and offline HDF5 caching.
      - Stage 3: Two-phase Multi-Anchor Causal Adapter training & Soft-ECE joint calibration.
    """

    def __init__(
        self,
        model: SFNKTModel,
        llm_reasoner: BaseLLMReasoner,
        scdt_regulator: CognitiveAnomalyRegulator,
        device: torch.device = torch.device("cpu"),
        checkpoint_dir: Union[str, Path] = "outputs/artifacts/sfn_kt",
        h5_cache_path: Union[str, Path] = "outputs/artifacts/sfn_kt/cognitive_qformer_cache.h5",
        metadata_manager: Optional[Any] = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        stage1_epochs: int = 10,
        stage2_epochs: int = 3,
        stage3_epochs: int = 15,
        warmup_epochs: int = 5,
        stage2_max_alignment_samples: int = 2000,
        alpha: float = 0.75,
        beta: float = 0.1,
        tau_s: float = 0.05,
        gradient_clip_norm: float = 1.0,
        wandb_logger: Optional[WandbLogger] = None,
    ):
        # Multi-GPU resolution: when 2+ CUDA devices are available and running on CUDA,
        # place Fast Backbone & Q-Former on cuda:0 and LLM Reasoner on cuda:1.
        # Otherwise (1 GPU, MPS, CPU), keep both on the same specified device.
        str_dev = str(device).lower()
        if torch.cuda.is_available() and torch.cuda.device_count() >= 2 and "cuda" in str_dev:
            self.device = torch.device("cuda:0")
            self.llm_device = torch.device("cuda:1")
            logger.info(
                "Multi-GPU detected (%d CUDA devices). Using Pipeline Allocation: "
                "Fast Backbone & Q-Former on %s, LLM Reasoner on %s.",
                torch.cuda.device_count(),
                self.device,
                self.llm_device,
            )
        else:
            self.device = device
            self.llm_device = device
            logger.info("Single-device mode: Model and LLM Reasoner running on %s.", self.device)

        self.model = model.to(self.device)
        self.llm_reasoner = llm_reasoner.to(self.llm_device)
        self.scdt = scdt_regulator
        self.checkpoint_dir = Path(checkpoint_dir)
        self.h5_cache_path = Path(h5_cache_path)
        self.metadata = metadata_manager

        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.stage1_epochs = stage1_epochs
        self.stage2_epochs = stage2_epochs
        self.stage3_epochs = stage3_epochs
        self.warmup_epochs = warmup_epochs
        self.stage2_max_alignment_samples = stage2_max_alignment_samples
        self.alpha = alpha
        self.beta = beta
        self.tau_s = tau_s
        self.gradient_clip_norm = gradient_clip_norm
        self.wandb_logger = wandb_logger

        self.loss_bce_masked = nn.BCEWithLogitsLoss(reduction="none")
        self.loss_weighted_bce = CognitiveWeightedBCELoss(alpha=alpha)
        self.loss_soft_ece = SoftECELoss(num_bins=10, tau_s=tau_s).to(device)
        self.prompt_builder = PedagogicalPromptBuilder(track="sparse")

        ensure_dir(self.checkpoint_dir)
        ensure_dir(self.h5_cache_path.parent)

    # -------------------------------------------------------------------------
    # STAGE 1: Fast Backbone Independent Training
    # -------------------------------------------------------------------------
    def train_stage1(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict[str, Any]:
        logger.info("=== Starting Stage 1: Fast Backbone Training (%d epochs) ===", self.stage1_epochs)

        # Optimize Fast Backbone & Base Prediction Head
        trainable_params = list(self.model.embedding_layer.parameters()) + \
                           list(self.model.fast_backbone.parameters()) + \
                           list(self.model.base_head.parameters())
        optimizer = torch.optim.AdamW(trainable_params, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.stage1_epochs, eta_min=1e-5)

        best_val_auc = 0.0
        best_path = self.checkpoint_dir / "fast_backbone_best.pt"

        history = []
        for epoch in range(1, self.stage1_epochs + 1):
            self.model.train()
            total_loss = 0.0
            n_batches = 0

            pbar = tqdm(train_loader, desc=f"Stage 1 [Epoch {epoch}/{self.stage1_epochs}]", leave=False)
            for batch in pbar:
                q_ids = batch["questions"].to(self.device)
                c_ids = batch["concepts"].to(self.device)
                responses = batch["responses"].to(self.device)
                selectmasks = batch["selectmasks"].to(self.device)
                padding_mask = ~batch["mask"].to(self.device)

                logits_base, _, _ = self.model.forward_fast_only(q_ids, c_ids, responses, padding_mask=padding_mask)

                # Target labels at step t+1: responses[:, 1:]
                targets = responses[:, 1:].float()
                # Eval mask: selectmasks == 1 and non-padded
                eval_mask = (selectmasks[:, 1:] == 1) & (~padding_mask[:, 1:])

                loss_matrix = self.loss_bce_masked(logits_base, targets)
                valid_loss = (loss_matrix * eval_mask.float()).sum() / (eval_mask.sum().float() + 1e-8)

                optimizer.zero_grad()
                valid_loss.backward()
                if self.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, self.gradient_clip_norm)
                optimizer.step()

                total_loss += valid_loss.item()
                n_batches += 1
                pbar.set_postfix({"loss": f"{valid_loss.item():.4f}"})

            scheduler.step()
            avg_train_loss = total_loss / max(1, n_batches)

            # Validation
            val_metrics = self._evaluate_fast_backbone(val_loader)
            val_auc = val_metrics["auc"]
            val_acc = val_metrics["accuracy"]
            logger.info(
                "Stage 1 Epoch %d/%d - Train Loss: %.4f | Val Loss: %.4f | Val AUC: %.4f | Val ACC: %.4f",
                epoch, self.stage1_epochs, avg_train_loss, val_metrics["logloss"], val_auc, val_acc
            )

            if self.wandb_logger and self.wandb_logger.enabled:
                self.wandb_logger.log_metrics({
                    "stage1/train_loss": avg_train_loss,
                    "stage1/val_loss": val_metrics["logloss"],
                    "stage1/val_auc": val_auc,
                    "stage1/val_acc": val_acc,
                    "stage1/lr": scheduler.get_last_lr()[0],
                }, step=epoch)

            history.append({"epoch": epoch, "train_loss": avg_train_loss, "val_metrics": val_metrics})

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "val_auc": val_auc,
                }, best_path)
                logger.info("  --> Saved best Fast Backbone checkpoint to %s (AUC=%.4f)", best_path, val_auc)

        # Load best weights
        if best_path.exists():
            checkpoint = torch.load(best_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            logger.info("Loaded best Fast Backbone checkpoint (AUC=%.4f)", checkpoint["val_auc"])

        # Permanently freeze Base Prediction Head as invariant sensor
        for param in self.model.base_head.parameters():
            param.requires_grad = False
        logger.info("Base Prediction Head frozen permanently as invariant diagnostic sensor.")

        return {"best_val_auc": best_val_auc, "history": history, "best_path": str(best_path)}

    @torch.no_grad()
    def _evaluate_fast_backbone(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        all_preds = []
        all_targets = []

        for batch in loader:
            q_ids = batch["questions"].to(self.device)
            c_ids = batch["concepts"].to(self.device)
            responses = batch["responses"].to(self.device)
            selectmasks = batch["selectmasks"].to(self.device)
            padding_mask = ~batch["mask"].to(self.device)

            _, p_base, _ = self.model.forward_fast_only(q_ids, c_ids, responses, padding_mask=padding_mask)

            targets = responses[:, 1:]
            eval_mask = (selectmasks[:, 1:] == 1) & (~padding_mask[:, 1:])

            p_valid = p_base[eval_mask].cpu().numpy()
            y_valid = targets[eval_mask].cpu().numpy()

            all_preds.extend(p_valid)
            all_targets.extend(y_valid)

        if len(all_targets) == 0:
            return {"auc": 0.5, "accuracy": 0.0, "logloss": 0.0, "brier": 0.0}

        return compute_metrics(np.array(all_targets), np.array(all_preds))

    # -------------------------------------------------------------------------
    # STAGE 2: SCDT Anomaly Scan, Q-Former Alignment & Tensor Caching
    # -------------------------------------------------------------------------
    def _collect_triggered_events(
        self,
        loader: DataLoader,
        max_samples: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Collects interactions where SCDT anomaly score S_t > tau*."""
        triggered_events = []
        with torch.no_grad():
            for batch in loader:
                q_ids = batch["questions"].to(self.device)
                c_ids = batch["concepts"].to(self.device)
                responses = batch["responses"].to(self.device)
                selectmasks = batch["selectmasks"].to(self.device)
                uids = batch["uid"].cpu().numpy()
                padding_mask = ~batch["mask"].to(self.device)

                _, p_base, _ = self.model.forward_fast_only(q_ids, c_ids, responses, padding_mask=padding_mask)
                targets = responses[:, 1:]
                eval_mask = (selectmasks[:, 1:] == 1) & (~padding_mask[:, 1:])

                scores = self.scdt.compute_anomaly_score(
                    p_base, targets, lambda_entropy=self.scdt.lambda_entropy
                )
                trigger_mask = self.scdt.get_trigger_mask(scores, eval_mask)  # [B, T-1]

                B, T_minus_1 = trigger_mask.shape
                for b in range(B):
                    uid = int(uids[b])
                    for t in range(T_minus_1):
                        if trigger_mask[b, t].item():
                            actual_t = t + 1  # 1-indexed in sequence
                            q_val = int(q_ids[b, actual_t].item())
                            c_val = int(c_ids[b, actual_t].item())
                            r_val = int(targets[b, t].item())

                            # Extract prior trajectory strictly 0 to t-1
                            prior_hist = []
                            for prev in range(actual_t):
                                if not padding_mask[b, prev].item():
                                    prev_q = int(q_ids[b, prev].item())
                                    prev_c = int(c_ids[b, prev].item())
                                    c_name = self.metadata.get_kc_name_from_encoded_id(prev_c) if self.metadata else f"Concept_{prev_c}"
                                    prior_hist.append({
                                        "question_id": prev_q,
                                        "concept_name": c_name,
                                        "correctness": int(responses[b, prev].item()),
                                    })

                            triggered_events.append({
                                "uid": uid,
                                "step": actual_t,
                                "qid": q_val,
                                "cid": c_val,
                                "response": r_val,
                                "prior_history": prior_hist,
                            })

                if max_samples and len(triggered_events) >= max_samples:
                    triggered_events = triggered_events[:max_samples]
                    break
        return triggered_events

    def _build_prompt_for_event(self, e: Dict[str, Any]) -> str:
        """Constructs prompt using real XES3G5M metadata if available."""
        if self.metadata is not None:
            q_info = self.metadata.get_question_info_from_encoded_id(e["qid"])
            kc_name = self.metadata.get_kc_name_from_encoded_id(e["cid"])
            q_text = q_info.get("content", f"Question {e['qid']}")
            analysis = q_info.get("analysis")
            options = q_info.get("options")
            ans = q_info.get("answer")
            student_ans = str(ans[0]) if (ans and e["response"] == 1) else None
        else:
            q_text = f"Math problem for question {e['qid']}"
            kc_name = f"Concept {e['cid']}"
            analysis = None
            options = None
            student_ans = None

        return self.prompt_builder.build_prompt(
            question_text=q_text,
            kc_name=kc_name,
            correctness=e["response"],
            prior_history=e["prior_history"],
            analysis=analysis,
            options=options,
            student_answer=student_ans,
        )

    def run_stage2_scan_and_cache(
        self,
        val_loader: DataLoader,
        train_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        max_cache_samples: Optional[int] = None,
    ) -> Dict[str, Any]:
        logger.info("=== Starting Stage 2: SCDT Anomaly Scan & Q-Former Offline Caching ===")
        self.model.eval()

        # Step 2.1: Calibrate dynamic threshold tau* on Validation set
        logger.info("Step 2.1: Scanning Validation set to calibrate tau*...")
        val_scores = []
        val_masks = []
        with torch.no_grad():
            for batch in val_loader:
                q_ids = batch["questions"].to(self.device)
                c_ids = batch["concepts"].to(self.device)
                responses = batch["responses"].to(self.device)
                selectmasks = batch["selectmasks"].to(self.device)
                padding_mask = ~batch["mask"].to(self.device)

                _, p_base, _ = self.model.forward_fast_only(q_ids, c_ids, responses, padding_mask=padding_mask)
                targets = responses[:, 1:]
                eval_mask = (selectmasks[:, 1:] == 1) & (~padding_mask[:, 1:])

                scores = self.scdt.compute_anomaly_score(
                    p_base, targets, lambda_entropy=self.scdt.lambda_entropy
                )
                val_scores.append(scores.cpu())
                val_masks.append(eval_mask.cpu())

        all_val_scores = torch.cat(val_scores, dim=0)
        all_val_masks = torch.cat(val_masks, dim=0)
        tau_star = self.scdt.calibrate_threshold(all_val_scores, all_val_masks)

        # Step 2.2: Identify triggered interactions on training set & align Cognitive Q-Former
        logger.info("Step 2.2: Identifying triggers on Train set and aligning Cognitive Q-Former...")
        train_triggered_events = self._collect_triggered_events(train_loader, max_samples=max_cache_samples)
        total_train_triggers = len(train_triggered_events)
        logger.info("Total Train anomaly triggers detected: %d", total_train_triggers)

        # Train Q-Former on representative alignment triggers as specified in model_arch.md
        # ("tối ưu hóa Q-Former thông qua hai đầu dò phụ siêu nhẹ trong vài nghìn bước đầu: khoảng 15 đến 25 phút")
        alignment_size = min(total_train_triggers, getattr(self, "stage2_max_alignment_samples", 2000))
        alignment_events = train_triggered_events[:alignment_size]
        logger.info(
            "Selected %d representative triggers for Cognitive Q-Former semantic alignment.",
            alignment_size,
        )

        cached_alignment_tensors = {}

        if alignment_size > 0:
            qformer_optimizer = torch.optim.AdamW(
                self.model.qformer.parameters(), lr=1e-3, weight_decay=1e-4
            )
            is_hf_llm = hasattr(self.llm_reasoner, "model")
            align_batch_size = min(
                16 if is_hf_llm else (32 if "mps" in str(self.device).lower() else 64),
                alignment_size,
            )

            # Step 2.2a: Pre-extract H_cot for the alignment subset with progress bar
            logger.info("Extracting H_cot representations for Q-Former alignment subset...")
            alignment_h_cot = []
            pbar_cot = tqdm(
                range(0, alignment_size, align_batch_size),
                desc="Stage 2 [Step 2.2: Extracting CoT for Alignment]",
                leave=False,
            )
            for s_idx in pbar_cot:
                chunk = alignment_events[s_idx : s_idx + align_batch_size]
                prompts = [self._build_prompt_for_event(e) for e in chunk]
                q_list = [e["qid"] for e in chunk]
                c_list = [e["cid"] for e in chunk]
                r_list = [e["response"] for e in chunk]

                with torch.no_grad():
                    h_chunk = self.llm_reasoner.extract_hidden_states(
                        prompts=prompts,
                        question_ids=q_list,
                        concept_ids=c_list,
                        responses=r_list,
                    )
                alignment_h_cot.append(h_chunk.cpu())

            all_alignment_h_cot = torch.cat(alignment_h_cot, dim=0)  # [N_align, K, d_llm]

            # Step 2.2b: Train Q-Former for stage2_epochs on GPU/VRAM with rapid convergence
            pbar_epochs = tqdm(range(1, self.stage2_epochs + 1), desc="Stage 2 [Step 2.2: Q-Former Alignment Training]")
            for q_epoch in pbar_epochs:
                self.model.qformer.train()
                perm = np.random.permutation(alignment_size)
                epoch_loss = 0.0
                n_align_batches = 0

                for s_idx in range(0, alignment_size, align_batch_size):
                    batch_idx = perm[s_idx : s_idx + align_batch_size]
                    h_batch = all_alignment_h_cot[batch_idx].to(self.device)
                    chunk_events = [alignment_events[i] for i in batch_idx]

                    c_list = [e["cid"] for e in chunk_events]
                    r_list = [e["response"] for e in chunk_events]
                    q_list = [e["qid"] for e in chunk_events]

                    z_cog = self.model.qformer(h_batch)
                    target_concepts = torch.tensor(c_list, dtype=torch.long, device=self.device)
                    target_responses = torch.tensor(r_list, dtype=torch.long, device=self.device)
                    target_q_tensor = torch.tensor(q_list, dtype=torch.long, device=self.device)

                    with torch.no_grad():
                        target_diffs = self.model.embedding_layer.question_diff(target_q_tensor).squeeze(-1)

                    qf_loss = self.model.qformer.compute_probe_loss(
                        z_cog, target_concepts, target_responses, target_difficulty=target_diffs, lambda_diff=0.5
                    )

                    qformer_optimizer.zero_grad()
                    qf_loss.backward()
                    qformer_optimizer.step()

                    epoch_loss += qf_loss.item()
                    n_align_batches += 1

                avg_qf_loss = epoch_loss / max(1, n_align_batches)
                pbar_epochs.set_postfix({"qf_loss": f"{avg_qf_loss:.4f}"})
                logger.info("Q-Former Alignment Epoch %d/%d - Loss: %.4f", q_epoch, self.stage2_epochs, avg_qf_loss)

            # Freeze Q-Former permanently as specified in model_arch.md
            for p in self.model.qformer.parameters():
                p.requires_grad = False
            self.model.qformer.eval()
            logger.info("Cognitive Q-Former successfully aligned and permanently frozen.")

            # Compute and cache Z_t^cog for all alignment events immediately without re-calling LLM
            with torch.no_grad():
                for s_idx in range(0, alignment_size, align_batch_size):
                    h_chunk = all_alignment_h_cot[s_idx : s_idx + align_batch_size].to(self.device)
                    chunk_events = alignment_events[s_idx : s_idx + align_batch_size]
                    z_cog = self.model.qformer(h_chunk)
                    z_np = z_cog.cpu().numpy().astype(np.float16)
                    for i_in_chunk, e in enumerate(chunk_events):
                        key = f"{e['uid']}_{e['step']}"
                        cached_alignment_tensors[key] = z_np[i_in_chunk]

            del all_alignment_h_cot, alignment_h_cot

        # Step 2.3: Generate and store compressed tensors Z_t^cog in HDF5 cache for train, val, and test splits
        logger.info("Step 2.3: Writing multi-split offline tensor cache to %s...", self.h5_cache_path)
        self.model.qformer.eval()

        val_triggered_events = self._collect_triggered_events(val_loader, max_samples=max_cache_samples)
        logger.info("Collected %d Validation anomaly triggers for offline caching.", len(val_triggered_events))

        test_triggered_events = []
        if test_loader is not None:
            test_triggered_events = self._collect_triggered_events(test_loader, max_samples=max_cache_samples)
            logger.info("Collected %d Test anomaly triggers for offline caching.", len(test_triggered_events))

        all_events_to_cache = list(train_triggered_events) + val_triggered_events + test_triggered_events
        total_to_cache = len(all_events_to_cache)
        logger.info("Total interactions to cache across splits: %d", total_to_cache)

        # Open in 'a' mode for resumable caching without duplicate re-inference
        with h5py.File(self.h5_cache_path, "a") as h5f:
            existing_keys = set(h5f.keys())

            # First write any pre-computed alignment tensors
            for key, tensor_np in cached_alignment_tensors.items():
                if key not in h5f:
                    h5f.create_dataset(key, data=tensor_np, dtype="float16")
                    existing_keys.add(key)

            # Filter remaining uncached events
            uncached_events = [e for e in all_events_to_cache if f"{e['uid']}_{e['step']}" not in existing_keys]
            n_uncached = len(uncached_events)
            logger.info(
                "Cache status: %d already cached, %d remaining to extract and store.",
                len(existing_keys),
                n_uncached,
            )

            if n_uncached > 0:
                is_hf_llm = hasattr(self.llm_reasoner, "model")
                cache_chunk_size = 16 if is_hf_llm else (32 if "mps" in str(self.device).lower() else 64)
                pbar_cache = tqdm(
                    range(0, n_uncached, cache_chunk_size),
                    desc="Stage 2 [Step 2.3: Offline HDF5 Caching]",
                    leave=True,
                )
                for start_idx in pbar_cache:
                    chunk_events = uncached_events[start_idx : start_idx + cache_chunk_size]
                    prompts = [self._build_prompt_for_event(e) for e in chunk_events]
                    q_list = [e["qid"] for e in chunk_events]
                    c_list = [e["cid"] for e in chunk_events]
                    r_list = [e["response"] for e in chunk_events]

                    with torch.no_grad():
                        h_cot = self.llm_reasoner.extract_hidden_states(
                            prompts=prompts,
                            question_ids=q_list,
                            concept_ids=c_list,
                            responses=r_list,
                        )
                        if h_cot.device != self.device:
                            h_cot = h_cot.to(self.device)
                        z_cog = self.model.qformer(h_cot)  # [chunk, M, d]
                        z_np = z_cog.cpu().numpy().astype(np.float16)

                    for idx_in_chunk, e in enumerate(chunk_events):
                        key = f"{e['uid']}_{e['step']}"
                        if key not in h5f:
                            h5f.create_dataset(key, data=z_np[idx_in_chunk], dtype="float16")

                    pbar_cache.set_postfix({"cached": min(start_idx + cache_chunk_size, n_uncached)})

            total_stored = len(h5f.keys())

        cache_size_mb = self.h5_cache_path.stat().st_size / (1024 * 1024)
        logger.info(
            "Stage 2 complete: Cached %d compressed tensors in %s (File size: %.2f MB)",
            total_stored, self.h5_cache_path.name, cache_size_mb
        )

        stage2_summary = {
            "tau_star": tau_star,
            "target_rho": self.scdt.target_trigger_rate,
            "train_triggers": total_train_triggers,
            "total_cached_tensors": total_stored,
            "cache_file": str(self.h5_cache_path),
            "cache_size_mb": round(cache_size_mb, 2),
        }
        save_json(stage2_summary, self.checkpoint_dir / "stage2_summary.json")
        return stage2_summary

    # -------------------------------------------------------------------------
    # STAGE 3: Multi-Anchor Causal Adapter Training & Calibration
    # -------------------------------------------------------------------------
    def train_stage3(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict[str, Any]:
        logger.info("=== Starting Stage 3: Multi-Anchor Causal Adapter Training (%d epochs) ===", self.stage3_epochs)

        if not self.h5_cache_path.exists():
            logger.warning("HDF5 cache not found at %s. Running with empty cognitive memory.", self.h5_cache_path)

        # Phase 1: Adapter Warmup (Freeze Fast Backbone completely)
        for param in self.model.fast_backbone.parameters():
            param.requires_grad = False
        for param in self.model.embedding_layer.parameters():
            param.requires_grad = False

        warmup_params = list(self.model.adapter.parameters()) + list(self.model.final_head.parameters())
        optimizer = torch.optim.AdamW(warmup_params, lr=self.lr, weight_decay=self.weight_decay)

        best_val_auc = 0.0
        best_path = self.checkpoint_dir / "sfn_kt_best.pt"

        # Open HDF5 cache for fast read
        h5_cache = h5py.File(self.h5_cache_path, "r") if self.h5_cache_path.exists() else None

        for epoch in range(1, self.stage3_epochs + 1):
            if epoch == self.warmup_epochs + 1:
                logger.info("Phase 2 Joint Training: Unfreezing final layer of Fast Backbone...")
                # Unfreeze last layer of transformer encoder
                for param in self.model.fast_backbone.encoder.layers[-1].parameters():
                    param.requires_grad = True

                joint_params = [p for p in self.model.parameters() if p.requires_grad]
                optimizer = torch.optim.AdamW(joint_params, lr=self.lr * 0.5, weight_decay=self.weight_decay)

            self.model.train()
            total_loss = 0.0
            n_batches = 0

            pbar = tqdm(train_loader, desc=f"Stage 3 [Epoch {epoch}/{self.stage3_epochs}]", leave=False)
            for batch in pbar:
                q_ids = batch["questions"].to(self.device)
                c_ids = batch["concepts"].to(self.device)
                responses = batch["responses"].to(self.device)
                selectmasks = batch["selectmasks"].to(self.device)
                uids = batch["uid"].cpu().numpy()
                padding_mask = ~batch["mask"].to(self.device)

                B, T = q_ids.size()
                M = self.model.num_queries
                d = self.model.d_model

                # Build z_memory and trigger_mask from cache
                z_memory = torch.zeros((B, T, M, d), dtype=torch.float32, device=self.device)
                trigger_mask = torch.zeros((B, T), dtype=torch.bool, device=self.device)

                if h5_cache is not None:
                    for b in range(B):
                        uid = int(uids[b])
                        for t in range(T):
                            key = f"{uid}_{t}"
                            if key in h5_cache:
                                z_memory[b, t] = torch.tensor(
                                    h5_cache[key][:], dtype=torch.float32, device=self.device
                                )
                                trigger_mask[b, t] = True

                logits_final, p_calibrated, _ = self.model.forward_calibrated(
                    q_ids, c_ids, responses, z_memory, trigger_mask, padding_mask=padding_mask
                )

                targets = responses[:, 1:]
                eval_mask = (selectmasks[:, 1:] == 1) & (~padding_mask[:, 1:])

                # Cognitive-Weighted BCE Loss
                loss_bce = self.loss_weighted_bce(
                    logits_final, targets, trigger_mask[:, :-1], eval_mask
                )

                # Soft-ECE Regularization
                loss_ece = self.loss_soft_ece(p_calibrated, targets, eval_mask)

                # Composite Stage 3 loss
                loss_stage3 = loss_bce + self.beta * loss_ece

                optimizer.zero_grad()
                loss_stage3.backward()
                if self.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                optimizer.step()

                total_loss += loss_stage3.item()
                n_batches += 1
                pbar.set_postfix({"loss": f"{loss_stage3.item():.4f}", "ece": f"{loss_ece.item():.4f}"})

            avg_loss = total_loss / max(1, n_batches)
            val_metrics = self.evaluate(val_loader, h5_cache=h5_cache)
            val_auc = val_metrics["calibrated"]["auc"]
            base_auc = val_metrics["base"]["auc"]
            gain_auc = val_auc - base_auc

            logger.info(
                "Stage 3 Epoch %d/%d - Loss: %.4f | Base AUC: %.4f | SFN-KT AUC: %.4f (Gain: %+.4f) | ECE: %.4f",
                epoch, self.stage3_epochs, avg_loss, base_auc, val_auc, gain_auc, val_metrics["calibrated"]["brier"]
            )

            if self.wandb_logger and self.wandb_logger.enabled:
                self.wandb_logger.log_metrics({
                    "stage3/train_loss": avg_loss,
                    "stage3/val_base_auc": base_auc,
                    "stage3/val_sfn_kt_auc": val_auc,
                    "stage3/val_gain_auc": gain_auc,
                    "stage3/val_ece_proxy": val_metrics["calibrated"]["brier"],
                }, step=epoch)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "val_auc": val_auc,
                    "base_auc": base_auc,
                    "gain_auc": gain_auc,
                }, best_path)
                logger.info("  --> Saved best SFN-KT checkpoint to %s (AUC=%.4f)", best_path, val_auc)

        if h5_cache is not None:
            h5_cache.close()

        return {"best_sfn_kt_auc": best_val_auc, "best_path": str(best_path)}

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
        h5_cache: Optional[Any] = None,
        save_predictions_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation comparing Base Fast Core vs Calibrated SFN-KT.
        Reports global metrics, active vs inactive breakdown, and calibration.
        """
        self.model.eval()
        close_cache_at_end = False
        if h5_cache is None and self.h5_cache_path.exists():
            h5_cache = h5py.File(self.h5_cache_path, "r")
            close_cache_at_end = True

        all_base_preds = []
        all_calibrated_preds = []
        all_targets = []
        all_active_flags = []
        all_uids = []
        all_qids = []

        for batch in loader:
            q_ids = batch["questions"].to(self.device)
            c_ids = batch["concepts"].to(self.device)
            responses = batch["responses"].to(self.device)
            selectmasks = batch["selectmasks"].to(self.device)
            uids = batch["uid"].cpu().numpy()
            padding_mask = ~batch["mask"].to(self.device)

            B, T = q_ids.size()
            M = self.model.num_queries
            d = self.model.d_model

            z_memory = torch.zeros((B, T, M, d), dtype=torch.float32, device=self.device)
            trigger_mask = torch.zeros((B, T), dtype=torch.bool, device=self.device)

            if h5_cache is not None:
                for b in range(B):
                    uid = int(uids[b])
                    for t in range(T):
                        key = f"{uid}_{t}"
                        if key in h5_cache:
                            z_memory[b, t] = torch.tensor(
                                h5_cache[key][:], dtype=torch.float32, device=self.device
                            )
                            trigger_mask[b, t] = True

            # Fast base predictions
            _, p_base, _ = self.model.forward_fast_only(q_ids, c_ids, responses, padding_mask=padding_mask)

            # Calibrated predictions
            _, p_calibrated, _ = self.model.forward_calibrated(
                q_ids, c_ids, responses, z_memory, trigger_mask, padding_mask=padding_mask
            )

            targets = responses[:, 1:]
            eval_mask = (selectmasks[:, 1:] == 1) & (~padding_mask[:, 1:])

            has_prior_trigger = (torch.cumsum(trigger_mask.float(), dim=1) > 0)[:, :-1]

            # Flatten valid entries
            p_base_valid = p_base[eval_mask].cpu().numpy()
            p_cal_valid = p_calibrated[eval_mask].cpu().numpy()
            y_valid = targets[eval_mask].cpu().numpy()
            active_valid = has_prior_trigger[eval_mask].cpu().numpy()

            all_base_preds.extend(p_base_valid)
            all_calibrated_preds.extend(p_cal_valid)
            all_targets.extend(y_valid)
            all_active_flags.extend(active_valid)

        if close_cache_at_end and h5_cache is not None:
            h5_cache.close()

        y_true = np.array(all_targets)
        y_base = np.array(all_base_preds)
        y_cal = np.array(all_calibrated_preds)
        active_arr = np.array(all_active_flags)

        base_metrics = compute_metrics(y_true, y_base)
        calibrated_metrics = compute_metrics(y_true, y_cal)

        # Active vs Inactive breakdown
        active_mask = active_arr == 1
        active_metrics = compute_metrics(y_true[active_mask], y_cal[active_mask]) if active_mask.sum() > 0 else {}
        inactive_metrics = compute_metrics(y_true[~active_mask], y_cal[~active_mask]) if (~active_mask).sum() > 0 else {}

        results = {
            "base": base_metrics,
            "calibrated": calibrated_metrics,
            "gain_auc": round(calibrated_metrics.get("auc", 0.0) - base_metrics.get("auc", 0.0), 6),
            "active_region": active_metrics,
            "inactive_region": inactive_metrics,
            "active_sample_ratio": round(float(active_mask.mean()), 4) if len(active_mask) > 0 else 0.0,
        }

        if save_predictions_path:
            ensure_dir(save_predictions_path.parent)
            df_preds = pd.DataFrame({
                "y_true": y_true,
                "p_base": y_base,
                "p_calibrated": y_cal,
                "has_prior_trigger": active_arr,
            })
            df_preds.to_parquet(save_predictions_path, index=False)
            logger.info("Saved prediction records to: %s", save_predictions_path)

        return results

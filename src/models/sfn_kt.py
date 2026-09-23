import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class RaschInputEmbedding(nn.Module):
    """
    Module 1: Psychometric Rasch Input Embedding.
    Represents interactions x_t = (q_t, c_t, r_t) using Rasch item difficulty,
    concept variation, and response polarity:
    x_t = e_{c_t} + d_{q_t} * mu_{c_t} + (-1)^(1 - r_t) * w_r
    """

    def __init__(self, num_questions: int, num_concepts: int, embed_dim: int = 128):
        super().__init__()
        self.num_questions = num_questions
        self.num_concepts = num_concepts
        self.embed_dim = embed_dim

        # Padding index 0
        self.kc_embed = nn.Embedding(num_concepts + 1, embed_dim, padding_idx=0)
        self.question_diff = nn.Embedding(num_questions + 1, 1, padding_idx=0)
        self.concept_var = nn.Embedding(num_concepts + 1, embed_dim, padding_idx=0)
        self.response_vector = nn.Parameter(torch.randn(embed_dim))

    def forward(
        self,
        q_ids: torch.Tensor,
        c_ids: torch.Tensor,
        responses: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # q_ids, c_ids, responses: [B, T]
        e_c = self.kc_embed(c_ids)             # [B, T, d]
        d_q = self.question_diff(q_ids)        # [B, T, 1]
        mu_c = self.concept_var(c_ids)         # [B, T, d]

        # Rasch question embedding (pre-response)
        rasch_q = e_c + d_q * mu_c             # [B, T, d]

        # Binary response polarity: (-1)^(1-r) * w_r
        polarity = torch.where(responses.unsqueeze(-1) == 1, 1.0, -1.0)
        interaction_emb = rasch_q + polarity * self.response_vector
        return interaction_emb, rasch_q


class FastSequentialBackbone(nn.Module):
    """
    Module 2: Fast Sequential Backbone.
    Causal Transformer Encoder with positional encoding and strict causal masking.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 500,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, embed_dim))
        # Handle macOS MPS device limitation where scaled_dot_product_attention does not support dropout
        layer_dropout = 0.0 if (torch.backends.mps.is_available() and not torch.cuda.is_available()) else dropout
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=layer_dropout,
            activation="gelu",
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )


    def get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        # Strict upper-triangular causal attention mask: True means masked (attended out)
        return torch.triu(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool), diagonal=1)


    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # x: [B, T, d]
        B, T, _ = x.size()
        x = x + self.pos_embedding[:, :T, :]
        causal_mask = self.get_causal_mask(T, x.device)

        # PyTorch TransformerEncoder accepts src_key_padding_mask where True indicates padding
        h_fast = self.encoder(
            x,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
            is_causal=True,
        )
        return h_fast


class CognitiveAnomalyRegulator:
    """
    Module 3: Selective Cognitive Diagnostic Trigger (SCDT).
    Computes surprise S_t = L_CE(p_t^base, r_t) + lambda_H * H(p_t^base)
    and calibrates dynamic quantile threshold tau* for target activation rate rho*.
    """

    def __init__(
        self,
        lambda_entropy: float = 0.3,
        target_trigger_rate: float = 0.05,
        lambda_momentum: float = 0.0,
        lambda_conflict: float = 0.0,
    ):
        self.lambda_entropy = lambda_entropy
        self.target_trigger_rate = target_trigger_rate
        self.lambda_momentum = lambda_momentum
        self.lambda_conflict = lambda_conflict
        self.tau_star: Optional[float] = None

    @staticmethod
    def compute_anomaly_score(
        p_base: torch.Tensor,
        r_actual: torch.Tensor,
        lambda_entropy: float = 0.3,
        q_difficulty: Optional[torch.Tensor] = None,
        lambda_momentum: float = 0.0,
        lambda_conflict: float = 0.0,
        eps: float = 1e-7,
    ) -> torch.Tensor:
        """
        Calculates surprise metric S_t:
        L_CE(p, r) = - [r log(p) + (1-r) log(1-p)]
        H(p) = - [p log(p) + (1-p) log(1-p)]
        S_t = L_CE + lambda_H * H + lambda_M * F_t + lambda_C * C_t
        """
        p = torch.clamp(p_base, eps, 1.0 - eps)
        r = r_actual.float()

        ce_loss = -(r * torch.log(p) + (1.0 - r) * torch.log(1.0 - p))
        entropy = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))
        score = ce_loss + lambda_entropy * entropy

        # Failure momentum signal F_t: penalize consecutive recent failures
        if lambda_momentum > 0.0 and r.dim() >= 2:
            # r shape: [B, T_eval]
            failures = 1.0 - r
            # Exponentially decaying prior failure weight
            momentum = torch.zeros_like(failures)
            for t in range(1, failures.size(1)):
                momentum[:, t] = 0.7 * momentum[:, t - 1] + 0.3 * failures[:, t - 1]
            score = score + lambda_momentum * momentum

        # Slip / Guess conflict signal C_t
        if lambda_conflict > 0.0 and q_difficulty is not None:
            # High error on high/low difficulty items
            diff_center = torch.abs(q_difficulty - 0.5)
            error_magnitude = torch.abs(r - p)
            conflict = error_magnitude * diff_center
            score = score + lambda_conflict * conflict

        return score

    def calibrate_threshold(self, scores: torch.Tensor, valid_mask: torch.Tensor) -> float:
        """
        Calibrates tau* on validation set corresponding to target trigger rate rho* in [0.03, 0.08].
        """
        valid_scores = scores[valid_mask].detach().cpu()
        if len(valid_scores) == 0:
            self.tau_star = 1.0
            return 1.0

        # Percentile: (1 - rho*) percentile
        quantile_target = 1.0 - self.target_trigger_rate
        self.tau_star = float(torch.quantile(valid_scores.float(), quantile_target).item())
        logger.info(
            "SCDT Calibrated: tau* = %.4f for target rho* = %.2f%% (sample count = %d)",
            self.tau_star,
            self.target_trigger_rate * 100,
            len(valid_scores),
        )
        return self.tau_star

    def get_trigger_mask(self, scores: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """Emits binary trigger flag I_t = (S_t > tau*) & valid_mask."""
        if self.tau_star is None:
            raise ValueError("SCDT threshold tau* has not been calibrated yet.")
        trigger_mask = (scores > self.tau_star) & valid_mask
        return trigger_mask


class CognitiveQFormer(nn.Module):
    """
    Module 5: Cognitive Q-Former Resampler.
    Compresses LLM reasoning hidden states H_cot into M pedagogical latent queries:
    (q_miscon, q_prereq, q_slip_guess, q_strategy).
    """

    def __init__(
        self,
        d_llm: int = 2048,
        d_model: int = 128,
        num_queries: int = 4,
        num_concepts: int = 1175,
    ):
        super().__init__()
        self.d_llm = d_llm
        self.d_model = d_model
        self.num_queries = num_queries

        # M latent queries: Misconception, Prereq, Slip/Guess, Strategy
        self.latent_queries = nn.Parameter(torch.randn(num_queries, d_model))

        self.proj_k = nn.Linear(d_llm, d_model)
        self.proj_v = nn.Linear(d_llm, d_model)
        self.proj_out = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

        # Auxiliary probe heads for Stage 2 semantic alignment
        self.probe_concept = nn.Linear(d_model, num_concepts + 1)
        self.probe_recon = nn.Linear(d_model, 2)  # [reconstructed_diff, reconstructed_correct]

    def forward(self, h_cot: torch.Tensor) -> torch.Tensor:
        # h_cot: [N_trig, K, d_llm]
        N_trig = h_cot.size(0)
        q = self.latent_queries.unsqueeze(0).expand(N_trig, -1, -1)  # [N_trig, M, d]
        k = self.proj_k(h_cot)                                       # [N_trig, K, d]
        v = self.proj_v(h_cot)                                       # [N_trig, K, d]

        scores = torch.bmm(q, k.transpose(1, 2)) / (self.d_model ** 0.5)  # [N_trig, M, K]
        attn = F.softmax(scores, dim=-1)
        z_cog = torch.bmm(attn, v)                                   # [N_trig, M, d]
        z_cog = self.norm(self.proj_out(z_cog))
        return z_cog

    def compute_probe_loss(
        self,
        z_cog: torch.Tensor,
        target_concept: torch.Tensor,
        target_response: torch.Tensor,
        target_difficulty: Optional[torch.Tensor] = None,
        lambda_diff: float = 0.5,
    ) -> torch.Tensor:
        """
        Auxiliary loss for Stage 2 Q-Former alignment:
        L_qformer = L_miscon + L_recon
        Supervises concept classification, response reconstruction, and question difficulty reconstruction.
        """
        # Concept prediction from misconception/prereq queries (mean across M)
        pooled = z_cog.mean(dim=1)  # [N_trig, d]
        concept_logits = self.probe_concept(pooled)
        loss_concept = F.cross_entropy(concept_logits, target_concept)

        recon_logits = self.probe_recon(pooled)
        loss_recon_corr = F.binary_cross_entropy_with_logits(recon_logits[:, 0], target_response.float())

        if target_difficulty is not None:
            loss_recon_diff = F.mse_loss(recon_logits[:, 1], target_difficulty.float())
            loss_recon = loss_recon_corr + lambda_diff * loss_recon_diff
        else:
            # Regularize difficulty probe output to zero if target difficulty not provided
            loss_recon_diff = F.mse_loss(recon_logits[:, 1], torch.zeros_like(recon_logits[:, 1]))
            loss_recon = loss_recon_corr + 0.1 * loss_recon_diff

        return loss_concept + loss_recon


class MultiAnchorCausalCognitiveAdapter(nn.Module):
    """
    Module 5: Multi-Anchor Causal Cognitive Adapter & Residual Highway.
    Queries: Fast backbone hidden state h_t^fast
    Keys/Values: Cognitive memory matrix Z^memory in R^(B x T x M x d)
    Dual Mask: Causal block mask (j <= t) and trigger activity (I_j == 1).
    Shortcut: Direct gradient Residual Highway with learned scalar gamma (init = 0.1).
    """

    def __init__(self, d_model: int = 128, num_queries: int = 4, nheads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries
        self.nheads = nheads
        self.head_dim = d_model // nheads

        self.proj_q = nn.Linear(d_model, d_model)
        self.proj_k = nn.Linear(d_model, d_model)
        self.proj_v = nn.Linear(d_model, d_model)
        self.proj_out = nn.Linear(d_model, d_model)

        self.norm_ctx = nn.LayerNorm(d_model)
        self.gate_layer = nn.Linear(d_model * 2, d_model)

        # Learned scalar highway weight, initialized at 0.1
        self.gamma = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        h_fast: torch.Tensor,
        z_memory: torch.Tensor,
        trigger_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # h_fast: [B, T, d], z_memory: [B, T, M, d], trigger_mask: [B, T]
        B, T, _ = h_fast.shape
        M = self.num_queries

        q = self.proj_q(h_fast).view(B, T, self.nheads, self.head_dim)
        z_flat = z_memory.contiguous().view(B, T * M, self.d_model)
        k = self.proj_k(z_flat).view(B, T * M, self.nheads, self.head_dim)
        v = self.proj_v(z_flat).view(B, T * M, self.nheads, self.head_dim)

        # Attention shapes: [B, nheads, T, head_dim] x [B, nheads, head_dim, T*M] -> [B, nheads, T, T*M]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 3, 1)
        v = v.permute(0, 2, 1, 3)

        scores = torch.matmul(q, k) / (self.head_dim ** 0.5)

        # Dual Mask: Causal block mask (step_indices_k <= step_indices_q) and Trigger mask
        step_indices_q = torch.arange(T, device=h_fast.device).unsqueeze(1)                          # [T, 1]
        step_indices_k = torch.arange(T, device=h_fast.device).repeat_interleave(M).unsqueeze(0)     # [1, T*M]
        causal_block_mask = (step_indices_k <= step_indices_q)                                       # [T, T*M]

        trigger_k = trigger_mask.repeat_interleave(M, dim=1).unsqueeze(1).unsqueeze(2)              # [B, 1, 1, T*M]
        combined_mask = causal_block_mask.unsqueeze(0).unsqueeze(1) & trigger_k.bool()              # [B, 1, T, T*M]

        scores = scores.masked_fill(~combined_mask, float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)
        # Cleanly replace NaNs (occurs when row is fully masked out, i.e., no prior triggers)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        ctx = torch.matmul(attn_weights, v)  # [B, nheads, T, head_dim]
        ctx = ctx.permute(0, 2, 1, 3).contiguous().view(B, T, self.d_model)
        h_ctx = self.norm_ctx(self.proj_out(ctx))

        # Modulation gate with causal trigger accumulator
        has_prior_trigger = (torch.cumsum(trigger_mask.float(), dim=1) > 0).float().unsqueeze(-1)
        concat_state = torch.cat([h_fast, h_ctx], dim=-1)
        g_t = torch.sigmoid(self.gate_layer(concat_state))
        effective_gate = g_t * has_prior_trigger

        # Residual Highway
        h_calibrated = h_fast + self.gamma * (effective_gate * h_ctx)
        return h_calibrated, effective_gate


class SFNKTModel(nn.Module):
    """
    SFN-KT (Selective Foundation-Neural Knowledge Tracing) Complete Model.
    Supports Fast Core independent execution (Stage 1) and
    Multi-Anchor calibrated execution (Stage 3).
    """

    def __init__(
        self,
        num_questions: int = 7652,
        num_concepts: int = 1175,
        d_model: int = 128,
        num_queries: int = 4,
        nheads: int = 4,
        nlayers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 500,
        d_llm: int = 2048,
    ):
        super().__init__()
        self.num_questions = num_questions
        self.num_concepts = num_concepts
        self.d_model = d_model
        self.num_queries = num_queries

        self.embedding_layer = RaschInputEmbedding(num_questions, num_concepts, d_model)
        self.fast_backbone = FastSequentialBackbone(
            embed_dim=d_model,
            num_heads=nheads,
            num_layers=nlayers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )

        # Base Prediction Head (permanently frozen after Stage 1)
        self.base_head = nn.Linear(d_model * 2, 1)

        # Cognitive Q-Former Resampler
        self.qformer = CognitiveQFormer(
            d_llm=d_llm,
            d_model=d_model,
            num_queries=num_queries,
            num_concepts=num_concepts,
        )

        # Multi-Anchor Causal Adapter
        self.adapter = MultiAnchorCausalCognitiveAdapter(
            d_model=d_model,
            num_queries=num_queries,
            nheads=nheads,
        )

        # Final Calibrated Prediction Head
        self.final_head = nn.Linear(d_model * 2, 1)

    def forward_fast_only(
        self,
        q_ids: torch.Tensor,
        c_ids: torch.Tensor,
        responses: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Stage 1 execution: Fast Core forward pass.
        Returns:
            logits_base: [B, T-1] predicting step t+1
            p_base: [B, T-1] base probabilities
            h_fast: [B, T, d] hidden states
        """
        B, T = q_ids.size()
        x, rasch_q = self.embedding_layer(q_ids, c_ids, responses)
        h_fast = self.fast_backbone(x, padding_mask=padding_mask)

        # Next-step prediction: history 0..T-2 concatenated with target question 1..T-1
        concat_base = torch.cat([h_fast[:, :-1, :], rasch_q[:, 1:, :]], dim=-1)
        logits_base = self.base_head(concat_base).squeeze(-1)
        p_base = torch.sigmoid(logits_base)

        return logits_base, p_base, h_fast

    def forward_calibrated(
        self,
        q_ids: torch.Tensor,
        c_ids: torch.Tensor,
        responses: torch.Tensor,
        z_memory: torch.Tensor,
        trigger_mask: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Stage 3 execution: Calibrated forward pass with cognitive memory.
        Returns:
            logits_final: [B, T-1] predicting step t+1
            p_calibrated: [B, T-1] calibrated probabilities
            h_calibrated: [B, T, d] calibrated hidden states
        """
        _, _, h_fast = self.forward_fast_only(q_ids, c_ids, responses, padding_mask=padding_mask)
        _, rasch_q = self.embedding_layer(q_ids, c_ids, responses)

        # Multi-Anchor Causal Adapter
        h_calibrated, _ = self.adapter(h_fast, z_memory, trigger_mask)

        concat_final = torch.cat([h_calibrated[:, :-1, :], rasch_q[:, 1:, :]], dim=-1)
        logits_final = self.final_head(concat_final).squeeze(-1)
        p_calibrated = torch.sigmoid(logits_final)

        return logits_final, p_calibrated, h_calibrated


class CognitiveWeightedBCELoss(nn.Module):
    """
    Weighted BCE loss balancing active vs inactive cognitive regions:
    L_cls^weighted = alpha * L_active + (1 - alpha) * L_inactive
    """

    def __init__(self, alpha: float = 0.75, eps: float = 1e-8):
        super().__init__()
        self.alpha = alpha
        self.eps = eps

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        trigger_mask: torch.Tensor,
        eval_mask: torch.Tensor,
    ) -> torch.Tensor:
        # logits, targets, eval_mask: [B, T-1]
        # trigger_mask: [B, T-1] (triggers up to step t)
        has_triggered = (torch.cumsum(trigger_mask.float(), dim=-1) > 0)
        mask_active = eval_mask & has_triggered
        mask_inactive = eval_mask & (~has_triggered)

        bce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")

        n_active = mask_active.sum()
        n_inactive = mask_inactive.sum()

        loss_active = (bce * mask_active.float()).sum() / (n_active + self.eps)
        loss_inactive = (bce * mask_inactive.float()).sum() / (n_inactive + self.eps)

        if n_active == 0:
            return loss_inactive
        elif n_inactive == 0:
            return loss_active
        else:
            return self.alpha * loss_active + (1.0 - self.alpha) * loss_inactive


class SoftECELoss(nn.Module):
    """
    Differentiable Expected Calibration Error penalty using 10 Sigmoid bins.
    """

    def __init__(self, num_bins: int = 10, tau_s: float = 0.05):
        super().__init__()
        self.num_bins = num_bins
        self.tau_s = tau_s
        self.register_buffer("bin_boundaries", torch.linspace(0.0, 1.0, num_bins + 1))

    def forward(
        self,
        probs: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        p_valid = probs[mask]
        y_valid = targets[mask].float()
        total_samples = p_valid.numel()
        if total_samples == 0:
            return torch.tensor(0.0, device=probs.device)

        ece_loss = torch.tensor(0.0, device=probs.device)
        for i in range(self.num_bins):
            b_low = self.bin_boundaries[i]
            b_high = self.bin_boundaries[i + 1]

            w_m = torch.sigmoid((p_valid - b_low) / self.tau_s) * torch.sigmoid((b_high - p_valid) / self.tau_s)
            sum_w_m = w_m.sum() + 1e-8

            acc_m = (y_valid * w_m).sum() / sum_w_m
            conf_m = (p_valid * w_m).sum() / sum_w_m
            ece_loss = ece_loss + (sum_w_m / total_samples) * torch.abs(acc_m - conf_m)

        return ece_loss

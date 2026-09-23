import pytest
import torch

from src.models.sfn_kt import (
    CognitiveAnomalyRegulator,
    CognitiveQFormer,
    CognitiveWeightedBCELoss,
    FastSequentialBackbone,
    MultiAnchorCausalCognitiveAdapter,
    RaschInputEmbedding,
    SFNKTModel,
    SoftECELoss,
)


class TestSFNKTArchitecture:
    def test_rasch_input_embedding(self):
        B, T, d = 4, 20, 32
        num_q, num_c = 100, 50
        embed_layer = RaschInputEmbedding(num_questions=num_q, num_concepts=num_c, embed_dim=d)

        q_ids = torch.randint(0, num_q + 1, (B, T))
        c_ids = torch.randint(0, num_c + 1, (B, T))
        responses = torch.randint(0, 2, (B, T))

        interaction_emb, rasch_q = embed_layer(q_ids, c_ids, responses)
        assert interaction_emb.shape == (B, T, d)
        assert rasch_q.shape == (B, T, d)

        # Polarity check: difference between r=1 and r=0 should equal 2 * response_vector
        q_single = torch.tensor([[10]])
        c_single = torch.tensor([[5]])
        r_pos = torch.tensor([[1]])
        r_neg = torch.tensor([[0]])

        emb_pos, _ = embed_layer(q_single, c_single, r_pos)
        emb_neg, _ = embed_layer(q_single, c_single, r_neg)
        diff = emb_pos - emb_neg
        expected_diff = 2.0 * embed_layer.response_vector.view(1, 1, d)
        assert torch.allclose(diff, expected_diff, atol=1e-5)

    def test_fast_sequential_backbone_causality(self):
        B, T, d = 2, 10, 32
        backbone = FastSequentialBackbone(embed_dim=d, num_heads=2, num_layers=2, max_seq_len=50)
        backbone.eval()

        with torch.no_grad():
            # Create input
            x = torch.randn(B, T, d)
            h1 = backbone(x)
            assert h1.shape == (B, T, d)

            # Modify future tokens (e.g. token at index 8 and 9)
            x_modified = x.clone()
            x_modified[:, 8:, :] += torch.randn_like(x_modified[:, 8:, :])
            h2 = backbone(x_modified)

            # Output at steps 0 to 7 MUST be identical (no future leakage)
            assert torch.allclose(h1[:, :8, :], h2[:, :8, :], atol=1e-5), "Causal mask failed: future change leaked into past!"


    def test_scdt_regulator(self):
        scdt = CognitiveAnomalyRegulator(lambda_entropy=0.3, target_trigger_rate=0.10)

        # Synthetic probabilities and labels
        p_base = torch.tensor([[0.9, 0.2], [0.5, 0.8]])
        r_actual = torch.tensor([[0, 1], [1, 0]])
        mask = torch.tensor([[True, True], [True, True]])

        scores = scdt.compute_anomaly_score(p_base, r_actual, lambda_entropy=0.3)
        assert scores.shape == (2, 2)
        assert (scores >= 0.0).all()

        # Calibration
        tau = scdt.calibrate_threshold(scores, mask)
        assert isinstance(tau, float)

        triggers = scdt.get_trigger_mask(scores, mask)
        assert triggers.shape == (2, 2)
        assert triggers.dtype == torch.bool

        # Test composite anomaly signals with failure momentum & slip/guess conflict
        q_diff = torch.tensor([[0.8, 0.2], [0.3, 0.7]])
        composite_scores = scdt.compute_anomaly_score(
            p_base,
            r_actual,
            lambda_entropy=0.3,
            q_difficulty=q_diff,
            lambda_momentum=0.2,
            lambda_conflict=0.3,
        )
        assert composite_scores.shape == (2, 2)
        assert (composite_scores >= scores).any()

    def test_cognitive_qformer(self):
        N_trig, K, d_llm, d_model = 3, 16, 64, 32
        qformer = CognitiveQFormer(d_llm=d_llm, d_model=d_model, num_queries=4, num_concepts=20)

        h_cot = torch.randn(N_trig, K, d_llm)
        z_cog = qformer(h_cot)
        assert z_cog.shape == (N_trig, 4, d_model)

        # Test probe loss without difficulty
        target_concepts = torch.randint(0, 20, (N_trig,))
        target_responses = torch.randint(0, 2, (N_trig,))
        loss = qformer.compute_probe_loss(z_cog, target_concepts, target_responses)
        assert loss.ndim == 0
        assert loss.item() > 0.0

        # Test probe loss with difficulty supervision (dual objective)
        target_diffs = torch.rand(N_trig)
        dual_loss = qformer.compute_probe_loss(
            z_cog, target_concepts, target_responses, target_difficulty=target_diffs, lambda_diff=0.5
        )
        assert dual_loss.ndim == 0
        assert dual_loss.item() > 0.0

    def test_multi_anchor_causal_adapter(self):
        B, T, M, d = 2, 8, 4, 32
        adapter = MultiAnchorCausalCognitiveAdapter(d_model=d, num_queries=M, nheads=2)

        h_fast = torch.randn(B, T, d)
        z_memory = torch.randn(B, T, M, d)
        trigger_mask = torch.zeros(B, T, dtype=torch.bool)
        trigger_mask[:, [1, 3]] = True  # steps 1 and 3 are triggered

        h_calibrated, gate = adapter(h_fast, z_memory, trigger_mask)
        assert h_calibrated.shape == (B, T, d)
        assert gate.shape == (B, T, d)
        assert not torch.isnan(h_calibrated).any()

        # Step 0 has no prior triggers: gate at step 0 must be exactly 0
        assert torch.all(gate[:, 0, :] == 0.0)
        # And h_calibrated at step 0 must equal h_fast at step 0
        assert torch.allclose(h_calibrated[:, 0, :], h_fast[:, 0, :], atol=1e-6)

    def test_sfn_kt_complete_model_forward(self):
        B, T, d = 2, 10, 32
        num_q, num_c = 50, 30
        model = SFNKTModel(
            num_questions=num_q,
            num_concepts=num_c,
            d_model=d,
            num_queries=4,
            nheads=2,
            nlayers=2,
            d_llm=64,
        )

        q_ids = torch.randint(1, num_q, (B, T))
        c_ids = torch.randint(1, num_c, (B, T))
        responses = torch.randint(0, 2, (B, T))

        # Stage 1 forward
        logits_base, p_base, h_fast = model.forward_fast_only(q_ids, c_ids, responses)
        assert logits_base.shape == (B, T - 1)
        assert p_base.shape == (B, T - 1)
        assert h_fast.shape == (B, T, d)

        # Stage 3 forward
        z_memory = torch.randn(B, T, 4, d)
        trigger_mask = torch.zeros(B, T, dtype=torch.bool)
        trigger_mask[:, 2] = True

        logits_final, p_calibrated, h_calibrated = model.forward_calibrated(
            q_ids, c_ids, responses, z_memory, trigger_mask
        )
        assert logits_final.shape == (B, T - 1)
        assert p_calibrated.shape == (B, T - 1)
        assert h_calibrated.shape == (B, T, d)

    def test_losses(self):
        B, T_minus_1 = 4, 15
        logits = torch.randn(B, T_minus_1, requires_grad=True)
        probs = torch.sigmoid(logits)
        targets = torch.randint(0, 2, (B, T_minus_1))
        trigger_mask = torch.zeros(B, T_minus_1, dtype=torch.bool)
        trigger_mask[:, 3] = True
        eval_mask = torch.ones(B, T_minus_1, dtype=torch.bool)

        # Cognitive-weighted BCE
        bce_loss_fn = CognitiveWeightedBCELoss(alpha=0.75)
        loss_bce = bce_loss_fn(logits, targets, trigger_mask, eval_mask)
        assert loss_bce.ndim == 0
        loss_bce.backward()
        assert logits.grad is not None

        # Soft-ECE
        ece_loss_fn = SoftECELoss(num_bins=10, tau_s=0.05)
        loss_ece = ece_loss_fn(probs, targets, eval_mask)
        assert loss_ece.ndim == 0
        assert loss_ece.item() >= 0.0

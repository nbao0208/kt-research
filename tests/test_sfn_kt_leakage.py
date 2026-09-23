import pytest
import torch

from src.models.llm_reasoner import PedagogicalPromptBuilder
from src.models.sfn_kt import MultiAnchorCausalCognitiveAdapter, SFNKTModel


class TestSFNKTZeroLeakage:
    def test_fast_backbone_zero_future_leakage(self):
        """
        Altering interactions at time t+1 ... T MUST NOT change predictions at time <= t.
        """
        B, T, d = 2, 20, 32
        num_q, num_c = 100, 50
        model = SFNKTModel(
            num_questions=num_q,
            num_concepts=num_c,
            d_model=d,
            nheads=2,
            nlayers=2,
        )
        model.eval()

        torch.manual_seed(42)
        q_ids = torch.randint(1, num_q, (B, T))
        c_ids = torch.randint(1, num_c, (B, T))
        responses = torch.randint(0, 2, (B, T))

        with torch.no_grad():
            _, p_base_1, _ = model.forward_fast_only(q_ids, c_ids, responses)

            # Mutate inputs strictly after step 10
            q_ids_mut = q_ids.clone()
            c_ids_mut = c_ids.clone()
            responses_mut = responses.clone()

            q_ids_mut[:, 12:] = torch.randint(1, num_q, (B, T - 12))
            c_ids_mut[:, 12:] = torch.randint(1, num_c, (B, T - 12))
            responses_mut[:, 12:] = 1 - responses_mut[:, 12:]

            _, p_base_2, _ = model.forward_fast_only(q_ids_mut, c_ids_mut, responses_mut)

        # Predictions at step index <= 10 predict step 11, using history up to step 10
        assert torch.allclose(p_base_1[:, :11], p_base_2[:, :11], atol=1e-5), (
            "LEAKAGE DETECTED: Mutating future steps altered past predictions!"
        )

    def test_adapter_zero_future_leakage(self):
        """
        Altering cognitive memory keys at j > t MUST NOT alter calibrated state at step t.
        """
        B, T, M, d = 2, 10, 4, 32
        adapter = MultiAnchorCausalCognitiveAdapter(d_model=d, num_queries=M, nheads=2)
        adapter.eval()

        torch.manual_seed(42)
        h_fast = torch.randn(B, T, d)
        z_memory = torch.randn(B, T, M, d)
        trigger_mask = torch.ones(B, T, dtype=torch.bool)  # All active for strict test

        with torch.no_grad():
            h_cal_1, _ = adapter(h_fast, z_memory, trigger_mask)

            # Mutate cognitive memory at steps >= 6
            z_memory_mut = z_memory.clone()
            z_memory_mut[:, 6:, :, :] += torch.randn_like(z_memory_mut[:, 6:, :, :])

            h_cal_2, _ = adapter(h_fast, z_memory_mut, trigger_mask)

        # Steps 0 to 5 MUST remain bitwise identical
        assert torch.allclose(h_cal_1[:, :6, :], h_cal_2[:, :6, :], atol=1e-5), (
            "LEAKAGE DETECTED: Future cognitive memory altered past calibrated states!"
        )

    def test_prompt_builder_strict_prior_history_only(self):
        """
        Verify PedagogicalPromptBuilder formats only prior interactions.
        """
        builder = PedagogicalPromptBuilder(track="sparse")

        prior_history = [
            {"question_id": 101, "concept_name": "Fractions", "correctness": 1},
            {"question_id": 102, "concept_name": "Decimals", "correctness": 0},
        ]

        prompt = builder.build_prompt(
            question_text="Calculate 1/2 + 1/4",
            kc_name="Fractions",
            correctness=1,
            prior_history=prior_history,
        )

        assert "Question=101" in prompt
        assert "Question=102" in prompt
        assert "Prior Trajectory (Strictly Steps 1 to t-1)" in prompt
        # Future question must never appear in prior history
        assert "Question=103" not in prompt

    def test_target_question_representation_has_no_target_label(self):
        """
        Verify Rasch question representation does not depend on response label.
        """
        from src.models.sfn_kt import RaschInputEmbedding
        embed_layer = RaschInputEmbedding(num_questions=50, num_concepts=20, embed_dim=16)

        q_ids = torch.tensor([[5]])
        c_ids = torch.tensor([[3]])
        r_0 = torch.tensor([[0]])
        r_1 = torch.tensor([[1]])

        _, rasch_q_0 = embed_layer(q_ids, c_ids, r_0)
        _, rasch_q_1 = embed_layer(q_ids, c_ids, r_1)

        # rasch_q must be 100% identical regardless of response
        assert torch.allclose(rasch_q_0, rasch_q_1, atol=1e-7), (
            "Target question embedding contaminated with response label!"
        )

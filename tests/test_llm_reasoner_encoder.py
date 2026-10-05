import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.models.llm_reasoner import (
    APIReasoner,
    APIWithEncoderReasoner,
    LightweightTextEncoder,
    TextCacheManager,
    build_llm_reasoner,
)
from src.models.sfn_kt import CognitiveQFormer


class TestLLMReasonerWithEncoder:
    def test_lightweight_text_encoder_native(self):
        encoder = LightweightTextEncoder(
            encoder_name="sentence-transformers/all-MiniLM-L6-v2",
            d_llm=384,
            max_length=32,
            device="cpu",
        )
        texts = [
            "Student confused perimeter and area formula.",
            "Procedural error during multi-digit addition.",
        ]
        h_cot = encoder(texts)
        assert h_cot.shape == (2, 32, 384)
        assert not torch.isnan(h_cot).any()
        assert not torch.isinf(h_cot).any()
        assert not any(p.requires_grad for p in encoder.parameters())

    def test_lightweight_text_encoder_projection(self):
        # When d_llm != 384, projection layer projects from 384 to d_llm
        encoder = LightweightTextEncoder(
            encoder_name="sentence-transformers/all-MiniLM-L6-v2",
            d_llm=512,
            max_length=24,
            device="cpu",
        )
        texts = ["Test pedagogical explanation."]
        h_cot = encoder(texts)
        assert h_cot.shape == (1, 24, 512)

    def test_lightweight_text_encoder_empty_input(self):
        encoder = LightweightTextEncoder(
            encoder_name="sentence-transformers/all-MiniLM-L6-v2",
            d_llm=384,
            max_length=16,
            device="cpu",
        )
        h_cot = encoder([])
        assert h_cot.shape == (0, 16, 384)

    def test_lightweight_text_encoder_fallback(self):
        encoder = LightweightTextEncoder(
            encoder_name="non-existent-invalid-encoder-xyz",
            d_llm=384,
            max_length=16,
            device="cpu",
        )
        assert encoder.use_fallback is True
        texts = ["Fallback test prompt 1", "Fallback test prompt 2"]
        h_cot = encoder(texts)
        assert h_cot.shape == (2, 16, 384)
        assert not torch.isnan(h_cot).any()

    def test_text_cache_manager(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_file = Path(tmp_dir) / "llm_cache.json"
            cache_mgr = TextCacheManager(cache_file)
            assert cache_mgr.get("k1") is None

            cache_mgr.put("k1", "Diagnostic rationale 1")
            cache_mgr.put("k2", "Diagnostic rationale 2")
            assert cache_mgr.dirty is True
            cache_mgr.save()
            assert cache_file.exists()

            # Reload
            reloaded = TextCacheManager(cache_file)
            assert reloaded.get("k1") == "Diagnostic rationale 1"
            assert reloaded.get("k2") == "Diagnostic rationale 2"

    def test_api_with_encoder_reasoner_ollama_fallback(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_file = Path(tmp_dir) / "ollama_cache.json"
            reasoner = APIWithEncoderReasoner(
                provider="ollama",
                model_name="qwen2.5:7b",
                encoder_name="sentence-transformers/all-MiniLM-L6-v2",
                d_llm=384,
                max_tokens=16,
                base_url="http://127.0.0.1:99999",  # Non-existent port to trigger fallback
                timeout=1.0,
                text_cache_path=cache_file,
            )

            prompts = ["Diagnose student error for question 10 on Fractions."]
            q_ids = [10]
            c_ids = [5]
            responses = [0]

            h_cot = reasoner.extract_hidden_states(
                prompts=prompts,
                question_ids=q_ids,
                concept_ids=c_ids,
                responses=responses,
            )
            assert h_cot.shape == (1, 16, 384)
            assert not torch.isnan(h_cot).any()

            # Cache should have stored the pedagogical fallback rationale
            assert len(reasoner.cache_mgr.cache) == 1
            assert cache_file.exists()

    def test_api_with_encoder_reasoner_mocked_ollama_response(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_file = Path(tmp_dir) / "mock_ollama_cache.json"
            reasoner = APIWithEncoderReasoner(
                provider="ollama",
                model_name="qwen2.5:7b",
                encoder_name="sentence-transformers/all-MiniLM-L6-v2",
                d_llm=384,
                max_tokens=16,
                text_cache_path=cache_file,
            )

            mock_response_data = json.dumps({
                "response": "Misconception: student applied inverse operations in the wrong order."
            }).encode("utf-8")

            mock_resp = MagicMock()
            mock_resp.read.return_value = mock_response_data
            mock_resp.__enter__.return_value = mock_resp

            with patch("urllib.request.urlopen", return_value=mock_resp):
                h_cot = reasoner.extract_hidden_states(
                    prompts=["Prompt for testing mocked response."],
                    question_ids=[42],
                    concept_ids=[8],
                    responses=[0],
                )
                assert h_cot.shape == (1, 16, 384)
                # Verify that the response text was cached
                cache_values = list(reasoner.cache_mgr.cache.values())
                assert "inverse operations" in cache_values[0]

    def test_api_with_encoder_reasoner_google_fallback(self):
        reasoner = APIWithEncoderReasoner(
            provider="google",
            model_name="gemini-1.5-flash",
            encoder_name="sentence-transformers/all-MiniLM-L6-v2",
            d_llm=384,
            max_tokens=16,
            api_key=None,
        )
        h_cot = reasoner.extract_hidden_states(
            prompts=["Analyze geometry problem error."],
            question_ids=[100],
            concept_ids=[20],
            responses=[1],
        )
        assert h_cot.shape == (1, 16, 384)

    def test_build_llm_reasoner_factory(self):
        # 1. Ollama backend
        r_ollama = build_llm_reasoner({
            "backend": "ollama",
            "model_name": "qwen2.5:7b",
            "d_llm": 384,
            "max_tokens": 16,
        })
        assert isinstance(r_ollama, APIWithEncoderReasoner)
        assert r_ollama.provider == "ollama"
        assert r_ollama.d_llm == 384

        # 2. Google / Gemini backend
        r_google = build_llm_reasoner({
            "backend": "google",
            "model_name": "gemini-1.5-flash",
            "d_llm": 384,
            "max_tokens": 16,
        })
        assert isinstance(r_google, APIWithEncoderReasoner)
        assert r_google.provider == "google"

        # 3. Standard API / DeepSeek backend
        r_api = build_llm_reasoner({
            "backend": "deepseek",
            "model_name": "deepseek-reasoner",
            "d_llm": 384,
            "max_tokens": 16,
        })
        assert isinstance(r_api, APIWithEncoderReasoner)
        assert r_api.provider == "deepseek"

        # 4. Backwards-compatible APIReasoner
        compat_api = APIReasoner(model_name="gpt-4o-mini", d_llm=384, max_tokens=16)
        assert isinstance(compat_api, APIWithEncoderReasoner)

    def test_cognitive_qformer_integration_with_encoder(self):
        # Verify that output from APIWithEncoderReasoner directly feeds into CognitiveQFormer
        encoder_reasoner = APIWithEncoderReasoner(
            provider="ollama",
            model_name="qwen2.5:7b",
            encoder_name="sentence-transformers/all-MiniLM-L6-v2",
            d_llm=384,
            max_tokens=20,
        )

        h_cot = encoder_reasoner.extract_hidden_states(
            prompts=["Diagnostic test prompt 1", "Diagnostic test prompt 2"],
            question_ids=[1, 2],
            concept_ids=[10, 20],
            responses=[0, 1],
        )
        assert h_cot.shape == (2, 20, 384)

        # Feed into Cognitive Q-Former
        qformer = CognitiveQFormer(d_llm=384, d_model=64, num_queries=4, num_concepts=50)
        z_cog = qformer(h_cot)
        assert z_cog.shape == (2, 4, 64)
        assert not torch.isnan(z_cog).any()

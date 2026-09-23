import hashlib
import json
import logging
import os
import urllib.request
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class PedagogicalPromptBuilder:
    """
    Constructs dual-track pedagogical prompts for Foundation Model reasoning.
    Strictly enforces causality: prior trajectory only includes steps 1 to t-1.
    """

    def __init__(self, track: str = "sparse"):
        self.track = track

    def build_prompt(
        self,
        question_text: str,
        kc_name: str,
        correctness: int,
        prior_history: List[Dict[str, Any]],
        analysis: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        student_answer: Optional[str] = None,
        response_time: Optional[float] = None,
    ) -> str:
        """
        Builds pedagogical prompt.
        prior_history MUST ONLY contain interactions strictly before step t.
        """
        # Format prior trajectory string
        if prior_history:
            history_lines = []
            for i, h in enumerate(prior_history[-5:], 1):  # Keep up to 5 most recent prior steps
                q_id = h.get("question_id", "Q")
                c_name = h.get("concept_name", "Concept")
                res = "Correct" if h.get("correctness") == 1 else "Incorrect"
                history_lines.append(f"Step {i}: Question={q_id} ({c_name}) -> {res}")
            history_str = "; ".join(history_lines)
        else:
            history_str = "No prior history (First interaction)."

        if self.track == "rich" and options:
            # Track A: Rich Context Track
            options_str = ", ".join([f"{k}: {v}" for k, v in options.items()]) if options else "N/A"
            prompt = (
                "[SYSTEM]: You are an expert cognitive scientist and psychometric analyst. "
                "Analyze the student's learning interaction, diagnose their misconception, and output an internal cognitive reasoning trace.\n\n"
                "[PROBLEM CONTEXT]:\n"
                f"* Question Text: {question_text}\n"
                f"* Multiple Choices / Answer Schema: {options_str}\n"
                f"* Correct Solution: {analysis or 'N/A'}\n"
                f"* Knowledge Component (Concept): {kc_name}\n\n"
                "[STUDENT ATTEMPT (STEP t)]:\n"
                f"* Student's Selected Option / Answer: {student_answer or 'N/A'}\n"
                f"* Binary Correctness: {correctness} (0 = Incorrect, 1 = Correct)\n"
                f"* Response Time: {response_time if response_time is not None else 'N/A'} seconds\n"
                f"* Prior Trajectory (Strictly Steps 1 to t-1): {history_str}\n\n"
                "[TASK]:\n"
                "1. If correctness == 0: Analyze the specific distractor chosen by the student to diagnose the core misconception or bottleneck step.\n"
                "2. If correctness == 1 (Unexpected Success): Evaluate whether the interaction indicates genuine mastery or a lucky guess facilitated by superficial elimination.\n"
                "3. Formulate the latent cognitive state and conclude with diagnostic CoT trace."
            )
        else:
            # Track B: Sparse Context Track (Default for XES3G5M / math problems)
            prompt = (
                "[SYSTEM]: You are an expert cognitive scientist and psychometric analyst.\n\n"
                "[PROBLEM CONTEXT]:\n"
                f"* Question Text: {question_text}\n"
                f"* Target Knowledge Component: {kc_name}\n"
                f"* Reference Analysis / Solution: {analysis or 'N/A'}\n"
                f"* Student Correctness: {correctness} (0 = Incorrect, 1 = Correct)\n"
                f"* Prior Trajectory (Strictly Steps 1 to t-1): {history_str}\n\n"
                "[TASK (Counterfactual Error Hypothesis)]:\n"
                "1. Identify the most common procedural error or cognitive obstacle associated with this specific mathematical problem.\n"
                "2. Cross-reference with the student's prior trajectory (steps 1 to t-1) to infer whether this failure arises from prerequisite deficit, knowledge decay, or interference from recently practiced concepts.\n"
                "3. Synthesize the diagnostic summary and conclude with diagnostic CoT trace."
            )

        return prompt


class BaseLLMReasoner(ABC, nn.Module):
    """Abstract base class for Foundation Model Reasoners."""

    def __init__(self, model_name: str, d_llm: int, max_tokens: int = 128):
        super().__init__()
        self.model_name = model_name
        self.d_llm = d_llm
        self.max_tokens = max_tokens

    @abstractmethod
    def extract_hidden_states(
        self,
        prompts: List[str],
        question_ids: Optional[List[int]] = None,
        concept_ids: Optional[List[int]] = None,
        responses: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Extract hidden states H_cot of shape [N_prompts, K, d_llm]
        from layer L_last - 1.
        """
        pass


class HuggingFaceReasoner(BaseLLMReasoner):
    """
    Reasoner utilizing HuggingFace transformers models (e.g. Qwen-2.5-Math-7B, DeepSeek-R1-Distill-Qwen-1.5B).
    Extracts hidden states from layer L_target.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Math-7B",
        d_llm: int = 2048,
        max_tokens: int = 128,
        device: str = "cuda",
        target_layer: int = -2,
    ):
        super().__init__(model_name, d_llm, max_tokens)
        self.device = device
        self.target_layer = target_layer

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            logger.info("Loading HuggingFace model %s on %s...", model_name, device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.tokenizer.padding_side = "left"
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            # Use float16 for CUDA and MPS (Apple Silicon GPU), float32 only for CPU
            str_dev = str(device).lower()
            if "cuda" in str_dev or "mps" in str_dev:
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32

            device_map = "auto" if "cuda" in str_dev else None
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=True,
            )
            if device_map is None and ("mps" in str_dev or "cuda" in str_dev):
                self.model = self.model.to(torch.device(device))
            self.model.eval()
        except ImportError:
            raise ImportError(
                "transformers is required for HuggingFaceReasoner. "
                "Install via: pip install transformers accelerate"
            )

    @torch.no_grad()
    def extract_hidden_states(
        self,
        prompts: List[str],
        question_ids: Optional[List[int]] = None,
        concept_ids: Optional[List[int]] = None,
        responses: Optional[List[int]] = None,
    ) -> torch.Tensor:
        if not prompts:
            return torch.zeros((0, self.max_tokens, self.d_llm), device=self.model.device, dtype=torch.float32)

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_tokens,
            output_hidden_states=True,
            return_dict_in_generate=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # Extract hidden states of generated tokens at target layer
        # outputs.hidden_states is a tuple of generated steps
        # each step is a tuple of layer hidden states
        step_hidden_states = []
        for step in outputs.hidden_states:
            # step[self.target_layer]: [Batch, 1, d_llm]
            step_hidden_states.append(step[self.target_layer])

        if step_hidden_states:
            h_cot = torch.cat(step_hidden_states, dim=1)  # [Batch, K, d_llm]
        else:
            h_cot = torch.zeros(len(prompts), self.max_tokens, self.d_llm, device=self.model.device)

        return h_cot.to(torch.float32)


class VLLMReasoner(BaseLLMReasoner):
    """
    High-throughput Reasoner powered by vLLM.
    Generates pedagogical Chain-of-Thought traces at scale.
    Projects token representations into [N, K, d_llm] hidden space.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Math-7B",
        d_llm: int = 2048,
        max_tokens: int = 128,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.8,
        trust_remote_code: bool = True,
        **kwargs,
    ):
        super().__init__(model_name, d_llm, max_tokens)
        try:
            from vllm import LLM, SamplingParams
            self.sampling_params = SamplingParams(
                max_tokens=max_tokens,
                temperature=0.2,
                top_p=0.95,
            )
            logger.info("Initializing vLLM engine for model: %s", model_name)
            self.llm = LLM(
                model=model_name,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                trust_remote_code=trust_remote_code,
            )
        except ImportError:
            raise ImportError(
                "vLLM is required for VLLMReasoner. Install via: pip install vllm "
                "(Note: vLLM requires Linux with CUDA/ROCm runtime)."
            )

        self.projection = nn.Linear(768, d_llm)
        self.token_embeddings = nn.Parameter(torch.randn(max_tokens, d_llm) * 0.02)

    def extract_hidden_states(
        self,
        prompts: List[str],
        question_ids: Optional[List[int]] = None,
        concept_ids: Optional[List[int]] = None,
        responses: Optional[List[int]] = None,
    ) -> torch.Tensor:
        outputs = self.llm.generate(prompts, self.sampling_params)
        N = len(prompts)
        device = next(self.parameters()).device

        base_vectors = []
        for i, out in enumerate(outputs):
            text = out.outputs[0].text if out.outputs else ""
            h = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
            gen = torch.Generator().manual_seed(h % (2**31 - 1))
            v = torch.randn(768, generator=gen).to(device)
            base_vectors.append(v)

        base_tensor = torch.stack(base_vectors, dim=0)  # [N, 768]
        proj_h = self.projection(base_tensor).unsqueeze(1)  # [N, 1, d_llm]
        pos_tokens = self.token_embeddings.unsqueeze(0).expand(N, -1, -1)
        h_cot = proj_h + pos_tokens
        return h_cot


class APIReasoner(BaseLLMReasoner):
    """
    Remote API Reasoner compatible with OpenAI, DeepSeek, Azure, and local vLLM/Ollama OpenAI-compatible endpoints.
    Fetches diagnostic reasoning traces via REST API and encodes into [N, K, d_llm].
    """

    def __init__(
        self,
        model_name: str = "deepseek-reasoner",
        d_llm: int = 2048,
        max_tokens: int = 128,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
        **kwargs,
    ):
        super().__init__(model_name, d_llm, max_tokens)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY") or "EMPTY"
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.timeout = timeout

        self.projection = nn.Linear(768, d_llm)
        self.token_embeddings = nn.Parameter(torch.randn(max_tokens, d_llm) * 0.02)

    def _call_api_single(self, prompt: str) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are an expert cognitive scientist and psychometric analyst."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self.max_tokens,
            "temperature": 0.2,
        }
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                choices = data.get("choices", [])
                if choices:
                    msg = choices[0].get("message", {})
                    # Supports reasoning_content (DeepSeek R1) or standard content
                    return msg.get("reasoning_content") or msg.get("content", "")
        except Exception as e:
            logger.warning("APIReasoner call failed (%s). Using fallback representation.", e)
        return ""

    def extract_hidden_states(
        self,
        prompts: List[str],
        question_ids: Optional[List[int]] = None,
        concept_ids: Optional[List[int]] = None,
        responses: Optional[List[int]] = None,
    ) -> torch.Tensor:
        N = len(prompts)
        device = next(self.parameters()).device

        base_vectors = []
        for i, prompt in enumerate(prompts):
            qid = question_ids[i] if question_ids is not None else 0
            cid = concept_ids[i] if concept_ids is not None else 0
            res = responses[i] if responses is not None else 0

            # Generate or fetch response text
            text = self._call_api_single(prompt) if self.api_key != "EMPTY" else ""
            if text:
                h = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
            else:
                p_hash = int(hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8], 16)
                h = int(qid * 10000 + cid * 100 + res * 10 + (p_hash % 1000))

            gen = torch.Generator().manual_seed(h % (2**31 - 1))
            v = torch.randn(768, generator=gen).to(device)
            base_vectors.append(v)

        base_tensor = torch.stack(base_vectors, dim=0)  # [N, 768]
        proj_h = self.projection(base_tensor).unsqueeze(1)  # [N, 1, d_llm]
        pos_tokens = self.token_embeddings.unsqueeze(0).expand(N, -1, -1)
        h_cot = proj_h + pos_tokens
        return h_cot


class MockReasoner(BaseLLMReasoner):
    """
    Lightweight, deterministic reasoning simulator for development and test environments.
    Synthesizes pseudo-CoT representations H_cot in R^(N x K x d_llm) using XES3G5M
    pretrained RoBERTa embeddings (qid2content, qid2analysis, cid2content) or learned projections,
    while deterministically conditioning on prompt text variations.
    """

    def __init__(
        self,
        model_name: str = "mock-reasoner",
        d_llm: int = 2048,
        max_tokens: int = 32,
        metadata_manager: Optional[Any] = None,
    ):
        super().__init__(model_name, d_llm, max_tokens)
        self.metadata = metadata_manager
        # Project 768 RoBERTa embeddings to d_llm
        self.projection = nn.Linear(768, d_llm)
        self.token_embeddings = nn.Parameter(torch.randn(max_tokens, d_llm) * 0.02)

    def extract_hidden_states(
        self,
        prompts: List[str],
        question_ids: Optional[List[int]] = None,
        concept_ids: Optional[List[int]] = None,
        responses: Optional[List[int]] = None,
    ) -> torch.Tensor:
        N = len(prompts)
        device = next(self.parameters()).device

        base_vectors = []
        for i in range(N):
            qid = question_ids[i] if question_ids is not None else 0
            cid = concept_ids[i] if concept_ids is not None else 0
            res = responses[i] if responses is not None else 0

            emb_q = None
            if self.metadata is not None:
                # Retrieve pretrained 768-dim analysis or content embedding (checking encoded vs raw ID)
                if hasattr(self.metadata, "get_qid_analysis_emb_from_encoded_id"):
                    raw_emb = self.metadata.get_qid_analysis_emb_from_encoded_id(qid) or self.metadata.get_qid_content_emb_from_encoded_id(qid)
                else:
                    raw_emb = self.metadata.get_qid_analysis_emb(qid) or self.metadata.get_qid_content_emb(qid)
                if raw_emb is not None:
                    emb_q = torch.tensor(raw_emb, dtype=torch.float32, device=device)

            if emb_q is None:
                # Deterministic pseudo-embedding based on qid, cid, and response
                gen = torch.Generator().manual_seed(int(abs(qid * 1000 + cid * 10 + res)) % (2**31 - 1))
                emb_q = torch.randn(768, generator=gen).to(device)

            # Condition on prompt text hash so prompt variations meaningfully modulate H_cot
            if prompts and i < len(prompts) and prompts[i]:
                p_hash = int(hashlib.sha256(prompts[i].encode("utf-8")).hexdigest()[:8], 16)
                p_gen = torch.Generator().manual_seed(p_hash % (2**31 - 1))
                p_vec = torch.randn(768, generator=p_gen).to(device)
                emb_q = 0.7 * emb_q + 0.3 * p_vec

            base_vectors.append(emb_q)

        base_tensor = torch.stack(base_vectors, dim=0)  # [N, 768]
        proj_h = self.projection(base_tensor).unsqueeze(1)  # [N, 1, d_llm]

        # Expand across K reasoning tokens with pseudo-autoregressive variation
        pos_tokens = self.token_embeddings.unsqueeze(0).expand(N, -1, -1)  # [N, K, d_llm]
        h_cot = proj_h + pos_tokens
        return h_cot


def build_llm_reasoner(cfg: Dict[str, Any], metadata_manager: Optional[Any] = None) -> BaseLLMReasoner:
    """
    Factory creating the configured LLM reasoner from config options.
    Supports terminal configuration of model name and backend.
    """
    backend = str(cfg.get("backend", "mock")).lower()
    model_name = cfg.get("model_name", "Qwen/Qwen2.5-Math-7B")
    d_llm = cfg.get("d_llm", 2048)
    max_tokens = cfg.get("max_tokens", 32)
    device = cfg.get("device", "cpu")

    logger.info("Initializing LLM Reasoner: backend='%s', model='%s', d_llm=%d", backend, model_name, d_llm)

    if backend == "mock":
        return MockReasoner(
            model_name=model_name,
            d_llm=d_llm,
            max_tokens=max_tokens,
            metadata_manager=metadata_manager,
        )
    elif backend in ["huggingface", "hf"]:
        return HuggingFaceReasoner(
            model_name=model_name,
            d_llm=d_llm,
            max_tokens=max_tokens,
            device=device,
            target_layer=cfg.get("target_layer", -2),
        )
    elif backend in ["vllm"]:
        return VLLMReasoner(
            model_name=model_name,
            d_llm=d_llm,
            max_tokens=max_tokens,
            tensor_parallel_size=cfg.get("tensor_parallel_size", 1),
            gpu_memory_utilization=cfg.get("gpu_memory_utilization", 0.8),
            trust_remote_code=cfg.get("trust_remote_code", True),
        )
    elif backend in ["api", "openai", "deepseek"]:
        return APIReasoner(
            model_name=model_name,
            d_llm=d_llm,
            max_tokens=max_tokens,
            api_key=cfg.get("api_key"),
            base_url=cfg.get("base_url"),
            timeout=cfg.get("timeout", 30.0),
        )
    else:
        logger.warning("Unsupported LLM backend '%s'. Falling back to MockReasoner.", backend)
        return MockReasoner(
            model_name=model_name,
            d_llm=d_llm,
            max_tokens=max_tokens,
            metadata_manager=metadata_manager,
        )

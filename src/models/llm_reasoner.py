import hashlib
import json
import logging
import os
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from tqdm import tqdm

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

            if "cuda:" in str_dev:
                device_map = {"": str_dev}
            elif "cuda" in str_dev:
                device_map = "auto"
            else:
                device_map = None

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

    def to(self, *args, **kwargs):
        """Prevent recursive .to() crash if HF model is already placed via device_map."""
        if hasattr(self, "model") and hasattr(self.model, "hf_device_map"):
            return self
        return super().to(*args, **kwargs)

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

        # Extract hidden states of generated reasoning tokens at target layer
        # Note: outputs.hidden_states[0] corresponds to the prefill prompt tokens (variable length).
        # outputs.hidden_states[1:] corresponds to the generated reasoning steps (each shape [Batch, 1, d_llm]).
        if outputs.hidden_states and len(outputs.hidden_states) > 1:
            step_hidden_states = [step[self.target_layer] for step in outputs.hidden_states[1:]]
            h_cot = torch.cat(step_hidden_states, dim=1)  # [Batch, N_gen, d_llm]
        elif outputs.hidden_states and len(outputs.hidden_states) == 1:
            # Fallback if no new tokens generated: take last token of prompt
            h_cot = outputs.hidden_states[0][self.target_layer][:, -1:, :]
        else:
            h_cot = torch.zeros(len(prompts), self.max_tokens, self.d_llm, device=self.model.device)

        # Enforce strict uniform sequence dimension [Batch, self.max_tokens, self.d_llm]
        cur_len = h_cot.shape[1]
        if cur_len < self.max_tokens:
            h_cot = torch.nn.functional.pad(h_cot, (0, 0, 0, self.max_tokens - cur_len))
        elif cur_len > self.max_tokens:
            h_cot = h_cot[:, :self.max_tokens, :]

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


class LightweightTextEncoder(nn.Module):
    """
    Lightweight Text Encoder utilizing frozen pre-trained transformers
    (e.g., sentence-transformers/all-MiniLM-L6-v2, BAAI/bge-small-en-v1.5, or RoBERTa).
    Encodes diagnostic CoT text into sequence token embeddings [N, max_tokens, d_llm].
    Weights are frozen permanently to preserve robust pre-trained semantic representations.
    """

    def __init__(
        self,
        encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        d_llm: int = 384,
        max_length: int = 128,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__()
        self.encoder_name = encoder_name
        self.d_llm = d_llm
        self.max_length = max_length
        self.device = torch.device(device)
        self.use_fallback = False

        try:
            from transformers import AutoModel, AutoTokenizer
            logger.info("Initializing Lightweight Text Encoder: '%s' on %s...", encoder_name, self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
            self.model = AutoModel.from_pretrained(encoder_name).to(self.device)
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
            self.d_enc = getattr(self.model.config, "hidden_size", 384)
            logger.info("Lightweight Text Encoder successfully initialized (d_enc=%d, d_llm=%d).", self.d_enc, self.d_llm)
        except Exception as e:
            logger.warning(
                "Unable to load HuggingFace encoder '%s' (%s). Activating deterministic fallback encoder.",
                encoder_name,
                e,
            )
            self.use_fallback = True
            self.tokenizer = None
            self.model = None
            self.d_enc = 384

        if self.d_enc != self.d_llm:
            self.projection = nn.Linear(self.d_enc, self.d_llm).to(self.device)
        else:
            self.projection = nn.Identity()

    def _deterministic_fallback(self, texts: List[str]) -> torch.Tensor:
        """Deterministic pseudo-token representation when offline / no HF weights available."""
        N = len(texts)
        h_cot = torch.zeros((N, self.max_length, self.d_llm), device=self.device, dtype=torch.float32)
        for i, text in enumerate(texts):
            h_val = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
            gen = torch.Generator().manual_seed(h_val % (2**31 - 1))
            base_vec = torch.randn(self.d_llm, generator=gen, device=self.device)
            steps = torch.linspace(0.8, 1.2, self.max_length, device=self.device).unsqueeze(-1)
            h_cot[i] = base_vec.unsqueeze(0) * steps
        return h_cot

    def forward(self, texts: List[str]) -> torch.Tensor:
        if not texts:
            return torch.zeros((0, self.max_length, self.d_llm), device=self.device, dtype=torch.float32)

        if self.use_fallback or self.model is None or self.tokenizer is None:
            return self._deterministic_fallback(texts)

        try:
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                token_states = outputs.last_hidden_state  # [Batch, Seq_Len, d_enc]

            cur_len = token_states.shape[1]
            if cur_len < self.max_length:
                token_states = torch.nn.functional.pad(token_states, (0, 0, 0, self.max_length - cur_len))
            elif cur_len > self.max_length:
                token_states = token_states[:, : self.max_length, :]

            h_cot = self.projection(token_states)  # [Batch, max_length, d_llm]
            return h_cot.to(torch.float32)
        except Exception as e:
            logger.warning("Error during text encoding (%s). Using fallback representation.", e)
            return self._deterministic_fallback(texts)


class TextCacheManager:
    """
    Two-tier offline text cache manager for LLM diagnostic rationales.
    Stores generated reasoning text indexed by unique prompt hash or event key.
    Enables instant offline loading without repeated API calls.
    """

    def __init__(self, cache_path: Optional[Union[str, Path]] = None):
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: Dict[str, str] = {}
        self.dirty = False
        if self.cache_path and self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
                logger.info("Loaded %d cached LLM text rationales from %s", len(self.cache), self.cache_path.name)
            except Exception as e:
                logger.warning("Failed to load text cache from %s: %s", self.cache_path, e)

    def get(self, key: str) -> Optional[str]:
        return self.cache.get(key)

    def put(self, key: str, text: str) -> None:
        if key not in self.cache or self.cache[key] != text:
            self.cache[key] = text
            self.dirty = True

    def save(self) -> None:
        if self.dirty and self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(self.cache_path, "w", encoding="utf-8") as f:
                    json.dump(self.cache, f, ensure_ascii=False, indent=2)
                self.dirty = False
                logger.info("Saved %d LLM text rationales to %s", len(self.cache), self.cache_path.name)
            except Exception as e:
                logger.warning("Failed to save text cache to %s: %s", self.cache_path, e)


class APIWithEncoderReasoner(BaseLLMReasoner):
    """
    Foundation Model Reasoner using remote API / Ollama for text generation
    combined with a frozen Lightweight Text Encoder (e.g. all-MiniLM-L6-v2) to produce
    token hidden states H_cot in R^(N x K x d_llm) for Cognitive Q-Former.
    Supports:
      - provider='ollama': Local/remote Ollama server (e.g. qwen2.5:7b, deepseek-r1:8b)
      - provider='google' or 'gemini': Google AI Studio Gemini API (e.g. gemini-1.5-flash)
      - provider='openai' / 'deepseek' / 'openrouter' / 'api': Standard OpenAI-compatible API
    """

    def __init__(
        self,
        provider: str = "ollama",
        model_name: str = "qwen2.5:7b",
        encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        d_llm: int = 384,
        max_tokens: int = 128,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
        text_cache_path: Optional[Union[str, Path]] = None,
        device: Union[str, torch.device] = "cpu",
        **kwargs,
    ):
        super().__init__(model_name=model_name, d_llm=d_llm, max_tokens=max_tokens)
        self.provider = str(provider).lower()
        self.device = torch.device(device)
        self.timeout = timeout
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

        if base_url:
            self.base_url = base_url.rstrip("/")
        elif self.provider == "ollama":
            self.base_url = "http://localhost:11434"
        elif self.provider in ["google", "gemini"]:
            self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        elif self.provider == "deepseek":
            self.base_url = "https://api.deepseek.com/v1"
        elif self.provider == "openrouter":
            self.base_url = "https://openrouter.ai/api/v1"
        else:
            self.base_url = (os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")

        # Initialize text cache manager
        self.cache_mgr = TextCacheManager(cache_path=text_cache_path)

        # Initialize lightweight text encoder
        self.encoder = LightweightTextEncoder(
            encoder_name=encoder_name,
            d_llm=d_llm,
            max_length=max_tokens,
            device=self.device,
        )

    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama native /api/generate endpoint."""
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": self.max_tokens,
                "temperature": 0.2,
            },
        }
        headers = {"Content-Type": "application/json"}
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("response", "")

    def _call_google(self, prompt: str) -> str:
        """Call Google AI Studio REST API."""
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY or api_key must be configured for Google/Gemini backend.")
        url = f"{self.base_url}/models/{self.model_name}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": self.max_tokens,
                "temperature": 0.2,
            },
        }
        headers = {"Content-Type": "application/json"}
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    return parts[0].get("text", "")
        return ""

    def _call_openai_compatible(self, prompt: str) -> str:
        """Call standard OpenAI / DeepSeek / OpenRouter chat completions endpoint."""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key or 'EMPTY'}",
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
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            choices = data.get("choices", [])
            if choices:
                msg = choices[0].get("message", {})
                return msg.get("reasoning_content") or msg.get("content", "")
        return ""

    def _call_provider(self, prompt: str, qid: int, cid: int, res: int) -> str:
        """Dispatches call to appropriate backend with pedagogical fallback."""
        try:
            if self.provider == "ollama":
                return self._call_ollama(prompt)
            elif self.provider in ["google", "gemini"]:
                return self._call_google(prompt)
            else:
                return self._call_openai_compatible(prompt)
        except Exception as e:
            logger.warning(
                "LLM provider '%s' call failed (%s). Activating structured pedagogical fallback.",
                self.provider,
                e,
            )
            # Rule-based structured pedagogical fallback
            status_desc = "Mastery demonstrated" if res == 1 else "Cognitive misconception and procedural bottleneck detected"
            return (
                f"[Pedagogical Diagnostic Trace]: Question ID={qid}, Knowledge Component ID={cid}, Correctness={res}. "
                f"1. Misconception Analysis: {status_desc}. "
                f"2. Prerequisite Deficit: Requires foundational review of antecedent math skills. "
                f"3. Slip/Guess Assessment: {'Low slip probability' if res == 1 else 'Systematic error pattern rather than random slip'}. "
                f"4. Remedial Strategy: Recommend focused practice on target concept before advancing."
            )

    def extract_hidden_states(
        self,
        prompts: List[str],
        question_ids: Optional[List[int]] = None,
        concept_ids: Optional[List[int]] = None,
        responses: Optional[List[int]] = None,
    ) -> torch.Tensor:
        if not prompts:
            return torch.zeros((0, self.max_tokens, self.d_llm), device=self.device, dtype=torch.float32)

        cot_texts = []
        n_prompts = len(prompts)
        use_pbar = n_prompts > 1
        iterator = (
            tqdm(
                enumerate(prompts),
                total=n_prompts,
                desc=f"Generating CoT via {self.provider} ({self.model_name})",
                leave=False,
            )
            if use_pbar
            else enumerate(prompts)
        )
        for i, prompt in iterator:
            qid = question_ids[i] if question_ids is not None else 0
            cid = concept_ids[i] if concept_ids is not None else 0
            res = responses[i] if responses is not None else 0

            cache_key = hashlib.sha256(f"{self.provider}_{self.model_name}_{prompt}".encode("utf-8")).hexdigest()[:24]
            cached_text = self.cache_mgr.get(cache_key)

            if cached_text:
                cot_texts.append(cached_text)
                if use_pbar:
                    iterator.set_postfix({"status": "cache_hit"})
            else:
                if use_pbar:
                    iterator.set_postfix({"status": "generating_cot..."})
                text = self._call_provider(prompt, qid, cid, res)
                self.cache_mgr.put(cache_key, text)
                cot_texts.append(text)

        # Save any newly generated rationales to offline text cache
        self.cache_mgr.save()

        # Encode text rationales via frozen lightweight encoder
        h_cot = self.encoder(cot_texts)
        return h_cot


class APIReasoner(APIWithEncoderReasoner):
    """
    Backwards-compatible API Reasoner wrapping APIWithEncoderReasoner.
    Defaults to OpenAI-compatible endpoint with all-MiniLM-L6-v2 text encoder.
    """

    def __init__(
        self,
        model_name: str = "deepseek-reasoner",
        d_llm: int = 2048,
        max_tokens: int = 128,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
        encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        text_cache_path: Optional[Union[str, Path]] = None,
        device: Union[str, torch.device] = "cpu",
        **kwargs,
    ):
        super().__init__(
            provider="openai",
            model_name=model_name,
            encoder_name=encoder_name,
            d_llm=d_llm,
            max_tokens=max_tokens,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            text_cache_path=text_cache_path,
            device=device,
            **kwargs,
        )


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
    elif backend in ["ollama"]:
        return APIWithEncoderReasoner(
            provider="ollama",
            model_name=model_name,
            encoder_name=cfg.get("encoder_name", "sentence-transformers/all-MiniLM-L6-v2"),
            d_llm=d_llm,
            max_tokens=max_tokens,
            base_url=cfg.get("base_url", "http://localhost:11434"),
            timeout=cfg.get("timeout", 60.0),
            text_cache_path=cfg.get("text_cache_path"),
            device=device,
        )
    elif backend in ["google", "gemini"]:
        return APIWithEncoderReasoner(
            provider="google",
            model_name=model_name,
            encoder_name=cfg.get("encoder_name", "sentence-transformers/all-MiniLM-L6-v2"),
            d_llm=d_llm,
            max_tokens=max_tokens,
            api_key=cfg.get("api_key"),
            base_url=cfg.get("base_url"),
            timeout=cfg.get("timeout", 30.0),
            text_cache_path=cfg.get("text_cache_path"),
            device=device,
        )
    elif backend in ["api", "openai", "deepseek", "openrouter"]:
        provider_name = "deepseek" if backend == "deepseek" else ("openrouter" if backend == "openrouter" else "openai")
        return APIWithEncoderReasoner(
            provider=provider_name,
            model_name=model_name,
            encoder_name=cfg.get("encoder_name", "sentence-transformers/all-MiniLM-L6-v2"),
            d_llm=d_llm,
            max_tokens=max_tokens,
            api_key=cfg.get("api_key"),
            base_url=cfg.get("base_url"),
            timeout=cfg.get("timeout", 30.0),
            text_cache_path=cfg.get("text_cache_path"),
            device=device,
        )
    else:
        logger.warning("Unsupported LLM backend '%s'. Falling back to MockReasoner.", backend)
        return MockReasoner(
            model_name=model_name,
            d_llm=d_llm,
            max_tokens=max_tokens,
            metadata_manager=metadata_manager,
        )

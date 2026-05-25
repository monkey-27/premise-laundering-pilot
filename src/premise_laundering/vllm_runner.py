from __future__ import annotations

from .schema import MODEL_ID, SEED


class VLLMGenerator:
    def __init__(
        self,
        model: str = MODEL_ID,
        max_model_len: int = 8192,
        max_new_tokens: int = 900,
        temperature: float = 0.2,
        top_p: float = 0.9,
        seed: int = SEED,
        dtype: str = "bfloat16",
    ) -> None:
        from vllm import LLM, SamplingParams

        try:
            self.sampling_params = SamplingParams(
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
            )
        except TypeError:
            self.sampling_params = SamplingParams(
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        try:
            self.llm = _make_llm(LLM, model, dtype, max_model_len, seed)
        except Exception:
            if dtype == "bfloat16":
                self.llm = _make_llm(LLM, model, "float16", max_model_len, seed)
            else:
                raise

    def __call__(self, prompts: list[str]) -> list[str]:
        outputs = self.llm.generate(prompts, self.sampling_params)
        return [output.outputs[0].text if output.outputs else "" for output in outputs]


def generation_config() -> dict:
    return {
        "engine": "vllm",
        "max_model_len": 8192,
        "max_new_tokens": 900,
        "temperature": 0.2,
        "top_p": 0.9,
        "seed": SEED,
        "dtype_preference": "bfloat16_then_float16",
    }


def _make_llm(llm_cls, model: str, dtype: str, max_model_len: int, seed: int):
    try:
        return llm_cls(
            model=model,
            dtype=dtype,
            max_model_len=max_model_len,
            trust_remote_code=True,
            seed=seed,
        )
    except TypeError:
        return llm_cls(
            model=model,
            dtype=dtype,
            max_model_len=max_model_len,
            trust_remote_code=True,
        )

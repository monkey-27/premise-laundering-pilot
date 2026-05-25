from __future__ import annotations

from .schema import SEED


class PrefixTransformersGenerator:
    def __init__(
        self,
        model: str,
        max_new_tokens: int = 220,
        temperature: float = 0.2,
        top_p: float = 0.9,
        seed: int = SEED,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

    def generate_from_user(self, prompts: list[str], max_new_tokens: int | None = None) -> list[str]:
        return [self._generate(prompt, None, max_new_tokens=max_new_tokens) for prompt in prompts]

    def continue_from_prefix(
        self,
        prompts_and_prefixes: list[tuple[str, str]],
        max_new_tokens: int | None = None,
    ) -> list[str]:
        return [
            self._generate(prompt, assistant_prefix, max_new_tokens=max_new_tokens)
            for prompt, assistant_prefix in prompts_and_prefixes
        ]

    def _generate(
        self,
        user_prompt: str,
        assistant_prefix: str | None,
        max_new_tokens: int | None = None,
    ) -> str:
        import torch

        messages = [{"role": "user", "content": user_prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if assistant_prefix:
            text += assistant_prefix
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=8192)
        if torch.cuda.is_available():
            inputs = {key: value.to("cuda") for key, value in inputs.items()}
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = outputs[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True)


def prefix_generation_config(model: str) -> dict:
    return {
        "engine": "transformers_prefix_continuation",
        "model": model,
        "max_model_len": 8192,
        "prefix_max_new_tokens": 120,
        "continuation_max_new_tokens": 220,
        "source_only_max_new_tokens": 180,
        "temperature": 0.2,
        "top_p": 0.9,
        "seed": SEED,
        "dtype_preference": "bfloat16_then_float16",
    }

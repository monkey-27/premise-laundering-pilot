from __future__ import annotations

from .schema import MODEL_ID, SEED


class TransformersGenerator:
    def __init__(
        self,
        model: str = MODEL_ID,
        max_new_tokens: int = 900,
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
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

    def __call__(self, prompts: list[str]) -> list[str]:
        if len(prompts) == 1:
            return [self._generate_one(prompts[0])]
        return self._generate_batch(prompts)

    def _generate_one(self, prompt: str) -> str:
        import torch

        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=8192)
        if torch.cuda.is_available():
            inputs = {key: value.to("cuda") for key, value in inputs.items()}
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = outputs[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def _generate_batch(self, prompts: list[str]) -> list[str]:
        import torch

        messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
        texts = [
            self.tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True,
            )
            for message in messages
        ]
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=8192,
            padding=True,
        )
        if torch.cuda.is_available():
            inputs = {key: value.to("cuda") for key, value in inputs.items()}
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        prompt_width = inputs["input_ids"].shape[-1]
        return [
            self.tokenizer.decode(output[prompt_width:], skip_special_tokens=True)
            for output in outputs
        ]


def generation_config() -> dict:
    return {
        "engine": "transformers",
        "max_model_len": 8192,
        "max_new_tokens": 900,
        "temperature": 0.2,
        "top_p": 0.9,
        "seed": SEED,
        "dtype_preference": "bfloat16_then_float16",
    }

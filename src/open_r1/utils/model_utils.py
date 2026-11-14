import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

from trl import ModelConfig, get_kbit_device_map, get_quantization_config

from ..configs import GRPOConfig, SFTConfig


def get_tokenizer(model_args: ModelConfig, training_args: SFTConfig | GRPOConfig) -> PreTrainedTokenizer:
    """Get the tokenizer for the model."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )

    if training_args.chat_template is not None:
        tokenizer.chat_template = training_args.chat_template
    elif tokenizer.chat_template is None:
        # Base models (e.g. Pythia) need a template for GRPO as well as SFT.
        # Reuse existing vocabulary/EOS; avoid resizing a large embedding table just for role delimiters.
        tokenizer.chat_template = (
            "{% for message in messages %}{{ message['role'] + ': ' + message['content'] + '\\n' }}"
            "{% if message['role'] == 'assistant' %}{{ eos_token }}{% endif %}{% endfor %}"
            "{% if add_generation_prompt %}{{ 'assistant: ' }}{% endif %}"
        )

    return tokenizer


def get_model(model_args: ModelConfig, training_args: SFTConfig | GRPOConfig) -> AutoModelForCausalLM:
    """Get the model"""
    device_type = training_args.device.type
    if device_type != "cuda":
        if getattr(model_args, "load_in_4bit", False) or getattr(model_args, "load_in_8bit", False):
            raise ValueError("bitsandbytes quantization requires the CUDA path; use float32 + LoRA on Mac/CPU.")
        if model_args.attn_implementation == "flash_attention_2":
            raise ValueError("FlashAttention requires CUDA; set --attn_implementation eager on Mac/CPU.")
        if getattr(training_args, "use_vllm", False):
            raise ValueError("This pinned vLLM backend requires CUDA; set --use_vllm false on Mac/CPU.")
        if getattr(training_args, "use_liger_kernel", False):
            raise ValueError("Liger requires CUDA; set --use_liger_kernel false on Mac/CPU.")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs,
    )
    return model

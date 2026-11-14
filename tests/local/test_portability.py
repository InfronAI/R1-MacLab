from types import SimpleNamespace

import pytest

from open_r1.utils import model_utils


def test_base_tokenizer_gets_template_without_resizing(monkeypatch):
    tokenizer = SimpleNamespace(chat_template=None)
    monkeypatch.setattr(model_utils.AutoTokenizer, "from_pretrained", lambda *a, **kw: tokenizer)
    model = SimpleNamespace(model_name_or_path="base", model_revision="main", trust_remote_code=False)
    result = model_utils.get_tokenizer(model, SimpleNamespace(chat_template=None))
    assert "add_generation_prompt" in result.chat_template
    result.chat_template = "existing"
    assert model_utils.get_tokenizer(model, SimpleNamespace(chat_template=None)).chat_template == "existing"


@pytest.mark.parametrize("flag,value", [("load_in_4bit", True), ("attn_implementation", "flash_attention_2")])
def test_cuda_only_options_fail_before_model_download(flag, value):
    model = SimpleNamespace(load_in_4bit=False, load_in_8bit=False, attn_implementation="eager")
    setattr(model, flag, value)
    with pytest.raises(ValueError, match="CUDA"):
        model_utils.get_model(model, SimpleNamespace(device=SimpleNamespace(type="mps")))

"""Protect scientific invariants rather than reproducing the implementation."""

from dataclasses import replace

import pytest
import torch

from open_r1.local.config import Config
from open_r1.local.data import correctness, expression_value, final_answer, load_tasks
from open_r1.local.evaluation import pass_at_k
from open_r1.local.model import Policy, batch_tensors, seed_all, token_logps
from open_r1.local.pipeline import Run
from open_r1.local.training import grpo_loss, group_advantages, train_sft


def small_config():
    return Config(
        device="cpu",
        width=16,
        student_width=16,
        layers=1,
        batch_size=2,
        pretrain_steps=1,
        cold_steps=1,
        sft_steps=1,
        distill_steps=1,
        rl_steps=1,
        alignment_steps=1,
        reward_steps=1,
        eval_samples=2,
        eval_limit=1,
        max_new_tokens=8,
        ablations=False,
    )


def test_splits_and_reference_verification():
    rows = load_tasks()
    assert {r["split"] for r in rows} == {"train", "validation", "test"}
    assert len({r["prompt"] for r in rows}) == len(rows)
    assert all(correctness(r["completion"], r) == 1 for r in rows)
    assert rows == load_tasks()


@pytest.mark.parametrize("text", ["__import__('os').system('id')", "x ** 100000", "open('file')", "[x]*10000", "True"])
def test_code_verifier_never_executes_programs(text):
    with pytest.raises((ValueError, SyntaxError)):
        expression_value(text, 2)


def test_code_semantic_equivalence():
    row = {"task": "code", "tests": [{"x": -1, "y": 2}, {"x": 7, "y": 10}]}
    assert correctness("<answer> 3 + x </answer>", row) == 1
    assert correctness("<answer> 3 - x </answer>", row) == 0
    assert final_answer("<answer>1</answer><answer>2</answer>") is None


def test_group_centering_and_constant_rewards():
    adv = group_advantages(torch.tensor([[1.0, 0.0, 0.0, 0.0], [2.0, 2.0, 2.0, 2.0]]))
    assert torch.allclose(adv.mean(-1), torch.zeros(2), atol=1e-6)
    assert adv[0, 0] > 0 and adv[0, 1] < 0
    assert torch.equal(adv[1], torch.zeros(4))


def test_clipping_and_prompt_padding_mask():
    # Advantage +1 must stop improving once likelihood ratio exceeds 1.2.
    logps = torch.tensor([[0.7, -99.0]], requires_grad=True)
    old = torch.zeros_like(logps)
    loss, kl = grpo_loss(logps, old, logps.detach(), torch.ones(1), torch.tensor([[1.0, 0.0]]), 0.0, 0.2)
    assert loss.item() == pytest.approx(-1.2)
    assert kl.item() == pytest.approx(0)
    loss.backward()
    assert torch.equal(logps.grad, torch.zeros_like(logps))


def test_k3_positive_and_per_sequence_normalization():
    current = torch.zeros(2, 2, requires_grad=True)
    ref = torch.full((2, 2), 0.5)
    loss, kl = grpo_loss(
        current, current.detach(), ref, torch.tensor([1.0, -1.0]), torch.tensor([[1.0, 0.0], [1.0, 1.0]]), 0.1, 0.2
    )
    assert kl.item() > 0
    assert loss.item() == pytest.approx(0.1 * kl.item(), abs=1e-6)


def test_masked_extreme_scores_do_not_poison_loss():
    logps = torch.tensor([[0.0, 1000.0]], requires_grad=True)
    loss, kl = grpo_loss(
        logps,
        torch.zeros_like(logps),
        torch.tensor([[0.0, -1000.0]]),
        torch.ones(1),
        torch.tensor([[1.0, 0.0]]),
        0.04,
        0.2,
    )
    assert loss.item() == pytest.approx(-1.0)
    assert kl.item() == 0
    loss.backward()
    assert logps.grad[0, 1].item() == 0


def test_pass_at_k_estimator():
    assert pass_at_k(4, 2, 1) == 0.5
    assert pass_at_k(4, 2, 2) == pytest.approx(5 / 6)
    assert pass_at_k(4, 0, 4) == 0
    with pytest.raises(ValueError):
        pass_at_k(2, 1, 3)


def test_masking_parameter_update_and_save_reload(tmp_path):
    seed_all(5)
    cfg = small_config()
    policy = Policy.create(cfg)
    row = load_tasks()[0]
    ids, attention, mask = batch_tensors(policy, [row, dict(row, completion="<answer> 1 </answer>")])
    prefix_len = len(policy.prompt_ids(row["prompt"]))
    assert not mask[:, : prefix_len - 1].any()
    assert mask[0, prefix_len - 1] == 1
    assert mask[1, -1] == 0
    before = {k: v.clone() for k, v in policy.model.state_dict().items()}
    train_sft(policy, [row], 1, cfg, lambda *a: None, "test")
    assert any(not torch.equal(v, before[k]) for k, v in policy.model.state_dict().items())
    policy.save(tmp_path / "model")
    reloaded = Policy.load(tmp_path / "model", "cpu")
    reloaded.model.eval()
    with torch.no_grad():
        assert torch.allclose(token_logps(policy, ids, attention), token_logps(reloaded, ids, attention))


def test_resume_guards_and_artifact_integrity(tmp_path):
    cfg = small_config()
    run = Run(cfg, tmp_path / "run")
    run.stage("dummy", [], lambda path: (path / "evidence.txt").write_text("real") and {})
    resumed = Run(cfg, tmp_path / "run", resume=True)
    resumed.stage("dummy", [], lambda p: pytest.fail("completed stage should be reused"))
    with pytest.raises(ValueError, match="identical configuration"):
        Run(replace(cfg, seed=99), tmp_path / "run", resume=True)
    (tmp_path / "run/stages/dummy/evidence.txt").write_text("modified")
    with pytest.raises(ValueError, match="artifact changed"):
        resumed.stage("dummy", [], lambda p: {})


def test_zero_accepted_traces_do_not_turn_into_reference_data(tmp_path, monkeypatch):
    run = Run(small_config(), tmp_path / "run")
    policy = Policy.create(small_config())
    monkeypatch.setattr(run, "load", lambda _: policy)
    monkeypatch.setattr(policy, "sample", lambda *a: [{"text": "incorrect", "ids": [2], "truncated": False}])
    with pytest.raises(RuntimeError, match="zero traces"):
        run.reject(tmp_path / "rejection")
    assert (tmp_path / "rejection/accepted.jsonl").read_text() == ""

"""Six-step CPU constructed-target optimization and 3+3 checkpoint parity.

This is a *software smoke experiment*, NOT a segmentation dataset benchmark.
"""
import copy
import json

import torch
from .bridge import RivalConditionedTransfer


def run():
    torch.set_num_threads(2)
    torch.manual_seed(103)
    x = torch.randn(1, 12, 4, 7, 7)
    scores = torch.randn(1, 4, 7, 7)
    target = (x + 0.3 * torch.sin(x)).detach()
    rows = []
    for arm in ("S", "M", "V", "R", "P"):
        torch.manual_seed(11)
        model = RivalConditionedTransfer(12, arm, rank=6, attn_dim=4, init_scale=0.15)
        initial_state = copy.deepcopy(model.state_dict())
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        losses = []
        for _ in range(6):
            optimizer.zero_grad(set_to_none=True)
            loss = (model(x, scores) - target).square().mean()
            losses.append(float(loss.detach()))
            loss.backward()
            optimizer.step()

        model_b = RivalConditionedTransfer(12, arm, rank=6, attn_dim=4, init_scale=0.15)
        model_b.load_state_dict(initial_state)
        opt_b = torch.optim.AdamW(model_b.parameters(), lr=0.01)
        for _ in range(3):
            opt_b.zero_grad(set_to_none=True)
            ((model_b(x, scores) - target).square().mean()).backward()
            opt_b.step()
        saved_model = copy.deepcopy(model_b.state_dict())
        saved_optimizer = copy.deepcopy(opt_b.state_dict())
        model_c = RivalConditionedTransfer(12, arm, rank=6, attn_dim=4, init_scale=0.15)
        model_c.load_state_dict(saved_model)
        opt_c = torch.optim.AdamW(model_c.parameters(), lr=0.01)
        opt_c.load_state_dict(saved_optimizer)
        for _ in range(3):
            opt_c.zero_grad(set_to_none=True)
            ((model_c(x, scores) - target).square().mean()).backward()
            opt_c.step()
        diff = max((model.state_dict()[key] - model_c.state_dict()[key]).abs().max().item()
                   for key in model.state_dict())
        if diff > 1e-7:
            raise AssertionError(f"{arm}: failed optimizer/model resume parity {diff}")
        rows.append({"arm": arm, "initial_loss": losses[0], "last_preupdate_loss": losses[-1],
                     "loss_decreased": losses[-1] < losses[0],
                     "max_resume_parameter_difference": diff})
    print(json.dumps({"experiment": "synthetic CPU optimization smoke (NOT segmentation)",
                      "torch": torch.__version__, "cuda": torch.cuda.is_available(),
                      "arms": rows}, indent=2))
    return rows


if __name__ == "__main__":
    run()

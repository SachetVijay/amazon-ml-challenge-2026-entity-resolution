"""LoRA fine-tune a decoder LLM (Qwen etc.) with a regression head on the last token.

  # one fold at a time (each run fills that fold's OOF rows + adds a test prediction)
  python -m amlc.finetune_llm --model Qwen/Qwen3-1.7B --fold 0 --name qwen17_lora
  python -m amlc.finetune_llm --model Qwen/Qwen3-8B --load-4bit --bs 4 --grad-accum 8 --fold 0 --name qwen8_lora

Expensive models: run 1-2 folds, stack.py fits blend weights on rows every model has.
Respect the competition's parameter cap / licence rules.
"""
import argparse
import math

import numpy as np
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig, get_cosine_schedule_with_warmup

from .data import ART, folds, fwd, inv, save_preds, test_df, texts, train_df, y_raw
from .metrics import greater_is_better, score


class Reg(nn.Module):
    def __init__(self, base, hid):
        super().__init__()
        self.base, self.head = base, nn.Linear(hid, 1)

    def forward(self, ids, mask):
        h = self.base(input_ids=ids, attention_mask=mask).last_hidden_state
        last = mask.sum(1) - 1  # right padding -> last real token
        return self.head(h[torch.arange(len(h)), last].float()).squeeze(-1)


def batches(enc, idx, bs, pad_id, dev, shuffle):
    idx = np.random.permutation(idx) if shuffle else idx
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        L = max(len(enc[j]) for j in b)
        ids = torch.full((len(b), L), pad_id)
        mask = torch.zeros((len(b), L), dtype=torch.long)
        for r, j in enumerate(b):
            ids[r, :len(enc[j])] = torch.tensor(enc[j])
            mask[r, :len(enc[j])] = 1
        yield b, ids.to(dev), mask.to(dev)


def predict(model, enc, idx, bs, pad_id, dev):
    model.eval()
    out = np.zeros(len(idx))
    order = idx[np.argsort([len(enc[j]) for j in idx])]  # length-sorted inference
    pos = {j: i for i, j in enumerate(idx)}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for b, ids, mask in batches(enc, order, bs * 2, pad_id, dev, False):
            p = model(ids, mask).float().cpu().numpy()
            for j, v in zip(b, p):
                out[pos[j]] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--evals-per-epoch", type=int, default=2)
    a = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(a.fold)

    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "right"
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    enc_tr = tok(texts(train_df()).tolist(), truncation=True, max_length=a.max_len)["input_ids"]
    enc_te = tok(texts(test_df()).tolist(), truncation=True, max_length=a.max_len)["input_ids"]

    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                           bnb_4bit_use_double_quant=True) if a.load_4bit else None
    base = AutoModel.from_pretrained(a.model, torch_dtype=torch.bfloat16, quantization_config=q, device_map={"": 0})
    if a.load_4bit:
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    else:
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        base.enable_input_require_grads()
    base = get_peft_model(base, LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.05,
                                           target_modules="all-linear"))
    model = Reg(base, base.config.hidden_size)
    model.head.float().to(dev)  # base is already on GPU via device_map; .to() on a 4-bit model is unsupported

    y, z, f = y_raw(), fwd(y_raw()), folds()
    tri, vai = np.where(f != a.fold)[0], np.where(f == a.fold)[0]
    zm, zs = z[tri].mean(), z[tri].std()
    zn = torch.tensor((z - zm) / zs, dtype=torch.float32)

    opt = torch.optim.AdamW([{"params": [p for p in base.parameters() if p.requires_grad], "lr": a.lr},
                             {"params": model.head.parameters(), "lr": a.head_lr}], weight_decay=0.01)
    steps = a.epochs * math.ceil(len(tri) / (a.bs * a.grad_accum))
    sched = get_cosine_schedule_with_warmup(opt, int(0.05 * steps), steps)
    eval_every = max(1, math.ceil(len(tri) / a.bs) // a.evals_per_epoch)
    gib, best, best_v = greater_is_better(), None, None
    idx_te = np.arange(len(enc_te))
    for ep in range(a.epochs):
        model.train()
        for it, (b, ids, mask) in enumerate(batches(enc_tr, tri, a.bs, pad_id, dev, True)):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = model(ids, mask)
            loss = nn.functional.huber_loss(p.float(), zn[b].to(dev), delta=1.0) / a.grad_accum
            loss.backward()
            if (it + 1) % a.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(), sched.step(), opt.zero_grad()
            if (it + 1) % eval_every == 0:
                pv = predict(model, enc_tr, vai, a.bs, pad_id, dev) * zs + zm
                sc = score(y[vai], inv(pv))
                print(f"ep {ep} it {it + 1}: val {sc:.4f} loss {loss.item() * a.grad_accum:.4f}", flush=True)
                if best is None or (sc > best if gib else sc < best):
                    best, best_v = sc, pv
                    torch.save({k: v for k, v in model.state_dict().items() if "lora" in k or "head" in k},
                               ART / f"{a.name}_f{a.fold}.pt")
                model.train()
    # reload best adapter + head, predict test
    model.load_state_dict(torch.load(ART / f"{a.name}_f{a.fold}.pt"), strict=False)
    pt = predict(model, enc_te, idx_te, a.bs, pad_id, dev) * zs + zm

    parts = ART / "parts"
    parts.mkdir(exist_ok=True)
    np.save(parts / f"{a.name}_f{a.fold}_oof.npy", _scatter(len(y), vai, inv(best_v)))
    np.save(parts / f"{a.name}_f{a.fold}_test.npy", inv(pt))
    _merge(a.name, len(y), len(pt))
    print(f"fold {a.fold} best {best:.4f}")


def _scatter(n, idx, v):
    o = np.full(n, np.nan)
    o[idx] = v
    return o


def _merge(name, n_tr, n_te):
    parts = sorted((ART / "parts").glob(f"{name}_f*_oof.npy"))
    oof = np.full(n_tr, np.nan)
    tests = []
    for p in parts:
        o = np.load(p)
        oof = np.where(np.isnan(o), oof, o)
        tests.append(np.load(str(p).replace("_oof.npy", "_test.npy")))
    save_preds(name, oof, np.mean(tests, 0))
    print(f"{name}: {len(parts)} fold(s) merged, OOF coverage {np.isfinite(oof).mean():.0%}")


if __name__ == "__main__":
    main()

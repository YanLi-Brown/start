import argparse
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel

from model import Model, ModelConfig
from sample import generate
from load import load_gpt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default="The capital of France is", type=str)
    p.add_argument("--max_new_tokens", default=80, type=int)
    p.add_argument("--temperature", default=0.8, type=float)
    p.add_argument("--top_k", default=None, type=int)
    p.add_argument("--top_p", default=0.9, type=float)
    p.add_argument(
        "--model", default="gpt2", type=str
    )  # gpt2 / gpt2-medium / gpt2-large ...
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--verify", action="store_true")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)

    model = Model(ModelConfig())
    model = load_gpt(model, args.model)
    model.eval()

    if args.verify:
        # 同一段输入，对比我们和官方 GPT-2 的 logits
        hf = GPT2LMHeadModel.from_pretrained(args.model)
        hf.eval()
        idx = tok(args.prompt, return_tensors="pt").input_ids
        with torch.no_grad():
            my_logits, _ = model(idx)
            hf_logits = hf(idx).logits
        diff = (my_logits - hf_logits).abs().max().item()
        print(f"max |diff| = {diff:.6e}   (理想情况 < 1e-4)")
        return

    idx = tok(args.prompt, return_tensors="pt").input_ids
    out = generate(
        model,
        idx,
        args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()

from transformers import GPT2LMHeadModel

_TRANSPOSED = (".c_attn.weight", ".c_proj.weight", ".c_fc.weight")


def load_gpt(model, name="gpt2"):
    hf = GPT2LMHeadModel.from_pretrained(name)

    sd = hf.state_dict()

    for k in list(sd.keys()):
        if k.endswith(_TRANSPOSED):
            sd[k] = sd[k].t()

    model.load_state_dict(sd, strict=True)

    return model

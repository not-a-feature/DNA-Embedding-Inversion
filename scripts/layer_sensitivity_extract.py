"""Extraction adapters for the layer study; foundation models remain frozen."""

import importlib

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from generate.generate_dnabert2_embeddings import _load_dnabert


def make_embedder(fm, layer, checkpoint, device):
    if fm == "dnabert2":
        tokenizer, model = _load_dnabert(checkpoint, device)
        model.requires_grad_(False)
        captured = []

        def capture(module, inputs, output):
            captured.append(output.detach())

        model.encoder.layer[layer - 1].register_forward_hook(capture)

        def embed(sequences):
            result = []
            for seq in sequences:
                ids = tokenizer(seq, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
                captured.clear()
                with torch.no_grad():
                    last = model(ids)[0]
                assert len(captured) == 1
                hidden = captured.pop().reshape(-1, model.config.hidden_size)
                assert hidden.shape[0] == ids.shape[1]
                if layer == 12:
                    torch.testing.assert_close(hidden, last[0])
                result.append(hidden.float().cpu().numpy().mean(axis=0))
            return result

        return embed

    if fm == "ntv2":
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
        model = AutoModelForMaskedLM.from_pretrained(checkpoint, trust_remote_code=True).to(device)
        model.eval().requires_grad_(False)

        def embed(sequences):
            result = []
            for start in range(0, len(sequences), 32):
                batch = sequences[start:start + 32]
                enc = tokenizer(batch, return_tensors="pt", padding=True,
                                truncation=True, max_length=2048, add_special_tokens=False).to(device)
                with torch.no_grad():
                    outputs = model(**enc, output_hidden_states=True)
                assert len(outputs.hidden_states) == 30
                hidden = outputs.hidden_states[layer].float().cpu().numpy()
                mask = enc["attention_mask"].bool().cpu().numpy()
                result.extend(hidden[i][mask[i]].mean(axis=0) for i in range(len(batch)))
            return result

        return embed

    assert fm == "evo2"
    evo2 = importlib.import_module("evo2")
    model = evo2.Evo2(checkpoint)
    model.model.eval().requires_grad_(False)
    layer_name = f"blocks.{layer}.mlp.l3"
    model.model.get_submodule(layer_name)

    def embed(sequences):
        result = []
        for start in range(0, len(sequences), 8):
            tokens = [model.tokenizer.tokenize(s) for s in sequences[start:start + 8]]
            assert len({len(t) for t in tokens}) == 1
            ids = torch.tensor(tokens, dtype=torch.long, device=device)
            with torch.no_grad():
                _, embeddings = model(ids, return_embeddings=True, layer_names=[layer_name])
            hidden = embeddings[layer_name].detach().float().cpu().numpy()
            assert hidden.shape[:2] == tuple(ids.shape)
            result.extend(hidden.mean(axis=1))
        return result

    return embed

"""Load the pinned JAX reference's parameters into the PyTorch transcription.

**By name, never by position.** A positional mapping is correct until someone
inserts a layer, and then it is silently wrong in a way every shape check passes.

The Flax tree and this module tree were given matching names on purpose, so the
mapping is nearly the identity:

    embed/embedding                    -> embed.weight
    pos_embed/embedding                -> pos_embed.weight
    blocks_3/ln_mem/scale              -> blocks.3.ln_mem.weight
    blocks_3/cross_attn/key/kernel     -> blocks.3.cross_attn.key.kernel
    srep_head/proj/kernel              -> srep_head.proj.kernel
    out_ln/bias                        -> out_ln.bias

Three renames are unavoidable and each is a place transcriptions go wrong:

* Flax LayerNorm calls its gain `scale`; `nn.LayerNorm` calls it `weight`.
* Flax `nn.Embed` calls its table `embedding`; `nn.Embedding` calls it `weight`.
* Flax `nn.Dense` holds `kernel` as `[in, out]`; `nn.Linear` holds `weight` as
  `[out, in]`. **This module keeps Flax's orientation** rather than transposing,
  so the loader copies rather than transposing and there is no chance of a
  transpose being applied twice or not at all. `tests/test_value_head_arithmetic`
  records what an undetected transpose costs elsewhere.

`load_reference_params` is **strict in both directions**: every reference array
must be consumed and every model parameter must be filled. A partial load is the
failure mode where the fidelity test compares a half-initialised model and the
mismatch is blamed on the transcription.
"""

from __future__ import annotations

import numpy as np
import torch

from rsr.model.tg.model import TGModel

__all__ = ["load_reference_params", "reference_name_to_torch"]

_LEAF_RENAMES = {"scale": "weight", "embedding": "weight"}


def reference_name_to_torch(name: str) -> str:
    """`blocks_3/ln_mem/scale` -> `blocks.3.ln_mem.weight`."""
    parts = name.split("/")
    out = []
    for part in parts:
        if part.startswith("blocks_") and part[7:].isdigit():
            out += ["blocks", part[7:]]
        else:
            out.append(part)
    if out[-1] in _LEAF_RENAMES:
        out[-1] = _LEAF_RENAMES[out[-1]]
    return ".".join(out)


def load_reference_params(model: TGModel, arrays: dict[str, np.ndarray]) -> None:
    """Copy `param/...` arrays from a golden fixture into `model`, in place.

    Raises on any unconsumed reference array or any unfilled model parameter.
    """
    reference = {
        key[len("param/") :]: value
        for key, value in arrays.items()
        if key.startswith("param/")
    }
    if not reference:
        raise KeyError("no 'param/...' arrays; is this a golden fixture?")

    own = dict(model.named_parameters())
    unfilled = set(own)
    unconsumed = set(reference)

    with torch.no_grad():
        for ref_name, value in reference.items():
            torch_name = reference_name_to_torch(ref_name)
            if torch_name not in own:
                raise KeyError(
                    f"reference parameter {ref_name!r} maps to {torch_name!r}, "
                    f"which the model does not have. Either the transcription is "
                    f"missing a layer or the name mapping is wrong -- do not "
                    f"'fix' this by skipping the array."
                )
            param = own[torch_name]
            array = torch.as_tensor(np.asarray(value), dtype=param.dtype)
            if tuple(array.shape) != tuple(param.shape):
                raise ValueError(
                    f"{ref_name} has shape {tuple(array.shape)} but "
                    f"{torch_name} has {tuple(param.shape)}. If these differ only "
                    f"by a transpose, the module is holding the wrong orientation "
                    f"-- fix the module, not the loader."
                )
            param.copy_(array)
            unfilled.discard(torch_name)
            unconsumed.discard(ref_name)

    if unfilled or unconsumed:
        raise ValueError(
            "the load was partial, which is the failure where the fidelity test "
            "compares a half-initialised model and blames the transcription.\n"
            f"  model parameters left at their init: {sorted(unfilled)}\n"
            f"  reference arrays not used: {sorted(unconsumed)}"
        )

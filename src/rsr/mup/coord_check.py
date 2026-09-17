"""E0a -- the muP coordinate check (spec section 6, kickoff T7).

Prediction: activation RMS width-invariant through the recurrence across
`d in {128, 192, 256, 384}`. **Kill gate.**

Run twice: bare TG, then TG with the value head attached.

**Check `psi_hat`'s scalar output specifically**, not only transformer
activations. That is the failure that otherwise surfaces in week 9 with no error
message.

---

**Operational note, found in Sprint 1 while building the head.** At
initialization the bilinear output RMS *decays* with width -- measured
0.0895 / 0.0701 / 0.0647 / 0.0512 for `d = 128 / 192 / 256 / 384`, roughly
`1/sqrt(d)`.

**That is expected and is not a bug.** It is the section 4.3 distinction made
concrete: with the `1/d` multiplier, the bilinear form is `Theta(sqrt(d))` at init
and so the multiplied output is `Theta(1/sqrt(d))`; it becomes `Theta(d)`, and the
multiplied output `Theta(1)`, only once `s`, `W` and `c` are correlated. **muP is
governed by the trained regime.**

The trap this creates: read the coordinate scale at step 0, see the decay, and
"fix" it by weakening or removing the `1/d` multiplier -- which then blows up as
`Theta(d)` once training correlates the factors. **E0a must therefore read the
coordinate scale after a number of real optimizer steps, not at initialization.**

A synthetic "correlated regime" constructed by aligning `c` with `s` through
`pinv(W)` was tried during Sprint 1 and does **not** substitute for this: it does
not control the correlation magnitude, and the resulting RMS values were
non-monotone in width. Measure it with real training steps. That is what E0a is.
"""

from __future__ import annotations

__all__ = ["run_coord_check"]


def run_coord_check(*args, **kwargs):
    raise NotImplementedError("E0a. See experiments/e0a/run.py.")

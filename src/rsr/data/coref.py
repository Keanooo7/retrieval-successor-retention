"""Coreference and mention detection for E0i (spec section 6, kickoff T3).

**fastcoref (MIT) is the pipeline of record; maverick-coref (CC BY-NC-SA 4.0) is
an agreement/ceiling check only.** See ADR-0003 -- the NC licence is acceptable
precisely because maverick never ships in the pipeline.

D-2 named coref as "an unnamed dependency with its own error rate feeding straight
into the dependent measure", then prescribed only a *population* check, which
cannot see precision. T3 closes that: hand-annotate 100 reintroductions and report
precision and recall.

Sentence segmentation uses **SaT**, matching section 10.1's "SaT-segmented
sentences" and the TG reference's `sat-3l-sm`.

No published CPU throughput figure exists for any candidate, so the 1M-token pilot
measures it rather than citing one.
"""

from __future__ import annotations

__all__ = ["segment_sentences", "detect_reintroductions"]


def segment_sentences(*args, **kwargs):
    raise NotImplementedError("E0i / kickoff T3.")


def detect_reintroductions(*args, **kwargs):
    raise NotImplementedError("E0i / kickoff T3.")

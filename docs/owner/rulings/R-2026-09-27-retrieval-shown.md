---
id: R-2026-09-27-retrieval-shown
date: 2026-09-27
stated_in: 'Interactive PM session, 2026-09-27. The draft is PLAN-v4 appendix C2, approved with the plan ("you are aproved").'
---
**Fresh-stream arm B, checkpoint `B.ckpt3000`, shows retrieval through memory.**

On held-out gap 2..16, answer accuracy across 3 seeds:

| Memory state | Accuracy (3 seeds) |
|---|---|
| Live | 0.92068 / 0.90859 / 0.93220 |
| Wiped (`slots_zeroed`) | 0.06091 / 0.05540 / 0.05226 |

Source: `runs/fresh-stream/ledger.json` on main, merged via PR #45.

This ruling is scoped to that substrate and checkpoint only. It satisfies `e0d`'s `ruling: R-*-retrieval-shown`. It opens nothing else.

**Separately:** I ratify, retroactively, the Stage-1 gate that was opened on 2026-09-26 without a ruling on record. That gate covers read-only measurements on frozen arm-B checkpoints.

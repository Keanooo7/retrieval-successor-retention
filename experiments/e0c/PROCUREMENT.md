# T8 — GPU procurement

| | |
|---|---|
| Status | ⚠️ **PARTIAL — comparison and decision recorded; no account opened, no hold placed** |
| Date | 2026-09-17 |
| Decides | §16 condition 8 — *"Parallel GPU capacity reserved, not just total hours priced"* |
| Owner action required | **Yes.** Opening an account, requesting a quote and placing any hold are outward-facing commercial acts and are not done here. |

---

## 🔴 The instrument the plan assumes may not exist

Both the delivery plan and the Sprint 1 kickoff resolve the §8-vs-§16 conflict the same way: week 1
takes *"a named provider, a written quote, and a **cancellable hold with a movable start date** —
nothing billable."* The reasoning is right — the reservation *is* the four-figure commitment §16
declines to approve, so a hold satisfies condition 8 without pre-empting the gate.

**But commodity GPU providers do not appear to sell that instrument.** What is actually on the
market is three things:

| Instrument | Commitment | Cancellable? | Guarantees capacity? |
|---|---|---|---|
| **On-demand** | none | n/a | 🔴 **no** — subject to availability at the moment you ask |
| **Savings plan / committed use** | **prepaid** | 🔴 **no** — RunPod's are *"prepaid, non-refundable, with fixed expiration dates"*, and *"if you stop it early, you still pay the reserved rate for the remaining hours"* | yes |
| **Enterprise reserved contract** | negotiated | contract-dependent | yes |

So the choice is not "hold vs book." It is:

- **On-demand** — no commitment, and **no guarantee that four 48 GB cards are simultaneously
  available** in weeks 6–8. That is the risk §4.1 was trying to retire by reserving.
- **Prepaid commitment** — a real four-figure outlay, which is exactly what §16 does not approve.
- **Enterprise quote** — a sales conversation. **Free, produces a written quote, and is the only
  route on which a genuinely cancellable or movable term is negotiable.**

### Recommendation

**Do the enterprise/sales route for the written quote, and plan to satisfy condition 8 by
*converting at the gate*, not by holding before it.**

Rewrite the week-1 exit as: **a named provider · a written quote · a documented availability check ·
a stated conversion decision.** Drop "cancellable hold" unless a provider confirms in writing that
they sell one — the current evidence is that the default product is non-refundable.

⚠️ **This changes what §16 condition 8 can mean before the gate.** It cannot mean "capacity is
locked." It can mean "we have a named provider, a price in writing, evidence the capacity exists,
and a decision ready to execute on the day the gate passes." That is weaker than the condition's
wording and should be recorded as weaker, not quietly relabelled.

---

## Provider comparison

⚠️ **Every figure below is a snapshot and the sources disagree with each other.** Prices moved
between the delivery plan's drafting and this check, in both directions. **Re-quote at the moment of
decision**; do not carry these numbers.

| Provider | Card | Quoted $/GPU-h | Source, September 2026 |
|---|---|---|---|
| RunPod | A40 48 GB | **$0.35 – $0.49** | sources disagree; ComputePrices $0.44, GetDeploying $0.49, gpuperhour $0.35 |
| RunPod | RTX A6000 48 GB | **$0.49** | gpuperhour |
| Thunder Compute | RTX A6000 48 GB | **$0.35** | provider's own pricing page |
| Vultr | A40 48 GB | **$1.86** | ComputePrices — *the delivery plan's $1.71 has moved* |
| — | A40 median across providers | **$1.05** | GetDeploying, 2026-09-15 |
| — | RTX A6000 median | **$0.56** | GetDeploying, 2026-09-09 |

Lambda Labs does **not** carry A6000 or A40 — it is not a candidate for this tier.

**Shortlist: RunPod and Thunder Compute**, with Vast.ai as a spot-market fallback. Both carry 48 GB
cards at the low end of the range, which is the class E0c will size against.

## What this costs, both ways

The band `$0.35 – $0.60` per GPU-hour survives the price movement, so the delivery plan's two
conclusions hold:

```
on-demand, corrected hours   ~1,140 GPU-h × $0.35-0.60   =   $400 -   $690   (three figures)
reserved, 4 GPUs × 504 h (weeks 6-8) + ~330 h elsewhere
                             ~2,346 billed GPU-h          =   $820 - $1,410   (four figures)
```

🔴 **"Four figures" is a property of the billing model, not of the experiments.** §16's unapproved
item is the *reservation*, not the experiment count.

🔴 **And the hour total itself is not yet trustworthy.** The 1,070 GPU-h in §4.1 — and the ~1,140
correction — both descend from [P2]'s 21 sent/sec, measured at `d_model = 768` / 85.6M params, about
four times wider than anything RSR trains (correction **B-2**). **E0c re-derives it.** Do not sign
anything against a number measured on the wrong configuration.

## Weeks 6–8 capacity, restated

The window is `3 × 7 × 24 = 504` wall-clock hours and must hold E3 (576) + A2/A4 (192) +
the E7 model (72) + A1 (2) = **842 GPU-h**:

| Assumption | GPUs |
|---|---|
| 100% utilisation, zero reruns | 1.67 |
| 85% utilisation, +25% rerun margin | 2.46 |
| 65% utilisation, +25% rerun margin | **3.21 — breaks the stated floor of 3** |

**Hold the floor at 4**, or move the E7 model out of the window.

📌 **And E0g may remove the E7 model entirely.** Three of the four NFRD stories run ~85–95 sentences,
over 2× `M = 40`, so E7 may be runnable on E3's own model with no second training run — which takes
72 GPU-h out of this window and drops the 65% figure to **2.94, inside a floor of 3.** Confirm by
measuring sentence counts under SaT before banking it.

## Owner actions

1. Open an account with RunPod **or** Thunder Compute. *(Not done here — account creation is yours.)*
2. Request a **written quote** for 4 × 48 GB cards, ~504 hours, start **week 6, movable**. Ask
   explicitly: **is any cancellable or deferrable term available**, and what is the notice period?
3. Record the answer here. If the honest answer is "no cancellable hold exists," say so in the
   GATE-1 report rather than marking condition 8 satisfied.
4. Check live availability of 4 simultaneous 48 GB cards in the target region. Availability is the
   risk reservation was meant to retire, and it is checkable for free, today.

## Sources

GetDeploying A40 and RTX A6000 comparison pages · ComputePrices provider pages for RunPod, Vultr and
Thunder Compute · Thunder Compute pricing page · RunPod pricing and savings-plan documentation ·
gpuperhour RunPod listing. All retrieved 2026-09-17.

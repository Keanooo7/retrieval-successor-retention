# RSR — End of Day Report, 17 September 2026

*What the project is, what we built today, and what we actually learned.*

---

## Part 1 — What this project is

### The one-sentence version

We are testing whether a language model, trained only to predict the next sentence, will
spontaneously discover a theory of human memory published in 1978.

### The longer version

The base model is called **Thought Gestalt**, or TG. It was built at Stanford by Nasim
Borazjanizadeh and James McClelland. It works differently from a normal language model:

- It reads **one sentence at a time**, not one word at a time.
- After reading each sentence, it compresses that whole sentence into a **single vector** — a
  "gestalt." Think of it as a one-line summary the model writes to itself.
- It keeps those summaries in a small **notebook** with a fixed number of pages. In the published
  version, 40 pages.
- To understand later sentences, it can only look back through that notebook. It cannot re-read the
  original text.

Now the important part: **the notebook fills up.** When it does, TG throws out the oldest page.
First in, first out. That is the entire rule.

**Our project changes that one rule.** Instead of throwing out the oldest page, we throw out the
page the model predicts it will need least, given where the story currently is. That is
**Retrieval-Successor Retention** — RSR.

### Why this is interesting

In 1978, two psychologists — Walter Kintsch and Teun van Dijk — proposed that human beings do
something very similar when reading. They argued we hold a limited number of ideas in working
memory, and when we run out of room we keep the ones that are *structurally important to the story*
and discard the rest. They called it the **leading-edge strategy**, and it successfully predicted
what people actually remember from narratives.

So the real question is not "can we build a better cache." It is:

> **If you build a system that has to forget, and you train it only to predict the next sentence —
> does it independently invent the rule that humans appear to use?**

If it does, that is a finding about memory, not about engineering.

### A crucial framing point

The project specification is emphatic that the **cognitive claim is primary** and the engineering
claim — "predicting future need beats measuring past use" — is **secondary**. This matters because
it is very easy to drift toward the engineering framing, since engineering results are easier to
measure. The specification treats that drift as a failure mode in itself.

---

## Part 2 — Where we started today and where we ended

**This morning**, the project was a specification document — a detailed 793-line plan — plus a
delivery schedule. There was no code. There was no model. Nobody had checked whether the
specification's own citations were accurate.

**Tonight**, there is:

- A working PyTorch implementation of TG, verified against the original authors' code
- A training loop that demonstrably learns
- 287 passing tests with zero skipped
- Measured hardware capacity on the actual machine
- Eleven substantive corrections to the specification, each traced to a primary source
- An automated overnight research loop with a researcher and a manager

Two independent sessions did this work in parallel — one on the Mac Studio, one on the MacBook Pro —
and then reviewed each other. That turned out to matter enormously, and Part 5 explains why.

---

## Part 3 — What we learned about the model

This is the substance. Each of these came from reading the original paper or the original source
code, not from assumption.

### 3.1 The original code exists — and it is not the same as the paper

The paper gives no code link. But the code is public, at `github.com/jlmcc94303/ThoughtGestaltCode`,
on James McClelland's personal account. It is written in **JAX**, a different framework from the
PyTorch we are using.

And its README says something important, verbatim:

> *"This version differs slightly from the version of the model described in (arXiv:2512.25026)."*

So there are now **two TGs**: the one in the paper, and the one in the code. They do not match. We
decided to build against the **code**, for a simple reason: you can check whether your copy matches
a piece of software. You cannot check whether it matches a description.

### 3.2 The summaries are normalised to length exactly 1 — and the paper never says so

This is the sharpest code-versus-paper divergence, and it has real mathematical consequences.

In the code, every sentence summary is divided by its own length so that it always has a magnitude
of exactly 1.0. The paper describes no such step.

Why it matters: the project's plan for scaling the model up relies on a piece of mathematics
(**μP**, or maximal update parameterization) that lets you tune settings on a small model and carry
them to a large one. That derivation assumes the summary vectors have "normal-sized" components.
A vector forced to length 1 in high dimensions has components that shrink as the model gets wider —
they are about `1/√d` rather than constant.

**Under the original prescription, one of our components would fade away as the model grows, in
exactly the regime the maths is supposed to protect.** We have deliberately *not* changed it — that
derivation belongs to Brendan, and an experiment will decide it. But it is flagged, and the
experiment now knows to measure both cases.

### 3.3 The best finding of the day: position is measured by rank, not by age

This one is subtle, and it is the most important thing we learned.

TG adds a **position signal** to each page of the notebook, so the model knows the order things were
written. Two details matter:

1. The signal is added to the **keys only** — the part used for matching — not to the content.
2. It is indexed by **position in the notebook**, not by how old the page is.

Under the original rule, those two things are the same. The oldest page is always page 1. Position
*is* age.

**But under our rule, they come apart.** If we throw out a page from the middle of the notebook,
every page behind it shifts up one position. So a page's "position signal" becomes a function of
*which other pages our policy chose to delete.*

This means our version and the original differ in **more than the forgetting rule** — which is the
one thing the whole experiment depends on being false.

And here is what makes it genuinely dangerous:

> **The test designed to catch exactly this cannot see it.**

That test works by switching our policy off, so it behaves like the original, and checking the two
match. But with the policy switched off, we *are* the original — the pages never get reshuffled, the
positions line up, and the test passes. **The problem is invisible in the one configuration we
check, and present in every configuration we care about.**

The response was not to argue about it but to **measure it**: log how many pages shift on every
deletion, add a comparison arm with the position signal switched off entirely, and record it all in
a written decision before the policy was built.

*Credit where due: the Mac Studio session found this one.*

### 3.4 The original model has never seen the corpus we plan to use

The paper trains on **WikiText-103** — encyclopedia text. Our headline experiment plans to use
**PG-19**, a collection of old books.

We counted: the words "PG-19" and "Gutenberg" appear **zero times** in the paper. "WikiText" appears
three times.

The specification's stated limitation was that our experiments go *deeper* than the original. The
truth is stronger: they go deeper **and** use an entirely different kind of text. Nobody knows how
TG behaves on book-length narrative at any depth.

### 3.5 The forgetting signal is measured in the wrong place

To decide what to forget, we need a measure of "how much is this page being used." The
specification measures the attention paid to each page.

But the code has a **learnable dial on each of the six layers that read from the notebook**, which
scales that layer's contribution before it is added in. The paper's own appendix shows those dials
**grow during training** and are **larger in the deeper layers**.

So the measurement weights every layer equally, while reality weights them by a factor that is both
uneven and *changing as training proceeds*. The fix is to compute it both ways and let the experiment
decide which agrees with ground truth.

Detail worth knowing: there are **six** of these layers, not twelve.

### 3.6 The comparison that could end the project is not a fair fight

The specification includes a rival method called **Expire-Span**. It is treated as a referendum: if
Expire-Span matches our approach, the whole design is unnecessary and the project stops.

But Expire-Span has its own tuning settings, and it *requires* a particular regularisation technique
to work properly. The plan runs it **once, untuned**, against nine tuned configurations of ours.

An untuned rival losing tells you nothing. A tuned rival winning ends the project. **The unfairness
runs in the direction that protects us**, which is the worst possible direction for a test whose
purpose is to kill the project if it deserves killing.

### 3.7 Two things the rival paper already knew

Our main comparison method is called **H2O**. Two discoveries:

**It already proved the thing we argue against.** H2O formalises cache eviction mathematically and
proves that a greedy strategy is near-optimal under stated assumptions. Our specification argues
*against* greedy strategies on those same mathematical grounds — without engaging their theorem at
all. They are not necessarily in conflict, but the argument has to be made, not skipped.

**Its own limitations section describes our bug, and says our fix failed.** H2O's appendix documents
that accumulating attention over time biases toward *older* items — and reports that they tried
replacing it with an average, which **made performance worse**. Our specification independently
identifies the same bias and proposes the same fix. That needs addressing, not ignoring.

### 3.8 The human-memory data we need exists, and it is free

To test the cognitive claim, we need stories that real people have read and tried to recall.

We found the **Naturalistic Free Recall Dataset** — four spoken stories, 229 participants, full
transcripts of what each person remembered. It is released **CC0**, meaning public domain: no
agreement, no waiting, no permission.

More importantly, it ships a measure of **semantic centrality** — how connected each event is to the
rest of the story. That is a measure of importance that has nothing to do with *when* something
happened, which is precisely what this experiment needs. The authors show it predicts recall.

⚠️ **One hazard found:** one of the four stories is chapter one of *Baseball Joe in the Big League*,
published **1915**, available on Project Gutenberg as book #27584. Our training corpus is books
published before 1919 from Project Gutenberg. **The model may have already read it.** It also has
the highest recall rate of the four, making it the story most likely to produce a convincing but
meaningless result. It must be checked before training.

### 3.9 The compute budget was wrong by about forty times

The specification estimated each training run would take **48 hours** and the project would need
about 1,070 GPU-hours — a four-figure rental bill.

We measured on the actual Mac Studio: **about one hour per run.**

The estimate came from the original paper's throughput, measured on a model **four times wider**
than anything we will train. Everything fits comfortably on hardware already owned. **No compute is
being rented, and none is planned** until the idea has proven itself locally.

Measured, with the memory system live:

| configuration | memory used | speed | time per full run |
|---|---|---|---|
| original rule | 32.2 GB | 374 sentences/sec | 0.9 hours |
| our rule, learned | 31.8 GB | 310 sentences/sec | 1.1 hours |

### 3.10 The model learns

The final check of the day. Starting a fresh model on structured data:

- Loss begins at **10.8171**. Pure chance for this vocabulary is **10.8249**. An untrained model
  sitting exactly at chance is exactly right.
- After 40 iterations it reaches **1.11**.
- Perplexity — roughly "how many options is it choosing between" — falls from **49,865 to 3.0**.

And we confirmed the memory system was actually active: 96 sentences written to the notebook, the
forgetting policy consulted 64 times. Both counted, not assumed. Section 5.2 explains why that last
sentence matters so much.

---

## Part 4 — What we built

**The model.** TG translated from JAX to PyTorch, checked against **438 reference tensors** captured
from the original code — including **206 gradient tensors**. Gradients matter specially here: TG
deliberately keeps the full calculation history of every summary, and the two frameworks handle that
differently. A model that matched on outputs but not gradients would look correct and be wrong in a
way that survives for weeks.

**The training loop.** Data → model → loss → optimizer → save, with the scaling mathematics wired
into the optimizer correctly rather than bolted on.

**The heartbeat.** One line written to a file per interval, forced to disk immediately so it survives
even a hard kill. It records loss, gradient size, speed, memory, and the identity of the exact code
version that produced each number. **If the run dies at 3am, the crash and its full error land in
that file** — so a run that died is distinguishable in the morning from a run that never started.

**Checkpointing and resume**, verified by stopping a run at step 10 and restarting: it continued at
step 10 with the loss descending exactly as it had been.

**The safety rails.** Several automatic checks that refuse to let quality slip: a test-count floor
that can rise but never fall, a limit on a statistical measure that must stay below 0.7, and a count
of unmeasured settings that may fall but never rise.

**A registry that refuses.** Any setting that is supposed to be measured rather than chosen will
**raise an error if read before its experiment has run**, and the error names the experiment that
owes the number. This exists because the predecessor project died from freezing a number nobody had
measured.

---

## Part 5 — The mistakes, and what they teach

This section is the most useful one, because every error below was caught, and each one generalises.

### 5.1 I said the code did not exist. It did.

I searched the paper thoroughly and correctly — no code link, confirmed across all 20 pages. Then I
searched the web for a repository, found a *different* model from 2018 with a similar name, and
concluded nothing existed. I recommended emailing the authors and starting a **three-to-six week**
reimplementation.

The Mac Studio session searched **GitHub's own repository index** for "thought gestalt" and found it
immediately.

> **The lesson: a search that finds nothing tells you nothing unless you have shown the search can
> find something.** One hit for a similar-but-wrong model should have read as information about the
> search tool, not as an answer. Checking would have cost one query.

### 5.2 I measured a model whose memory was switched off

I measured hardware capacity using randomly generated text. But TG only writes a summary to its
notebook when it sees an **end-of-sentence marker** — and my random text was drawn from a range that
could never contain that marker.

So no summary was ever written. The notebook stayed empty. The entire memory system — the thing this
project is *about* — returned exactly zero. **And it still produced a perfectly plausible loss
value.**

The Mac Studio session caught this and re-measured. They even reproduced my numbers first, to prove
the *scripts* worked and the *configuration* was wrong.

> **The lesson: a system can be fully disabled and still produce confident output.** The fix is to
> count the thing you assume is happening. The corrected measurement now *asserts* that sentences
> reach memory and the policy is consulted.

The project's own source code contains a comment warning about this exact trap. I walked into it one
file over.

### 5.3 A five-seed result from a one-seed run

My results file reported an average across five random seeds, with a standard deviation and a range.
The committed code ran **one seed**. It even printed a command to reproduce the result — a command
that produces one seed.

The numbers happened to be correct; I had run five seeds earlier in a scratch terminal. But **the
reproducibility was fictional.**

Fixing it exposed a second layer: my "fix" made the code request five seeds while an inner function
reset the randomness to the same value every time — so all five runs were byte-for-byte identical.

**The thing that exposed it was a reported standard deviation of exactly 0.0000.**

> **The lesson: always report spread, never just an average.** A standard deviation of exactly zero
> is not an unusually clean result; it is a broken one. And a result is not reproducible because a
> command is printed next to it.

### 5.4 Two safety checks that passed when there was nothing to check

One check, on finding no recorded standard, **recorded whatever it happened to see as the standard
and reported success** — a safety rail that certifies any environment on first sight.

Another had never once worked. A flag was passed twice, which made the tool silent, which meant the
number being looked for was never printed. **It had reported "could not measure" since the day it
was written**, and nobody had noticed because the project had no automated checking running.

> **The lesson: a safety check that has never been observed failing is not known to work.** And
> "could not measure" must never be allowed to look like "measured and fine."

### 5.5 Reading a result through a pipe

I checked one of these exit codes by piping the output through another command — and read *that*
command's result instead. It reported success while the actual check was reporting failure. In the
same hour I had written a document about this precise class of error.

> **The lesson: knowing a rule is not the same as executing it.**

### 5.6 Why two sessions mattered

The honest scoreboard: the Mac Studio session was **ahead on the science** and this session was
**ahead on the tooling and the checking.** The Studio found the code, the unit-norm issue, the
position confound, and the corpus mismatch. This session brought the mutation testing, the safety
rails, and four literature findings.

> **The clearest lesson of the day: having better context made my *instruments* better and my
> *scholarship* no better** — and it did not stop me making the single most expensive error.

---

## Part 6 — What is running tonight

An automated loop with two roles and a deliberate separation between them.

**The researcher** runs one experiment at a time, knows every number in detail, and reports. It
**cannot decide what runs next.**

**The manager** knows the whole project, reviews every report, and decides the next run. It **never
runs an experiment** — so it cannot approve its own work. Its one required exception: each cycle it
must **re-run one of the researcher's measurements itself** to see whether it gets the same answer.

Protections against the system drifting overnight:

- **Every number must trace to a machine-written record.** No typed numbers. This is a direct
  response to 5.3.
- **Every run states what it expects before it runs**, and which falsifier it addresses.
- **A canary every fourth cycle**: rerun one fixed configuration and check the number has not moved.
  If it has, the environment moved and everything since is suspect.
- **"We learned nothing this cycle" must be a writable outcome.** A loop that cannot report a null
  night will invent a result instead.

---

## Part 7 — What to look at tomorrow

**Read the overnight log first**, not the final summary message. Look for **rejections and
re-executions**. A night with zero of either is not a clean night — it is an unexamined one.

Three things are genuinely yours to decide:

1. **What counts as "where the story is now"** — the current sentence, or a running summary of
   several? Because summaries are normalised to length 1, this choice changes the mathematics. A
   default has been set so nothing stalls, but it is yours to confirm.
2. **The position-signal decision** (3.3), now written up as a formal decision record.
3. **Section 15 of the specification** — a section reserved for your own original contribution. It
   explicitly must not be filled in by an AI, and no session has touched it. That remains true.

### The one risk nothing tonight can protect against

The Mac Studio has **FileVault** enabled. If power flickers, the machine stops at a password screen
and becomes completely unreachable until someone types it in physically. Everything else is
configured correctly — it never sleeps, restarts automatically, has 720 GB free.

---

## Quick reference — numbers worth remembering

| | |
|---|---|
| Original TG model | 12 layers, width 768, 85.6M parameters |
| Original reported quality | 29.8 test perplexity, trained on WikiText-103 |
| Our model size | width 128–384, 2.4M–21.3M parameters |
| Notebook size | 40 pages on real text, 16 on synthetic |
| Summary vector length | exactly 1.0 (in code; unmentioned in the paper) |
| Layers reading the notebook | 6, at positions 2, 4, 6, 8, 10, 12 |
| Measured speed | ~310–374 sentences/sec on the Mac Studio |
| Measured memory | ~32 GB of 64 GB |
| Time per full training run | about 1 hour (spec estimated 48) |
| Training loss today | 10.82 (chance) → 1.11 |
| Tests passing | 287, zero skipped |
| Human-memory dataset | 4 stories, 229 participants, public domain |

*Every figure above was measured or read from a primary source today. None is an estimate.*

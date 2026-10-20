# RESULTS: RSR run 1 -- psi_hat trained online on frozen arm B ckpt3000

**Descriptive. No gate, no pass/fail threshold.** Every contrast is read only as CI above 0 / CI below 0 / CI includes 0.

## 1. HEADLINE (the one designated contrast)

`acc.all.psiC_minus_fifo`: psiC - FIFO, all-query held-out answer accuracy.

> This interval resamples EVAL documents only. It holds fixed the trained φ (one φ
> initialisation, one data order), the substrate seed and the pool. It is a within-seed interval.
> The three per-seed values show the **combined** variation of substrate seed, φ initialisation,
> data order and EVAL documents, which change together across seeds. They cannot separate
> run-to-run variation from substrate-seed variation.

- seed 0: -0.2053 [-0.2096, -0.2007] CI below 0 (within-seed)
- seed 1: -0.2279 [-0.2323, -0.2236] CI below 0 (within-seed)
- seed 2: -0.1683 [-0.1726, -0.1639] CI below 0 (within-seed)
- across seeds: n = 3, points [-0.205294, -0.227911, -0.168332], mean -0.2005, sd +0.0301, sign count {'above_0': 0, 'below_0': 3, 'exactly_0': 0} (no CI across seeds, no test)

> SECONDARY and DIAGNOSTIC intervals are unadjusted for multiplicity (A1.4); only the headline is designated.

## 2. Commands, exit codes, provenance

- finalize -> exit `0`, status `ok`; tree `ed6302aae37a2831c9b8567b9f80a093b4c00f70` dirty `True`; PREREG {'base': '7c58777', 'A1': '6a4c5a8', 'A2': '7fcd3f9', 'A3': '58a9216', 'A3b': '7196a52', 'A3c': '34f495f', 'A3d': 'fe0e01f', 'A4': '43de9f5'}, reviews ['787875b', 'c02cae7', '7f7e2ad', '3ef97c8', '33ce631', 'de8f813', 'b543e96', '42609b2']
- owner values: `{'gamma': 0.9, 'phi_weight_decay': 0.01, 'iters': 3000, 'o5': 'eval', 'seeds': (0, 1, 2)}`; oracle_gamma `0.9` (A2.2)
- seed 0: owner values registered in the PREREG (`### A4.1 Owner values (fills A2.8's slots)`): status `filled`, values `{'gamma': 0.9, 'phi_weight_decay': 0.01, 'iters': 3000, 'o5': 'eval'}` (C10 compares them, A3.7a); steps recorded per arm `{'psiC': 3000, 'fifo': 3000, 'age': 3000}`, seeds_actually_run `[0]`
- seed 1: owner values registered in the PREREG (`### A4.1 Owner values (fills A2.8's slots)`): status `filled`, values `{'gamma': 0.9, 'phi_weight_decay': 0.01, 'iters': 3000, 'o5': 'eval'}` (C10 compares them, A3.7a); steps recorded per arm `{'psiC': 3000, 'fifo': 3000, 'age': 3000}`, seeds_actually_run `[1]`
- seed 2: owner values registered in the PREREG (`### A4.1 Owner values (fills A2.8's slots)`): status `filled`, values `{'gamma': 0.9, 'phi_weight_decay': 0.01, 'iters': 3000, 'o5': 'eval'}` (C10 compares them, A3.7a); steps recorded per arm `{'psiC': 3000, 'fifo': 3000, 'age': 3000}`, seeds_actually_run `[2]`
- seed 0: launch-git `{'run_sha': 'ed6302aae37a2831c9b8567b9f80a093b4c00f70', 'dirty': False, 'gamma': '0.9', 'wd': '0.01', 'steps': '3000', 'ckpt': '1000', 'o5': 'eval', 'c1_flags': '--frozen-model-mode eval', 'c2_flags': '', 'launched_utc': '2026-10-06T07:12:30Z', 'env': None}`
  - **psiC** rc `0` (ran, live), run_id `rsr-d128-s48-b16-12977d9de91a9898`, config_hash `12977d9de91a9898`, label `rsr[b=off,beta=1,gamma=0.9,nu=0,shadow=off]+frozen[eval]+warm[ckpt-003000.pt@0ee3f8a69b50,src_seed=0]`, job pinned_sha `ed6302aae37a2831c9b8567b9f80a093b4c00f70`, steps recorded `3000`
    - `.venv/bin/python -m rsr.train.loop --d 128 --batch 16 --iters 3000 --steps-per-stream 48 --memory-slots 16 --lr 0.001 --seed 0 --beat-every 1 --ckpt-every 40 --device cpu --freeze-transformer --frozen-model-mode eval --srep-norm-reg-weight 0.0 --pool-offset 993000 --pool-length 256 --pool-vocab-documents 64 --warm-start /Users/keanooo7/retrieval-successor-retention/.worktrees/fresh-stream/runs/fresh-stream/B/seed0/ckpt-003000.pt --warm-start-sha256 0ee3f8a69b507d927c631eb85116e1cd9739ee4472c44eb7f145d739a118da60 --warm-start-manifest /Users/keanooo7/rsr-substrate/2026-09-27/MANIFEST.sha256 --warm-start-root /Users/keanooo7/retrieval-successor-retention --warm-start-source-config /Users/keanooo7/retrieval-successor-retention/.worktrees/fresh-stream/runs/fresh-stream/B/seed0/result.json --policy rsr --policy-override b_enabled=false --policy-override shadow_enabled=false --policy-override nu=0 --policy-override beta=1 --policy-override gamma=0.9 --out-dir runs/rsr-run1/psiC/seed0` (job record)
  - **fifo** rc `0` (ran, live), run_id `fifo-d128-s48-b16-050c2dd027ecd1c5`, config_hash `050c2dd027ecd1c5`, label `fifo+frozen[eval]+warm[ckpt-003000.pt@0ee3f8a69b50,src_seed=0]`, job pinned_sha `ed6302aae37a2831c9b8567b9f80a093b4c00f70`, steps recorded `3000`
    - `.venv/bin/python -m rsr.train.loop --d 128 --batch 16 --iters 3000 --steps-per-stream 48 --memory-slots 16 --lr 0.001 --seed 0 --beat-every 1 --ckpt-every 40 --device cpu --freeze-transformer --frozen-model-mode eval --srep-norm-reg-weight 0.0 --pool-offset 993000 --pool-length 256 --pool-vocab-documents 64 --warm-start /Users/keanooo7/retrieval-successor-retention/.worktrees/fresh-stream/runs/fresh-stream/B/seed0/ckpt-003000.pt --warm-start-sha256 0ee3f8a69b507d927c631eb85116e1cd9739ee4472c44eb7f145d739a118da60 --warm-start-manifest /Users/keanooo7/rsr-substrate/2026-09-27/MANIFEST.sha256 --warm-start-root /Users/keanooo7/retrieval-successor-retention --warm-start-source-config /Users/keanooo7/retrieval-successor-retention/.worktrees/fresh-stream/runs/fresh-stream/B/seed0/result.json --policy fifo --out-dir runs/rsr-run1/fifo/seed0` (job record)
  - **age** rc `0` (ran, live), run_id `rsr-d128-s48-b16-047c2b89be22593d`, config_hash `047c2b89be22593d`, label `rsr[b=off,beta=1,gamma=0.9,nu=0,psi=age_only,shadow=off]+frozen[eval]+warm[ckpt-003000.pt@0ee3f8a69b50,src_seed=0]`, job pinned_sha `ed6302aae37a2831c9b8567b9f80a093b4c00f70`, steps recorded `3000`
    - `.venv/bin/python -m rsr.train.loop --d 128 --batch 16 --iters 3000 --steps-per-stream 48 --memory-slots 16 --lr 0.001 --seed 0 --beat-every 1 --ckpt-every 40 --device cpu --freeze-transformer --frozen-model-mode eval --srep-norm-reg-weight 0.0 --pool-offset 993000 --pool-length 256 --pool-vocab-documents 64 --warm-start /Users/keanooo7/retrieval-successor-retention/.worktrees/fresh-stream/runs/fresh-stream/B/seed0/ckpt-003000.pt --warm-start-sha256 0ee3f8a69b507d927c631eb85116e1cd9739ee4472c44eb7f145d739a118da60 --warm-start-manifest /Users/keanooo7/rsr-substrate/2026-09-27/MANIFEST.sha256 --warm-start-root /Users/keanooo7/retrieval-successor-retention --warm-start-source-config /Users/keanooo7/retrieval-successor-retention/.worktrees/fresh-stream/runs/fresh-stream/B/seed0/result.json --policy rsr --policy-override b_enabled=false --policy-override shadow_enabled=false --policy-override nu=0 --policy-override beta=1 --policy-override gamma=0.9 --policy-override psi_override=age_only --out-dir runs/rsr-run1/age/seed0` (job record)
- seed 1: launch-git `{'run_sha': 'ed6302aae37a2831c9b8567b9f80a093b4c00f70', 'dirty': False, 'gamma': '0.9', 'wd': '0.01', 'steps': '3000', 'ckpt': '1000', 'o5': 'eval', 'c1_flags': '--frozen-model-mode eval', 'c2_flags': '', 'launched_utc': '2026-10-06T07:12:30Z', 'env': None}`
  - **psiC** rc `0` (ran, live), run_id `rsr-d128-s48-b16-9ccf359832496fee`, config_hash `9ccf359832496fee`, label `rsr[b=off,beta=1,gamma=0.9,nu=0,shadow=off]+frozen[eval]+warm[ckpt-003000.pt@dadd1e08a384,src_seed=1]`, job pinned_sha `ed6302aae37a2831c9b8567b9f80a093b4c00f70`, steps recorded `3000`
    - `.venv/bin/python -m rsr.train.loop --d 128 --batch 16 --iters 3000 --steps-per-stream 48 --memory-slots 16 --lr 0.001 --seed 1 --beat-every 1 --ckpt-every 40 --device cpu --freeze-transformer --frozen-model-mode eval --srep-norm-reg-weight 0.0 --pool-offset 993000 --pool-length 256 --pool-vocab-documents 64 --warm-start /Users/keanooo7/retrieval-successor-retention/.worktrees/fresh-stream/runs/fresh-stream/B/seed1/ckpt-003000.pt --warm-start-sha256 dadd1e08a3849c1211c8479394df4060f91cd2a6e188b97701b783f2068a3da8 --warm-start-manifest /Users/keanooo7/rsr-substrate/2026-09-27/MANIFEST.sha256 --warm-start-root /Users/keanooo7/retrieval-successor-retention --warm-start-source-config /Users/keanooo7/retrieval-successor-retention/.worktrees/fresh-stream/runs/fresh-stream/B/seed1/result.json --policy rsr --policy-override b_enabled=false --policy-override shadow_enabled=false --policy-override nu=0 --policy-override beta=1 --policy-override gamma=0.9 --out-dir runs/rsr-run1/psiC/seed1` (job record)
  - **fifo** rc `0` (ran, live), run_id `fifo-d128-s48-b16-35e967fdcc38001e`, config_hash `35e967fdcc38001e`, label `fifo+frozen[eval]+warm[ckpt-003000.pt@dadd1e08a384,src_seed=1]`, job pinned_sha `ed6302aae37a2831c9b8567b9f80a093b4c00f70`, steps recorded `3000`
    - `.venv/bin/python -m rsr.train.loop --d 128 --batch 16 --iters 3000 --steps-per-stream 48 --memory-slots 16 --lr 0.001 --seed 1 --beat-every 1 --ckpt-every 40 --device cpu --freeze-transformer --frozen-model-mode eval --srep-norm-reg-weight 0.0 --pool-offset 993000 --pool-length 256 --pool-vocab-documents 64 --warm-start /Users/keanooo7/retrieval-successor-retention/.worktrees/fresh-stream/runs/fresh-stream/B/seed1/ckpt-003000.pt --warm-start-sha256 dadd1e08a3849c1211c8479394df4060f91cd2a6e188b97701b783f2068a3da8 --warm-start-manifest /Users/keanooo7/rsr-substrate/2026-09-27/MANIFEST.sha256 --warm-start-root /Users/keanooo7/retrieval-successor-retention --warm-start-source-config /Users/keanooo7/retrieval-successor-retention/.worktrees/fresh-stream/runs/fresh-stream/B/seed1/result.json --policy fifo --out-dir runs/rsr-run1/fifo/seed1` (job record)
  - **age** rc `0` (ran, live), run_id `rsr-d128-s48-b16-e83b3e13d70c6722`, config_hash `e83b3e13d70c6722`, label `rsr[b=off,beta=1,gamma=0.9,nu=0,psi=age_only,shadow=off]+frozen[eval]+warm[ckpt-003000.pt@dadd1e08a384,src_seed=1]`, job pinned_sha `ed6302aae37a2831c9b8567b9f80a093b4c00f70`, steps recorded `3000`
    - `.venv/bin/python -m rsr.train.loop --d 128 --batch 16 --iters 3000 --steps-per-stream 48 --memory-slots 16 --lr 0.001 --seed 1 --beat-every 1 --ckpt-every 40 --device cpu --freeze-transformer --frozen-model-mode eval --srep-norm-reg-weight 0.0 --pool-offset 993000 --pool-length 256 --pool-vocab-documents 64 --warm-start /Users/keanooo7/retrieval-successor-retention/.worktrees/fresh-stream/runs/fresh-stream/B/seed1/ckpt-003000.pt --warm-start-sha256 dadd1e08a3849c1211c8479394df4060f91cd2a6e188b97701b783f2068a3da8 --warm-start-manifest /Users/keanooo7/rsr-substrate/2026-09-27/MANIFEST.sha256 --warm-start-root /Users/keanooo7/retrieval-successor-retention --warm-start-source-config /Users/keanooo7/retrieval-successor-retention/.worktrees/fresh-stream/runs/fresh-stream/B/seed1/result.json --policy rsr --policy-override b_enabled=false --policy-override shadow_enabled=false --policy-override nu=0 --policy-override beta=1 --policy-override gamma=0.9 --policy-override psi_override=age_only --out-dir runs/rsr-run1/age/seed1` (job record)
- seed 2: launch-git `{'run_sha': 'ed6302aae37a2831c9b8567b9f80a093b4c00f70', 'dirty': False, 'gamma': '0.9', 'wd': '0.01', 'steps': '3000', 'ckpt': '1000', 'o5': 'eval', 'c1_flags': '--frozen-model-mode eval', 'c2_flags': '', 'launched_utc': '2026-10-06T07:12:30Z', 'env': None}`
  - **psiC** rc `0` (ran, live), run_id `rsr-d128-s48-b16-79d38829323c91af`, config_hash `79d38829323c91af`, label `rsr[b=off,beta=1,gamma=0.9,nu=0,shadow=off]+frozen[eval]+warm[ckpt-003000.pt@b507ebc573a5,src_seed=2]`, job pinned_sha `ed6302aae37a2831c9b8567b9f80a093b4c00f70`, steps recorded `3000`
    - `.venv/bin/python -m rsr.train.loop --d 128 --batch 16 --iters 3000 --steps-per-stream 48 --memory-slots 16 --lr 0.001 --seed 2 --beat-every 1 --ckpt-every 40 --device cpu --freeze-transformer --frozen-model-mode eval --srep-norm-reg-weight 0.0 --pool-offset 993000 --pool-length 256 --pool-vocab-documents 64 --warm-start /Users/keanooo7/retrieval-successor-retention/.worktrees/fresh-stream/runs/fresh-stream/B/seed2/ckpt-003000.pt --warm-start-sha256 b507ebc573a54316f9d3c20337f49e664bb9c414388af925c5a8e87efd6b63b8 --warm-start-manifest /Users/keanooo7/rsr-substrate/2026-09-27/MANIFEST.sha256 --warm-start-root /Users/keanooo7/retrieval-successor-retention --warm-start-source-config /Users/keanooo7/retrieval-successor-retention/.worktrees/fresh-stream/runs/fresh-stream/B/seed2/result.json --policy rsr --policy-override b_enabled=false --policy-override shadow_enabled=false --policy-override nu=0 --policy-override beta=1 --policy-override gamma=0.9 --out-dir runs/rsr-run1/psiC/seed2` (job record)
  - **fifo** rc `0` (ran, live), run_id `fifo-d128-s48-b16-6a47ccbc60167560`, config_hash `6a47ccbc60167560`, label `fifo+frozen[eval]+warm[ckpt-003000.pt@b507ebc573a5,src_seed=2]`, job pinned_sha `ed6302aae37a2831c9b8567b9f80a093b4c00f70`, steps recorded `3000`
    - `.venv/bin/python -m rsr.train.loop --d 128 --batch 16 --iters 3000 --steps-per-stream 48 --memory-slots 16 --lr 0.001 --seed 2 --beat-every 1 --ckpt-every 40 --device cpu --freeze-transformer --frozen-model-mode eval --srep-norm-reg-weight 0.0 --pool-offset 993000 --pool-length 256 --pool-vocab-documents 64 --warm-start /Users/keanooo7/retrieval-successor-retention/.worktrees/fresh-stream/runs/fresh-stream/B/seed2/ckpt-003000.pt --warm-start-sha256 b507ebc573a54316f9d3c20337f49e664bb9c414388af925c5a8e87efd6b63b8 --warm-start-manifest /Users/keanooo7/rsr-substrate/2026-09-27/MANIFEST.sha256 --warm-start-root /Users/keanooo7/retrieval-successor-retention --warm-start-source-config /Users/keanooo7/retrieval-successor-retention/.worktrees/fresh-stream/runs/fresh-stream/B/seed2/result.json --policy fifo --out-dir runs/rsr-run1/fifo/seed2` (job record)
  - **age** rc `0` (ran, live), run_id `rsr-d128-s48-b16-887bd240c17a35ca`, config_hash `887bd240c17a35ca`, label `rsr[b=off,beta=1,gamma=0.9,nu=0,psi=age_only,shadow=off]+frozen[eval]+warm[ckpt-003000.pt@b507ebc573a5,src_seed=2]`, job pinned_sha `ed6302aae37a2831c9b8567b9f80a093b4c00f70`, steps recorded `3000`
    - `.venv/bin/python -m rsr.train.loop --d 128 --batch 16 --iters 3000 --steps-per-stream 48 --memory-slots 16 --lr 0.001 --seed 2 --beat-every 1 --ckpt-every 40 --device cpu --freeze-transformer --frozen-model-mode eval --srep-norm-reg-weight 0.0 --pool-offset 993000 --pool-length 256 --pool-vocab-documents 64 --warm-start /Users/keanooo7/retrieval-successor-retention/.worktrees/fresh-stream/runs/fresh-stream/B/seed2/ckpt-003000.pt --warm-start-sha256 b507ebc573a54316f9d3c20337f49e664bb9c414388af925c5a8e87efd6b63b8 --warm-start-manifest /Users/keanooo7/rsr-substrate/2026-09-27/MANIFEST.sha256 --warm-start-root /Users/keanooo7/retrieval-successor-retention --warm-start-source-config /Users/keanooo7/retrieval-successor-retention/.worktrees/fresh-stream/runs/fresh-stream/B/seed2/result.json --policy rsr --policy-override b_enabled=false --policy-override shadow_enabled=false --policy-override nu=0 --policy-override beta=1 --policy-override gamma=0.9 --policy-override psi_override=age_only --out-dir runs/rsr-run1/age/seed2` (job record)

**Hardware:** Apple M4 Max, macOS-26.6.2-arm64-arm-64bit, CPU, torch 2.14.0, Python 3.12.13.

The §10 step_s projections were measured with one run-1 process beside the battery; per-step cost with ten concurrent single-thread processes was not measured before the run. The measured step_s and wall-clock are below (A1.6).

## Seed 0 (status `ok`, exit `0`). Everything below is SECONDARY or DIAGNOSTIC.

### Controls (§8 as amended)

- **C0**: ok = `True` -- `{"pool_overlaps": [], "pool_below_K": true, "eval_overlaps": [], "eval_below_K": true, "pool_eval_disjoint": true, "vocab_sha256": "afbaf994e9d1f617fdc2e68579bf5f55622ee17aee056d33d8667d218d245031", "pool_closure": true, "eval_closure": true, "ok": true, "failed": []}`
- **C1**: ok = `True` -- `{"per_arm": {"psiC": {"sha256": "0ee3f8a69b507d927c631eb85116e1cd9739ee4472c44eb7f145d739a118da60", "source_vocab_sha256": "afbaf994e9d1f617fdc2e68579bf5f55622ee17aee056d33d8667d218d245031", "run_vocab_sha256": "afbaf994e9d1f617fdc2e68579bf5f55622ee17aee056d33d8667d218d245031", "vocab_checked": true, "seed_matches": true, "source_seed": 0, "allow_unrecorded_vocabulary": false, "allow_seed_mismatch": false, "ok": true}, "fifo": {"sha256": "0ee3f8a69b507d927c631eb85116e1cd9739ee4472c44eb7f145d739a118da60", "source_vocab_sha256": "afbaf994e9d1f617fdc2e68579bf5f55622ee17aee056d33d8667d218d245031", "run_vocab_sha256": "afbaf994e9d1f617fdc2e68579bf5f55622ee17aee056d33d8667d218d245031", "vocab_chec`
- **C2**: ok = `True` -- `{"steps_compared": 3000, "warm_steps_compared": 16, "a_pool_ids": {"present": true, "required": true, "cross_arm_mismatch_steps": [], "epochs_not_exactly_the_pool": []}, "b_fingerprint": {"cross_arm_mismatch_steps": [], "replay_mismatch_steps": [], "sampling": "per-epoch permutation (each document once per epoch)", "sampling_per_arm": {"psiC": "per-epoch permutation (each document once per epoch)", "fifo": "per-epoch permutation (each document once per epoch)", "age": "per-epoch permutation (each document once per epoch)"}, "sampling_agrees": true}, "c_loss_lm_warm_mismatch_steps": [], "same_step_sets": true, "ok": true}`
- **C3**: ok = `True` -- `{"per_checkpoint": {"psiC": {"n_tensors": 206, "keys_equal": true, "differ": [], "ok": true}, "fifo": {"n_tensors": 206, "keys_equal": true, "differ": [], "ok": true}, "age": {"n_tensors": 206, "keys_equal": true, "differ": [], "ok": true}, "psiC@160": {"n_tensors": 206, "keys_equal": true, "differ": [], "ok": true}, "age@160": {"n_tensors": 206, "keys_equal": true, "differ": [], "ok": true}, "psiC@1000": {"n_tensors": 206, "keys_equal": true, "differ": [], "ok": true}, "age@1000": {"n_tensors": 206, "keys_equal": true, "differ": [], "ok": true}, "psiC@2000": {"n_tensors": 206, "keys_equal": true, "differ": [], "ok": true}, "age@2000": {"n_tensors": 206, "keys_equal": true, "differ": [], "ok`
- **C4**: ok = `True` -- `{"R_quantity": 0.8597733527421951, "reference": 0.8597733527421951, "abs_diff": 0.0, "tolerance": 1e-06, "seconds": 25.886796000006143, "torch_threads": 1, "ok": true}`
- **C5**: ok = `True` -- `{"per_arm": {"psiC": {"band": "live", "ratio": 50.306607642343124, "exit_code": 0}, "fifo": {"band": "live", "ratio": 50.306607642343124, "exit_code": 0}, "age": {"band": "live", "ratio": 50.306607642343124, "exit_code": 0}}, "ok": true, "inert": [], "ratio_note": "the ratio is reported, never a control (A2.3)"}`
- **C6**: ok = `True` -- `{"per_arm": {"psiC": {"steps_requested": 3000, "steps_done": 3000, "status": "completed", "n_beats": 3000, "ok": true}, "fifo": {"steps_requested": 3000, "steps_done": 3000, "status": "completed", "n_beats": 3000, "ok": true}, "age": {"steps_requested": 3000, "steps_done": 3000, "status": "completed", "n_beats": 3000, "ok": true}}, "ok": true}`
- **C7**: ok = `True` -- `{"per_pass": {"psiC": {"n_documents": 2048, "mismatched_docs": [], "n_mismatched": 0}, "fifo": {"n_documents": 2048, "mismatched_docs": [], "n_mismatched": 0}, "age": {"n_documents": 2048, "mismatched_docs": [], "n_mismatched": 0}, "oracle": {"n_documents": 2048, "mismatched_docs": [], "n_mismatched": 0}, "lru": {"n_documents": 2048, "mismatched_docs": [], "n_mismatched": 0}, "random0": {"n_documents": 2048, "mismatched_docs": [], "n_mismatched": 0}, "random1": {"n_documents": 2048, "mismatched_docs": [], "n_mismatched": 0}, "random2": {"n_documents": 2048, "mismatched_docs": [], "n_mismatched": 0}, "random3": {"n_documents": 2048, "mismatched_docs": [], "n_mismatched": 0}, "random4": {"n_do`
- **C8**: ok = `True` -- `{"per_arm": {"psiC": {"n_steps": 3000, "nonfinite_steps": [], "ok": true}, "age": {"n_steps": 3000, "nonfinite_steps": [], "ok": true}}, "ok": true}`
- **C10**: ok = `True` -- `{"derived_ckpt": 1000, "derived_every": 40, "registered": {"file": "/Users/keanooo7/retrieval-successor-retention/.worktrees/rsr-run1-run/experiments/rsr-run1/PREREG.md", "heading": "### A4.1 Owner values (fills A2.8's slots)", "values": {"gamma": 0.9, "phi_weight_decay": 0.01, "iters": 3000, "o5": "eval"}, "raw": {"O1": "0.9", "O4": "0.01", "R-1": "3000", "O5": "eval"}, "problems": [], "status": "filled"}, "launch": {"run_sha": "ed6302aae37a2831c9b8567b9f80a093b4c00f70", "dirty": false, "gamma": "0.9", "wd": "0.01", "steps": "3000", "ckpt": "1000", "ckpt_every": "40", "ckpt_every_rule": "gcd(160, CKPT) if STEPS>=160 and CKPT>0 else CKPT (A4.2)", "diag_step": "160", "o5": "eval", "c1_flags":`
- **C9 / C9b** (scoped to the oracle numbers): ok = `True` -- `{'documents': [995000, 995016], 'C9': {'gammas': [0.5, 0.9], 'differing_docs': [], 'ok': True}, 'C9b': {'gamma': 0.9, 'differing_docs': [], 'ok': True}, 'ok': True, 'scope': "a failure marks oracle regret, headroom share and the oracle's own accuracy NOT READ; exit and status unchanged (A3.6)"}`

### What ran

- psiC: steps_requested `3000`, steps_done `3000`, status `completed`
- fifo: steps_requested `3000`, steps_done `3000`, status `completed`
- age: steps_requested `3000`, steps_done `3000`, status `completed`
- psiC: T_warm `16.0` steps = `1.0` epoch; total epochs `187.5`; T_warm / total epochs `0.005333333333333333` (31(b)); sampling `per-epoch permutation (each document once per epoch)`
- age: T_warm `16.0` steps = `1.0` epoch; total epochs `187.5`; T_warm / total epochs `0.005333333333333333` (31(b)); sampling `per-epoch permutation (each document once per epoch)`
- frozen model mode during training: `{'psiC': 'eval (dropout off)', 'fifo': 'eval (dropout off)', 'age': 'eval (dropout off)'}`
- final checkpoint bytes (T6): `{'psiC': 10066443, 'fifo': 9861915, 'age': 9865491}`

### Held-out accuracy (§7.1; SECONDARY), EVAL n = 2048

> This interval resamples EVAL documents only. It holds fixed the trained φ (one φ
> initialisation, one data order), the substrate seed and the pool. It is a within-seed interval.
> The three per-seed values show the **combined** variation of substrate seed, φ initialisation,
> data order and EVAL documents, which change together across seeds. They cannot separate
> run-to-run variation from substrate-seed variation.

| bucket | n | psiC | fifo | age | lru | oracle | random |
|---|---|---|---|---|---|---|---|
| all | 37853 | +0.5697 [+0.5652, +0.5741] | +0.7750 [+0.7705, +0.7793] | +0.7750 [+0.7705, +0.7793] | +0.7857 [+0.7815, +0.7899] | +0.9224 [+0.9198, +0.9250] | +0.7498 [+0.7463, +0.7532] |
| gap_eq_1 | 8054 | +0.9927 [+0.9907, +0.9945] | +0.9918 [+0.9898, +0.9937] | +0.9918 [+0.9898, +0.9937] | +0.9914 [+0.9894, +0.9933] | +0.9917 [+0.9897, +0.9935] | +0.9918 [+0.9898, +0.9935] |
| gap_ge_2 | 29799 | +0.4554 [+0.4501, +0.4604] | +0.7164 [+0.7110, +0.7216] | +0.7164 [+0.7110, +0.7216] | +0.7301 [+0.7248, +0.7350] | +0.9037 [+0.9004, +0.9069] | +0.6844 [+0.6802, +0.6884] |
| gap_2_to_M | 22542 | +0.5511 [+0.5449, +0.5570] | +0.9241 [+0.9206, +0.9275] | +0.9241 [+0.9206, +0.9275] | +0.8961 [+0.8920, +0.9000] | +0.9210 [+0.9174, +0.9244] | +0.8051 [+0.8012, +0.8091] |
| gap_gt_M | 7257 | +0.1579 [+0.1495, +0.1663] | +0.0712 [+0.0652, +0.0769] | +0.0712 [+0.0652, +0.0769] | +0.2144 [+0.2049, +0.2236] | +0.8498 [+0.8423, +0.8578] | +0.3095 [+0.3035, +0.3156] |

`LRUPolicy` falls back to the write step for a never-attended slot, so it degrades toward FIFO by design; its distance from FIFO is `acc.*.lru_minus_fifo` (A1.1).

### Contrasts (SECONDARY; the headline is §1)

- `R.age`: +0.8607 [+0.8559, +0.8655] CI above 0
- `R.fifo`: +0.8607 [+0.8559, +0.8655] CI above 0
- `R.psiC`: +0.4878 [+0.4807, +0.4945] CI above 0
- `R_own_floor.age`: +0.8616 [+0.8569, +0.8661] CI above 0
- `R_own_floor.fifo`: +0.8607 [+0.8559, +0.8655] CI above 0
- `R_own_floor.psiC`: +0.4879 [+0.4811, +0.4949] CI above 0
- `acc.all.age_minus_fifo`: +0.0000 [+0.0000, +0.0000] CI includes 0 **sd exactly 0: broken, not clean**
- `acc.all.age_minus_lru`: -0.0107 [-0.0130, -0.0083] CI below 0
- `acc.all.lru_minus_fifo`: +0.0107 [+0.0083, +0.0130] CI above 0
- `acc.all.psiC_minus_age`: -0.2053 [-0.2096, -0.2007] CI below 0
- `acc.all.psiC_minus_fifo` **(HEADLINE)**: -0.2053 [-0.2096, -0.2007] CI below 0
- `acc.all.psiC_minus_lru`: -0.2160 [-0.2205, -0.2115] CI below 0
- `acc.gap_2_to_M.age_minus_fifo`: +0.0000 [+0.0000, +0.0000] CI includes 0 **sd exactly 0: broken, not clean**
- `acc.gap_2_to_M.age_minus_lru`: +0.0279 [+0.0256, +0.0303] CI above 0
- `acc.gap_2_to_M.lru_minus_fifo`: -0.0279 [-0.0303, -0.0256] CI below 0
- `acc.gap_2_to_M.psiC_minus_age`: -0.3729 [-0.3785, -0.3672] CI below 0
- `acc.gap_2_to_M.psiC_minus_fifo`: -0.3729 [-0.3785, -0.3672] CI below 0
- `acc.gap_2_to_M.psiC_minus_lru`: -0.3450 [-0.3507, -0.3390] CI below 0
- `acc.gap_eq_1.age_minus_fifo`: +0.0000 [+0.0000, +0.0000] CI includes 0 **sd exactly 0: broken, not clean**
- `acc.gap_eq_1.age_minus_lru`: +0.0004 [-0.0002, +0.0010] CI includes 0
- `acc.gap_eq_1.lru_minus_fifo`: -0.0004 [-0.0010, +0.0002] CI includes 0
- `acc.gap_eq_1.psiC_minus_age`: +0.0009 [+0.0002, +0.0016] CI above 0
- `acc.gap_eq_1.psiC_minus_fifo`: +0.0009 [+0.0002, +0.0016] CI above 0
- `acc.gap_eq_1.psiC_minus_lru`: +0.0012 [+0.0005, +0.0021] CI above 0
- `acc.gap_ge_2.age_minus_fifo`: +0.0000 [+0.0000, +0.0000] CI includes 0 **sd exactly 0: broken, not clean**
- `acc.gap_ge_2.age_minus_lru`: -0.0137 [-0.0166, -0.0107] CI below 0
- `acc.gap_ge_2.lru_minus_fifo`: +0.0137 [+0.0107, +0.0166] CI above 0
- `acc.gap_ge_2.psiC_minus_age`: -0.2610 [-0.2664, -0.2554] CI below 0
- `acc.gap_ge_2.psiC_minus_fifo`: -0.2610 [-0.2664, -0.2554] CI below 0
- `acc.gap_ge_2.psiC_minus_lru`: -0.2747 [-0.2803, -0.2691] CI below 0
- `acc.gap_gt_M.age_minus_fifo`: +0.0000 [+0.0000, +0.0000] CI includes 0 **sd exactly 0: broken, not clean**
- `acc.gap_gt_M.age_minus_lru`: -0.1432 [-0.1525, -0.1336] CI below 0
- `acc.gap_gt_M.lru_minus_fifo`: +0.1432 [+0.1336, +0.1525] CI above 0
- `acc.gap_gt_M.psiC_minus_age`: +0.0867 [+0.0778, +0.0958] CI above 0
- `acc.gap_gt_M.psiC_minus_fifo`: +0.0867 [+0.0778, +0.0958] CI above 0
- `acc.gap_gt_M.psiC_minus_lru`: -0.0565 [-0.0681, -0.0458] CI below 0
- `floor_difference.sz_age_minus_sz_fifo`: -0.0008 [-0.0040, +0.0024] CI includes 0
- `floor_difference.sz_psiC_minus_sz_fifo`: -0.0001 [-0.0033, +0.0033] CI includes 0
- `nll_answer.age_minus_fifo`: +0.0000 [+0.0000, +0.0000] CI includes 0 **sd exactly 0: broken, not clean**
- `nll_answer.age_minus_lru`: -0.0260 [-0.0342, -0.0180] CI below 0
- `nll_answer.lru_minus_fifo`: +0.0260 [+0.0180, +0.0342] CI above 0
- `nll_answer.psiC_minus_age`: +0.6242 [+0.6089, +0.6392] CI above 0
- `nll_answer.psiC_minus_fifo`: +0.6242 [+0.6089, +0.6392] CI above 0
- `nll_answer.psiC_minus_lru`: +0.5982 [+0.5819, +0.6140] CI above 0
- `nll_real.age_minus_fifo`: +0.0000 [+0.0000, +0.0000] CI includes 0 **sd exactly 0: broken, not clean**
- `nll_real.age_minus_lru`: -0.0272 [-0.0285, -0.0259] CI below 0
- `nll_real.lru_minus_fifo`: +0.0272 [+0.0259, +0.0285] CI above 0
- `nll_real.psiC_minus_age`: +0.1688 [+0.1662, +0.1713] CI above 0
- `nll_real.psiC_minus_fifo`: +0.1688 [+0.1662, +0.1713] CI above 0
- `nll_real.psiC_minus_lru`: +0.1416 [+0.1388, +0.1444] CI above 0
- `oracle_regret.all.age`: +0.1474 [+0.1438, +0.1510] CI above 0
- `oracle_regret.all.fifo`: +0.1474 [+0.1438, +0.1510] CI above 0
- `oracle_regret.all.psiC`: +0.3527 [+0.3485, +0.3569] CI above 0
- `oracle_regret.gap_2_to_M.age`: -0.0030 [-0.0041, -0.0019] CI below 0
- `oracle_regret.gap_2_to_M.fifo`: -0.0030 [-0.0041, -0.0019] CI below 0
- `oracle_regret.gap_2_to_M.psiC`: +0.3699 [+0.3641, +0.3755] CI above 0
- `oracle_regret.gap_eq_1.age`: -0.0001 [-0.0005, +0.0002] CI includes 0
- `oracle_regret.gap_eq_1.fifo`: -0.0001 [-0.0005, +0.0002] CI includes 0
- `oracle_regret.gap_eq_1.psiC`: -0.0010 [-0.0019, -0.0002] CI below 0
- `oracle_regret.gap_ge_2.age`: +0.1873 [+0.1828, +0.1917] CI above 0
- `oracle_regret.gap_ge_2.fifo`: +0.1873 [+0.1828, +0.1917] CI above 0
- `oracle_regret.gap_ge_2.psiC`: +0.4483 [+0.4432, +0.4531] CI above 0
- `oracle_regret.gap_gt_M.age`: +0.7786 [+0.7692, +0.7876] CI above 0
- `oracle_regret.gap_gt_M.fifo`: +0.7786 [+0.7692, +0.7876] CI above 0
- `oracle_regret.gap_gt_M.psiC`: +0.6919 [+0.6809, +0.7033] CI above 0
- NLL `real` level (pooled per target): psiC +1.5202; fifo +1.3515; age +1.3515; lru +1.3787
- NLL `answer` level (pooled per target): psiC +1.3992; fifo +0.7750; age +0.7750; lru +0.8010
- headroom share `psiC` (all): -1.3924 [-1.4496, -1.3369], dropped 0
- headroom share `age` (all): +0.0000 [+0.0000, +0.0000], dropped 0
- DIAGNOSTIC snapshot `psiC@160` (A4.2): accuracy +0.5450 [+0.5406, +0.5494]; minus final fifo -0.2300 [-0.2344, -0.2255] CI below 0
- DIAGNOSTIC snapshot `age@160` (A4.2): accuracy +0.7750 [+0.7705, +0.7793]; minus final fifo +0.0000 [+0.0000, +0.0000] CI includes 0 **sd exactly 0: broken, not clean**
- DIAGNOSTIC snapshot `psiC@1000` (A1.8 m-9): accuracy +0.5366 [+0.5323, +0.5411]; minus final fifo -0.2384 [-0.2428, -0.2339] CI below 0
- DIAGNOSTIC snapshot `age@1000` (A1.8 m-9): accuracy +0.7750 [+0.7705, +0.7793]; minus final fifo +0.0000 [+0.0000, +0.0000] CI includes 0 **sd exactly 0: broken, not clean**
- DIAGNOSTIC snapshot `psiC@2000` (A1.8 m-9): accuracy +0.5499 [+0.5453, +0.5545]; minus final fifo -0.2251 [-0.2294, -0.2208] CI below 0
- DIAGNOSTIC snapshot `age@2000` (A1.8 m-9): accuracy +0.7750 [+0.7705, +0.7793]; minus final fifo +0.0000 [+0.0000, +0.0000] CI includes 0 **sd exactly 0: broken, not clean**

### Step-160 phi reading -- DIAGNOSTIC (A4.2): never the headline, never gated

psiC and age read with their step-160 phi (`psiC@160`, `age@160`, floors `sz_psiC@160`, `sz_age@160`); fifo, LRU and sz_fifo are the final reading's, which phi does not touch. Same EVAL documents, batch and bootstrap draw.

- DIAGNOSTIC step 160 accuracy `all`: psiC@160 +0.5450 [+0.5406, +0.5494]; age@160 +0.7750 [+0.7705, +0.7793]; sz_psiC@160 +0.0639 [+0.0614, +0.0664]; sz_age@160 +0.0623 [+0.0598, +0.0645]
- DIAGNOSTIC step 160 accuracy `gap_eq_1`: psiC@160 +0.9927 [+0.9907, +0.9945]; age@160 +0.9918 [+0.9898, +0.9937]; sz_psiC@160 +0.0659 [+0.0609, +0.0717]; sz_age@160 +0.0670 [+0.0615, +0.0725]
- DIAGNOSTIC step 160 accuracy `gap_ge_2`: psiC@160 +0.4240 [+0.4191, +0.4289]; age@160 +0.7164 [+0.7110, +0.7216]; sz_psiC@160 +0.0634 [+0.0606, +0.0662]; sz_age@160 +0.0610 [+0.0583, +0.0636]
- DIAGNOSTIC step 160 accuracy `gap_2_to_M`: psiC@160 +0.5147 [+0.5087, +0.5206]; age@160 +0.9241 [+0.9206, +0.9275]; sz_psiC@160 +0.0643 [+0.0612, +0.0677]; sz_age@160 +0.0617 [+0.0585, +0.0648]
- DIAGNOSTIC step 160 accuracy `gap_gt_M`: psiC@160 +0.1421 [+0.1340, +0.1500]; age@160 +0.0712 [+0.0652, +0.0769]; sz_psiC@160 +0.0605 [+0.0548, +0.0658]; sz_age@160 +0.0587 [+0.0532, +0.0639]
- DIAGNOSTIC step 160 `R.age`: +0.8607 [+0.8559, +0.8655] CI above 0
- DIAGNOSTIC step 160 `R.psiC`: +0.4514 [+0.4450, +0.4580] CI above 0
- DIAGNOSTIC step 160 `R_own_floor.age`: +0.8623 [+0.8576, +0.8670] CI above 0
- DIAGNOSTIC step 160 `R_own_floor.psiC`: +0.4504 [+0.4439, +0.4569] CI above 0
- DIAGNOSTIC step 160 `acc.all.age_minus_fifo`: +0.0000 [+0.0000, +0.0000] CI includes 0 **sd exactly 0: broken, not clean**
- DIAGNOSTIC step 160 `acc.all.psiC_minus_age`: -0.2300 [-0.2344, -0.2255] CI below 0
- DIAGNOSTIC step 160 `acc.all.psiC_minus_fifo`: -0.2300 [-0.2344, -0.2255] CI below 0
- DIAGNOSTIC step 160 `acc.gap_2_to_M.age_minus_fifo`: +0.0000 [+0.0000, +0.0000] CI includes 0 **sd exactly 0: broken, not clean**
- DIAGNOSTIC step 160 `acc.gap_2_to_M.psiC_minus_age`: -0.4093 [-0.4147, -0.4036] CI below 0
- DIAGNOSTIC step 160 `acc.gap_2_to_M.psiC_minus_fifo`: -0.4093 [-0.4147, -0.4036] CI below 0
- DIAGNOSTIC step 160 `acc.gap_eq_1.age_minus_fifo`: +0.0000 [+0.0000, +0.0000] CI includes 0 **sd exactly 0: broken, not clean**
- DIAGNOSTIC step 160 `acc.gap_eq_1.psiC_minus_age`: +0.0009 [+0.0001, +0.0016] CI above 0
- DIAGNOSTIC step 160 `acc.gap_eq_1.psiC_minus_fifo`: +0.0009 [+0.0001, +0.0016] CI above 0
- DIAGNOSTIC step 160 `acc.gap_ge_2.age_minus_fifo`: +0.0000 [+0.0000, +0.0000] CI includes 0 **sd exactly 0: broken, not clean**
- DIAGNOSTIC step 160 `acc.gap_ge_2.psiC_minus_age`: -0.2924 [-0.2980, -0.2868] CI below 0
- DIAGNOSTIC step 160 `acc.gap_ge_2.psiC_minus_fifo`: -0.2924 [-0.2980, -0.2868] CI below 0
- DIAGNOSTIC step 160 `acc.gap_gt_M.age_minus_fifo`: +0.0000 [+0.0000, +0.0000] CI includes 0 **sd exactly 0: broken, not clean**
- DIAGNOSTIC step 160 `acc.gap_gt_M.psiC_minus_age`: +0.0708 [+0.0626, +0.0798] CI above 0
- DIAGNOSTIC step 160 `acc.gap_gt_M.psiC_minus_fifo`: +0.0708 [+0.0626, +0.0798] CI above 0
- DIAGNOSTIC step 160 `floor_difference.sz_age_minus_sz_fifo`: -0.0016 [-0.0048, +0.0013] CI includes 0
- DIAGNOSTIC step 160 `floor_difference.sz_psiC_minus_sz_fifo`: +0.0010 [-0.0023, +0.0042] CI includes 0
- DIAGNOSTIC step 160 `nll_answer.age_minus_fifo`: +0.0000 [+0.0000, +0.0000] CI includes 0 **sd exactly 0: broken, not clean**
- DIAGNOSTIC step 160 `nll_answer.psiC_minus_age`: +0.6840 [+0.6691, +0.6990] CI above 0
- DIAGNOSTIC step 160 `nll_answer.psiC_minus_fifo`: +0.6840 [+0.6691, +0.6990] CI above 0
- DIAGNOSTIC step 160 `nll_real.age_minus_fifo`: +0.0000 [+0.0000, +0.0000] CI includes 0 **sd exactly 0: broken, not clean**
- DIAGNOSTIC step 160 `nll_real.psiC_minus_age`: +0.1646 [+0.1621, +0.1668] CI above 0
- DIAGNOSTIC step 160 `nll_real.psiC_minus_fifo`: +0.1646 [+0.1621, +0.1668] CI above 0
- DIAGNOSTIC held-out evictions `psiC`: n 65536, fraction oldest +0.0497, victim ages `{'1': 24924, '2': 5199, '3': 4045, '4': 2503, '5': 2022, '6': 1771, '7': 1594, '8': 1489, '9': 1477, '10': 1363, '11': 1403, '12': 1269, '13': 1284, '14': 1303, '15': 1262, '16': 1278, '17': 1066, '18': 1024, '19': 1038, '20': 890, '21': 831, '22': 785, '23': 721, '24': 698, '25': 614, '26': 585, '27': 473, '28': 408, '29': 383, '30': 310, '31': 295, '32': 236, '33': 182, '34': 154, '35': 142, '36': 119, '37': 107, '38': 82, '39': 65, '40': 48, '41': 40, '42': 18, '43': 18, '44': 10, '45': 5, '46': 2, '47': 1}`
- DIAGNOSTIC held-out evictions `fifo`: n 65536, fraction oldest +1.0000, victim ages `{'16': 65536}`
- DIAGNOSTIC held-out evictions `age`: n 65536, fraction oldest +1.0000, victim ages `{'16': 65536}`
- DIAGNOSTIC held-out evictions `oracle`: n 65536, fraction oldest +0.3783, victim ages `{'7': 4, '8': 33, '9': 116, '10': 366, '11': 1019, '12': 2598, '13': 5836, '14': 11323, '15': 16741, '16': 20243, '17': 445, '18': 437, '19': 397, '20': 421, '21': 401, '22': 375, '23': 399, '24': 401, '25': 350, '26': 341, '27': 306, '28': 319, '29': 276, '30': 279, '31': 272, '32': 252, '33': 271, '34': 197, '35': 234, '36': 220, '37': 184, '38': 178, '39': 147, '40': 155}`
- DIAGNOSTIC held-out evictions `lru`: n 65536, fraction oldest +0.0215, victim ages `{'9': 3, '10': 262, '11': 3112, '12': 12235, '13': 18364, '14': 11557, '15': 5904, '16': 3515, '17': 2453, '18': 1744, '19': 1270, '20': 859, '21': 603, '22': 482, '23': 412, '24': 438, '25': 403, '26': 375, '27': 285, '28': 236, '29': 189, '30': 158, '31': 124, '32': 79, '33': 79, '34': 65, '35': 64, '36': 49, '37': 46, '38': 36, '39': 29, '40': 26, '41': 22, '42': 22, '43': 15, '44': 9, '45': 5, '46': 5, '47': 2}`

### Training time, on the pool (§7.3: SECONDARY; in-sample for phi)

- delta loss_lm `psiC` - fifo, post-warm: mean +0.1788, sd +0.0158; 500-step blocks [([16, 500], '+0.1696'), ([500, 1000], '+0.1777'), ([1000, 1500], '+0.1860'), ([1500, 2000], '+0.1857'), ([2000, 2500], '+0.1791'), ([2500, 3000], '+0.1743')]
- delta loss_lm `age` - fifo, post-warm: mean +0.0000, sd +0.0000 **sd exactly 0: broken, not clean**; 500-step blocks [([16, 500], '+0.0000'), ([500, 1000], '+0.0000'), ([1000, 1500], '+0.0000'), ([1500, 2000], '+0.0000'), ([2000, 2500], '+0.0000'), ([2500, 3000], '+0.0000')]
- DIAGNOSTIC `psiC`: L_MC warm mean +1.6347, post-warm mean +1.3096; corr_psi_age post-warm median +0.3172 (None on 0 steps); phi grad norm median +0.0086; eval_forward_s total +3147.1; fill fraction (regression rows) +0.8404
- DIAGNOSTIC `age`: L_MC warm mean +1.5789, post-warm mean +0.2375; corr_psi_age post-warm median -0.9857 (None on 1 steps); phi grad norm median +0.0129; eval_forward_s total +3149.0; fill fraction (regression rows) +0.8404
- DIAGNOSTIC `psiC` attribution (mean per-step fraction): warm `{'fifo_warmup': 1.0}`, post-warm `{'psi': 1.0}`
- DIAGNOSTIC `age` attribution (mean per-step fraction): warm `{'fifo_warmup': 1.0}`, post-warm `{'age_only': 1.0}`
- run-level split histograms (train()'s phase_summary) equal the per-step sums: `{'psiC': True, 'age': True}`

### §7.1 diagnostics (§7.4: DIAGNOSTIC; never a gate; warm / scored split)

- `psiC` warm: steps 16, evictions 8192, fraction oldest +1.0000, fraction_shifting +0.0000, mean_shift +0.0000; raw Spearman rho(score, age) median -0.0763; partial rho (a diagnostic, never a gate) median -0.0695 IQR [-0.0788, -0.0575], None on 0 steps
  - victim ages `{'16': 8192}`; displacement `{'0': 8192}`
- `psiC` scored: steps 2984, evictions 1527808, fraction oldest +0.0718, fraction_shifting +0.9282, mean_shift +9.5862; raw Spearman rho(score, age) median +0.2008; partial rho (a diagnostic, never a gate) median -0.2121 IQR [-0.2913, -0.1806], None on 0 steps
  - victim ages `{'1': 480009, '2': 91336, '3': 79365, '4': 57821, '5': 51600, '6': 43133, '7': 38779, '8': 36742, '9': 35479, '10': 33379, '11': 33599, '12': 32581, '13': 33787, '14': 34946, '15': 34837, '16': 30883, '17': 32379, '18': 31130, '19': 29409, '20': 26473, '21': 27434, '22': 26156, '23': 21652, '24': 22560, '25': 21041, '26': 18371, '27': 16261, '28': 15356, '29': 14412, '30': 12991, '31': 11634, '32': 10391, '33': 7777, '34': 6585, '35': 5928, '36': 4762, '37': 3468, '38': 3415, '39': 3221, '40': 2126, '41': 1634, '42': 1247, '43': 857, '44': 368, '45': 399, '46': 95}`; displacement `{'0': 109765, '1': 81277, '2': 74774, '3': 65447, '4': 63227, '5': 57650, '6': 50247, '7': 46035, '8': 54930, '9': 47427, '10': 49945, '11': 55058, '12': 62533, '13': 80818, '14': 148666, '15': 480009}`
- `psiC` first_scored_block: steps 500, evictions 256000, fraction oldest +0.1183, fraction_shifting +0.8817, mean_shift +7.6271; raw Spearman rho(score, age) median -0.1933; partial rho (a diagnostic, never a gate) median -0.3772 IQR [-0.3956, -0.3642], None on 0 steps
  - victim ages `{'1': 52275, '2': 12331, '3': 10171, '4': 8571, '5': 7657, '6': 6829, '7': 6502, '8': 6034, '9': 6549, '10': 6305, '11': 5964, '12': 6009, '13': 6287, '14': 6775, '15': 6803, '16': 5987, '17': 6318, '18': 6217, '19': 5930, '20': 5831, '21': 6271, '22': 5873, '23': 5497, '24': 5415, '25': 4948, '26': 4711, '27': 4526, '28': 4300, '29': 4048, '30': 3791, '31': 3521, '32': 3435, '33': 2211, '34': 2247, '35': 1979, '36': 1558, '37': 1355, '38': 1191, '39': 956, '40': 752, '41': 692, '42': 578, '43': 363, '44': 217, '45': 141, '46': 79}`; displacement `{'0': 30280, '1': 22041, '2': 19110, '3': 15168, '4': 13469, '5': 10795, '6': 10001, '7': 9857, '8': 10182, '9': 8630, '10': 8527, '11': 9375, '12': 9555, '13': 10667, '14': 16068, '15': 52275}`
  - uniform random over the live slots would give `{'0': 16000.0, '1': 16000.0, '2': 16000.0, '3': 16000.0, '4': 16000.0, '5': 16000.0, '6': 16000.0, '7': 16000.0, '8': 16000.0, '9': 16000.0, '10': 16000.0, '11': 16000.0, '12': 16000.0, '13': 16000.0, '14': 16000.0, '15': 16000.0}` (A1.8 m-2)
- `age` warm: steps 16, evictions 8192, fraction oldest +1.0000, fraction_shifting +0.0000, mean_shift +0.0000; raw Spearman rho(score, age) median -0.9765; partial rho (a diagnostic, never a gate) median -0.9712 IQR [-0.9784, -0.9678], None on 1 steps
  - victim ages `{'16': 8192}`; displacement `{'0': 8192}`
- `age` scored: steps 2984, evictions 1527808, fraction oldest +1.0000, fraction_shifting +0.0000, mean_shift +0.0000; raw Spearman rho(score, age) median -1.0000; partial rho (a diagnostic, never a gate) median -1.0000 IQR [-1.0000, -1.0000], None on 0 steps
  - victim ages `{'16': 1527808}`; displacement `{'0': 1527808}`
- `age` first_scored_block: steps 500, evictions 256000, fraction oldest +1.0000, fraction_shifting +0.0000, mean_shift +0.0000; raw Spearman rho(score, age) median -1.0000; partial rho (a diagnostic, never a gate) median -1.0000 IQR [-1.0000, -1.0000], None on 0 steps
  - victim ages `{'16': 256000}`; displacement `{'0': 256000}`
  - uniform random over the live slots would give `{'0': 16000.0, '1': 16000.0, '2': 16000.0, '3': 16000.0, '4': 16000.0, '5': 16000.0, '6': 16000.0, '7': 16000.0, '8': 16000.0, '9': 16000.0, '10': 16000.0, '11': 16000.0, '12': 16000.0, '13': 16000.0, '14': 16000.0, '15': 16000.0}` (A1.8 m-2)
- Partial rho partials out the slot gestalts only (linear), not `c_t`; no verdict is drawn from any comparison with the spec's E2 value. Leading-edge statistic: not computed (the spec defines a baseline, not a statistic; `leading_edge.py` is a stub). Attention-share Gini: not available.

### Step-160 recency statistics (§7.4) -- DIAGNOSTIC (A4.2): never the headline, never gated

- DIAGNOSTIC step 160 `psiC` warm: steps 16, evictions 8192, fraction oldest +1.0000, mean_shift +0.0000; raw Spearman rho(score, age) median -0.0763
- DIAGNOSTIC step 160 `psiC` scored: steps 144, evictions 73728, fraction oldest +0.1290, mean_shift +7.4427; raw Spearman rho(score, age) median -0.2085
- DIAGNOSTIC step 160 `psiC` first_scored_block: steps 144, evictions 73728, fraction oldest +0.1290, mean_shift +7.4427; raw Spearman rho(score, age) median -0.2085
- DIAGNOSTIC step 160 `age` warm: steps 16, evictions 8192, fraction oldest +1.0000, mean_shift +0.0000; raw Spearman rho(score, age) median -0.9765
- DIAGNOSTIC step 160 `age` scored: steps 144, evictions 73728, fraction oldest +1.0000, mean_shift +0.0000; raw Spearman rho(score, age) median -1.0000
- DIAGNOSTIC step 160 `age` first_scored_block: steps 144, evictions 73728, fraction oldest +1.0000, mean_shift +0.0000; raw Spearman rho(score, age) median -1.0000

### §7.5 reporting (SECONDARY)

- fill fraction, held-out in-loop: +0.8229
- The censored-target caveat, verbatim from the PREREG:

  > The shadow buffer is off (ADR-0009 L6(a)). Every regression target `G_i(t)` is cut at the
  > slot's eviction. Before `T_warm` the cut is FIFO's, at age `M`, so the target itself is
  > age-shaped although ψ̂ has no age input. After `T_warm` the cut is the learned policy's own,
  > and self-confirming (§3.4, §7.2). Spec §3.4's description of pre-`T_warm` data (spec :280)
  > holds for the policy, not for the target. The rescaled `r_i` is described as rescaled, never
  > as anything stronger (§3.2.1).

### T5 (held-out cost, measured in this run; DIAGNOSTIC)

The step-160 DIAGNOSTIC passes (psiC@160, age@160, sz_psiC@160, sz_age@160; A4.2) add about 1,090 s per seed. That figure is EXTRAPOLATED from build-time T5 (one batch per pass, x 128 batches; sz_age assumed equal to sz_psiC), not measured; this run's own per-pass timing follows.

- `psiC`: first batch +1.85 s; projected full EVAL +237.3 s; measured +228.6 s over 128 batches
- `fifo`: first batch +0.89 s; projected full EVAL +114.5 s; measured +115.5 s over 128 batches
- `age`: first batch +1.75 s; projected full EVAL +224.3 s; measured +225.7 s over 128 batches
- `oracle`: first batch +0.79 s; projected full EVAL +100.7 s; measured +116.3 s over 128 batches
- `lru`: first batch +0.81 s; projected full EVAL +103.3 s; measured +123.3 s over 128 batches
- `random0`: first batch +0.81 s; projected full EVAL +104.2 s; measured +115.4 s over 128 batches
- `random1`: first batch +0.78 s; projected full EVAL +99.8 s; measured +115.4 s over 128 batches
- `random2`: first batch +0.84 s; projected full EVAL +107.3 s; measured +115.7 s over 128 batches
- `random3`: first batch +0.77 s; projected full EVAL +98.7 s; measured +115.2 s over 128 batches
- `random4`: first batch +0.77 s; projected full EVAL +98.6 s; measured +115.2 s over 128 batches
- `sz_psiC`: first batch +2.27 s; projected full EVAL +290.7 s; measured +304.9 s over 128 batches
- `sz_fifo`: first batch +1.00 s; projected full EVAL +127.6 s; measured +116.0 s over 128 batches
- `sz_age`: first batch +2.35 s; projected full EVAL +301.3 s; measured +304.7 s over 128 batches
- `psiC@160`: first batch +1.75 s; projected full EVAL +224.5 s; measured +227.2 s over 128 batches
- `age@160`: first batch +1.72 s; projected full EVAL +219.6 s; measured +224.6 s over 128 batches
- `psiC@1000`: first batch +1.77 s; projected full EVAL +226.3 s; measured +226.4 s over 128 batches
- `age@1000`: first batch +1.72 s; projected full EVAL +220.4 s; measured +224.3 s over 128 batches
- `psiC@2000`: first batch +1.72 s; projected full EVAL +220.5 s; measured +225.8 s over 128 batches
- `age@2000`: first batch +1.77 s; projected full EVAL +226.0 s; measured +224.4 s over 128 batches
- `sz_psiC@160`: first batch +2.25 s; projected full EVAL +288.3 s; measured +305.4 s over 128 batches
- `sz_age@160`: first batch +2.31 s; projected full EVAL +295.1 s; measured +305.8 s over 128 batches

## Seed 1 (status `ok`, exit `0`). Everything below is SECONDARY or DIAGNOSTIC.

### Controls (§8 as amended)

- **C0**: ok = `True` -- `{"pool_overlaps": [], "pool_below_K": true, "eval_overlaps": [], "eval_below_K": true, "pool_eval_disjoint": true, "vocab_sha256": "afbaf994e9d1f617fdc2e68579bf5f55622ee17aee056d33d8667d218d245031", "pool_closure": true, "eval_closure": true, "ok": true, "failed": []}`
- **C1**: ok = `True` -- `{"per_arm": {"psiC": {"sha256": "dadd1e08a3849c1211c8479394df4060f91cd2a6e188b97701b783f2068a3da8", "source_vocab_sha256": "afbaf994e9d1f617fdc2e68579bf5f55622ee17aee056d33d8667d218d245031", "run_vocab_sha256": "afbaf994e9d1f617fdc2e68579bf5f55622ee17aee056d33d8667d218d245031", "vocab_checked": true, "seed_matches": true, "source_seed": 1, "allow_unrecorded_vocabulary": false, "allow_seed_mismatch": false, "ok": true}, "fifo": {"sha256": "dadd1e08a3849c1211c8479394df4060f91cd2a6e188b97701b783f2068a3da8", "source_vocab_sha256": "afbaf994e9d1f617fdc2e68579bf5f55622ee17aee056d33d8667d218d245031", "run_vocab_sha256": "afbaf994e9d1f617fdc2e68579bf5f55622ee17aee056d33d8667d218d245031", "vocab_chec`
- **C2**: ok = `True` -- `{"steps_compared": 3000, "warm_steps_compared": 16, "a_pool_ids": {"present": true, "required": true, "cross_arm_mismatch_steps": [], "epochs_not_exactly_the_pool": []}, "b_fingerprint": {"cross_arm_mismatch_steps": [], "replay_mismatch_steps": [], "sampling": "per-epoch permutation (each document once per epoch)", "sampling_per_arm": {"psiC": "per-epoch permutation (each document once per epoch)", "fifo": "per-epoch permutation (each document once per epoch)", "age": "per-epoch permutation (each document once per epoch)"}, "sampling_agrees": true}, "c_loss_lm_warm_mismatch_steps": [], "same_step_sets": true, "ok": true}`
- **C3**: ok = `True` -- `{"per_checkpoint": {"psiC": {"n_tensors": 206, "keys_equal": true, "differ": [], "ok": true}, "fifo": {"n_tensors": 206, "keys_equal": true, "differ": [], "ok": true}, "age": {"n_tensors": 206, "keys_equal": true, "differ": [], "ok": true}, "psiC@160": {"n_tensors": 206, "keys_equal": true, "differ": [], "ok": true}, "age@160": {"n_tensors": 206, "keys_equal": true, "differ": [], "ok": true}, "psiC@1000": {"n_tensors": 206, "keys_equal": true, "differ": [], "ok": true}, "age@1000": {"n_tensors": 206, "keys_equal": true, "differ": [], "ok": true}, "psiC@2000": {"n_tensors": 206, "keys_equal": true, "differ": [], "ok": true}, "age@2000": {"n_tensors": 206, "keys_equal": true, "differ": [], "ok`
- **C4**: ok = `True` -- `{"R_quantity": 0.8531856164336205, "reference": 0.8531856164336205, "abs_diff": 0.0, "tolerance": 1e-06, "seconds": 21.87109195801895, "torch_threads": 1, "ok": true}`
- **C5**: ok = `True` -- `{"per_arm": {"psiC": {"band": "live", "ratio": 86.69945157035805, "exit_code": 0}, "fifo": {"band": "live", "ratio": 86.69945157035805, "exit_code": 0}, "age": {"band": "live", "ratio": 86.69945157035805, "exit_code": 0}}, "ok": true, "inert": [], "ratio_note": "the ratio is reported, never a control (A2.3)"}`
- **C6**: ok = `True` -- `{"per_arm": {"psiC": {"steps_requested": 3000, "steps_done": 3000, "status": "completed", "n_beats": 3000, "ok": true}, "fifo": {"steps_requested": 3000, "steps_done": 3000, "status": "completed", "n_beats": 3000, "ok": true}, "age": {"steps_requested": 3000, "steps_done": 3000, "status": "completed", "n_beats": 3000, "ok": true}}, "ok": true}`
- **C7**: ok = `True` -- `{"per_pass": {"psiC": {"n_documents": 2048, "mismatched_docs": [], "n_mismatched": 0}, "fifo": {"n_documents": 2048, "mismatched_docs": [], "n_mismatched": 0}, "age": {"n_documents": 2048, "mismatched_docs": [], "n_mismatched": 0}, "oracle": {"n_documents": 2048, "mismatched_docs": [], "n_mismatched": 0}, "lru": {"n_documents": 2048, "mismatched_docs": [], "n_mismatched": 0}, "random0": {"n_documents": 2048, "mismatched_docs": [], "n_mismatched": 0}, "random1": {"n_documents": 2048, "mismatched_docs": [], "n_mismatched": 0}, "random2": {"n_documents": 2048, "mismatched_docs": [], "n_mismatched": 0}, "random3": {"n_documents": 2048, "mismatched_docs": [], "n_mismatched": 0}, "random4": {"n_do`
- **C8**: ok = `True` -- `{"per_arm": {"psiC": {"n_steps": 3000, "nonfinite_steps": [], "ok": true}, "age": {"n_steps": 3000, "nonfinite_steps": [], "ok": true}}, "ok": true}`
- **C10**: ok = `True` -- `{"derived_ckpt": 1000, "derived_every": 40, "registered": {"file": "/Users/keanooo7/retrieval-successor-retention/.worktrees/rsr-run1-run/experiments/rsr-run1/PREREG.md", "heading": "### A4.1 Owner values (fills A2.8's slots)", "values": {"gamma": 0.9, "phi_weight_decay": 0.01, "iters": 3000, "o5": "eval"}, "raw": {"O1": "0.9", "O4": "0.01", "R-1": "3000", "O5": "eval"}, "problems": [], "status": "filled"}, "launch": {"run_sha": "ed6302aae37a2831c9b8567b9f80a093b4c00f70", "dirty": false, "gamma": "0.9", "wd": "0.01", "steps": "3000", "ckpt": "1000", "ckpt_every": "40", "ckpt_every_rule": "gcd(160, CKPT) if STEPS>=160 and CKPT>0 else CKPT (A4.2)", "diag_step": "160", "o5": "eval", "c1_flags":`
- **C9 / C9b** (scoped to the oracle numbers): ok = `True` -- `{'documents': [995000, 995016], 'C9': {'gammas': [0.5, 0.9], 'differing_docs': [], 'ok': True}, 'C9b': {'gamma': 0.9, 'differing_docs': [], 'ok': True}, 'ok': True, 'scope': "a failure marks oracle regret, headroom share and the oracle's own accuracy NOT READ; exit and status unchanged (A3.6)"}`

### What ran

- psiC: steps_requested `3000`, steps_done `3000`, status `completed`
- fifo: steps_requested `3000`, steps_done `3000`, status `completed`
- age: steps_requested `3000`, steps_done `3000`, status `completed`
- psiC: T_warm `16.0` steps = `1.0` epoch; total epochs `187.5`; T_warm / total epochs `0.005333333333333333` (31(b)); sampling `per-epoch permutation (each document once per epoch)`
- age: T_warm `16.0` steps = `1.0` epoch; total epochs `187.5`; T_warm / total epochs `0.005333333333333333` (31(b)); sampling `per-epoch permutation (each document once per epoch)`
- frozen model mode during training: `{'psiC': 'eval (dropout off)', 'fifo': 'eval (dropout off)', 'age': 'eval (dropout off)'}`
- final checkpoint bytes (T6): `{'psiC': 10066443, 'fifo': 9861851, 'age': 9865427}`

### Held-out accuracy (§7.1; SECONDARY), EVAL n = 2048

> This interval resamples EVAL documents only. It holds fixed the trained φ (one φ
> initialisation, one data order), the substrate seed and the pool. It is a within-seed interval.
> The three per-seed values show the **combined** variation of substrate seed, φ initialisation,
> data order and EVAL documents, which change together across seeds. They cannot separate
> run-to-run variation from substrate-seed variation.

| bucket | n | psiC | fifo | age | lru | oracle | random |
|---|---|---|---|---|---|---|---|
| all | 37870 | +0.5386 [+0.5343, +0.5429] | +0.7665 [+0.7623, +0.7707] | +0.7665 [+0.7623, +0.7707] | +0.7754 [+0.7711, +0.7796] | +0.9117 [+0.9087, +0.9144] | +0.7429 [+0.7395, +0.7465] |
| gap_eq_1 | 7980 | +0.9883 [+0.9860, +0.9906] | +0.9858 [+0.9832, +0.9884] | +0.9858 [+0.9832, +0.9884] | +0.9858 [+0.9832, +0.9883] | +0.9858 [+0.9832, +0.9883] | +0.9854 [+0.9828, +0.9878] |
| gap_ge_2 | 29890 | +0.4185 [+0.4138, +0.4231] | +0.7080 [+0.7029, +0.7131] | +0.7080 [+0.7029, +0.7131] | +0.7192 [+0.7142, +0.7243] | +0.8919 [+0.8882, +0.8955] | +0.6781 [+0.6742, +0.6822] |
| gap_2_to_M | 22677 | +0.5137 [+0.5081, +0.5196] | +0.9146 [+0.9110, +0.9184] | +0.9146 [+0.9110, +0.9184] | +0.8874 [+0.8830, +0.8915] | +0.9115 [+0.9077, +0.9151] | +0.7973 [+0.7936, +0.8013] |
| gap_gt_M | 7213 | +0.1194 [+0.1121, +0.1266] | +0.0582 [+0.0528, +0.0635] | +0.0582 [+0.0528, +0.0635] | +0.1904 [+0.1814, +0.1995] | +0.8303 [+0.8216, +0.8387] | +0.3034 [+0.2973, +0.3097] |

`LRUPolicy` falls back to the write step for a never-attended slot, so it degrades toward FIFO by design; its distance from FIFO is `acc.*.lru_minus_fifo` (A1.1).

### Contrasts (SECONDARY; the headline is §1)

- `R.age`: +0.8525 [+0.8477, +0.8571] CI above 0
- `R.fifo`: +0.8525 [+0.8477, +0.8571] CI above 0
- `R.psiC`: +0.4516 [+0.4454, +0.4580] CI above 0
- `R_own_floor.age`: +0.8524 [+0.8477, +0.8571] CI above 0
- `R_own_floor.fifo`: +0.8525 [+0.8477, +0.8571] CI above 0
- `R_own_floor.psiC`: +0.4540 [+0.4475, +0.4606] CI above 0
- `acc.all.age_minus_fifo`: +0.0000 [+0.0000, +0.0000] CI includes 0 **sd exactly 0: broken, not clean**
- `acc.all.age_minus_lru`: -0.0088 [-0.0110, -0.0066] CI below 0
- `acc.all.lru_minus_fifo`: +0.0088 [+0.0066, +0.0110] CI above 0
- `acc.all.psiC_minus_age`: -0.2279 [-0.2323, -0.2236] CI below 0
- `acc.all.psiC_minus_fifo` **(HEADLINE)**: -0.2279 [-0.2323, -0.2236] CI below 0
- `acc.all.psiC_minus_lru`: -0.2368 [-0.2414, -0.2324] CI below 0
- `acc.gap_2_to_M.age_minus_fifo`: +0.0000 [+0.0000, +0.0000] CI includes 0 **sd exactly 0: broken, not clean**
- `acc.gap_2_to_M.age_minus_lru`: +0.0273 [+0.0250, +0.0296] CI above 0
- `acc.gap_2_to_M.lru_minus_fifo`: -0.0273 [-0.0296, -0.0250] CI below 0
- `acc.gap_2_to_M.psiC_minus_age`: -0.4009 [-0.4064, -0.3953] CI below 0
- `acc.gap_2_to_M.psiC_minus_fifo`: -0.4009 [-0.4064, -0.3953] CI below 0
- `acc.gap_2_to_M.psiC_minus_lru`: -0.3737 [-0.3794, -0.3677] CI below 0
- `acc.gap_eq_1.age_minus_fifo`: +0.0000 [+0.0000, +0.0000] CI includes 0 **sd exactly 0: broken, not clean**
- `acc.gap_eq_1.age_minus_lru`: +0.0000 [-0.0004, +0.0004] CI includes 0
- `acc.gap_eq_1.lru_minus_fifo`: +0.0000 [-0.0004, +0.0004] CI includes 0
- `acc.gap_eq_1.psiC_minus_age`: +0.0025 [+0.0014, +0.0038] CI above 0
- `acc.gap_eq_1.psiC_minus_fifo`: +0.0025 [+0.0014, +0.0038] CI above 0
- `acc.gap_eq_1.psiC_minus_lru`: +0.0025 [+0.0015, +0.0038] CI above 0
- `acc.gap_ge_2.age_minus_fifo`: +0.0000 [+0.0000, +0.0000] CI includes 0 **sd exactly 0: broken, not clean**
- `acc.gap_ge_2.age_minus_lru`: -0.0112 [-0.0139, -0.0084] CI below 0
- `acc.gap_ge_2.lru_minus_fifo`: +0.0112 [+0.0084, +0.0139] CI above 0
- `acc.gap_ge_2.psiC_minus_age`: -0.2894 [-0.2948, -0.2840] CI below 0
- `acc.gap_ge_2.psiC_minus_fifo`: -0.2894 [-0.2948, -0.2840] CI below 0
- `acc.gap_ge_2.psiC_minus_lru`: -0.3006 [-0.3061, -0.2950] CI below 0
- `acc.gap_gt_M.age_minus_fifo`: +0.0000 [+0.0000, +0.0000] CI includes 0 **sd exactly 0: broken, not clean**
- `acc.gap_gt_M.age_minus_lru`: -0.1321 [-0.1410, -0.1234] CI below 0
- `acc.gap_gt_M.lru_minus_fifo`: +0.1321 [+0.1234, +0.1410] CI above 0
- `acc.gap_gt_M.psiC_minus_age`: +0.0611 [+0.0525, +0.0698] CI above 0
- `acc.gap_gt_M.psiC_minus_fifo`: +0.0611 [+0.0525, +0.0698] CI above 0
- `acc.gap_gt_M.psiC_minus_lru`: -0.0710 [-0.0824, -0.0600] CI below 0
- `floor_difference.sz_age_minus_sz_fifo`: +0.0000 [-0.0034, +0.0035] CI includes 0
- `floor_difference.sz_psiC_minus_sz_fifo`: -0.0024 [-0.0060, +0.0011] CI includes 0
- `nll_answer.age_minus_fifo`: +0.0000 [+0.0000, +0.0000] CI includes 0 **sd exactly 0: broken, not clean**
- `nll_answer.age_minus_lru`: +0.0180 [+0.0105, +0.0248] CI above 0
- `nll_answer.lru_minus_fifo`: -0.0180 [-0.0248, -0.0105] CI below 0
- `nll_answer.psiC_minus_age`: +0.6526 [+0.6382, +0.6664] CI above 0
- `nll_answer.psiC_minus_fifo`: +0.6526 [+0.6382, +0.6664] CI above 0
- `nll_answer.psiC_minus_lru`: +0.6705 [+0.6556, +0.6850] CI above 0
- `nll_real.age_minus_fifo`: +0.0000 [+0.0000, +0.0000] CI includes 0 **sd exactly 0: broken, not clean**
- `nll_real.age_minus_lru`: -0.0027 [-0.0035, -0.0019] CI below 0
- `nll_real.lru_minus_fifo`: +0.0027 [+0.0019, +0.0035] CI above 0
- `nll_real.psiC_minus_age`: +0.2085 [+0.2061, +0.2109] CI above 0
- `nll_real.psiC_minus_fifo`: +0.2085 [+0.2061, +0.2109] CI above 0
- `nll_real.psiC_minus_lru`: +0.2058 [+0.2033, +0.2083] CI above 0
- `oracle_regret.all.age`: +0.1452 [+0.1416, +0.1489] CI above 0
- `oracle_regret.all.fifo`: +0.1452 [+0.1416, +0.1489] CI above 0
- `oracle_regret.all.psiC`: +0.3731 [+0.3687, +0.3775] CI above 0
- `oracle_regret.gap_2_to_M.age`: -0.0032 [-0.0044, -0.0019] CI below 0
- `oracle_regret.gap_2_to_M.fifo`: -0.0032 [-0.0044, -0.0019] CI below 0
- `oracle_regret.gap_2_to_M.psiC`: +0.3978 [+0.3921, +0.4036] CI above 0
- `oracle_regret.gap_eq_1.age`: +0.0000 [-0.0004, +0.0004] CI includes 0
- `oracle_regret.gap_eq_1.fifo`: +0.0000 [-0.0004, +0.0004] CI includes 0
- `oracle_regret.gap_eq_1.psiC`: -0.0025 [-0.0038, -0.0014] CI below 0
- `oracle_regret.gap_ge_2.age`: +0.1839 [+0.1793, +0.1885] CI above 0
- `oracle_regret.gap_ge_2.fifo`: +0.1839 [+0.1793, +0.1885] CI above 0
- `oracle_regret.gap_ge_2.psiC`: +0.4733 [+0.4684, +0.4784] CI above 0
- `oracle_regret.gap_gt_M.age`: +0.7721 [+0.7618, +0.7816] CI above 0
- `oracle_regret.gap_gt_M.fifo`: +0.7721 [+0.7618, +0.7816] CI above 0
- `oracle_regret.gap_gt_M.psiC`: +0.7109 [+0.7001, +0.7222] CI above 0
- NLL `real` level (pooled per target): psiC +1.5640; fifo +1.3555; age +1.3555; lru +1.3582
- NLL `answer` level (pooled per target): psiC +1.4602; fifo +0.8076; age +0.8076; lru +0.7896
- headroom share `psiC` (all): -1.5701 [-1.6292, -1.5130], dropped 0
- headroom share `age` (all): +0.0000 [+0.0000, +0.0000], dropped 0
- DIAGNOSTIC snapshot `psiC@160` (A4.2): accuracy +0.5569 [+0.5524, +0.5613]; minus final fifo -0.2096 [-0.2138, -0.2053] CI below 0
- DIAGNOSTIC snapshot `age@160` (A4.2): accuracy +0.7665 [+0.7623, +0.7707]; minus final fifo +0.0000 [+0.0000, +0.0000] CI includes 0 **sd exactly 0: broken, not clean**
- DIAGNOSTIC snapshot `psiC@1000` (A1.8 m-9): accuracy +0.5412 [+0.5368, +0.5456]; minus final fifo -0.2253 [-0.2297, -0.2210] CI below 0
- DIAGNOSTIC snapshot `age@1000` (A1.8 m-9): accuracy +0.7665 [+0.7623, +0.7707]; minus final fifo +0.0000 [+0.0000, +0.0000] CI includes 0 **sd exactly 0: broken, not clean**
- DIAGNOSTIC snapshot `psiC@2000` (A1.8 m-9): accuracy +0.5360 [+0.5317, +0.5403]; minus final fifo -0.2305 [-0.2350, -0.2262] CI below 0
- DIAGNOSTIC snapshot `age@2000` (A1.8 m-9): accuracy +0.7665 [+0.7623, +0.7707]; minus final fifo +0.0000 [+0.0000, +0.0000] CI includes 0 **sd exactly 0: broken, not clean**

### Step-160 phi reading -- DIAGNOSTIC (A4.2): never the headline, never gated

psiC and age read with their step-160 phi (`psiC@160`, `age@160`, floors `sz_psiC@160`, `sz_age@160`); fifo, LRU and sz_fifo are the final reading's, which phi does not touch. Same EVAL documents, batch and bootstrap draw.

- DIAGNOSTIC step 160 accuracy `all`: psiC@160 +0.5569 [+0.5524, +0.5613]; age@160 +0.7665 [+0.7623, +0.7707]; sz_psiC@160 +0.0629 [+0.0604, +0.0654]; sz_age@160 +0.0634 [+0.0608, +0.0660]
- DIAGNOSTIC step 160 accuracy `gap_eq_1`: psiC@160 +0.9875 [+0.9849, +0.9898]; age@160 +0.9858 [+0.9832, +0.9884]; sz_psiC@160 +0.0588 [+0.0540, +0.0644]; sz_age@160 +0.0653 [+0.0598, +0.0712]
- DIAGNOSTIC step 160 accuracy `gap_ge_2`: psiC@160 +0.4420 [+0.4371, +0.4468]; age@160 +0.7080 [+0.7029, +0.7131]; sz_psiC@160 +0.0640 [+0.0611, +0.0667]; sz_age@160 +0.0629 [+0.0600, +0.0658]
- DIAGNOSTIC step 160 accuracy `gap_2_to_M`: psiC@160 +0.5351 [+0.5296, +0.5408]; age@160 +0.9146 [+0.9110, +0.9184]; sz_psiC@160 +0.0654 [+0.0620, +0.0685]; sz_age@160 +0.0606 [+0.0575, +0.0638]
- DIAGNOSTIC step 160 accuracy `gap_gt_M`: psiC@160 +0.1493 [+0.1414, +0.1570]; age@160 +0.0582 [+0.0528, +0.0635]; sz_psiC@160 +0.0598 [+0.0546, +0.0654]; sz_age@160 +0.0700 [+0.0642, +0.0760]
- DIAGNOSTIC step 160 `R.age`: +0.8525 [+0.8477, +0.8571] CI above 0
- DIAGNOSTIC step 160 `R.psiC`: +0.4729 [+0.4666, +0.4792] CI above 0
- DIAGNOSTIC step 160 `R_own_floor.age`: +0.8540 [+0.8492, +0.8589] CI above 0
- DIAGNOSTIC step 160 `R_own_floor.psiC`: +0.4697 [+0.4629, +0.4763] CI above 0
- DIAGNOSTIC step 160 `acc.all.age_minus_fifo`: +0.0000 [+0.0000, +0.0000] CI includes 0 **sd exactly 0: broken, not clean**
- DIAGNOSTIC step 160 `acc.all.psiC_minus_age`: -0.2096 [-0.2138, -0.2053] CI below 0
- DIAGNOSTIC step 160 `acc.all.psiC_minus_fifo`: -0.2096 [-0.2138, -0.2053] CI below 0
- DIAGNOSTIC step 160 `acc.gap_2_to_M.age_minus_fifo`: +0.0000 [+0.0000, +0.0000] CI includes 0 **sd exactly 0: broken, not clean**
- DIAGNOSTIC step 160 `acc.gap_2_to_M.psiC_minus_age`: -0.3795 [-0.3851, -0.3740] CI below 0
- DIAGNOSTIC step 160 `acc.gap_2_to_M.psiC_minus_fifo`: -0.3795 [-0.3851, -0.3740] CI below 0
- DIAGNOSTIC step 160 `acc.gap_eq_1.age_minus_fifo`: +0.0000 [+0.0000, +0.0000] CI includes 0 **sd exactly 0: broken, not clean**
- DIAGNOSTIC step 160 `acc.gap_eq_1.psiC_minus_age`: +0.0016 [+0.0006, +0.0028] CI above 0
- DIAGNOSTIC step 160 `acc.gap_eq_1.psiC_minus_fifo`: +0.0016 [+0.0006, +0.0028] CI above 0
- DIAGNOSTIC step 160 `acc.gap_ge_2.age_minus_fifo`: +0.0000 [+0.0000, +0.0000] CI includes 0 **sd exactly 0: broken, not clean**
- DIAGNOSTIC step 160 `acc.gap_ge_2.psiC_minus_age`: -0.2660 [-0.2711, -0.2606] CI below 0
- DIAGNOSTIC step 160 `acc.gap_ge_2.psiC_minus_fifo`: -0.2660 [-0.2711, -0.2606] CI below 0
- DIAGNOSTIC step 160 `acc.gap_gt_M.age_minus_fifo`: +0.0000 [+0.0000, +0.0000] CI includes 0 **sd exactly 0: broken, not clean**
- DIAGNOSTIC step 160 `acc.gap_gt_M.psiC_minus_age`: +0.0911 [+0.0820, +0.1000] CI above 0
- DIAGNOSTIC step 160 `acc.gap_gt_M.psiC_minus_fifo`: +0.0911 [+0.0820, +0.1000] CI above 0
- DIAGNOSTIC step 160 `floor_difference.sz_age_minus_sz_fifo`: -0.0015 [-0.0051, +0.0019] CI includes 0
- DIAGNOSTIC step 160 `floor_difference.sz_psiC_minus_sz_fifo`: +0.0032 [-0.0004, +0.0068] CI includes 0
- DIAGNOSTIC step 160 `nll_answer.age_minus_fifo`: +0.0000 [+0.0000, +0.0000] CI includes 0 **sd exactly 0: broken, not clean**
- DIAGNOSTIC step 160 `nll_answer.psiC_minus_age`: +0.5954 [+0.5810, +0.6097] CI above 0
- DIAGNOSTIC step 160 `nll_answer.psiC_minus_fifo`: +0.5954 [+0.5810, +0.6097] CI above 0
- DIAGNOSTIC step 160 `nll_real.age_minus_fifo`: +0.0000 [+0.0000, +0.0000] CI includes 0 **sd exactly 0: broken, not clean**
- DIAGNOSTIC step 160 `nll_real.psiC_minus_age`: +0.1566 [+0.1543, +0.1589] CI above 0
- DIAGNOSTIC step 160 `nll_real.psiC_minus_fifo`: +0.1566 [+0.1543, +0.1589] CI above 0
- DIAGNOSTIC held-out evictions `psiC`: n 65536, fraction oldest +0.0715, victim ages `{'1': 20954, '2': 3809, '3': 3213, '4': 2254, '5': 1893, '6': 1757, '7': 1607, '8': 1551, '9': 1545, '10': 1644, '11': 1472, '12': 1527, '13': 1534, '14': 1596, '15': 1626, '16': 1728, '17': 1578, '18': 1524, '19': 1413, '20': 1321, '21': 1216, '22': 1107, '23': 1009, '24': 938, '25': 829, '26': 745, '27': 628, '28': 576, '29': 466, '30': 405, '31': 338, '32': 267, '33': 209, '34': 203, '35': 181, '36': 167, '37': 147, '38': 127, '39': 116, '40': 85, '41': 77, '42': 60, '43': 40, '44': 28, '45': 16, '46': 10}`
- DIAGNOSTIC held-out evictions `fifo`: n 65536, fraction oldest +1.0000, victim ages `{'16': 65536}`
- DIAGNOSTIC held-out evictions `age`: n 65536, fraction oldest +1.0000, victim ages `{'16': 65536}`
- DIAGNOSTIC held-out evictions `oracle`: n 65536, fraction oldest +0.3796, victim ages `{'5': 1, '6': 11, '7': 4, '8': 26, '9': 77, '10': 293, '11': 940, '12': 2433, '13': 5957, '14': 10995, '15': 17248, '16': 20338, '17': 487, '18': 445, '19': 422, '20': 421, '21': 392, '22': 414, '23': 358, '24': 337, '25': 379, '26': 328, '27': 339, '28': 300, '29': 287, '30': 278, '31': 250, '32': 241, '33': 229, '34': 199, '35': 235, '36': 202, '37': 187, '38': 179, '39': 160, '40': 144}`
- DIAGNOSTIC held-out evictions `lru`: n 65536, fraction oldest +0.1084, victim ages `{'9': 9, '10': 247, '11': 2689, '12': 10023, '13': 16273, '14': 12718, '15': 6859, '16': 4574, '17': 2675, '18': 1840, '19': 1252, '20': 1035, '21': 765, '22': 587, '23': 519, '24': 469, '25': 423, '26': 392, '27': 351, '28': 320, '29': 230, '30': 215, '31': 167, '32': 149, '33': 114, '34': 108, '35': 90, '36': 86, '37': 72, '38': 58, '39': 49, '40': 50, '41': 36, '42': 27, '43': 24, '44': 16, '45': 11, '46': 12, '47': 2}`

### Training time, on the pool (§7.3: SECONDARY; in-sample for phi)

- delta loss_lm `psiC` - fifo, post-warm: mean +0.1898, sd +0.0234; 500-step blocks [([16, 500], '+0.1565'), ([500, 1000], '+0.1729'), ([1000, 1500], '+0.1969'), ([1500, 2000], '+0.2016'), ([2000, 2500], '+0.2042'), ([2500, 3000], '+0.2058')]
- delta loss_lm `age` - fifo, post-warm: mean +0.0000, sd +0.0000 **sd exactly 0: broken, not clean**; 500-step blocks [([16, 500], '+0.0000'), ([500, 1000], '+0.0000'), ([1000, 1500], '+0.0000'), ([1500, 2000], '+0.0000'), ([2000, 2500], '+0.0000'), ([2500, 3000], '+0.0000')]
- DIAGNOSTIC `psiC`: L_MC warm mean +1.4860, post-warm mean +1.0689; corr_psi_age post-warm median +0.2533 (None on 0 steps); phi grad norm median +0.0072; eval_forward_s total +3138.7; fill fraction (regression rows) +0.8404
- DIAGNOSTIC `age`: L_MC warm mean +1.4339, post-warm mean +0.2072; corr_psi_age post-warm median -0.9921 (None on 1 steps); phi grad norm median +0.0127; eval_forward_s total +3142.8; fill fraction (regression rows) +0.8404
- DIAGNOSTIC `psiC` attribution (mean per-step fraction): warm `{'fifo_warmup': 1.0}`, post-warm `{'psi': 1.0}`
- DIAGNOSTIC `age` attribution (mean per-step fraction): warm `{'fifo_warmup': 1.0}`, post-warm `{'age_only': 1.0}`
- run-level split histograms (train()'s phase_summary) equal the per-step sums: `{'psiC': True, 'age': True}`

### §7.1 diagnostics (§7.4: DIAGNOSTIC; never a gate; warm / scored split)

- `psiC` warm: steps 16, evictions 8192, fraction oldest +1.0000, fraction_shifting +0.0000, mean_shift +0.0000; raw Spearman rho(score, age) median -0.0305; partial rho (a diagnostic, never a gate) median -0.0311 IQR [-0.0540, +0.0323], None on 0 steps
  - victim ages `{'16': 8192}`; displacement `{'0': 8192}`
- `psiC` scored: steps 2984, evictions 1527808, fraction oldest +0.0821, fraction_shifting +0.9179, mean_shift +8.8663; raw Spearman rho(score, age) median +0.1223; partial rho (a diagnostic, never a gate) median -0.3132 IQR [-0.3328, -0.2950], None on 0 steps
  - victim ages `{'1': 408997, '2': 81448, '3': 70295, '4': 53296, '5': 42146, '6': 37164, '7': 38187, '8': 40473, '9': 38582, '10': 39864, '11': 38952, '12': 39161, '13': 40131, '14': 39750, '15': 43429, '16': 41353, '17': 40673, '18': 39787, '19': 34262, '20': 34457, '21': 29311, '22': 30154, '23': 26359, '24': 24139, '25': 23512, '26': 18406, '27': 18137, '28': 17038, '29': 14214, '30': 14051, '31': 12037, '32': 10860, '33': 8273, '34': 7691, '35': 6570, '36': 5033, '37': 4409, '38': 3616, '39': 2892, '40': 3442, '41': 1936, '42': 1259, '43': 1183, '44': 529, '45': 240, '46': 110}`; displacement `{'0': 125473, '1': 89502, '2': 77240, '3': 72606, '4': 74685, '5': 76441, '6': 67147, '7': 62887, '8': 60336, '9': 54261, '10': 54331, '11': 54140, '12': 55962, '13': 69369, '14': 124431, '15': 408997}`
- `psiC` first_scored_block: steps 500, evictions 256000, fraction oldest +0.1128, fraction_shifting +0.8872, mean_shift +7.7169; raw Spearman rho(score, age) median -0.1621; partial rho (a diagnostic, never a gate) median -0.3227 IQR [-0.3363, -0.3082], None on 0 steps
  - victim ages `{'1': 51637, '2': 12650, '3': 10516, '4': 8182, '5': 7209, '6': 5919, '7': 6228, '8': 6394, '9': 6776, '10': 6458, '11': 6688, '12': 6666, '13': 6772, '14': 6981, '15': 7684, '16': 7328, '17': 6360, '18': 6822, '19': 6268, '20': 6703, '21': 5703, '22': 5572, '23': 5548, '24': 4876, '25': 4572, '26': 3488, '27': 4048, '28': 3834, '29': 3395, '30': 3264, '31': 3181, '32': 3199, '33': 2376, '34': 2335, '35': 2172, '36': 1819, '37': 1395, '38': 1296, '39': 1031, '40': 1000, '41': 717, '42': 468, '43': 293, '44': 103, '45': 23, '46': 51}`; displacement `{'0': 28873, '1': 19298, '2': 16808, '3': 15092, '4': 14845, '5': 13487, '6': 12072, '7': 10975, '8': 9780, '9': 9354, '10': 8562, '11': 8423, '12': 8869, '13': 11358, '14': 16567, '15': 51637}`
  - uniform random over the live slots would give `{'0': 16000.0, '1': 16000.0, '2': 16000.0, '3': 16000.0, '4': 16000.0, '5': 16000.0, '6': 16000.0, '7': 16000.0, '8': 16000.0, '9': 16000.0, '10': 16000.0, '11': 16000.0, '12': 16000.0, '13': 16000.0, '14': 16000.0, '15': 16000.0}` (A1.8 m-2)
- `age` warm: steps 16, evictions 8192, fraction oldest +1.0000, fraction_shifting +0.0000, mean_shift +0.0000; raw Spearman rho(score, age) median -0.9941; partial rho (a diagnostic, never a gate) median -0.9926 IQR [-0.9963, -0.7664], None on 1 steps
  - victim ages `{'16': 8192}`; displacement `{'0': 8192}`
- `age` scored: steps 2984, evictions 1527808, fraction oldest +1.0000, fraction_shifting +0.0000, mean_shift +0.0000; raw Spearman rho(score, age) median -1.0000; partial rho (a diagnostic, never a gate) median -1.0000 IQR [-1.0000, -1.0000], None on 0 steps
  - victim ages `{'16': 1527808}`; displacement `{'0': 1527808}`
- `age` first_scored_block: steps 500, evictions 256000, fraction oldest +1.0000, fraction_shifting +0.0000, mean_shift +0.0000; raw Spearman rho(score, age) median -1.0000; partial rho (a diagnostic, never a gate) median -1.0000 IQR [-1.0000, -1.0000], None on 0 steps
  - victim ages `{'16': 256000}`; displacement `{'0': 256000}`
  - uniform random over the live slots would give `{'0': 16000.0, '1': 16000.0, '2': 16000.0, '3': 16000.0, '4': 16000.0, '5': 16000.0, '6': 16000.0, '7': 16000.0, '8': 16000.0, '9': 16000.0, '10': 16000.0, '11': 16000.0, '12': 16000.0, '13': 16000.0, '14': 16000.0, '15': 16000.0}` (A1.8 m-2)
- Partial rho partials out the slot gestalts only (linear), not `c_t`; no verdict is drawn from any comparison with the spec's E2 value. Leading-edge statistic: not computed (the spec defines a baseline, not a statistic; `leading_edge.py` is a stub). Attention-share Gini: not available.

### Step-160 recency statistics (§7.4) -- DIAGNOSTIC (A4.2): never the headline, never gated

- DIAGNOSTIC step 160 `psiC` warm: steps 16, evictions 8192, fraction oldest +1.0000, mean_shift +0.0000; raw Spearman rho(score, age) median -0.0305
- DIAGNOSTIC step 160 `psiC` scored: steps 144, evictions 73728, fraction oldest +0.1183, mean_shift +7.6439; raw Spearman rho(score, age) median -0.1623
- DIAGNOSTIC step 160 `psiC` first_scored_block: steps 144, evictions 73728, fraction oldest +0.1183, mean_shift +7.6439; raw Spearman rho(score, age) median -0.1623
- DIAGNOSTIC step 160 `age` warm: steps 16, evictions 8192, fraction oldest +1.0000, mean_shift +0.0000; raw Spearman rho(score, age) median -0.9941
- DIAGNOSTIC step 160 `age` scored: steps 144, evictions 73728, fraction oldest +1.0000, mean_shift +0.0000; raw Spearman rho(score, age) median -1.0000
- DIAGNOSTIC step 160 `age` first_scored_block: steps 144, evictions 73728, fraction oldest +1.0000, mean_shift +0.0000; raw Spearman rho(score, age) median -1.0000

### §7.5 reporting (SECONDARY)

- fill fraction, held-out in-loop: +0.8229
- The censored-target caveat, verbatim from the PREREG:

  > The shadow buffer is off (ADR-0009 L6(a)). Every regression target `G_i(t)` is cut at the
  > slot's eviction. Before `T_warm` the cut is FIFO's, at age `M`, so the target itself is
  > age-shaped although ψ̂ has no age input. After `T_warm` the cut is the learned policy's own,
  > and self-confirming (§3.4, §7.2). Spec §3.4's description of pre-`T_warm` data (spec :280)
  > holds for the policy, not for the target. The rescaled `r_i` is described as rescaled, never
  > as anything stronger (§3.2.1).

### T5 (held-out cost, measured in this run; DIAGNOSTIC)

The step-160 DIAGNOSTIC passes (psiC@160, age@160, sz_psiC@160, sz_age@160; A4.2) add about 1,090 s per seed. That figure is EXTRAPOLATED from build-time T5 (one batch per pass, x 128 batches; sz_age assumed equal to sz_psiC), not measured; this run's own per-pass timing follows.

- `psiC`: first batch +1.76 s; projected full EVAL +225.0 s; measured +227.1 s over 128 batches
- `fifo`: first batch +0.77 s; projected full EVAL +98.2 s; measured +115.6 s over 128 batches
- `age`: first batch +1.53 s; projected full EVAL +195.5 s; measured +225.8 s over 128 batches
- `oracle`: first batch +0.77 s; projected full EVAL +99.0 s; measured +116.6 s over 128 batches
- `lru`: first batch +0.82 s; projected full EVAL +104.7 s; measured +122.8 s over 128 batches
- `random0`: first batch +0.77 s; projected full EVAL +98.6 s; measured +116.0 s over 128 batches
- `random1`: first batch +0.77 s; projected full EVAL +98.7 s; measured +116.7 s over 128 batches
- `random2`: first batch +0.80 s; projected full EVAL +102.3 s; measured +116.6 s over 128 batches
- `random3`: first batch +0.90 s; projected full EVAL +115.8 s; measured +116.9 s over 128 batches
- `random4`: first batch +0.92 s; projected full EVAL +117.8 s; measured +116.9 s over 128 batches
- `sz_psiC`: first batch +2.52 s; projected full EVAL +322.3 s; measured +306.5 s over 128 batches
- `sz_fifo`: first batch +0.90 s; projected full EVAL +115.2 s; measured +117.3 s over 128 batches
- `sz_age`: first batch +2.33 s; projected full EVAL +298.0 s; measured +306.4 s over 128 batches
- `psiC@160`: first batch +1.75 s; projected full EVAL +224.4 s; measured +228.4 s over 128 batches
- `age@160`: first batch +1.74 s; projected full EVAL +223.3 s; measured +226.1 s over 128 batches
- `psiC@1000`: first batch +1.74 s; projected full EVAL +222.4 s; measured +228.1 s over 128 batches
- `age@1000`: first batch +1.78 s; projected full EVAL +227.9 s; measured +227.1 s over 128 batches
- `psiC@2000`: first batch +1.82 s; projected full EVAL +232.4 s; measured +228.6 s over 128 batches
- `age@2000`: first batch +1.75 s; projected full EVAL +224.0 s; measured +225.8 s over 128 batches
- `sz_psiC@160`: first batch +2.25 s; projected full EVAL +288.3 s; measured +307.4 s over 128 batches
- `sz_age@160`: first batch +2.24 s; projected full EVAL +286.6 s; measured +309.3 s over 128 batches

## Seed 2 (status `ok`, exit `0`). Everything below is SECONDARY or DIAGNOSTIC.

### Controls (§8 as amended)

- **C0**: ok = `True` -- `{"pool_overlaps": [], "pool_below_K": true, "eval_overlaps": [], "eval_below_K": true, "pool_eval_disjoint": true, "vocab_sha256": "afbaf994e9d1f617fdc2e68579bf5f55622ee17aee056d33d8667d218d245031", "pool_closure": true, "eval_closure": true, "ok": true, "failed": []}`
- **C1**: ok = `True` -- `{"per_arm": {"psiC": {"sha256": "b507ebc573a54316f9d3c20337f49e664bb9c414388af925c5a8e87efd6b63b8", "source_vocab_sha256": "afbaf994e9d1f617fdc2e68579bf5f55622ee17aee056d33d8667d218d245031", "run_vocab_sha256": "afbaf994e9d1f617fdc2e68579bf5f55622ee17aee056d33d8667d218d245031", "vocab_checked": true, "seed_matches": true, "source_seed": 2, "allow_unrecorded_vocabulary": false, "allow_seed_mismatch": false, "ok": true}, "fifo": {"sha256": "b507ebc573a54316f9d3c20337f49e664bb9c414388af925c5a8e87efd6b63b8", "source_vocab_sha256": "afbaf994e9d1f617fdc2e68579bf5f55622ee17aee056d33d8667d218d245031", "run_vocab_sha256": "afbaf994e9d1f617fdc2e68579bf5f55622ee17aee056d33d8667d218d245031", "vocab_chec`
- **C2**: ok = `True` -- `{"steps_compared": 3000, "warm_steps_compared": 16, "a_pool_ids": {"present": true, "required": true, "cross_arm_mismatch_steps": [], "epochs_not_exactly_the_pool": []}, "b_fingerprint": {"cross_arm_mismatch_steps": [], "replay_mismatch_steps": [], "sampling": "per-epoch permutation (each document once per epoch)", "sampling_per_arm": {"psiC": "per-epoch permutation (each document once per epoch)", "fifo": "per-epoch permutation (each document once per epoch)", "age": "per-epoch permutation (each document once per epoch)"}, "sampling_agrees": true}, "c_loss_lm_warm_mismatch_steps": [], "same_step_sets": true, "ok": true}`
- **C3**: ok = `True` -- `{"per_checkpoint": {"psiC": {"n_tensors": 206, "keys_equal": true, "differ": [], "ok": true}, "fifo": {"n_tensors": 206, "keys_equal": true, "differ": [], "ok": true}, "age": {"n_tensors": 206, "keys_equal": true, "differ": [], "ok": true}, "psiC@160": {"n_tensors": 206, "keys_equal": true, "differ": [], "ok": true}, "age@160": {"n_tensors": 206, "keys_equal": true, "differ": [], "ok": true}, "psiC@1000": {"n_tensors": 206, "keys_equal": true, "differ": [], "ok": true}, "age@1000": {"n_tensors": 206, "keys_equal": true, "differ": [], "ok": true}, "psiC@2000": {"n_tensors": 206, "keys_equal": true, "differ": [], "ok": true}, "age@2000": {"n_tensors": 206, "keys_equal": true, "differ": [], "ok`
- **C4**: ok = `True` -- `{"R_quantity": 0.879943523555994, "reference": 0.879943523555994, "abs_diff": 0.0, "tolerance": 1e-06, "seconds": 25.599167875014246, "torch_threads": 1, "ok": true}`
- **C5**: ok = `True` -- `{"per_arm": {"psiC": {"band": "live", "ratio": 77.85591251012764, "exit_code": 0}, "fifo": {"band": "live", "ratio": 77.85591251012764, "exit_code": 0}, "age": {"band": "live", "ratio": 77.85591251012764, "exit_code": 0}}, "ok": true, "inert": [], "ratio_note": "the ratio is reported, never a control (A2.3)"}`
- **C6**: ok = `True` -- `{"per_arm": {"psiC": {"steps_requested": 3000, "steps_done": 3000, "status": "completed", "n_beats": 3000, "ok": true}, "fifo": {"steps_requested": 3000, "steps_done": 3000, "status": "completed", "n_beats": 3000, "ok": true}, "age": {"steps_requested": 3000, "steps_done": 3000, "status": "completed", "n_beats": 3000, "ok": true}}, "ok": true}`
- **C7**: ok = `True` -- `{"per_pass": {"psiC": {"n_documents": 2048, "mismatched_docs": [], "n_mismatched": 0}, "fifo": {"n_documents": 2048, "mismatched_docs": [], "n_mismatched": 0}, "age": {"n_documents": 2048, "mismatched_docs": [], "n_mismatched": 0}, "oracle": {"n_documents": 2048, "mismatched_docs": [], "n_mismatched": 0}, "lru": {"n_documents": 2048, "mismatched_docs": [], "n_mismatched": 0}, "random0": {"n_documents": 2048, "mismatched_docs": [], "n_mismatched": 0}, "random1": {"n_documents": 2048, "mismatched_docs": [], "n_mismatched": 0}, "random2": {"n_documents": 2048, "mismatched_docs": [], "n_mismatched": 0}, "random3": {"n_documents": 2048, "mismatched_docs": [], "n_mismatched": 0}, "random4": {"n_do`
- **C8**: ok = `True` -- `{"per_arm": {"psiC": {"n_steps": 3000, "nonfinite_steps": [], "ok": true}, "age": {"n_steps": 3000, "nonfinite_steps": [], "ok": true}}, "ok": true}`
- **C10**: ok = `True` -- `{"derived_ckpt": 1000, "derived_every": 40, "registered": {"file": "/Users/keanooo7/retrieval-successor-retention/.worktrees/rsr-run1-run/experiments/rsr-run1/PREREG.md", "heading": "### A4.1 Owner values (fills A2.8's slots)", "values": {"gamma": 0.9, "phi_weight_decay": 0.01, "iters": 3000, "o5": "eval"}, "raw": {"O1": "0.9", "O4": "0.01", "R-1": "3000", "O5": "eval"}, "problems": [], "status": "filled"}, "launch": {"run_sha": "ed6302aae37a2831c9b8567b9f80a093b4c00f70", "dirty": false, "gamma": "0.9", "wd": "0.01", "steps": "3000", "ckpt": "1000", "ckpt_every": "40", "ckpt_every_rule": "gcd(160, CKPT) if STEPS>=160 and CKPT>0 else CKPT (A4.2)", "diag_step": "160", "o5": "eval", "c1_flags":`
- **C9 / C9b** (scoped to the oracle numbers): ok = `True` -- `{'documents': [995000, 995016], 'C9': {'gammas': [0.5, 0.9], 'differing_docs': [], 'ok': True}, 'C9b': {'gamma': 0.9, 'differing_docs': [], 'ok': True}, 'ok': True, 'scope': "a failure marks oracle regret, headroom share and the oracle's own accuracy NOT READ; exit and status unchanged (A3.6)"}`

### What ran

- psiC: steps_requested `3000`, steps_done `3000`, status `completed`
- fifo: steps_requested `3000`, steps_done `3000`, status `completed`
- age: steps_requested `3000`, steps_done `3000`, status `completed`
- psiC: T_warm `16.0` steps = `1.0` epoch; total epochs `187.5`; T_warm / total epochs `0.005333333333333333` (31(b)); sampling `per-epoch permutation (each document once per epoch)`
- age: T_warm `16.0` steps = `1.0` epoch; total epochs `187.5`; T_warm / total epochs `0.005333333333333333` (31(b)); sampling `per-epoch permutation (each document once per epoch)`
- frozen model mode during training: `{'psiC': 'eval (dropout off)', 'fifo': 'eval (dropout off)', 'age': 'eval (dropout off)'}`
- final checkpoint bytes (T6): `{'psiC': 10066571, 'fifo': 9861979, 'age': 9865491}`

### Held-out accuracy (§7.1; SECONDARY), EVAL n = 2048

> This interval resamples EVAL documents only. It holds fixed the trained φ (one φ
> initialisation, one data order), the substrate seed and the pool. It is a within-seed interval.
> The three per-seed values show the **combined** variation of substrate seed, φ initialisation,
> data order and EVAL documents, which change together across seeds. They cannot separate
> run-to-run variation from substrate-seed variation.

| bucket | n | psiC | fifo | age | lru | oracle | random |
|---|---|---|---|---|---|---|---|
| all | 37836 | +0.6348 [+0.6304, +0.6392] | +0.8031 [+0.7991, +0.8070] | +0.8031 [+0.7991, +0.8070] | +0.7899 [+0.7858, +0.7939] | +0.9280 [+0.9252, +0.9307] | +0.7581 [+0.7548, +0.7613] |
| gap_eq_1 | 7960 | +0.9888 [+0.9864, +0.9910] | +0.9877 [+0.9853, +0.9901] | +0.9877 [+0.9853, +0.9901] | +0.9866 [+0.9839, +0.9891] | +0.9877 [+0.9852, +0.9902] | +0.9872 [+0.9849, +0.9895] |
| gap_ge_2 | 29876 | +0.5404 [+0.5354, +0.5456] | +0.7539 [+0.7491, +0.7588] | +0.7539 [+0.7491, +0.7588] | +0.7375 [+0.7323, +0.7425] | +0.9121 [+0.9086, +0.9155] | +0.6971 [+0.6931, +0.7010] |
| gap_2_to_M | 22663 | +0.6291 [+0.6235, +0.6350] | +0.9278 [+0.9243, +0.9312] | +0.9278 [+0.9243, +0.9312] | +0.8912 [+0.8870, +0.8953] | +0.9207 [+0.9169, +0.9242] | +0.8098 [+0.8061, +0.8136] |
| gap_gt_M | 7213 | +0.2619 [+0.2524, +0.2721] | +0.2075 [+0.1990, +0.2166] | +0.2075 [+0.1990, +0.2166] | +0.2547 [+0.2453, +0.2642] | +0.8853 [+0.8780, +0.8927] | +0.3429 [+0.3370, +0.3491] |

`LRUPolicy` falls back to the write step for a never-attended slot, so it degrades toward FIFO by design; its distance from FIFO is `acc.*.lru_minus_fifo` (A1.1).

### Contrasts (SECONDARY; the headline is §1)

- `R.age`: +0.8661 [+0.8613, +0.8709] CI above 0
- `R.fifo`: +0.8661 [+0.8613, +0.8709] CI above 0
- `R.psiC`: +0.5674 [+0.5607, +0.5742] CI above 0
- `R_own_floor.age`: +0.8654 [+0.8604, +0.8702] CI above 0
- `R_own_floor.fifo`: +0.8661 [+0.8613, +0.8709] CI above 0
- `R_own_floor.psiC`: +0.5659 [+0.5590, +0.5729] CI above 0
- `acc.all.age_minus_fifo`: +0.0000 [+0.0000, +0.0000] CI includes 0 **sd exactly 0: broken, not clean**
- `acc.all.age_minus_lru`: +0.0132 [+0.0105, +0.0159] CI above 0
- `acc.all.lru_minus_fifo`: -0.0132 [-0.0159, -0.0105] CI below 0
- `acc.all.psiC_minus_age`: -0.1683 [-0.1726, -0.1639] CI below 0
- `acc.all.psiC_minus_fifo` **(HEADLINE)**: -0.1683 [-0.1726, -0.1639] CI below 0
- `acc.all.psiC_minus_lru`: -0.1551 [-0.1592, -0.1506] CI below 0
- `acc.gap_2_to_M.age_minus_fifo`: +0.0000 [+0.0000, +0.0000] CI includes 0 **sd exactly 0: broken, not clean**
- `acc.gap_2_to_M.age_minus_lru`: +0.0366 [+0.0340, +0.0395] CI above 0
- `acc.gap_2_to_M.lru_minus_fifo`: -0.0366 [-0.0395, -0.0340] CI below 0
- `acc.gap_2_to_M.psiC_minus_age`: -0.2987 [-0.3041, -0.2931] CI below 0
- `acc.gap_2_to_M.psiC_minus_fifo`: -0.2987 [-0.3041, -0.2931] CI below 0
- `acc.gap_2_to_M.psiC_minus_lru`: -0.2621 [-0.2673, -0.2563] CI below 0
- `acc.gap_eq_1.age_minus_fifo`: +0.0000 [+0.0000, +0.0000] CI includes 0 **sd exactly 0: broken, not clean**
- `acc.gap_eq_1.age_minus_lru`: +0.0011 [+0.0002, +0.0021] CI above 0
- `acc.gap_eq_1.lru_minus_fifo`: -0.0011 [-0.0021, -0.0002] CI below 0
- `acc.gap_eq_1.psiC_minus_age`: +0.0011 [+0.0002, +0.0020] CI above 0
- `acc.gap_eq_1.psiC_minus_fifo`: +0.0011 [+0.0002, +0.0020] CI above 0
- `acc.gap_eq_1.psiC_minus_lru`: +0.0023 [+0.0010, +0.0035] CI above 0
- `acc.gap_ge_2.age_minus_fifo`: +0.0000 [+0.0000, +0.0000] CI includes 0 **sd exactly 0: broken, not clean**
- `acc.gap_ge_2.age_minus_lru`: +0.0164 [+0.0130, +0.0199] CI above 0
- `acc.gap_ge_2.lru_minus_fifo`: -0.0164 [-0.0199, -0.0130] CI below 0
- `acc.gap_ge_2.psiC_minus_age`: -0.2135 [-0.2187, -0.2080] CI below 0
- `acc.gap_ge_2.psiC_minus_fifo`: -0.2135 [-0.2187, -0.2080] CI below 0
- `acc.gap_ge_2.psiC_minus_lru`: -0.1971 [-0.2022, -0.1914] CI below 0
- `acc.gap_gt_M.age_minus_fifo`: +0.0000 [+0.0000, +0.0000] CI includes 0 **sd exactly 0: broken, not clean**
- `acc.gap_gt_M.age_minus_lru`: -0.0471 [-0.0580, -0.0358] CI below 0
- `acc.gap_gt_M.lru_minus_fifo`: +0.0471 [+0.0358, +0.0580] CI above 0
- `acc.gap_gt_M.psiC_minus_age`: +0.0543 [+0.0414, +0.0670] CI above 0
- `acc.gap_gt_M.psiC_minus_fifo`: +0.0543 [+0.0414, +0.0670] CI above 0
- `acc.gap_gt_M.psiC_minus_lru`: +0.0072 [-0.0054, +0.0200] CI includes 0
- `floor_difference.sz_age_minus_sz_fifo`: +0.0007 [-0.0027, +0.0040] CI includes 0
- `floor_difference.sz_psiC_minus_sz_fifo`: +0.0014 [-0.0019, +0.0047] CI includes 0
- `nll_answer.age_minus_fifo`: +0.0000 [+0.0000, +0.0000] CI includes 0 **sd exactly 0: broken, not clean**
- `nll_answer.age_minus_lru`: -0.0932 [-0.1027, -0.0842] CI below 0
- `nll_answer.lru_minus_fifo`: +0.0932 [+0.0842, +0.1027] CI above 0
- `nll_answer.psiC_minus_age`: +0.5806 [+0.5646, +0.5951] CI above 0
- `nll_answer.psiC_minus_fifo`: +0.5806 [+0.5646, +0.5951] CI above 0
- `nll_answer.psiC_minus_lru`: +0.4874 [+0.4703, +0.5033] CI above 0
- `nll_real.age_minus_fifo`: +0.0000 [+0.0000, +0.0000] CI includes 0 **sd exactly 0: broken, not clean**
- `nll_real.age_minus_lru`: -0.0204 [-0.0216, -0.0194] CI below 0
- `nll_real.lru_minus_fifo`: +0.0204 [+0.0194, +0.0216] CI above 0
- `nll_real.psiC_minus_age`: +0.1733 [+0.1704, +0.1760] CI above 0
- `nll_real.psiC_minus_fifo`: +0.1733 [+0.1704, +0.1760] CI above 0
- `nll_real.psiC_minus_lru`: +0.1529 [+0.1499, +0.1557] CI above 0
- `oracle_regret.all.age`: +0.1249 [+0.1217, +0.1283] CI above 0
- `oracle_regret.all.fifo`: +0.1249 [+0.1217, +0.1283] CI above 0
- `oracle_regret.all.psiC`: +0.2933 [+0.2890, +0.2972] CI above 0
- `oracle_regret.gap_2_to_M.age`: -0.0071 [-0.0087, -0.0057] CI below 0
- `oracle_regret.gap_2_to_M.fifo`: -0.0071 [-0.0087, -0.0057] CI below 0
- `oracle_regret.gap_2_to_M.psiC`: +0.2916 [+0.2859, +0.2968] CI above 0
- `oracle_regret.gap_eq_1.age`: +0.0000 [-0.0006, +0.0007] CI includes 0
- `oracle_regret.gap_eq_1.fifo`: +0.0000 [-0.0006, +0.0007] CI includes 0
- `oracle_regret.gap_eq_1.psiC`: -0.0011 [-0.0021, -0.0002] CI below 0
- `oracle_regret.gap_ge_2.age`: +0.1582 [+0.1543, +0.1626] CI above 0
- `oracle_regret.gap_ge_2.fifo`: +0.1582 [+0.1543, +0.1626] CI above 0
- `oracle_regret.gap_ge_2.psiC`: +0.3717 [+0.3667, +0.3764] CI above 0
- `oracle_regret.gap_gt_M.age`: +0.6778 [+0.6671, +0.6880] CI above 0
- `oracle_regret.gap_gt_M.fifo`: +0.6778 [+0.6671, +0.6880] CI above 0
- `oracle_regret.gap_gt_M.psiC`: +0.6235 [+0.6118, +0.6345] CI above 0
- NLL `real` level (pooled per target): psiC +1.5181; fifo +1.3448; age +1.3448; lru +1.3653
- NLL `answer` level (pooled per target): psiC +1.2724; fifo +0.6918; age +0.6918; lru +0.7850
- headroom share `psiC` (all): -1.3474 [-1.4065, -1.2878], dropped 0
- headroom share `age` (all): +0.0000 [+0.0000, +0.0000], dropped 0
- DIAGNOSTIC snapshot `psiC@160` (A4.2): accuracy +0.5944 [+0.5901, +0.5989]; minus final fifo -0.2087 [-0.2129, -0.2044] CI below 0
- DIAGNOSTIC snapshot `age@160` (A4.2): accuracy +0.8031 [+0.7991, +0.8070]; minus final fifo +0.0000 [+0.0000, +0.0000] CI includes 0 **sd exactly 0: broken, not clean**
- DIAGNOSTIC snapshot `psiC@1000` (A1.8 m-9): accuracy +0.6154 [+0.6110, +0.6199]; minus final fifo -0.1877 [-0.1918, -0.1833] CI below 0
- DIAGNOSTIC snapshot `age@1000` (A1.8 m-9): accuracy +0.8031 [+0.7991, +0.8070]; minus final fifo +0.0000 [+0.0000, +0.0000] CI includes 0 **sd exactly 0: broken, not clean**
- DIAGNOSTIC snapshot `psiC@2000` (A1.8 m-9): accuracy +0.6284 [+0.6239, +0.6329]; minus final fifo -0.1747 [-0.1792, -0.1702] CI below 0
- DIAGNOSTIC snapshot `age@2000` (A1.8 m-9): accuracy +0.8031 [+0.7991, +0.8070]; minus final fifo +0.0000 [+0.0000, +0.0000] CI includes 0 **sd exactly 0: broken, not clean**

### Step-160 phi reading -- DIAGNOSTIC (A4.2): never the headline, never gated

psiC and age read with their step-160 phi (`psiC@160`, `age@160`, floors `sz_psiC@160`, `sz_age@160`); fifo, LRU and sz_fifo are the final reading's, which phi does not touch. Same EVAL documents, batch and bootstrap draw.

- DIAGNOSTIC step 160 accuracy `all`: psiC@160 +0.5944 [+0.5901, +0.5989]; age@160 +0.8031 [+0.7991, +0.8070]; sz_psiC@160 +0.0639 [+0.0614, +0.0663]; sz_age@160 +0.0634 [+0.0611, +0.0658]
- DIAGNOSTIC step 160 accuracy `gap_eq_1`: psiC@160 +0.9888 [+0.9865, +0.9911]; age@160 +0.9877 [+0.9853, +0.9901]; sz_psiC@160 +0.0682 [+0.0626, +0.0738]; sz_age@160 +0.0678 [+0.0624, +0.0731]
- DIAGNOSTIC step 160 accuracy `gap_ge_2`: psiC@160 +0.4893 [+0.4843, +0.4946]; age@160 +0.7539 [+0.7491, +0.7588]; sz_psiC@160 +0.0627 [+0.0600, +0.0654]; sz_age@160 +0.0622 [+0.0596, +0.0650]
- DIAGNOSTIC step 160 accuracy `gap_2_to_M`: psiC@160 +0.5745 [+0.5691, +0.5804]; age@160 +0.9278 [+0.9243, +0.9312]; sz_psiC@160 +0.0624 [+0.0593, +0.0654]; sz_age@160 +0.0614 [+0.0584, +0.0647]
- DIAGNOSTIC step 160 accuracy `gap_gt_M`: psiC@160 +0.2215 [+0.2120, +0.2315]; age@160 +0.2075 [+0.1990, +0.2166]; sz_psiC@160 +0.0638 [+0.0584, +0.0691]; sz_age@160 +0.0646 [+0.0593, +0.0702]
- DIAGNOSTIC step 160 `R.age`: +0.8661 [+0.8613, +0.8709] CI above 0
- DIAGNOSTIC step 160 `R.psiC`: +0.5128 [+0.5064, +0.5196] CI above 0
- DIAGNOSTIC step 160 `R_own_floor.age`: +0.8664 [+0.8617, +0.8711] CI above 0
- DIAGNOSTIC step 160 `R_own_floor.psiC`: +0.5121 [+0.5061, +0.5189] CI above 0
- DIAGNOSTIC step 160 `acc.all.age_minus_fifo`: +0.0000 [+0.0000, +0.0000] CI includes 0 **sd exactly 0: broken, not clean**
- DIAGNOSTIC step 160 `acc.all.psiC_minus_age`: -0.2087 [-0.2129, -0.2044] CI below 0
- DIAGNOSTIC step 160 `acc.all.psiC_minus_fifo`: -0.2087 [-0.2129, -0.2044] CI below 0
- DIAGNOSTIC step 160 `acc.gap_2_to_M.age_minus_fifo`: +0.0000 [+0.0000, +0.0000] CI includes 0 **sd exactly 0: broken, not clean**
- DIAGNOSTIC step 160 `acc.gap_2_to_M.psiC_minus_age`: -0.3533 [-0.3583, -0.3476] CI below 0
- DIAGNOSTIC step 160 `acc.gap_2_to_M.psiC_minus_fifo`: -0.3533 [-0.3583, -0.3476] CI below 0
- DIAGNOSTIC step 160 `acc.gap_eq_1.age_minus_fifo`: +0.0000 [+0.0000, +0.0000] CI includes 0 **sd exactly 0: broken, not clean**
- DIAGNOSTIC step 160 `acc.gap_eq_1.psiC_minus_age`: +0.0011 [+0.0003, +0.0021] CI above 0
- DIAGNOSTIC step 160 `acc.gap_eq_1.psiC_minus_fifo`: +0.0011 [+0.0003, +0.0021] CI above 0
- DIAGNOSTIC step 160 `acc.gap_ge_2.age_minus_fifo`: +0.0000 [+0.0000, +0.0000] CI includes 0 **sd exactly 0: broken, not clean**
- DIAGNOSTIC step 160 `acc.gap_ge_2.psiC_minus_age`: -0.2646 [-0.2699, -0.2591] CI below 0
- DIAGNOSTIC step 160 `acc.gap_ge_2.psiC_minus_fifo`: -0.2646 [-0.2699, -0.2591] CI below 0
- DIAGNOSTIC step 160 `acc.gap_gt_M.age_minus_fifo`: +0.0000 [+0.0000, +0.0000] CI includes 0 **sd exactly 0: broken, not clean**
- DIAGNOSTIC step 160 `acc.gap_gt_M.psiC_minus_age`: +0.0140 [+0.0014, +0.0263] CI above 0
- DIAGNOSTIC step 160 `acc.gap_gt_M.psiC_minus_fifo`: +0.0140 [+0.0014, +0.0263] CI above 0
- DIAGNOSTIC step 160 `floor_difference.sz_age_minus_sz_fifo`: -0.0003 [-0.0035, +0.0031] CI includes 0
- DIAGNOSTIC step 160 `floor_difference.sz_psiC_minus_sz_fifo`: +0.0007 [-0.0026, +0.0038] CI includes 0
- DIAGNOSTIC step 160 `nll_answer.age_minus_fifo`: +0.0000 [+0.0000, +0.0000] CI includes 0 **sd exactly 0: broken, not clean**
- DIAGNOSTIC step 160 `nll_answer.psiC_minus_age`: +0.6960 [+0.6811, +0.7105] CI above 0
- DIAGNOSTIC step 160 `nll_answer.psiC_minus_fifo`: +0.6960 [+0.6811, +0.7105] CI above 0
- DIAGNOSTIC step 160 `nll_real.age_minus_fifo`: +0.0000 [+0.0000, +0.0000] CI includes 0 **sd exactly 0: broken, not clean**
- DIAGNOSTIC step 160 `nll_real.psiC_minus_age`: +0.1800 [+0.1773, +0.1826] CI above 0
- DIAGNOSTIC step 160 `nll_real.psiC_minus_fifo`: +0.1800 [+0.1773, +0.1826] CI above 0
- DIAGNOSTIC held-out evictions `psiC`: n 65536, fraction oldest +0.0415, victim ages `{'1': 20536, '2': 4665, '3': 3667, '4': 2692, '5': 2268, '6': 2075, '7': 1922, '8': 1854, '9': 1801, '10': 1706, '11': 1583, '12': 1601, '13': 1564, '14': 1521, '15': 1541, '16': 1473, '17': 1286, '18': 1270, '19': 1105, '20': 995, '21': 877, '22': 818, '23': 761, '24': 652, '25': 598, '26': 562, '27': 511, '28': 467, '29': 431, '30': 406, '31': 311, '32': 322, '33': 284, '34': 230, '35': 199, '36': 214, '37': 160, '38': 134, '39': 102, '40': 103, '41': 81, '42': 57, '43': 62, '44': 36, '45': 15, '46': 12, '47': 6}`
- DIAGNOSTIC held-out evictions `fifo`: n 65536, fraction oldest +1.0000, victim ages `{'16': 65536}`
- DIAGNOSTIC held-out evictions `age`: n 65536, fraction oldest +1.0000, victim ages `{'16': 65536}`
- DIAGNOSTIC held-out evictions `oracle`: n 65536, fraction oldest +0.3806, victim ages `{'7': 3, '8': 33, '9': 72, '10': 307, '11': 997, '12': 2425, '13': 5925, '14': 11223, '15': 16912, '16': 20426, '17': 461, '18': 423, '19': 434, '20': 460, '21': 407, '22': 392, '23': 371, '24': 339, '25': 321, '26': 335, '27': 337, '28': 307, '29': 303, '30': 279, '31': 268, '32': 236, '33': 224, '34': 226, '35': 238, '36': 182, '37': 187, '38': 179, '39': 160, '40': 144}`
- DIAGNOSTIC held-out evictions `lru`: n 65536, fraction oldest +0.0541, victim ages `{'9': 6, '10': 384, '11': 4278, '12': 14451, '13': 16868, '14': 9190, '15': 5415, '16': 3883, '17': 2513, '18': 1603, '19': 1167, '20': 815, '21': 620, '22': 514, '23': 513, '24': 468, '25': 504, '26': 388, '27': 337, '28': 296, '29': 208, '30': 178, '31': 156, '32': 132, '33': 97, '34': 97, '35': 66, '36': 84, '37': 63, '38': 49, '39': 34, '40': 43, '41': 31, '42': 22, '43': 23, '44': 13, '45': 12, '46': 12, '47': 3}`

### Training time, on the pool (§7.3: SECONDARY; in-sample for phi)

- delta loss_lm `psiC` - fifo, post-warm: mean +0.1737, sd +0.0157; 500-step blocks [([16, 500], '+0.1802'), ([500, 1000], '+0.1708'), ([1000, 1500], '+0.1681'), ([1500, 2000], '+0.1718'), ([2000, 2500], '+0.1761'), ([2500, 3000], '+0.1753')]
- delta loss_lm `age` - fifo, post-warm: mean +0.0000, sd +0.0000 **sd exactly 0: broken, not clean**; 500-step blocks [([16, 500], '+0.0000'), ([500, 1000], '+0.0000'), ([1000, 1500], '+0.0000'), ([1500, 2000], '+0.0000'), ([2000, 2500], '+0.0000'), ([2500, 3000], '+0.0000')]
- DIAGNOSTIC `psiC`: L_MC warm mean +1.6744, post-warm mean +1.5091; corr_psi_age post-warm median +0.2465 (None on 0 steps); phi grad norm median +0.0082; eval_forward_s total +3146.2; fill fraction (regression rows) +0.8404
- DIAGNOSTIC `age`: L_MC warm mean +1.6185, post-warm mean +0.3007; corr_psi_age post-warm median -0.9772 (None on 1 steps); phi grad norm median +0.0150; eval_forward_s total +3133.1; fill fraction (regression rows) +0.8404
- DIAGNOSTIC `psiC` attribution (mean per-step fraction): warm `{'fifo_warmup': 1.0}`, post-warm `{'psi': 1.0}`
- DIAGNOSTIC `age` attribution (mean per-step fraction): warm `{'fifo_warmup': 1.0}`, post-warm `{'age_only': 1.0}`
- run-level split histograms (train()'s phase_summary) equal the per-step sums: `{'psiC': True, 'age': True}`

### §7.1 diagnostics (§7.4: DIAGNOSTIC; never a gate; warm / scored split)

- `psiC` warm: steps 16, evictions 8192, fraction oldest +1.0000, fraction_shifting +0.0000, mean_shift +0.0000; raw Spearman rho(score, age) median -0.0590; partial rho (a diagnostic, never a gate) median -0.0473 IQR [-0.0519, -0.0254], None on 0 steps
  - victim ages `{'16': 8192}`; displacement `{'0': 8192}`
- `psiC` scored: steps 2984, evictions 1527808, fraction oldest +0.0574, fraction_shifting +0.9426, mean_shift +8.8117; raw Spearman rho(score, age) median +0.0841; partial rho (a diagnostic, never a gate) median -0.2083 IQR [-0.2409, -0.1826], None on 0 steps
  - victim ages `{'1': 336130, '2': 93792, '3': 72385, '4': 55698, '5': 52163, '6': 54493, '7': 47772, '8': 49476, '9': 48824, '10': 46897, '11': 46170, '12': 46128, '13': 44920, '14': 41223, '15': 45691, '16': 40855, '17': 37506, '18': 38836, '19': 29793, '20': 32553, '21': 28906, '22': 24062, '23': 23179, '24': 21557, '25': 17723, '26': 16792, '27': 15699, '28': 14536, '29': 14688, '30': 13468, '31': 10918, '32': 9817, '33': 8230, '34': 7062, '35': 7279, '36': 5718, '37': 4656, '38': 4394, '39': 4775, '40': 3245, '41': 3213, '42': 1848, '43': 1677, '44': 1339, '45': 875, '46': 608, '47': 239}`; displacement `{'0': 87658, '1': 86255, '2': 83065, '3': 77638, '4': 74793, '5': 77433, '6': 74724, '7': 75183, '8': 74241, '9': 70380, '10': 68884, '11': 69355, '12': 67096, '13': 79428, '14': 125545, '15': 336130}`
- `psiC` first_scored_block: steps 500, evictions 256000, fraction oldest +0.0828, fraction_shifting +0.9172, mean_shift +7.5390; raw Spearman rho(score, age) median -0.0944; partial rho (a diagnostic, never a gate) median -0.2693 IQR [-0.2867, -0.2536], None on 0 steps
  - victim ages `{'1': 38769, '2': 14082, '3': 11152, '4': 8091, '5': 7955, '6': 8487, '7': 7382, '8': 7970, '9': 7703, '10': 7838, '11': 8238, '12': 8268, '13': 8301, '14': 7857, '15': 8681, '16': 7867, '17': 7912, '18': 7770, '19': 6265, '20': 6632, '21': 5809, '22': 5366, '23': 5135, '24': 4768, '25': 4285, '26': 3694, '27': 3501, '28': 3470, '29': 3190, '30': 2678, '31': 2538, '32': 2106, '33': 1746, '34': 1810, '35': 1601, '36': 1132, '37': 1088, '38': 888, '39': 1110, '40': 793, '41': 623, '42': 425, '43': 374, '44': 383, '45': 157, '46': 69, '47': 41}`; displacement `{'0': 21191, '1': 18784, '2': 19403, '3': 17031, '4': 16013, '5': 15275, '6': 13914, '7': 13362, '8': 11913, '9': 10398, '10': 10754, '11': 10420, '12': 9866, '13': 11818, '14': 17089, '15': 38769}`
  - uniform random over the live slots would give `{'0': 16000.0, '1': 16000.0, '2': 16000.0, '3': 16000.0, '4': 16000.0, '5': 16000.0, '6': 16000.0, '7': 16000.0, '8': 16000.0, '9': 16000.0, '10': 16000.0, '11': 16000.0, '12': 16000.0, '13': 16000.0, '14': 16000.0, '15': 16000.0}` (A1.8 m-2)
- `age` warm: steps 16, evictions 8192, fraction oldest +1.0000, fraction_shifting +0.0000, mean_shift +0.0000; raw Spearman rho(score, age) median -0.6265; partial rho (a diagnostic, never a gate) median -0.5915 IQR [-0.7203, -0.1126], None on 1 steps
  - victim ages `{'16': 8192}`; displacement `{'0': 8192}`
- `age` scored: steps 2984, evictions 1527808, fraction oldest +1.0000, fraction_shifting +0.0000, mean_shift +0.0000; raw Spearman rho(score, age) median -1.0000; partial rho (a diagnostic, never a gate) median -1.0000 IQR [-1.0000, -1.0000], None on 0 steps
  - victim ages `{'16': 1527808}`; displacement `{'0': 1527808}`
- `age` first_scored_block: steps 500, evictions 256000, fraction oldest +1.0000, fraction_shifting +0.0000, mean_shift +0.0000; raw Spearman rho(score, age) median -1.0000; partial rho (a diagnostic, never a gate) median -1.0000 IQR [-1.0000, -1.0000], None on 0 steps
  - victim ages `{'16': 256000}`; displacement `{'0': 256000}`
  - uniform random over the live slots would give `{'0': 16000.0, '1': 16000.0, '2': 16000.0, '3': 16000.0, '4': 16000.0, '5': 16000.0, '6': 16000.0, '7': 16000.0, '8': 16000.0, '9': 16000.0, '10': 16000.0, '11': 16000.0, '12': 16000.0, '13': 16000.0, '14': 16000.0, '15': 16000.0}` (A1.8 m-2)
- Partial rho partials out the slot gestalts only (linear), not `c_t`; no verdict is drawn from any comparison with the spec's E2 value. Leading-edge statistic: not computed (the spec defines a baseline, not a statistic; `leading_edge.py` is a stub). Attention-share Gini: not available.

### Step-160 recency statistics (§7.4) -- DIAGNOSTIC (A4.2): never the headline, never gated

- DIAGNOSTIC step 160 `psiC` warm: steps 16, evictions 8192, fraction oldest +1.0000, mean_shift +0.0000; raw Spearman rho(score, age) median -0.0590
- DIAGNOSTIC step 160 `psiC` scored: steps 144, evictions 73728, fraction oldest +0.0921, mean_shift +7.4217; raw Spearman rho(score, age) median -0.1267
- DIAGNOSTIC step 160 `psiC` first_scored_block: steps 144, evictions 73728, fraction oldest +0.0921, mean_shift +7.4217; raw Spearman rho(score, age) median -0.1267
- DIAGNOSTIC step 160 `age` warm: steps 16, evictions 8192, fraction oldest +1.0000, mean_shift +0.0000; raw Spearman rho(score, age) median -0.6265
- DIAGNOSTIC step 160 `age` scored: steps 144, evictions 73728, fraction oldest +1.0000, mean_shift +0.0000; raw Spearman rho(score, age) median -1.0000
- DIAGNOSTIC step 160 `age` first_scored_block: steps 144, evictions 73728, fraction oldest +1.0000, mean_shift +0.0000; raw Spearman rho(score, age) median -1.0000

### §7.5 reporting (SECONDARY)

- fill fraction, held-out in-loop: +0.8229
- The censored-target caveat, verbatim from the PREREG:

  > The shadow buffer is off (ADR-0009 L6(a)). Every regression target `G_i(t)` is cut at the
  > slot's eviction. Before `T_warm` the cut is FIFO's, at age `M`, so the target itself is
  > age-shaped although ψ̂ has no age input. After `T_warm` the cut is the learned policy's own,
  > and self-confirming (§3.4, §7.2). Spec §3.4's description of pre-`T_warm` data (spec :280)
  > holds for the policy, not for the target. The rescaled `r_i` is described as rescaled, never
  > as anything stronger (§3.2.1).

### T5 (held-out cost, measured in this run; DIAGNOSTIC)

The step-160 DIAGNOSTIC passes (psiC@160, age@160, sz_psiC@160, sz_age@160; A4.2) add about 1,090 s per seed. That figure is EXTRAPOLATED from build-time T5 (one batch per pass, x 128 batches; sz_age assumed equal to sz_psiC), not measured; this run's own per-pass timing follows.

- `psiC`: first batch +1.82 s; projected full EVAL +232.5 s; measured +228.7 s over 128 batches
- `fifo`: first batch +0.91 s; projected full EVAL +115.9 s; measured +115.7 s over 128 batches
- `age`: first batch +1.80 s; projected full EVAL +230.7 s; measured +226.3 s over 128 batches
- `oracle`: first batch +0.92 s; projected full EVAL +117.4 s; measured +116.5 s over 128 batches
- `lru`: first batch +0.94 s; projected full EVAL +120.9 s; measured +122.8 s over 128 batches
- `random0`: first batch +0.89 s; projected full EVAL +113.3 s; measured +115.5 s over 128 batches
- `random1`: first batch +0.87 s; projected full EVAL +111.6 s; measured +115.5 s over 128 batches
- `random2`: first batch +0.89 s; projected full EVAL +114.0 s; measured +116.0 s over 128 batches
- `random3`: first batch +0.89 s; projected full EVAL +113.9 s; measured +115.8 s over 128 batches
- `random4`: first batch +0.78 s; projected full EVAL +100.4 s; measured +115.5 s over 128 batches
- `sz_psiC`: first batch +2.24 s; projected full EVAL +286.4 s; measured +304.6 s over 128 batches
- `sz_fifo`: first batch +0.94 s; projected full EVAL +120.9 s; measured +115.4 s over 128 batches
- `sz_age`: first batch +2.41 s; projected full EVAL +308.1 s; measured +303.5 s over 128 batches
- `psiC@160`: first batch +1.68 s; projected full EVAL +214.6 s; measured +228.2 s over 128 batches
- `age@160`: first batch +1.80 s; projected full EVAL +231.0 s; measured +224.6 s over 128 batches
- `psiC@1000`: first batch +1.79 s; projected full EVAL +229.3 s; measured +226.2 s over 128 batches
- `age@1000`: first batch +1.65 s; projected full EVAL +210.6 s; measured +224.5 s over 128 batches
- `psiC@2000`: first batch +1.72 s; projected full EVAL +220.3 s; measured +226.0 s over 128 batches
- `age@2000`: first batch +1.80 s; projected full EVAL +229.9 s; measured +225.3 s over 128 batches
- `sz_psiC@160`: first batch +2.38 s; projected full EVAL +304.1 s; measured +306.1 s over 128 batches
- `sz_age@160`: first batch +2.24 s; projected full EVAL +286.2 s; measured +305.6 s over 128 batches

## What this does not show

Not E1 and not E2 (ADR-0009 L6). Not evidence on the cognitive claim. No scaling claim. One corpus (synthetic S0-03), `M = 16`, one width, a frozen substrate, the shadow buffer off, `b` off. The rescaled target is reported as rescaled and nothing stronger. Training-time differences are in-sample for phi. Within-seed CIs do not estimate run-to-run or seed-to-seed variation.

The heartbeat's `attribution` is recorded for the RSR arms (RSRPolicy.attribution, feat/rsr-run1-ready c02b16d) and is None for FIFO, which has no terms. With b off and nu = 0 the scored decomposition has a single term, so the attribution only separates FIFO-warmup evictions from scored ones (A1.8 m-6's text predates c02b16d). Per-arm fractions are in the training section (DIAGNOSTIC).

In-loop `loss_lm` values are eval-mode values and are not comparable with arm B's train-mode heartbeat losses (A1.3).

## The author's expectation (PREREG §7.6, verbatim), and the outcome beside it

### 7.6 Author's expectation (before any run-1 number; allowed to be wrong)

These are written for `γ` in `(0, 1)`. If O1 rules `γ = 0`, an amendment restates them before any
run.

- **psiC.** It mostly evicts the oldest slot, though not always:
  - `fraction_shifting` on scored steps is between 0.05 and 0.5;
  - raw ρ(score, age) is clearly negative (below −0.5);
  - partial ρ is small in magnitude (`|ρ| < 0.3`);
  - held-out all-query accuracy is within ±0.01 of fifo, more likely above than below, with the
    CI possibly including 0.

  The reason is the nearest offline analogue, B2's ψ̂-C (the censored target). At γ = 0.9 it was
  EQUIV to FIFO with Δ +0.0035 / +0.0082 / +0.0071 (verified results row 20). Online,
  self-censoring after `T_warm` should push the head further toward recency, not away from it.
- **age.** The table learns `G` decreasing with age, because under FIFO censoring a slot of age
  `a` has at most `M − a` observable terms. So it evicts the oldest slot: `displacement = 0` on at
  least 95% of scored evictions, and accuracy within ±0.005 of fifo.
- **Losses.** Held-out real-token NLL differs across arms by less than 0.01 nats.
- **Controls.** Every control in §8 passes.

The outcome, as numbers in the expectation's terms (reported as text; never scored as met or missed, A1.8 m-10):

- seed 0 `psiC` scored: fraction_shifting +0.9282, fraction oldest +0.0718, raw rho median +0.2008, partial rho median -0.2121
- seed 0 `age` scored: fraction_shifting +0.0000, fraction oldest +1.0000, raw rho median -1.0000, partial rho median -1.0000
- seed 0 `acc.all.psiC_minus_fifo`: -0.2053 [-0.2096, -0.2007]
- seed 0 `acc.all.age_minus_fifo`: +0.0000 [+0.0000, +0.0000] **sd exactly 0: broken, not clean**
- seed 0 `nll_real.psiC_minus_fifo`: +0.1688 [+0.1662, +0.1713]
- seed 0 `nll_real.age_minus_fifo`: +0.0000 [+0.0000, +0.0000] **sd exactly 0: broken, not clean**
- seed 0 `nll_real.psiC_minus_age`: +0.1688 [+0.1662, +0.1713]
- seed 0 controls failed: `[]`
- seed 1 `psiC` scored: fraction_shifting +0.9179, fraction oldest +0.0821, raw rho median +0.1223, partial rho median -0.3132
- seed 1 `age` scored: fraction_shifting +0.0000, fraction oldest +1.0000, raw rho median -1.0000, partial rho median -1.0000
- seed 1 `acc.all.psiC_minus_fifo`: -0.2279 [-0.2323, -0.2236]
- seed 1 `acc.all.age_minus_fifo`: +0.0000 [+0.0000, +0.0000] **sd exactly 0: broken, not clean**
- seed 1 `nll_real.psiC_minus_fifo`: +0.2085 [+0.2061, +0.2109]
- seed 1 `nll_real.age_minus_fifo`: +0.0000 [+0.0000, +0.0000] **sd exactly 0: broken, not clean**
- seed 1 `nll_real.psiC_minus_age`: +0.2085 [+0.2061, +0.2109]
- seed 1 controls failed: `[]`
- seed 2 `psiC` scored: fraction_shifting +0.9426, fraction oldest +0.0574, raw rho median +0.0841, partial rho median -0.2083
- seed 2 `age` scored: fraction_shifting +0.0000, fraction oldest +1.0000, raw rho median -1.0000, partial rho median -1.0000
- seed 2 `acc.all.psiC_minus_fifo`: -0.1683 [-0.1726, -0.1639]
- seed 2 `acc.all.age_minus_fifo`: +0.0000 [+0.0000, +0.0000] **sd exactly 0: broken, not clean**
- seed 2 `nll_real.psiC_minus_fifo`: +0.1733 [+0.1704, +0.1760]
- seed 2 `nll_real.age_minus_fifo`: +0.0000 [+0.0000, +0.0000] **sd exactly 0: broken, not clean**
- seed 2 `nll_real.psiC_minus_age`: +0.1733 [+0.1704, +0.1760]
- seed 2 controls failed: `[]`

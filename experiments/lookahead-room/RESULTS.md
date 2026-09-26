# RESULTS — lookahead room (W10)

Rendered from `runs/lookahead-room/ledger.json` by `experiments/lookahead-room/run.py --render-results`. PREREG `experiments/lookahead-room/PREREG.md` (commit `527daf4`). Run sha `95a96fc49e0d6c239aab8e6515a7bacf4368ee11`, platform `macOS-26.6.2-arm64-arm-64bit`, device `cpu`.

Command: `uv run python experiments/lookahead-room/run.py` (children under `commands` in the ledger).

**classification: INCONCLUSIVE** (headline B.ckpt3000); verdict outcome `inconclusive`.

Per checkpoint (`per_checkpoint`): `B.ckpt2500` PARTIAL, `B.ckpt3000` PARTIAL

Controls failed (`controls_failed`): ['C1 failed on B.ckpt2500.seed0: {"ok": false, "max_abs_acc_diff": 2.8209831914871586e-08, "max_abs_nll_diff": 1.4570024275961657e-07, "out_of_tolerance": ["B.ckpt2500.heldout.live.all.answer_acc", "B.ckpt2500.heldout.live.gap_eq_1.answer_acc", "B.ckpt2500.heldout.live.gap_ge_2.answer_acc", "B.ckpt2500.heldout.live.gap_2_to_M.answer_acc", "B.ckpt2500.heldout.live.gap_lt_M.answer_acc", "B.ckpt2500.heldout.live.gap_eq_M.answer_acc"', 'C1 failed on B.ckpt2500.seed1: {"ok": false, "max_abs_acc_diff": 2.3610881916269477e-08, "max_abs_nll_diff": 5.527916882019923e-07, "out_of_tolerance": ["B.ckpt2500.heldout.live.all.answer_acc", "B.ckpt2500.heldout.live.gap_eq_1.answer_acc", "B.ckpt2500.heldout.live.gap_ge_2.answer_acc", "B.ckpt2500.heldout.live.gap_2_to_M.answer_acc", "B.ckpt2500.heldout.live.gap_lt_M.answer_acc", "B.ckpt2500.heldout.live.gap_eq_M.answer_acc",', 'C1 failed on B.ckpt2500.seed2: {"ok": false, "max_abs_acc_diff": 2.9394071399124755e-08, "max_abs_nll_diff": 7.40675365484833e-08, "out_of_tolerance": ["B.ckpt2500.heldout.live.all.answer_acc", "B.ckpt2500.heldout.live.gap_eq_1.answer_acc", "B.ckpt2500.heldout.live.gap_ge_2.answer_acc", "B.ckpt2500.heldout.live.gap_2_to_M.answer_acc", "B.ckpt2500.heldout.live.gap_lt_M.answer_acc", "B.ckpt2500.heldout.live.gap_eq_M.answer_acc", ', 'C1 failed on B.ckpt3000.seed0: {"ok": false, "max_abs_acc_diff": 2.948861377305434e-08, "max_abs_nll_diff": 2.518710162657811e-07, "out_of_tolerance": ["B.ckpt3000.heldout.live.all.answer_acc", "B.ckpt3000.heldout.live.gap_eq_1.answer_acc", "B.ckpt3000.heldout.live.gap_ge_2.answer_acc", "B.ckpt3000.heldout.live.gap_2_to_M.answer_acc", "B.ckpt3000.heldout.live.gap_lt_M.answer_acc", "B.ckpt3000.heldout.live.gap_eq_M.answer_acc", ', 'C1 failed on B.ckpt3000.seed1: {"ok": false, "max_abs_acc_diff": 1.931784887609922e-08, "max_abs_nll_diff": 1.374218199234889e-07, "out_of_tolerance": ["B.ckpt3000.heldout.live.all.answer_acc", "B.ckpt3000.heldout.live.gap_eq_1.answer_acc", "B.ckpt3000.heldout.live.gap_ge_2.answer_acc", "B.ckpt3000.heldout.live.gap_2_to_M.answer_acc", "B.ckpt3000.heldout.live.gap_lt_M.answer_acc", "B.ckpt3000.heldout.live.gap_eq_M.answer_acc", ', 'C1 failed on B.ckpt3000.seed2: {"ok": false, "max_abs_acc_diff": 2.2225460805103125e-08, "max_abs_nll_diff": 2.919970189807497e-07, "out_of_tolerance": ["B.ckpt3000.heldout.live.all.answer_acc", "B.ckpt3000.heldout.live.gap_eq_1.answer_acc", "B.ckpt3000.heldout.live.gap_ge_2.answer_acc", "B.ckpt3000.heldout.live.gap_2_to_M.answer_acc", "B.ckpt3000.heldout.live.gap_lt_M.answer_acc", "B.ckpt3000.heldout.live.gap_le_M.answer_acc",']

🔴 The rule hit rates are an upper-bound-style proxy from FIFO-world targets (demand-if-resident probes), not RSR's result; the discounted-return targets are hindsight. See PREREG.

## B.ckpt2500

Differences, set U (point [paired per-document percentile CI]):

| key | seed0 | seed1 | seed2 |
|---|---|---|---|
| `B.ckpt2500.U.room_3b` | 0.0205 [0.0156, 0.0254] | 0.0367 [0.0327, 0.0410] | 0.0215 [0.0169, 0.0260] |
| `B.ckpt2500.U.room_3b_097` | -0.0376 [-0.0433, -0.0318] | -0.0030 [-0.0079, 0.0020] | -0.0186 [-0.0241, -0.0132] |
| `B.ckpt2500.U.oracle_minus_fifo` | 0.1941 [0.1888, 0.1994] | 0.1907 [0.1855, 0.1963] | 0.1912 [0.1859, 0.1967] |
| `B.ckpt2500.U.factfiller_minus_fifo` | 0.1595 [0.1542, 0.1651] | 0.1560 [0.1508, 0.1617] | 0.1578 [0.1524, 0.1634] |
| `B.ckpt2500.U.pending_fifo_minus_fifo` | 0.1941 [0.1888, 0.1994] | 0.1907 [0.1855, 0.1963] | 0.1912 [0.1859, 0.1967] |
| `B.ckpt2500.U.rule_g0_minus_fifo` | -0.0031 [-0.0085, 0.0021] | -0.0324 [-0.0367, -0.0283] | 0.0345 [0.0297, 0.0394] |
| `B.ckpt2500.U.rule_g09_minus_fifo` | 0.0174 [0.0107, 0.0243] | 0.0043 [-0.0010, 0.0098] | 0.0560 [0.0501, 0.0619] |
| `B.ckpt2500.U.online_g0_minus_rule_g0` | -0.0026 [-0.0062, 0.0011] | 0.0131 [0.0102, 0.0160] | 0.0104 [0.0070, 0.0141] |
| `B.ckpt2500.U.lit_room_3b` | 0.0051 [0.0031, 0.0071] | -0.0003 [-0.0021, 0.0015] | -0.0032 [-0.0046, -0.0018] |
| `B.ckpt2500.U.next_minus_g0` | 0.0155 [0.0135, 0.0175] | 0.0151 [0.0133, 0.0169] | 0.0106 [0.0085, 0.0127] |

Hit rates (set U):

| key | seed0 | seed1 | seed2 |
|---|---|---|---|
| `B.ckpt2500.U.hit.fifo` | 0.8059 | 0.8093 | 0.8088 |
| `B.ckpt2500.U.hit.fifo.gap_gt_M` | 0.0000 | 0.0000 | 0.0000 |
| `B.ckpt2500.U.hit.oracle` | 1.0000 | 1.0000 | 1.0000 |
| `B.ckpt2500.U.hit.oracle.gap_gt_M` | 1.0000 | 1.0000 | 1.0000 |
| `B.ckpt2500.U.hit.pending_fifo` | 1.0000 | 1.0000 | 1.0000 |
| `B.ckpt2500.U.hit.pending_fifo.gap_gt_M` | 1.0000 | 1.0000 | 1.0000 |
| `B.ckpt2500.U.hit.factfiller` | 0.9655 | 0.9653 | 0.9666 |
| `B.ckpt2500.U.hit.factfiller.gap_gt_M` | 0.9027 | 0.9009 | 0.9020 |
| `B.ckpt2500.U.hit.rule_g0` | 0.8028 | 0.7768 | 0.8433 |
| `B.ckpt2500.U.hit.rule_g0.gap_gt_M` | 0.3734 | 0.1550 | 0.4403 |
| `B.ckpt2500.U.hit.rule_g09` | 0.8233 | 0.8136 | 0.8648 |
| `B.ckpt2500.U.hit.rule_g09.gap_gt_M` | 0.6172 | 0.3638 | 0.6929 |
| `B.ckpt2500.U.hit.rule_g097` | 0.7652 | 0.7739 | 0.8247 |
| `B.ckpt2500.U.hit.rule_g097.gap_gt_M` | 0.6310 | 0.3521 | 0.6835 |
| `B.ckpt2500.U.hit.rule_next` | 0.8182 | 0.7919 | 0.8539 |
| `B.ckpt2500.U.hit.rule_next.gap_gt_M` | 0.4092 | 0.2034 | 0.4791 |
| `B.ckpt2500.U.hit.lit_g0` | 0.8032 | 0.8112 | 0.8154 |
| `B.ckpt2500.U.hit.lit_g0.gap_gt_M` | 0.0525 | 0.0533 | 0.0550 |
| `B.ckpt2500.U.hit.lit_g09` | 0.8083 | 0.8109 | 0.8121 |
| `B.ckpt2500.U.hit.lit_g09.gap_gt_M` | 0.0128 | 0.0112 | 0.0175 |
| `B.ckpt2500.U.hit.lit_g097` | 0.8079 | 0.8106 | 0.8117 |
| `B.ckpt2500.U.hit.lit_g097.gap_gt_M` | 0.0108 | 0.0094 | 0.0154 |
| `B.ckpt2500.U.hit.online_g0` | 0.8002 | 0.7900 | 0.8537 |
| `B.ckpt2500.U.hit.online_g0.gap_gt_M` | 0.2589 | 0.1365 | 0.4048 |

Hit rates (set P):

| key | seed0 | seed1 | seed2 |
|---|---|---|---|
| `B.ckpt2500.P.hit.fifo` | 0.8148 | 0.8261 | 0.8104 |
| `B.ckpt2500.P.hit.oracle` | 1.0000 | 1.0000 | 1.0000 |
| `B.ckpt2500.P.hit.pending_fifo` | 1.0000 | 1.0000 | 1.0000 |
| `B.ckpt2500.P.hit.factfiller` | 0.9579 | 0.9655 | 0.9639 |
| `B.ckpt2500.P.hit.rule_g0` | 0.8199 | 0.7874 | 0.8398 |
| `B.ckpt2500.P.hit.rule_g09` | 0.8140 | 0.8336 | 0.8482 |
| `B.ckpt2500.P.hit.rule_g097` | 0.7584 | 0.7798 | 0.8037 |
| `B.ckpt2500.P.hit.rule_next` | 0.8308 | 0.8034 | 0.8532 |
| `B.ckpt2500.P.hit.lit_g0` | 0.8114 | 0.8286 | 0.8129 |
| `B.ckpt2500.P.hit.lit_g09` | 0.8190 | 0.8286 | 0.8138 |
| `B.ckpt2500.P.hit.lit_g097` | 0.8173 | 0.8286 | 0.8121 |
| `B.ckpt2500.P.hit.online_g0` | 0.8081 | 0.7958 | 0.8406 |

Live answer accuracy, FIFO vs online_g0 (set U):

| key | seed0 | seed1 | seed2 |
|---|---|---|---|
| `B.ckpt2500.U.acc_fifo` | 0.7664 | 0.7671 | 0.7794 |
| `B.ckpt2500.U.acc_online_g0` | 0.7585 | 0.7554 | 0.8022 |

r_i (= D) by class, FIFO-resident slots at full-memory steps:

| key | seed0 | seed1 | seed2 |
|---|---|---|---|
| `B.ckpt2500.class.res.pending.D.n` | 68789.0000 | 69293.0000 | 68208.0000 |
| `B.ckpt2500.class.res.pending.D.mean` | 0.0703 | 0.0674 | 0.0789 |
| `B.ckpt2500.class.res.pending.D.sd` | 0.0357 | 0.0396 | 0.0426 |
| `B.ckpt2500.class.res.pending.D.q10` | 0.0321 | 0.0254 | 0.0349 |
| `B.ckpt2500.class.res.pending.D.q50` | 0.0635 | 0.0589 | 0.0694 |
| `B.ckpt2500.class.res.pending.D.q90` | 0.1167 | 0.1207 | 0.1356 |
| `B.ckpt2500.class.res.pending.G09.n` | 68789.0000 | 69293.0000 | 68208.0000 |
| `B.ckpt2500.class.res.pending.G09.mean` | 0.6719 | 0.5237 | 0.6933 |
| `B.ckpt2500.class.res.pending.G09.sd` | 0.1682 | 0.1419 | 0.1934 |
| `B.ckpt2500.class.res.pending.G09.q10` | 0.4856 | 0.3621 | 0.4795 |
| `B.ckpt2500.class.res.pending.G09.q50` | 0.6509 | 0.5047 | 0.6699 |
| `B.ckpt2500.class.res.pending.G09.q90` | 0.8946 | 0.7123 | 0.9435 |
| `B.ckpt2500.class.res.pending.G097.n` | 68789.0000 | 69293.0000 | 68208.0000 |
| `B.ckpt2500.class.res.pending.G097.mean` | 1.2962 | 0.9357 | 1.3099 |
| `B.ckpt2500.class.res.pending.G097.sd` | 0.4204 | 0.2838 | 0.4421 |
| `B.ckpt2500.class.res.pending.G097.q10` | 0.7412 | 0.5800 | 0.7301 |
| `B.ckpt2500.class.res.pending.G097.q50` | 1.2936 | 0.9280 | 1.3062 |
| `B.ckpt2500.class.res.pending.G097.q90` | 1.8309 | 1.2958 | 1.8596 |
| `B.ckpt2500.class.res.querying.D.n` | 10838.0000 | 10866.0000 | 10762.0000 |
| `B.ckpt2500.class.res.querying.D.mean` | 0.2076 | 0.1925 | 0.2114 |
| `B.ckpt2500.class.res.querying.D.sd` | 0.0596 | 0.0590 | 0.0711 |
| `B.ckpt2500.class.res.querying.D.q10` | 0.1248 | 0.1108 | 0.1208 |
| `B.ckpt2500.class.res.querying.D.q50` | 0.2116 | 0.1990 | 0.2083 |
| `B.ckpt2500.class.res.querying.D.q90` | 0.2815 | 0.2641 | 0.3042 |
| `B.ckpt2500.class.res.querying.G09.n` | 10838.0000 | 10866.0000 | 10762.0000 |
| `B.ckpt2500.class.res.querying.G09.mean` | 0.6458 | 0.5529 | 0.6582 |
| `B.ckpt2500.class.res.querying.G09.sd` | 0.1944 | 0.1634 | 0.2213 |
| `B.ckpt2500.class.res.querying.G09.q10` | 0.3778 | 0.3378 | 0.3663 |
| `B.ckpt2500.class.res.querying.G09.q50` | 0.6494 | 0.5576 | 0.6553 |
| `B.ckpt2500.class.res.querying.G09.q90` | 0.8871 | 0.7559 | 0.9423 |
| `B.ckpt2500.class.res.querying.G097.n` | 10838.0000 | 10866.0000 | 10762.0000 |
| `B.ckpt2500.class.res.querying.G097.mean` | 1.0630 | 0.8422 | 1.0824 |
| `B.ckpt2500.class.res.querying.G097.sd` | 0.4686 | 0.3325 | 0.4998 |
| `B.ckpt2500.class.res.querying.G097.q10` | 0.4172 | 0.3798 | 0.4054 |
| `B.ckpt2500.class.res.querying.G097.q50` | 1.0831 | 0.8583 | 1.0899 |
| `B.ckpt2500.class.res.querying.G097.q90` | 1.6565 | 1.2526 | 1.7216 |
| `B.ckpt2500.class.res.answered.D.n` | 132276.0000 | 133705.0000 | 133399.0000 |
| `B.ckpt2500.class.res.answered.D.mean` | 0.0596 | 0.0513 | 0.0630 |
| `B.ckpt2500.class.res.answered.D.sd` | 0.0279 | 0.0289 | 0.0305 |
| `B.ckpt2500.class.res.answered.D.q10` | 0.0298 | 0.0229 | 0.0307 |
| `B.ckpt2500.class.res.answered.D.q50` | 0.0541 | 0.0438 | 0.0566 |
| `B.ckpt2500.class.res.answered.D.q90` | 0.0969 | 0.0902 | 0.1037 |
| `B.ckpt2500.class.res.answered.G09.n` | 132276.0000 | 133705.0000 | 133399.0000 |
| `B.ckpt2500.class.res.answered.G09.mean` | 0.5334 | 0.3821 | 0.5465 |
| `B.ckpt2500.class.res.answered.G09.sd` | 0.2080 | 0.1430 | 0.2192 |
| `B.ckpt2500.class.res.answered.G09.q10` | 0.2330 | 0.1793 | 0.2304 |
| `B.ckpt2500.class.res.answered.G09.q50` | 0.5451 | 0.3915 | 0.5604 |
| `B.ckpt2500.class.res.answered.G09.q90` | 0.7832 | 0.5532 | 0.8078 |
| `B.ckpt2500.class.res.answered.G097.n` | 132276.0000 | 133705.0000 | 133399.0000 |
| `B.ckpt2500.class.res.answered.G097.mean` | 0.9955 | 0.6770 | 1.0076 |
| `B.ckpt2500.class.res.answered.G097.sd` | 0.5093 | 0.3315 | 0.5149 |
| `B.ckpt2500.class.res.answered.G097.q10` | 0.2664 | 0.2044 | 0.2659 |
| `B.ckpt2500.class.res.answered.G097.q50` | 1.0297 | 0.7003 | 1.0491 |
| `B.ckpt2500.class.res.answered.G097.q90` | 1.6314 | 1.0888 | 1.6386 |
| `B.ckpt2500.class.res.filler.D.n` | 345153.0000 | 343192.0000 | 344687.0000 |
| `B.ckpt2500.class.res.filler.D.mean` | 0.0575 | 0.0618 | 0.0544 |
| `B.ckpt2500.class.res.filler.D.sd` | 0.0224 | 0.0330 | 0.0290 |
| `B.ckpt2500.class.res.filler.D.q10` | 0.0351 | 0.0278 | 0.0243 |
| `B.ckpt2500.class.res.filler.D.q50` | 0.0521 | 0.0539 | 0.0480 |
| `B.ckpt2500.class.res.filler.D.q90` | 0.0870 | 0.1070 | 0.0929 |
| `B.ckpt2500.class.res.filler.G09.n` | 345153.0000 | 343192.0000 | 344687.0000 |
| `B.ckpt2500.class.res.filler.G09.mean` | 0.4896 | 0.4447 | 0.4193 |
| `B.ckpt2500.class.res.filler.G09.sd` | 0.2486 | 0.2476 | 0.2228 |
| `B.ckpt2500.class.res.filler.G09.q10` | 0.1417 | 0.1295 | 0.1265 |
| `B.ckpt2500.class.res.filler.G09.q50` | 0.5003 | 0.4088 | 0.3907 |
| `B.ckpt2500.class.res.filler.G09.q90` | 0.8064 | 0.7828 | 0.7113 |
| `B.ckpt2500.class.res.filler.G097.n` | 345153.0000 | 343192.0000 | 344687.0000 |
| `B.ckpt2500.class.res.filler.G097.mean` | 0.8913 | 0.7852 | 0.7392 |
| `B.ckpt2500.class.res.filler.G097.sd` | 0.5830 | 0.5477 | 0.4911 |
| `B.ckpt2500.class.res.filler.G097.q10` | 0.1547 | 0.1428 | 0.1395 |
| `B.ckpt2500.class.res.filler.G097.q50` | 0.8547 | 0.6797 | 0.6565 |
| `B.ckpt2500.class.res.filler.G097.q90` | 1.6926 | 1.6065 | 1.4459 |
| `B.ckpt2500.class.probed.pending.D.n` | 36800.0000 | 35856.0000 | 36032.0000 |
| `B.ckpt2500.class.probed.pending.D.mean` | 0.0864 | 0.0519 | 0.0801 |
| `B.ckpt2500.class.probed.pending.D.sd` | 0.0331 | 0.0305 | 0.0360 |
| `B.ckpt2500.class.probed.pending.D.q10` | 0.0497 | 0.0235 | 0.0404 |
| `B.ckpt2500.class.probed.pending.D.q50` | 0.0807 | 0.0432 | 0.0742 |
| `B.ckpt2500.class.probed.pending.D.q90` | 0.1302 | 0.0924 | 0.1258 |
| `B.ckpt2500.class.probed.querying.D.n` | 3905.0000 | 3845.0000 | 3836.0000 |
| `B.ckpt2500.class.probed.querying.D.mean` | 0.1831 | 0.1294 | 0.1746 |
| `B.ckpt2500.class.probed.querying.D.sd` | 0.0480 | 0.0409 | 0.0550 |
| `B.ckpt2500.class.probed.querying.D.q10` | 0.1195 | 0.0712 | 0.1078 |
| `B.ckpt2500.class.probed.querying.D.q50` | 0.1842 | 0.1327 | 0.1697 |
| `B.ckpt2500.class.probed.querying.D.q90` | 0.2452 | 0.1788 | 0.2469 |
| `B.ckpt2500.class.probed.answered.D.n` | 261023.0000 | 261544.0000 | 261871.0000 |
| `B.ckpt2500.class.probed.answered.D.mean` | 0.0920 | 0.0545 | 0.0857 |
| `B.ckpt2500.class.probed.answered.D.sd` | 0.0360 | 0.0310 | 0.0381 |
| `B.ckpt2500.class.probed.answered.D.q10` | 0.0524 | 0.0247 | 0.0427 |
| `B.ckpt2500.class.probed.answered.D.q50` | 0.0855 | 0.0458 | 0.0798 |
| `B.ckpt2500.class.probed.answered.D.q90` | 0.1408 | 0.0964 | 0.1346 |
| `B.ckpt2500.class.probed.filler.D.n` | 237920.0000 | 238403.0000 | 237909.0000 |
| `B.ckpt2500.class.probed.filler.D.mean` | 0.1037 | 0.0864 | 0.0789 |
| `B.ckpt2500.class.probed.filler.D.sd` | 0.0343 | 0.0357 | 0.0366 |
| `B.ckpt2500.class.probed.filler.D.q10` | 0.0649 | 0.0449 | 0.0362 |
| `B.ckpt2500.class.probed.filler.D.q50` | 0.0981 | 0.0816 | 0.0738 |
| `B.ckpt2500.class.probed.filler.D.q90` | 0.1505 | 0.1343 | 0.1253 |

Pending asserts by k (steps to query; the last bin pools every k above M):

| key | seed0 | seed1 | seed2 |
|---|---|---|---|
| `B.ckpt2500.pend_k.1.mean` | 0.0825 | 0.0836 | 0.0940 |
| `B.ckpt2500.pend_k.1.n` | 8220.0000 | 8231.0000 | 8137.0000 |
| `B.ckpt2500.pend_k.1.q50` | 0.0758 | 0.0774 | 0.0850 |
| `B.ckpt2500.pend_k.2.mean` | 0.0797 | 0.0787 | 0.0917 |
| `B.ckpt2500.pend_k.2.n` | 6297.0000 | 6336.0000 | 6271.0000 |
| `B.ckpt2500.pend_k.2.q50` | 0.0728 | 0.0726 | 0.0826 |
| `B.ckpt2500.pend_k.3.mean` | 0.0763 | 0.0757 | 0.0865 |
| `B.ckpt2500.pend_k.3.n` | 4940.0000 | 4999.0000 | 4949.0000 |
| `B.ckpt2500.pend_k.3.q50` | 0.0691 | 0.0685 | 0.0766 |
| `B.ckpt2500.pend_k.4.mean` | 0.0723 | 0.0718 | 0.0839 |
| `B.ckpt2500.pend_k.4.n` | 4028.0000 | 4148.0000 | 4019.0000 |
| `B.ckpt2500.pend_k.4.q50` | 0.0655 | 0.0628 | 0.0740 |
| `B.ckpt2500.pend_k.5.mean` | 0.0706 | 0.0681 | 0.0790 |
| `B.ckpt2500.pend_k.5.n` | 3429.0000 | 3576.0000 | 3448.0000 |
| `B.ckpt2500.pend_k.5.q50` | 0.0633 | 0.0585 | 0.0696 |
| `B.ckpt2500.pend_k.6.mean` | 0.0682 | 0.0636 | 0.0762 |
| `B.ckpt2500.pend_k.6.n` | 3130.0000 | 3154.0000 | 3072.0000 |
| `B.ckpt2500.pend_k.6.q50` | 0.0625 | 0.0550 | 0.0666 |
| `B.ckpt2500.pend_k.7.mean` | 0.0664 | 0.0617 | 0.0732 |
| `B.ckpt2500.pend_k.7.n` | 2867.0000 | 2921.0000 | 2816.0000 |
| `B.ckpt2500.pend_k.7.q50` | 0.0593 | 0.0529 | 0.0648 |
| `B.ckpt2500.pend_k.8.mean` | 0.0653 | 0.0619 | 0.0717 |
| `B.ckpt2500.pend_k.8.n` | 2720.0000 | 2748.0000 | 2668.0000 |
| `B.ckpt2500.pend_k.8.q50` | 0.0582 | 0.0531 | 0.0627 |
| `B.ckpt2500.pend_k.9.mean` | 0.0660 | 0.0619 | 0.0726 |
| `B.ckpt2500.pend_k.9.n` | 2632.0000 | 2639.0000 | 2591.0000 |
| `B.ckpt2500.pend_k.9.q50` | 0.0585 | 0.0538 | 0.0637 |
| `B.ckpt2500.pend_k.10.mean` | 0.0657 | 0.0634 | 0.0736 |
| `B.ckpt2500.pend_k.10.n` | 2605.0000 | 2603.0000 | 2553.0000 |
| `B.ckpt2500.pend_k.10.q50` | 0.0591 | 0.0545 | 0.0657 |
| `B.ckpt2500.pend_k.11.mean` | 0.0664 | 0.0630 | 0.0737 |
| `B.ckpt2500.pend_k.11.n` | 2578.0000 | 2565.0000 | 2527.0000 |
| `B.ckpt2500.pend_k.11.q50` | 0.0600 | 0.0545 | 0.0649 |
| `B.ckpt2500.pend_k.12.mean` | 0.0670 | 0.0633 | 0.0737 |
| `B.ckpt2500.pend_k.12.n` | 2450.0000 | 2410.0000 | 2417.0000 |
| `B.ckpt2500.pend_k.12.q50` | 0.0613 | 0.0552 | 0.0643 |
| `B.ckpt2500.pend_k.13.mean` | 0.0667 | 0.0613 | 0.0735 |
| `B.ckpt2500.pend_k.13.n` | 2310.0000 | 2241.0000 | 2289.0000 |
| `B.ckpt2500.pend_k.13.q50` | 0.0605 | 0.0523 | 0.0648 |
| `B.ckpt2500.pend_k.14.mean` | 0.0658 | 0.0619 | 0.0726 |
| `B.ckpt2500.pend_k.14.n` | 2174.0000 | 2144.0000 | 2171.0000 |
| `B.ckpt2500.pend_k.14.q50` | 0.0597 | 0.0529 | 0.0642 |
| `B.ckpt2500.pend_k.15.mean` | 0.0662 | 0.0603 | 0.0737 |
| `B.ckpt2500.pend_k.15.n` | 2060.0000 | 2040.0000 | 2062.0000 |
| `B.ckpt2500.pend_k.15.q50` | 0.0598 | 0.0518 | 0.0652 |
| `B.ckpt2500.pend_k.16.mean` | 0.0660 | 0.0602 | 0.0744 |
| `B.ckpt2500.pend_k.16.n` | 1939.0000 | 1922.0000 | 1937.0000 |
| `B.ckpt2500.pend_k.16.q50` | 0.0600 | 0.0510 | 0.0653 |
| `B.ckpt2500.pend_k.17.mean` | 0.0640 | 0.0591 | 0.0706 |
| `B.ckpt2500.pend_k.17.n` | 14410.0000 | 14616.0000 | 14281.0000 |
| `B.ckpt2500.pend_k.17.q50` | 0.0581 | 0.0510 | 0.0622 |

AUCs (mean over full-memory steps):

| key | seed0 | seed1 | seed2 |
|---|---|---|---|
| `B.ckpt2500.auc.all.pend_vs_ans.rule_g0` | 0.5323 | 0.6047 | 0.5656 |
| `B.ckpt2500.auc.all.pend_vs_ans.rule_g0.n_steps` | 31131.0000 | 31117.0000 | 31076.0000 |
| `B.ckpt2500.auc.all.pend_vs_ans.rule_g09` | 0.5746 | 0.6898 | 0.5997 |
| `B.ckpt2500.auc.all.pend_vs_ans.rule_g09.n_steps` | 31131.0000 | 31117.0000 | 31076.0000 |
| `B.ckpt2500.auc.all.pend_vs_ans.rule_g097` | 0.5591 | 0.6734 | 0.5810 |
| `B.ckpt2500.auc.all.pend_vs_ans.rule_g097.n_steps` | 31131.0000 | 31117.0000 | 31076.0000 |
| `B.ckpt2500.auc.all.pend_vs_ans.rule_next` | 0.5821 | 0.6397 | 0.6044 |
| `B.ckpt2500.auc.all.pend_vs_ans.rule_next.n_steps` | 31131.0000 | 31117.0000 | 31076.0000 |
| `B.ckpt2500.auc.all.pend_vs_fill.rule_g0` | 0.5621 | 0.4457 | 0.6670 |
| `B.ckpt2500.auc.all.pend_vs_fill.rule_g0.n_steps` | 31131.0000 | 31117.0000 | 31076.0000 |
| `B.ckpt2500.auc.all.pend_vs_fill.rule_g09` | 0.5767 | 0.3981 | 0.7315 |
| `B.ckpt2500.auc.all.pend_vs_fill.rule_g09.n_steps` | 31131.0000 | 31117.0000 | 31076.0000 |
| `B.ckpt2500.auc.all.pend_vs_fill.rule_g097` | 0.5397 | 0.3583 | 0.7057 |
| `B.ckpt2500.auc.all.pend_vs_fill.rule_g097.n_steps` | 31131.0000 | 31117.0000 | 31076.0000 |
| `B.ckpt2500.auc.all.pend_vs_fill.rule_next` | 0.6123 | 0.5016 | 0.6992 |
| `B.ckpt2500.auc.all.pend_vs_fill.rule_next.n_steps` | 31131.0000 | 31117.0000 | 31076.0000 |
| `B.ckpt2500.auc.res.pend_vs_ans.rule_g0` | 0.6299 | 0.6559 | 0.6392 |
| `B.ckpt2500.auc.res.pend_vs_ans.rule_g0.n_steps` | 27031.0000 | 27224.0000 | 27071.0000 |
| `B.ckpt2500.auc.res.pend_vs_ans.rule_g09` | 0.6617 | 0.7351 | 0.6461 |
| `B.ckpt2500.auc.res.pend_vs_ans.rule_g09.n_steps` | 27031.0000 | 27224.0000 | 27071.0000 |
| `B.ckpt2500.auc.res.pend_vs_ans.rule_g097` | 0.6276 | 0.7120 | 0.6154 |
| `B.ckpt2500.auc.res.pend_vs_ans.rule_g097.n_steps` | 27031.0000 | 27224.0000 | 27071.0000 |
| `B.ckpt2500.auc.res.pend_vs_ans.rule_next` | 0.6573 | 0.6804 | 0.6608 |
| `B.ckpt2500.auc.res.pend_vs_ans.rule_next.n_steps` | 27031.0000 | 27224.0000 | 27071.0000 |
| `B.ckpt2500.auc.res.pend_vs_fill.rule_g0` | 0.6268 | 0.5584 | 0.7154 |
| `B.ckpt2500.auc.res.pend_vs_fill.rule_g0.n_steps` | 27079.0000 | 27268.0000 | 27124.0000 |
| `B.ckpt2500.auc.res.pend_vs_fill.rule_g09` | 0.6358 | 0.4979 | 0.7732 |
| `B.ckpt2500.auc.res.pend_vs_fill.rule_g09.n_steps` | 27079.0000 | 27268.0000 | 27124.0000 |
| `B.ckpt2500.auc.res.pend_vs_fill.rule_g097` | 0.5820 | 0.4423 | 0.7425 |
| `B.ckpt2500.auc.res.pend_vs_fill.rule_g097.n_steps` | 27079.0000 | 27268.0000 | 27124.0000 |
| `B.ckpt2500.auc.res.pend_vs_fill.rule_next` | 0.6594 | 0.5929 | 0.7365 |
| `B.ckpt2500.auc.res.pend_vs_fill.rule_next.n_steps` | 27079.0000 | 27268.0000 | 27124.0000 |

Controls:

| key | seed0 | seed1 | seed2 |
|---|---|---|---|
| `B.ckpt2500.C1.max_abs_acc_diff` | 2.8209831914871586e-08 | 2.3610881916269477e-08 | 2.9394071399124755e-08 |
| `B.ckpt2500.C1.max_abs_nll_diff` | 1.4570024275961657e-07 | 5.527916882019923e-07 | 7.40675365484833e-08 |
| `B.ckpt2500.C3_sum_worst` | 2.384185791015625e-07 | 2.384185791015625e-07 | 2.384185791015625e-07 |
| `B.ckpt2500.C4_identity_worst` | 4.470348358154297e-08 | 4.470348358154297e-08 | 5.960464477539063e-08 |
| `B.ckpt2500.seconds` | 1153.8777980804443 | 1148.754497051239 | 1152.0943002700806 |

## B.ckpt3000

Differences, set U (point [paired per-document percentile CI]):

| key | seed0 | seed1 | seed2 |
|---|---|---|---|
| `B.ckpt3000.U.room_3b` | 0.0355 [0.0304, 0.0405] | 0.0420 [0.0376, 0.0463] | 0.0478 [0.0431, 0.0526] |
| `B.ckpt3000.U.room_3b_097` | -0.0249 [-0.0311, -0.0189] | -0.0012 [-0.0064, 0.0040] | 0.0224 [0.0171, 0.0281] |
| `B.ckpt3000.U.oracle_minus_fifo` | 0.1941 [0.1888, 0.1994] | 0.1907 [0.1855, 0.1963] | 0.1912 [0.1859, 0.1967] |
| `B.ckpt3000.U.factfiller_minus_fifo` | 0.1595 [0.1542, 0.1651] | 0.1560 [0.1508, 0.1617] | 0.1578 [0.1524, 0.1634] |
| `B.ckpt3000.U.pending_fifo_minus_fifo` | 0.1941 [0.1888, 0.1994] | 0.1907 [0.1855, 0.1963] | 0.1912 [0.1859, 0.1967] |
| `B.ckpt3000.U.rule_g0_minus_fifo` | -0.0062 [-0.0114, -0.0011] | -0.0212 [-0.0253, -0.0172] | 0.0369 [0.0319, 0.0418] |
| `B.ckpt3000.U.rule_g09_minus_fifo` | 0.0293 [0.0230, 0.0361] | 0.0208 [0.0158, 0.0261] | 0.0847 [0.0791, 0.0905] |
| `B.ckpt3000.U.online_g0_minus_rule_g0` | -0.0049 [-0.0086, -0.0015] | 0.0054 [0.0024, 0.0083] | 0.0034 [-0.0001, 0.0068] |
| `B.ckpt3000.U.lit_room_3b` | 0.0045 [0.0025, 0.0064] | 0.0022 [0.0003, 0.0041] | -0.0015 [-0.0031, 0.0000] |
| `B.ckpt3000.U.next_minus_g0` | 0.0142 [0.0122, 0.0163] | 0.0107 [0.0092, 0.0121] | 0.0078 [0.0059, 0.0097] |

Hit rates (set U):

| key | seed0 | seed1 | seed2 |
|---|---|---|---|
| `B.ckpt3000.U.hit.fifo` | 0.8059 | 0.8093 | 0.8088 |
| `B.ckpt3000.U.hit.fifo.gap_gt_M` | 0.0000 | 0.0000 | 0.0000 |
| `B.ckpt3000.U.hit.oracle` | 1.0000 | 1.0000 | 1.0000 |
| `B.ckpt3000.U.hit.oracle.gap_gt_M` | 1.0000 | 1.0000 | 1.0000 |
| `B.ckpt3000.U.hit.pending_fifo` | 1.0000 | 1.0000 | 1.0000 |
| `B.ckpt3000.U.hit.pending_fifo.gap_gt_M` | 1.0000 | 1.0000 | 1.0000 |
| `B.ckpt3000.U.hit.factfiller` | 0.9655 | 0.9653 | 0.9666 |
| `B.ckpt3000.U.hit.factfiller.gap_gt_M` | 0.9027 | 0.9009 | 0.9020 |
| `B.ckpt3000.U.hit.rule_g0` | 0.7997 | 0.7881 | 0.8457 |
| `B.ckpt3000.U.hit.rule_g0.gap_gt_M` | 0.3380 | 0.1745 | 0.4346 |
| `B.ckpt3000.U.hit.rule_g09` | 0.8352 | 0.8301 | 0.8935 |
| `B.ckpt3000.U.hit.rule_g09.gap_gt_M` | 0.6353 | 0.3875 | 0.7727 |
| `B.ckpt3000.U.hit.rule_g097` | 0.7748 | 0.7869 | 0.8681 |
| `B.ckpt3000.U.hit.rule_g097.gap_gt_M` | 0.6284 | 0.3737 | 0.7982 |
| `B.ckpt3000.U.hit.rule_next` | 0.8139 | 0.7988 | 0.8535 |
| `B.ckpt3000.U.hit.rule_next.gap_gt_M` | 0.3798 | 0.2117 | 0.4661 |
| `B.ckpt3000.U.hit.lit_g0` | 0.8035 | 0.8082 | 0.8138 |
| `B.ckpt3000.U.hit.lit_g0.gap_gt_M` | 0.0528 | 0.0502 | 0.0534 |
| `B.ckpt3000.U.hit.lit_g09` | 0.8080 | 0.8104 | 0.8123 |
| `B.ckpt3000.U.hit.lit_g09.gap_gt_M` | 0.0113 | 0.0078 | 0.0185 |
| `B.ckpt3000.U.hit.lit_g097` | 0.8076 | 0.8102 | 0.8120 |
| `B.ckpt3000.U.hit.lit_g097.gap_gt_M` | 0.0090 | 0.0070 | 0.0169 |
| `B.ckpt3000.U.hit.online_g0` | 0.7948 | 0.7935 | 0.8491 |
| `B.ckpt3000.U.hit.online_g0.gap_gt_M` | 0.2213 | 0.1589 | 0.3929 |

Hit rates (set P):

| key | seed0 | seed1 | seed2 |
|---|---|---|---|
| `B.ckpt3000.P.hit.fifo` | 0.8148 | 0.8261 | 0.8104 |
| `B.ckpt3000.P.hit.oracle` | 1.0000 | 1.0000 | 1.0000 |
| `B.ckpt3000.P.hit.pending_fifo` | 1.0000 | 1.0000 | 1.0000 |
| `B.ckpt3000.P.hit.factfiller` | 0.9579 | 0.9655 | 0.9639 |
| `B.ckpt3000.P.hit.rule_g0` | 0.8123 | 0.7975 | 0.8523 |
| `B.ckpt3000.P.hit.rule_g09` | 0.8308 | 0.8395 | 0.8725 |
| `B.ckpt3000.P.hit.rule_g097` | 0.7660 | 0.8042 | 0.8482 |
| `B.ckpt3000.P.hit.rule_next` | 0.8258 | 0.8084 | 0.8532 |
| `B.ckpt3000.P.hit.lit_g0` | 0.8114 | 0.8202 | 0.8163 |
| `B.ckpt3000.P.hit.lit_g09` | 0.8165 | 0.8269 | 0.8112 |
| `B.ckpt3000.P.hit.lit_g097` | 0.8173 | 0.8261 | 0.8112 |
| `B.ckpt3000.P.hit.online_g0` | 0.8022 | 0.7983 | 0.8331 |

Live answer accuracy, FIFO vs online_g0 (set U):

| key | seed0 | seed1 | seed2 |
|---|---|---|---|
| `B.ckpt3000.U.acc_fifo` | 0.7703 | 0.7664 | 0.8037 |
| `B.ckpt3000.U.acc_online_g0` | 0.7602 | 0.7565 | 0.8043 |

r_i (= D) by class, FIFO-resident slots at full-memory steps:

| key | seed0 | seed1 | seed2 |
|---|---|---|---|
| `B.ckpt3000.class.res.pending.D.n` | 68789.0000 | 69293.0000 | 68208.0000 |
| `B.ckpt3000.class.res.pending.D.mean` | 0.0711 | 0.0688 | 0.0787 |
| `B.ckpt3000.class.res.pending.D.sd` | 0.0351 | 0.0368 | 0.0431 |
| `B.ckpt3000.class.res.pending.D.q10` | 0.0340 | 0.0288 | 0.0332 |
| `B.ckpt3000.class.res.pending.D.q50` | 0.0640 | 0.0614 | 0.0701 |
| `B.ckpt3000.class.res.pending.D.q90` | 0.1170 | 0.1181 | 0.1351 |
| `B.ckpt3000.class.res.pending.G09.n` | 68789.0000 | 69293.0000 | 68208.0000 |
| `B.ckpt3000.class.res.pending.G09.mean` | 0.6696 | 0.5462 | 0.7088 |
| `B.ckpt3000.class.res.pending.G09.sd` | 0.1605 | 0.1289 | 0.1832 |
| `B.ckpt3000.class.res.pending.G09.q10` | 0.4827 | 0.3916 | 0.4948 |
| `B.ckpt3000.class.res.pending.G09.q50` | 0.6559 | 0.5352 | 0.6945 |
| `B.ckpt3000.class.res.pending.G09.q90` | 0.8831 | 0.7188 | 0.9469 |
| `B.ckpt3000.class.res.pending.G097.n` | 68789.0000 | 69293.0000 | 68208.0000 |
| `B.ckpt3000.class.res.pending.G097.mean` | 1.2806 | 0.9816 | 1.3460 |
| `B.ckpt3000.class.res.pending.G097.sd` | 0.4111 | 0.2718 | 0.4308 |
| `B.ckpt3000.class.res.pending.G097.q10` | 0.7237 | 0.6233 | 0.7565 |
| `B.ckpt3000.class.res.pending.G097.q50` | 1.2876 | 0.9903 | 1.3642 |
| `B.ckpt3000.class.res.pending.G097.q90` | 1.8082 | 1.3213 | 1.8727 |
| `B.ckpt3000.class.res.querying.D.n` | 10838.0000 | 10866.0000 | 10762.0000 |
| `B.ckpt3000.class.res.querying.D.mean` | 0.2074 | 0.1954 | 0.2233 |
| `B.ckpt3000.class.res.querying.D.sd` | 0.0562 | 0.0557 | 0.0679 |
| `B.ckpt3000.class.res.querying.D.q10` | 0.1283 | 0.1181 | 0.1312 |
| `B.ckpt3000.class.res.querying.D.q50` | 0.2130 | 0.2021 | 0.2270 |
| `B.ckpt3000.class.res.querying.D.q90` | 0.2751 | 0.2619 | 0.3068 |
| `B.ckpt3000.class.res.querying.G09.n` | 10838.0000 | 10866.0000 | 10762.0000 |
| `B.ckpt3000.class.res.querying.G09.mean` | 0.6372 | 0.5688 | 0.6668 |
| `B.ckpt3000.class.res.querying.G09.sd` | 0.1915 | 0.1560 | 0.2120 |
| `B.ckpt3000.class.res.querying.G09.q10` | 0.3667 | 0.3522 | 0.3777 |
| `B.ckpt3000.class.res.querying.G09.q50` | 0.6490 | 0.5803 | 0.6725 |
| `B.ckpt3000.class.res.querying.G09.q90` | 0.8664 | 0.7581 | 0.9394 |
| `B.ckpt3000.class.res.querying.G097.n` | 10838.0000 | 10866.0000 | 10762.0000 |
| `B.ckpt3000.class.res.querying.G097.mean` | 1.0484 | 0.8726 | 1.1006 |
| `B.ckpt3000.class.res.querying.G097.sd` | 0.4692 | 0.3323 | 0.4963 |
| `B.ckpt3000.class.res.querying.G097.q10` | 0.3959 | 0.3948 | 0.4113 |
| `B.ckpt3000.class.res.querying.G097.q50` | 1.0677 | 0.8965 | 1.1171 |
| `B.ckpt3000.class.res.querying.G097.q90` | 1.6521 | 1.2875 | 1.7380 |
| `B.ckpt3000.class.res.answered.D.n` | 132276.0000 | 133705.0000 | 133399.0000 |
| `B.ckpt3000.class.res.answered.D.mean` | 0.0598 | 0.0534 | 0.0622 |
| `B.ckpt3000.class.res.answered.D.sd` | 0.0282 | 0.0278 | 0.0331 |
| `B.ckpt3000.class.res.answered.D.q10` | 0.0309 | 0.0259 | 0.0276 |
| `B.ckpt3000.class.res.answered.D.q50` | 0.0531 | 0.0467 | 0.0551 |
| `B.ckpt3000.class.res.answered.D.q90` | 0.0978 | 0.0894 | 0.1067 |
| `B.ckpt3000.class.res.answered.G09.n` | 132276.0000 | 133705.0000 | 133399.0000 |
| `B.ckpt3000.class.res.answered.G09.mean` | 0.5317 | 0.4014 | 0.5546 |
| `B.ckpt3000.class.res.answered.G09.sd` | 0.2085 | 0.1427 | 0.2184 |
| `B.ckpt3000.class.res.answered.G09.q10` | 0.2263 | 0.1925 | 0.2288 |
| `B.ckpt3000.class.res.answered.G09.q50` | 0.5467 | 0.4168 | 0.5760 |
| `B.ckpt3000.class.res.answered.G09.q90` | 0.7851 | 0.5680 | 0.8109 |
| `B.ckpt3000.class.res.answered.G097.n` | 132276.0000 | 133705.0000 | 133399.0000 |
| `B.ckpt3000.class.res.answered.G097.mean` | 0.9843 | 0.7115 | 1.0329 |
| `B.ckpt3000.class.res.answered.G097.sd` | 0.5035 | 0.3351 | 0.5172 |
| `B.ckpt3000.class.res.answered.G097.q10` | 0.2593 | 0.2189 | 0.2663 |
| `B.ckpt3000.class.res.answered.G097.q50` | 1.0180 | 0.7484 | 1.0905 |
| `B.ckpt3000.class.res.answered.G097.q90` | 1.6272 | 1.1258 | 1.6608 |
| `B.ckpt3000.class.res.filler.D.n` | 345153.0000 | 343192.0000 | 344687.0000 |
| `B.ckpt3000.class.res.filler.D.mean` | 0.0573 | 0.0606 | 0.0544 |
| `B.ckpt3000.class.res.filler.D.sd` | 0.0216 | 0.0251 | 0.0285 |
| `B.ckpt3000.class.res.filler.D.q10` | 0.0359 | 0.0343 | 0.0247 |
| `B.ckpt3000.class.res.filler.D.q50` | 0.0521 | 0.0551 | 0.0485 |
| `B.ckpt3000.class.res.filler.D.q90` | 0.0853 | 0.0950 | 0.0912 |
| `B.ckpt3000.class.res.filler.G09.n` | 345153.0000 | 343192.0000 | 344687.0000 |
| `B.ckpt3000.class.res.filler.G09.mean` | 0.4845 | 0.4380 | 0.4231 |
| `B.ckpt3000.class.res.filler.G09.sd` | 0.2322 | 0.2022 | 0.2110 |
| `B.ckpt3000.class.res.filler.G09.q10` | 0.1428 | 0.1444 | 0.1263 |
| `B.ckpt3000.class.res.filler.G09.q50` | 0.5134 | 0.4499 | 0.4280 |
| `B.ckpt3000.class.res.filler.G09.q90` | 0.7674 | 0.6941 | 0.6862 |
| `B.ckpt3000.class.res.filler.G097.n` | 345153.0000 | 343192.0000 | 344687.0000 |
| `B.ckpt3000.class.res.filler.G097.mean` | 0.8699 | 0.7703 | 0.7408 |
| `B.ckpt3000.class.res.filler.G097.sd` | 0.5356 | 0.4681 | 0.4584 |
| `B.ckpt3000.class.res.filler.G097.q10` | 0.1560 | 0.1577 | 0.1397 |
| `B.ckpt3000.class.res.filler.G097.q50` | 0.8670 | 0.7496 | 0.7238 |
| `B.ckpt3000.class.res.filler.G097.q90` | 1.5739 | 1.4212 | 1.3486 |
| `B.ckpt3000.class.probed.pending.D.n` | 36800.0000 | 35856.0000 | 36032.0000 |
| `B.ckpt3000.class.probed.pending.D.mean` | 0.0843 | 0.0544 | 0.0829 |
| `B.ckpt3000.class.probed.pending.D.sd` | 0.0346 | 0.0287 | 0.0439 |
| `B.ckpt3000.class.probed.pending.D.q10` | 0.0466 | 0.0269 | 0.0337 |
| `B.ckpt3000.class.probed.pending.D.q50` | 0.0775 | 0.0474 | 0.0740 |
| `B.ckpt3000.class.probed.pending.D.q90` | 0.1327 | 0.0904 | 0.1436 |
| `B.ckpt3000.class.probed.querying.D.n` | 3905.0000 | 3845.0000 | 3836.0000 |
| `B.ckpt3000.class.probed.querying.D.mean` | 0.1764 | 0.1375 | 0.1829 |
| `B.ckpt3000.class.probed.querying.D.sd` | 0.0443 | 0.0407 | 0.0640 |
| `B.ckpt3000.class.probed.querying.D.q10` | 0.1174 | 0.0777 | 0.1052 |
| `B.ckpt3000.class.probed.querying.D.q50` | 0.1777 | 0.1427 | 0.1754 |
| `B.ckpt3000.class.probed.querying.D.q90` | 0.2326 | 0.1866 | 0.2697 |
| `B.ckpt3000.class.probed.answered.D.n` | 261023.0000 | 261544.0000 | 261871.0000 |
| `B.ckpt3000.class.probed.answered.D.mean` | 0.0883 | 0.0574 | 0.0886 |
| `B.ckpt3000.class.probed.answered.D.sd` | 0.0372 | 0.0290 | 0.0462 |
| `B.ckpt3000.class.probed.answered.D.q10` | 0.0478 | 0.0285 | 0.0354 |
| `B.ckpt3000.class.probed.answered.D.q50` | 0.0804 | 0.0509 | 0.0806 |
| `B.ckpt3000.class.probed.answered.D.q90` | 0.1411 | 0.0941 | 0.1513 |
| `B.ckpt3000.class.probed.filler.D.n` | 237920.0000 | 238403.0000 | 237909.0000 |
| `B.ckpt3000.class.probed.filler.D.mean` | 0.0962 | 0.0817 | 0.0757 |
| `B.ckpt3000.class.probed.filler.D.sd` | 0.0324 | 0.0288 | 0.0366 |
| `B.ckpt3000.class.probed.filler.D.q10` | 0.0605 | 0.0499 | 0.0349 |
| `B.ckpt3000.class.probed.filler.D.q50` | 0.0901 | 0.0765 | 0.0706 |
| `B.ckpt3000.class.probed.filler.D.q90` | 0.1410 | 0.1211 | 0.1214 |

Pending asserts by k (steps to query; the last bin pools every k above M):

| key | seed0 | seed1 | seed2 |
|---|---|---|---|
| `B.ckpt3000.pend_k.1.mean` | 0.0823 | 0.0843 | 0.0922 |
| `B.ckpt3000.pend_k.1.n` | 8220.0000 | 8231.0000 | 8137.0000 |
| `B.ckpt3000.pend_k.1.q50` | 0.0761 | 0.0806 | 0.0846 |
| `B.ckpt3000.pend_k.2.mean` | 0.0801 | 0.0797 | 0.0902 |
| `B.ckpt3000.pend_k.2.n` | 6297.0000 | 6336.0000 | 6271.0000 |
| `B.ckpt3000.pend_k.2.q50` | 0.0738 | 0.0757 | 0.0836 |
| `B.ckpt3000.pend_k.3.mean` | 0.0769 | 0.0766 | 0.0860 |
| `B.ckpt3000.pend_k.3.n` | 4940.0000 | 4999.0000 | 4949.0000 |
| `B.ckpt3000.pend_k.3.q50` | 0.0704 | 0.0715 | 0.0775 |
| `B.ckpt3000.pend_k.4.mean` | 0.0732 | 0.0733 | 0.0835 |
| `B.ckpt3000.pend_k.4.n` | 4028.0000 | 4148.0000 | 4019.0000 |
| `B.ckpt3000.pend_k.4.q50` | 0.0663 | 0.0658 | 0.0748 |
| `B.ckpt3000.pend_k.5.mean` | 0.0711 | 0.0694 | 0.0795 |
| `B.ckpt3000.pend_k.5.n` | 3429.0000 | 3576.0000 | 3448.0000 |
| `B.ckpt3000.pend_k.5.q50` | 0.0633 | 0.0617 | 0.0704 |
| `B.ckpt3000.pend_k.6.mean` | 0.0688 | 0.0650 | 0.0767 |
| `B.ckpt3000.pend_k.6.n` | 3130.0000 | 3154.0000 | 3072.0000 |
| `B.ckpt3000.pend_k.6.q50` | 0.0617 | 0.0570 | 0.0680 |
| `B.ckpt3000.pend_k.7.mean` | 0.0675 | 0.0635 | 0.0737 |
| `B.ckpt3000.pend_k.7.n` | 2867.0000 | 2921.0000 | 2816.0000 |
| `B.ckpt3000.pend_k.7.q50` | 0.0599 | 0.0561 | 0.0653 |
| `B.ckpt3000.pend_k.8.mean` | 0.0664 | 0.0636 | 0.0724 |
| `B.ckpt3000.pend_k.8.n` | 2720.0000 | 2748.0000 | 2668.0000 |
| `B.ckpt3000.pend_k.8.q50` | 0.0588 | 0.0561 | 0.0635 |
| `B.ckpt3000.pend_k.9.mean` | 0.0667 | 0.0636 | 0.0732 |
| `B.ckpt3000.pend_k.9.n` | 2632.0000 | 2639.0000 | 2591.0000 |
| `B.ckpt3000.pend_k.9.q50` | 0.0586 | 0.0557 | 0.0644 |
| `B.ckpt3000.pend_k.10.mean` | 0.0664 | 0.0651 | 0.0736 |
| `B.ckpt3000.pend_k.10.n` | 2605.0000 | 2603.0000 | 2553.0000 |
| `B.ckpt3000.pend_k.10.q50` | 0.0596 | 0.0567 | 0.0647 |
| `B.ckpt3000.pend_k.11.mean` | 0.0672 | 0.0646 | 0.0736 |
| `B.ckpt3000.pend_k.11.n` | 2578.0000 | 2565.0000 | 2527.0000 |
| `B.ckpt3000.pend_k.11.q50` | 0.0599 | 0.0575 | 0.0656 |
| `B.ckpt3000.pend_k.12.mean` | 0.0680 | 0.0648 | 0.0743 |
| `B.ckpt3000.pend_k.12.n` | 2450.0000 | 2410.0000 | 2417.0000 |
| `B.ckpt3000.pend_k.12.q50` | 0.0614 | 0.0575 | 0.0654 |
| `B.ckpt3000.pend_k.13.mean` | 0.0676 | 0.0627 | 0.0740 |
| `B.ckpt3000.pend_k.13.n` | 2310.0000 | 2241.0000 | 2289.0000 |
| `B.ckpt3000.pend_k.13.q50` | 0.0600 | 0.0540 | 0.0660 |
| `B.ckpt3000.pend_k.14.mean` | 0.0668 | 0.0635 | 0.0730 |
| `B.ckpt3000.pend_k.14.n` | 2174.0000 | 2144.0000 | 2171.0000 |
| `B.ckpt3000.pend_k.14.q50` | 0.0609 | 0.0552 | 0.0653 |
| `B.ckpt3000.pend_k.15.mean` | 0.0671 | 0.0617 | 0.0739 |
| `B.ckpt3000.pend_k.15.n` | 2060.0000 | 2040.0000 | 2062.0000 |
| `B.ckpt3000.pend_k.15.q50` | 0.0608 | 0.0535 | 0.0652 |
| `B.ckpt3000.pend_k.16.mean` | 0.0677 | 0.0615 | 0.0741 |
| `B.ckpt3000.pend_k.16.n` | 1939.0000 | 1922.0000 | 1937.0000 |
| `B.ckpt3000.pend_k.16.q50` | 0.0606 | 0.0527 | 0.0648 |
| `B.ckpt3000.pend_k.17.mean` | 0.0655 | 0.0606 | 0.0712 |
| `B.ckpt3000.pend_k.17.n` | 14410.0000 | 14616.0000 | 14281.0000 |
| `B.ckpt3000.pend_k.17.q50` | 0.0591 | 0.0530 | 0.0623 |

AUCs (mean over full-memory steps):

| key | seed0 | seed1 | seed2 |
|---|---|---|---|
| `B.ckpt3000.auc.all.pend_vs_ans.rule_g0` | 0.5408 | 0.6100 | 0.5688 |
| `B.ckpt3000.auc.all.pend_vs_ans.rule_g0.n_steps` | 31131.0000 | 31117.0000 | 31076.0000 |
| `B.ckpt3000.auc.all.pend_vs_ans.rule_g09` | 0.5865 | 0.7074 | 0.6185 |
| `B.ckpt3000.auc.all.pend_vs_ans.rule_g09.n_steps` | 31131.0000 | 31117.0000 | 31076.0000 |
| `B.ckpt3000.auc.all.pend_vs_ans.rule_g097` | 0.5704 | 0.6967 | 0.5977 |
| `B.ckpt3000.auc.all.pend_vs_ans.rule_g097.n_steps` | 31131.0000 | 31117.0000 | 31076.0000 |
| `B.ckpt3000.auc.all.pend_vs_ans.rule_next` | 0.5889 | 0.6439 | 0.6100 |
| `B.ckpt3000.auc.all.pend_vs_ans.rule_next.n_steps` | 31131.0000 | 31117.0000 | 31076.0000 |
| `B.ckpt3000.auc.all.pend_vs_fill.rule_g0` | 0.5705 | 0.4622 | 0.6734 |
| `B.ckpt3000.auc.all.pend_vs_fill.rule_g0.n_steps` | 31131.0000 | 31117.0000 | 31076.0000 |
| `B.ckpt3000.auc.all.pend_vs_fill.rule_g09` | 0.6098 | 0.4360 | 0.8201 |
| `B.ckpt3000.auc.all.pend_vs_fill.rule_g09.n_steps` | 31131.0000 | 31117.0000 | 31076.0000 |
| `B.ckpt3000.auc.all.pend_vs_fill.rule_g097` | 0.5740 | 0.3650 | 0.8188 |
| `B.ckpt3000.auc.all.pend_vs_fill.rule_g097.n_steps` | 31131.0000 | 31117.0000 | 31076.0000 |
| `B.ckpt3000.auc.all.pend_vs_fill.rule_next` | 0.6204 | 0.5209 | 0.7064 |
| `B.ckpt3000.auc.all.pend_vs_fill.rule_next.n_steps` | 31131.0000 | 31117.0000 | 31076.0000 |
| `B.ckpt3000.auc.res.pend_vs_ans.rule_g0` | 0.6378 | 0.6668 | 0.6458 |
| `B.ckpt3000.auc.res.pend_vs_ans.rule_g0.n_steps` | 27031.0000 | 27224.0000 | 27071.0000 |
| `B.ckpt3000.auc.res.pend_vs_ans.rule_g09` | 0.6611 | 0.7537 | 0.6662 |
| `B.ckpt3000.auc.res.pend_vs_ans.rule_g09.n_steps` | 27031.0000 | 27224.0000 | 27071.0000 |
| `B.ckpt3000.auc.res.pend_vs_ans.rule_g097` | 0.6244 | 0.7379 | 0.6309 |
| `B.ckpt3000.auc.res.pend_vs_ans.rule_g097.n_steps` | 27031.0000 | 27224.0000 | 27071.0000 |
| `B.ckpt3000.auc.res.pend_vs_ans.rule_next` | 0.6617 | 0.6886 | 0.6679 |
| `B.ckpt3000.auc.res.pend_vs_ans.rule_next.n_steps` | 27031.0000 | 27224.0000 | 27071.0000 |
| `B.ckpt3000.auc.res.pend_vs_fill.rule_g0` | 0.6356 | 0.5753 | 0.7108 |
| `B.ckpt3000.auc.res.pend_vs_fill.rule_g0.n_steps` | 27079.0000 | 27268.0000 | 27124.0000 |
| `B.ckpt3000.auc.res.pend_vs_fill.rule_g09` | 0.6530 | 0.5495 | 0.8226 |
| `B.ckpt3000.auc.res.pend_vs_fill.rule_g09.n_steps` | 27079.0000 | 27268.0000 | 27124.0000 |
| `B.ckpt3000.auc.res.pend_vs_fill.rule_g097` | 0.5979 | 0.4603 | 0.8162 |
| `B.ckpt3000.auc.res.pend_vs_fill.rule_g097.n_steps` | 27079.0000 | 27268.0000 | 27124.0000 |
| `B.ckpt3000.auc.res.pend_vs_fill.rule_next` | 0.6648 | 0.6105 | 0.7305 |
| `B.ckpt3000.auc.res.pend_vs_fill.rule_next.n_steps` | 27079.0000 | 27268.0000 | 27124.0000 |

Controls:

| key | seed0 | seed1 | seed2 |
|---|---|---|---|
| `B.ckpt3000.C1.max_abs_acc_diff` | 2.948861377305434e-08 | 1.931784887609922e-08 | 2.2225460805103125e-08 |
| `B.ckpt3000.C1.max_abs_nll_diff` | 2.518710162657811e-07 | 1.374218199234889e-07 | 2.919970189807497e-07 |
| `B.ckpt3000.C3_sum_worst` | 2.384185791015625e-07 | 2.384185791015625e-07 | 2.384185791015625e-07 |
| `B.ckpt3000.C4_identity_worst` | 5.960464477539063e-08 | 4.470348358154297e-08 | 7.450580596923828e-08 |
| `B.ckpt3000.seconds` | 1155.1233689785004 | 1151.5227680206299 | 1150.5261878967285 |


## Reading (hand-written; not rendered — a re-render drops this section)

**The pre-registered verdict is INCONCLUSIVE, because control C1 failed on every child.**
C1 required the FIFO rollout's live `answer_acc` on P to equal the fresh-stream ledger
to the PREREG's `ACC_TOL` (manifest `thresholds`). The worst gaps are in
`B.ckpt3000.C1.max_abs_acc_diff`, all float32-rounding sized. Every argmax matched. The
reference ledger stores `float(ok.float().mean())`, which is a float32 mean, while this
harness computes the mean in float64. Recomputing the harness's accuracies as float32
means reproduces every reference bucket exactly (the C1 diagnostic claim in claims.json).
So the defect is in the PREREG's tolerance, not in the rollout. The tolerance is not
changed after data, so the verdict stays INCONCLUSIVE.

**What the rule would have said had C1 passed:** PARTIAL at both checkpoints. At
B.ckpt3000, `room_3b` is 0.0355 / 0.0420 / 0.0478 and every CI lower bound is above
zero, but every point is below 0.05. At B.ckpt2500 it is 0.0205 / 0.0367 / 0.0215. The
upper bounds exceed 0.02, so this is not NO_ROOM either.

**The class picture.** Pending asserts draw only slightly more demand before their
query than filler does (resident mean 0.0711 / 0.0688 / 0.0787 vs 0.0573 / 0.0606 /
0.0544 at ckpt3000). The query step itself draws 0.2074 / 0.1954 / 0.2233. Probed at
the oldest rank, filler draws more than pending asserts on every seed. `T0` ranks pending above
filler at AUC 0.6356 / 0.5753 / 0.7108 (resident); `G_0.9` at 0.653 / 0.5495 / 0.8226.

**Hit rates.** On seed0 and seed1, `rule_g0` is at or below FIFO. `rule_g09`
beats FIFO by 0.0293 / 0.0208 / 0.0847 at ckpt3000, a small share of the oracle's 0.1941
/ 0.1907 / 0.1912. `rule_g097` is worse than `rule_g09` on seed0 (`B.ckpt3000.U.room_3b_097.point`
-0.0249). `online_g0` stays close to `rule_g0` (difference -0.0049 / 0.0054 /
0.0034). The literal variants sit on FIFO, as predicted by construction.

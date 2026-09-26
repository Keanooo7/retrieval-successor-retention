# RESULTS — retention readability

Rendered from `runs/retention-readability/ledger.json` by `experiments/retention-readability/run.py --render-results`. PREREG `experiments/retention-readability/PREREG.md` (commit `ab7cbd2`). Run sha `a6ec1487cf2847cd6260cfe008ac72f9243b5bc8`, platform `macOS-26.6.2-arm64-arm-64bit`, device `cpu`.

Command: `uv run python experiments/retention-readability/run.py` (children listed under `commands` in the ledger).

**classification: READABLE_AT_SHIFTED_RANK** (headline B.ckpt3000); verdict outcome `survived`; moving `False`.

Per checkpoint (`per_checkpoint`):

- `B.ckpt2500`: (a) NO_PENALTY, (b) READABLE, combined READABLE_AT_SHIFTED_RANK
- `B.ckpt3000`: (a) NO_PENALTY, (b) READABLE, combined READABLE_AT_SHIFTED_RANK

Controls failed (`controls_failed`): none

## B.ckpt2500, set U (point [percentile CI], paired per-document bootstrap)

| readout | seed0 | seed1 | seed2 |
|---|---|---|---|
| `B.ckpt2500.U.RP` | 0.0032 [0.0017, 0.0047] | 0.0035 [0.0019, 0.0052] | 0.0054 [0.0036, 0.0073] |
| `B.ckpt2500.U.D_ref` | -0.0102 [-0.0541, 0.0341] | 0.0835 [0.0315, 0.1422] | -0.0286 [-0.0665, 0.0132] |
| `B.ckpt2500.U.D_floor` | 0.7741 [0.7610, 0.7873] | 0.7691 [0.7565, 0.7825] | 0.7633 [0.7494, 0.7766] |
| `B.ckpt2500.U.H_model` | 0.1484 [0.1436, 0.1533] | 0.1446 [0.1401, 0.1497] | 0.1428 [0.1377, 0.1479] |
| `B.ckpt2500.U.H_ff_model` | 0.0711 [0.0663, 0.0760] | 0.0724 [0.0679, 0.0774] | 0.0601 [0.0553, 0.0647] |
| `B.ckpt2500.U.D_ref_rank0` | 0.0044 [-0.0408, 0.0495] | 0.0861 [0.0326, 0.1439] | -0.0115 [-0.0492, 0.0292] |
| `B.ckpt2500.U.RP_nll` | 0.0030 [0.0007, 0.0054] | 0.0090 [0.0062, 0.0119] | 0.0112 [0.0084, 0.0142] |
| `B.ckpt2500.U.RP_brier16` | 0.0023 [0.0011, 0.0036] | 0.0043 [0.0028, 0.0059] | 0.0064 [0.0048, 0.0081] |
| `B.ckpt2500.U.D_ref_nll` | -0.0027 [-0.1672, 0.1405] | -0.1709 [-0.3608, 0.0004] | 0.0661 [-0.0871, 0.2044] |
| `B.ckpt2500.U.D_ref_brier16` | -0.0003 [-0.0638, 0.0564] | -0.0995 [-0.1764, -0.0289] | 0.0405 [-0.0199, 0.0968] |

| accuracy | seed0 | seed1 | seed2 |
|---|---|---|---|
| `B.ckpt2500.U.fifo.all.acc` | 0.7664 | 0.7671 | 0.7794 |
| `B.ckpt2500.U.fifo.gap_2_to_M.acc` | 0.9167 | 0.9110 | 0.9257 |
| `B.ckpt2500.U.fifo.gap_eq_M.acc` | 0.8484 | 0.7529 | 0.8821 |
| `B.ckpt2500.U.fifo.gap_gt_M.acc` | 0.0640 | 0.0674 | 0.0902 |
| `B.ckpt2500.U.fifo.rescued.acc` | nan | nan | nan |
| `B.ckpt2500.U.fifo.rescued_rank0.acc` | nan | nan | nan |
| `B.ckpt2500.U.fifo.gap_2_to_M_shifted.acc` | 0.8995 | 0.9027 | 0.9137 |
| `B.ckpt2500.U.oracle.all.acc` | 0.9148 | 0.9117 | 0.9222 |
| `B.ckpt2500.U.oracle.gap_2_to_M.acc` | 0.9135 | 0.9075 | 0.9202 |
| `B.ckpt2500.U.oracle.gap_eq_M.acc` | 0.8375 | 0.7529 | 0.8555 |
| `B.ckpt2500.U.oracle.gap_gt_M.acc` | 0.8382 | 0.8364 | 0.8535 |
| `B.ckpt2500.U.oracle.rescued.acc` | 0.8382 | 0.8364 | 0.8535 |
| `B.ckpt2500.U.oracle.rescued_rank0.acc` | 0.8528 | 0.8390 | 0.8706 |
| `B.ckpt2500.U.oracle.gap_2_to_M_shifted.acc` | 0.8914 | 0.8934 | 0.9069 |
| `B.ckpt2500.U.factfiller.all.acc` | 0.8375 | 0.8395 | 0.8394 |
| `B.ckpt2500.U.factfiller.gap_2_to_M.acc` | 0.8831 | 0.8793 | 0.8879 |
| `B.ckpt2500.U.factfiller.gap_eq_M.acc` | 0.6968 | 0.6332 | 0.7338 |
| `B.ckpt2500.U.factfiller.gap_gt_M.acc` | 0.5490 | 0.5618 | 0.5352 |
| `B.ckpt2500.U.factfiller.rescued.acc` | 0.6006 | 0.6163 | 0.5850 |
| `B.ckpt2500.U.factfiller.rescued_rank0.acc` | 0.4849 | 0.5249 | 0.4324 |
| `B.ckpt2500.U.factfiller.gap_2_to_M_shifted.acc` | 0.8897 | 0.8849 | 0.8957 |

| model-free | seed0 | seed1 | seed2 |
|---|---|---|---|
| `B.ckpt2500.U.model_free_hit.fifo` | 0.8059 | 0.8093 | 0.8088 |
| `B.ckpt2500.U.model_free_hit.oracle` | 1.0000 | 1.0000 | 1.0000 |
| `B.ckpt2500.U.model_free_hit.factfiller` | 0.9655 | 0.9653 | 0.9666 |
| `B.ckpt2500.U.H_free` | 0.1941 | 0.1907 | 0.1912 |

Set P:

| readout | seed0 | seed1 | seed2 |
|---|---|---|---|
| `B.ckpt2500.P.RP` | 0.0028 [-0.0027, 0.0085] | 0.0014 [-0.0029, 0.0057] | 0.0169 [0.0071, 0.0289] |
| `B.ckpt2500.P.D_ref` | -0.0343 [-0.1659, 0.1607] | 0.2585 [0.0332, 0.4948] | -0.0651 [-0.1591, 0.0773] |
| `B.ckpt2500.P.D_floor` | 0.7955 [0.7387, 0.8484] | 0.7971 [0.7424, 0.8512] | 0.8097 [0.7511, 0.8631] |
| `B.ckpt2500.P.H_model` | 0.1456 [0.1264, 0.1649] | 0.1378 [0.1206, 0.1553] | 0.1435 [0.1234, 0.1620] |

Set E:

| readout | seed0 | seed1 | seed2 |
|---|---|---|---|
| `B.ckpt2500.E.RP` | 0.0032 [0.0016, 0.0048] | 0.0036 [0.0019, 0.0055] | 0.0047 [0.0029, 0.0065] |
| `B.ckpt2500.E.D_ref` | -0.0084 [-0.0522, 0.0412] | 0.0710 [0.0147, 0.1264] | -0.0260 [-0.0649, 0.0182] |
| `B.ckpt2500.E.D_floor` | 0.7729 [0.7595, 0.7864] | 0.7675 [0.7543, 0.7809] | 0.7604 [0.7460, 0.7746] |
| `B.ckpt2500.E.H_model` | 0.1485 [0.1436, 0.1536] | 0.1451 [0.1400, 0.1506] | 0.1427 [0.1376, 0.1481] |

## B.ckpt3000, set U (point [percentile CI], paired per-document bootstrap)

| readout | seed0 | seed1 | seed2 |
|---|---|---|---|
| `B.ckpt3000.U.RP` | 0.0016 [0.0000, 0.0032] | 0.0033 [0.0016, 0.0051] | 0.0084 [0.0063, 0.0106] |
| `B.ckpt3000.U.D_ref` | 0.0139 [-0.0292, 0.0616] | 0.0573 [0.0067, 0.1103] | -0.0242 [-0.0586, 0.0126] |
| `B.ckpt3000.U.D_floor` | 0.7839 [0.7709, 0.7964] | 0.7761 [0.7633, 0.7884] | 0.6775 [0.6618, 0.6930] |
| `B.ckpt3000.U.H_model` | 0.1512 [0.1463, 0.1561] | 0.1459 [0.1411, 0.1506] | 0.1244 [0.1198, 0.1290] |
| `B.ckpt3000.U.H_ff_model` | 0.0691 [0.0641, 0.0739] | 0.0689 [0.0641, 0.0739] | 0.0441 [0.0395, 0.0485] |
| `B.ckpt3000.U.D_ref_rank0` | 0.0241 [-0.0202, 0.0728] | 0.0575 [0.0073, 0.1120] | -0.0045 [-0.0401, 0.0322] |
| `B.ckpt3000.U.RP_nll` | 0.0041 [0.0017, 0.0065] | 0.0089 [0.0060, 0.0121] | 0.0207 [0.0169, 0.0248] |
| `B.ckpt3000.U.RP_brier16` | 0.0026 [0.0013, 0.0039] | 0.0049 [0.0033, 0.0066] | 0.0113 [0.0092, 0.0135] |
| `B.ckpt3000.U.D_ref_nll` | -0.0300 [-0.2252, 0.1393] | -0.1636 [-0.3416, 0.0013] | -0.0223 [-0.1793, 0.1225] |
| `B.ckpt3000.U.D_ref_brier16` | -0.0037 [-0.0734, 0.0586] | -0.0804 [-0.1532, -0.0139] | 0.0124 [-0.0470, 0.0689] |

| accuracy | seed0 | seed1 | seed2 |
|---|---|---|---|
| `B.ckpt3000.U.fifo.all.acc` | 0.7703 | 0.7664 | 0.8037 |
| `B.ckpt3000.U.fifo.gap_2_to_M.acc` | 0.9207 | 0.9116 | 0.9317 |
| `B.ckpt3000.U.fifo.gap_eq_M.acc` | 0.8412 | 0.7838 | 0.9011 |
| `B.ckpt3000.U.fifo.gap_gt_M.acc` | 0.0712 | 0.0650 | 0.1994 |
| `B.ckpt3000.U.fifo.rescued.acc` | nan | nan | nan |
| `B.ckpt3000.U.fifo.rescued_rank0.acc` | nan | nan | nan |
| `B.ckpt3000.U.fifo.gap_2_to_M_shifted.acc` | 0.9052 | 0.9011 | 0.9217 |
| `B.ckpt3000.U.oracle.all.acc` | 0.9215 | 0.9123 | 0.9281 |
| `B.ckpt3000.U.oracle.gap_2_to_M.acc` | 0.9191 | 0.9083 | 0.9233 |
| `B.ckpt3000.U.oracle.gap_eq_M.acc` | 0.8303 | 0.7838 | 0.8441 |
| `B.ckpt3000.U.oracle.gap_gt_M.acc` | 0.8551 | 0.8411 | 0.8770 |
| `B.ckpt3000.U.oracle.rescued.acc` | 0.8551 | 0.8411 | 0.8770 |
| `B.ckpt3000.U.oracle.rescued_rank0.acc` | 0.8653 | 0.8413 | 0.8966 |
| `B.ckpt3000.U.oracle.gap_2_to_M_shifted.acc` | 0.8983 | 0.8904 | 0.9142 |
| `B.ckpt3000.U.factfiller.all.acc` | 0.8394 | 0.8353 | 0.8478 |
| `B.ckpt3000.U.factfiller.gap_2_to_M.acc` | 0.8879 | 0.8716 | 0.8966 |
| `B.ckpt3000.U.factfiller.gap_eq_M.acc` | 0.6859 | 0.6062 | 0.7529 |
| `B.ckpt3000.U.factfiller.gap_gt_M.acc` | 0.5455 | 0.5709 | 0.5508 |
| `B.ckpt3000.U.factfiller.rescued.acc` | 0.5969 | 0.6267 | 0.6032 |
| `B.ckpt3000.U.factfiller.rescued_rank0.acc` | 0.4849 | 0.5415 | 0.4595 |
| `B.ckpt3000.U.factfiller.gap_2_to_M_shifted.acc` | 0.8937 | 0.8747 | 0.9041 |

| model-free | seed0 | seed1 | seed2 |
|---|---|---|---|
| `B.ckpt3000.U.model_free_hit.fifo` | 0.8059 | 0.8093 | 0.8088 |
| `B.ckpt3000.U.model_free_hit.oracle` | 1.0000 | 1.0000 | 1.0000 |
| `B.ckpt3000.U.model_free_hit.factfiller` | 0.9655 | 0.9653 | 0.9666 |
| `B.ckpt3000.U.H_free` | 0.1941 | 0.1907 | 0.1912 |

Set P:

| readout | seed0 | seed1 | seed2 |
|---|---|---|---|
| `B.ckpt3000.P.RP` | 0.0042 [0.0000, 0.0090] | 0.0028 [-0.0042, 0.0113] | 0.0127 [0.0028, 0.0245] |
| `B.ckpt3000.P.D_ref` | 0.0813 [-0.1030, 0.2996] | -0.0290 [-0.1644, 0.1450] | -0.0929 [-0.1322, -0.0561] |
| `B.ckpt3000.P.D_floor` | 0.8000 [0.7479, 0.8523] | 0.7633 [0.7107, 0.8187] | 0.7522 [0.6986, 0.8009] |
| `B.ckpt3000.P.H_model` | 0.1456 [0.1270, 0.1646] | 0.1311 [0.1151, 0.1483] | 0.1351 [0.1171, 0.1522] |

Set E:

| readout | seed0 | seed1 | seed2 |
|---|---|---|---|
| `B.ckpt3000.E.RP` | 0.0014 [-0.0004, 0.0032] | 0.0034 [0.0016, 0.0052] | 0.0082 [0.0060, 0.0103] |
| `B.ckpt3000.E.D_ref` | 0.0093 [-0.0371, 0.0588] | 0.0641 [0.0085, 0.1225] | -0.0192 [-0.0555, 0.0225] |
| `B.ckpt3000.E.D_floor` | 0.7829 [0.7695, 0.7961] | 0.7768 [0.7635, 0.7899] | 0.6729 [0.6558, 0.6888] |
| `B.ckpt3000.E.H_model` | 0.1515 [0.1469, 0.1567] | 0.1468 [0.1416, 0.1523] | 0.1237 [0.1188, 0.1284] |

## A.ckpt3000, set U (point [percentile CI], paired per-document bootstrap)

| readout | seed0 | seed1 | seed2 |
|---|---|---|---|
| `A.ckpt3000.U.RP` | -0.0008 [-0.0024, 0.0008] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| `A.ckpt3000.U.D_ref` | -0.0355 [-0.0729, -0.0007] | 0.0019 [-0.0289, 0.0312] | -0.0343 [-0.0699, -0.0010] |
| `A.ckpt3000.U.D_floor` | 0.0010 [-0.0042, 0.0062] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| `A.ckpt3000.U.H_model` | 0.0001 [-0.0013, 0.0017] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| `A.ckpt3000.U.H_ff_model` | -0.0001 [-0.0023, 0.0018] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| `A.ckpt3000.U.D_ref_rank0` | -0.0339 [-0.0722, 0.0007] | 0.0002 [-0.0312, 0.0287] | -0.0309 [-0.0668, 0.0026] |
| `A.ckpt3000.U.RP_nll` | 0.0000 [-0.0001, 0.0001] | -0.0000 [-0.0001, 0.0000] | 0.0002 [0.0001, 0.0003] |
| `A.ckpt3000.U.RP_brier16` | 0.0000 [-0.0000, 0.0000] | -0.0000 [-0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| `A.ckpt3000.U.D_ref_nll` | 0.0002 [-0.0097, 0.0100] | -0.0037 [-0.0157, 0.0081] | 0.0123 [0.0024, 0.0226] |
| `A.ckpt3000.U.D_ref_brier16` | 0.0001 [-0.0011, 0.0013] | -0.0005 [-0.0020, 0.0010] | 0.0016 [0.0003, 0.0029] |

| accuracy | seed0 | seed1 | seed2 |
|---|---|---|---|
| `A.ckpt3000.U.fifo.all.acc` | 0.0647 | 0.0626 | 0.0606 |
| `A.ckpt3000.U.fifo.gap_2_to_M.acc` | 0.0669 | 0.0634 | 0.0623 |
| `A.ckpt3000.U.fifo.gap_eq_M.acc` | 0.0903 | 0.0579 | 0.0875 |
| `A.ckpt3000.U.fifo.gap_gt_M.acc` | 0.0538 | 0.0598 | 0.0532 |
| `A.ckpt3000.U.fifo.rescued.acc` | nan | nan | nan |
| `A.ckpt3000.U.fifo.rescued_rank0.acc` | nan | nan | nan |
| `A.ckpt3000.U.fifo.gap_2_to_M_shifted.acc` | 0.0696 | 0.0598 | 0.0622 |
| `A.ckpt3000.U.oracle.all.acc` | 0.0648 | 0.0626 | 0.0606 |
| `A.ckpt3000.U.oracle.gap_2_to_M.acc` | 0.0677 | 0.0634 | 0.0623 |
| `A.ckpt3000.U.oracle.gap_eq_M.acc` | 0.0794 | 0.0579 | 0.0875 |
| `A.ckpt3000.U.oracle.gap_gt_M.acc` | 0.0548 | 0.0598 | 0.0532 |
| `A.ckpt3000.U.oracle.rescued.acc` | 0.0548 | 0.0598 | 0.0532 |
| `A.ckpt3000.U.oracle.rescued_rank0.acc` | 0.0564 | 0.0581 | 0.0565 |
| `A.ckpt3000.U.oracle.gap_2_to_M_shifted.acc` | 0.0696 | 0.0596 | 0.0622 |
| `A.ckpt3000.U.factfiller.all.acc` | 0.0645 | 0.0626 | 0.0606 |
| `A.ckpt3000.U.factfiller.gap_2_to_M.acc` | 0.0670 | 0.0634 | 0.0623 |
| `A.ckpt3000.U.factfiller.gap_eq_M.acc` | 0.0830 | 0.0579 | 0.0875 |
| `A.ckpt3000.U.factfiller.gap_gt_M.acc` | 0.0566 | 0.0598 | 0.0532 |
| `A.ckpt3000.U.factfiller.rescued.acc` | 0.0567 | 0.0589 | 0.0549 |
| `A.ckpt3000.U.factfiller.rescued_rank0.acc` | 0.0633 | 0.0498 | 0.0330 |
| `A.ckpt3000.U.factfiller.gap_2_to_M_shifted.acc` | 0.0690 | 0.0596 | 0.0616 |

| model-free | seed0 | seed1 | seed2 |
|---|---|---|---|
| `A.ckpt3000.U.model_free_hit.fifo` | 0.8059 | 0.8093 | 0.8088 |
| `A.ckpt3000.U.model_free_hit.oracle` | 1.0000 | 1.0000 | 1.0000 |
| `A.ckpt3000.U.model_free_hit.factfiller` | 0.9655 | 0.9653 | 0.9666 |
| `A.ckpt3000.U.H_free` | 0.1941 | 0.1907 | 0.1912 |

Set P:

| readout | seed0 | seed1 | seed2 |
|---|---|---|---|
| `A.ckpt3000.P.RP` | -0.0014 [-0.0072, 0.0043] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| `A.ckpt3000.P.D_ref` | -0.0192 [-0.1373, 0.0543] | 0.0628 [0.0350, 0.0931] | -0.0190 [-0.1706, 0.0583] |
| `A.ckpt3000.P.D_floor` | -0.0045 [-0.0324, 0.0219] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| `A.ckpt3000.P.H_model` | -0.0008 [-0.0078, 0.0059] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |

Set E:

| readout | seed0 | seed1 | seed2 |
|---|---|---|---|
| `A.ckpt3000.E.RP` | -0.0008 [-0.0025, 0.0008] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| `A.ckpt3000.E.D_ref` | -0.0368 [-0.0760, -0.0031] | -0.0026 [-0.0346, 0.0270] | -0.0354 [-0.0741, -0.0009] |
| `A.ckpt3000.E.D_floor` | 0.0014 [-0.0036, 0.0066] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| `A.ckpt3000.E.H_model` | 0.0002 [-0.0013, 0.0017] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |


## Reading (hand-written; appended after rendering, re-append if re-rendered)

Hardware: the Mac Studio (Apple M-series, see `provenance.platform`), CPU, one thread per child, nine children in parallel.
The run sha `a6ec1487cf2847cd6260cfe008ac72f9243b5bc8` is two commits behind this file's: `f7330c3`
changed only `main()`'s return statements (`status(...)`, required by
`tests/test_exit_codes.py`), and the render commit changed only this renderer's heading text.

- **Controls.** Every child: the bit-exact control held with zero NLL difference and no argmax mismatch, the ledger
  reproduction held inside 1e-6 (per-child values under `<ckpt>.seed<s>.controls`), harness
  residency equalled model-free residency, the document checks passed, and the arm A null held.
- **(a) Rank penalty exists but is small.** `B.ckpt3000.U.RP.point` 0.0016 / 0.0033 / 0.0084; the
  CI excludes zero on seed1 and seed2 (seed0's lower bound rounds to 0.0000) but every upper bound
  is under RP_MAX. So moving facts FIFO also keeps to other ranks and neighbours costs well under a
  point of accuracy on this model — a real, measurable, sub-threshold effect, not zero. Its NLL
  version (`B.ckpt3000.U.RP_nll`) is positive on every seed too.
- **(b) Rescued facts are read about as well as FIFO reads its own rank-0 fact.**
  `B.ckpt3000.U.oracle.rescued.acc` 0.8551 / 0.8411 / 0.8770 against
  `B.ckpt3000.U.fifo.gap_eq_M.acc` 0.8412 / 0.7838 / 0.9011 and `B.ckpt3000.U.fifo.gap_gt_M.acc`
  0.0712 / 0.0650 / 0.1994.
- **(c) Descriptive only, not a gate.** `B.ckpt3000.U.H_model` 0.1512 / 0.1459 / 0.1244 against
  the model-free `B.ckpt3000.U.H_free` 0.1941 / 0.1907 / 0.1912 on the same documents; on P alone
  `B.ckpt3000.P.H_free` reproduces the red team's 0.1852 / 0.1739 / 0.1896.
- **Fact/filler is not read like the oracle (secondary, no verdict).**
  `B.ckpt3000.U.factfiller.rescued.acc` 0.5969 / 0.6267 / 0.6032, far below the oracle's rescued
  accuracy, although fact/filler keeps rescued facts resident too. Its model-based gain
  `B.ckpt3000.U.H_ff_model` 0.0691 / 0.0689 / 0.0441 is well under half its model-free gain. Which
  of rank spread, stale (already-queried) asserts as neighbours, or fact age causes this is NOT
  separated by this run.
- **Arm A.** `A.ckpt3000.U.H_model` is exactly 0.0000 with a zero-width CI on seed1 and seed2: the
  argmax never changes with the memory's content there (NLL does move slightly,
  `A.ckpt3000.U.RP_nll`). Chance-level, inert memory, as the control needs; not a broken
  instrument.
- **Instrument defect in a secondary bucket.** `gap_2_to_M_shifted` counts rank != M − gap, which
  also fires for answers read while the memory is still filling (the first sentences of a
  document), for FIFO as for the oracle. It is a descriptive bucket only; no primary readout uses
  it.

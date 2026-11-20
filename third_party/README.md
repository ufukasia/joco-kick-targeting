# Third-party code used in the exploratory reviewer blocks (PROTOCOL_v4_exploratory)

Neither competitor's code is redistributed here. Fetch it from the original sources.

## NeuroLKH (block B2)

- Source: https://github.com/liangxinedu/NeuroLKH, commit `32ab884`. Clone it to `third_party/NeuroLKH`.
- Model: the authors' pretrained `pretrained/neurolkh_m.pt`, used unchanged.
- `neurolkh_prep.py` builds the input graph exactly as the authors' `tsplib_test.py` does. Its `selftest` command compares the graph with their dense code.
- `neurolkh_infer.py` runs the forward pass. It needs PyTorch and a GPU.
- The repository has no license file. We therefore ship only what our runs produced: the network outputs in `results/kick_targeting/neurolkh_scores/*.neurolkh_m.npz` and the forward times. The search runs can be reproduced from these files without a GPU. The input graphs (`*.graph.npz`) are not shipped; `py -3.13 neurolkh_prep.py prep` rebuilds them.

## LKH-3.0.13 (block B3)

- LKH is free for academic and non-commercial use and must be obtained from the author: http://webhotel4.ruc.dk/~keld/research/LKH-3/
- The official site was unreachable during this work. We used two public copies, https://github.com/HuaYuXiao/LKH-3 and https://github.com/Seth9976/LKH-3.0.13. The kick-related files in the two copies are byte-identical.
- `LKH-3.0.13_KSwapKick_targeting.patch` is our only change, to `SRC/KSwapKick.c`. Apply it with `patch -p1 < LKH-3.0.13_KSwapKick_targeting.patch` inside the LKH-3.0.13 folder, then run `make`.
- The patch does nothing unless the environment variable `LKH_KICK_DREF` is set. When it is unset, the patched binary reproduced the official one exactly on dka1376 (300 trials, `KICK_TYPE = 4`).
- The runs used gcc 12.2 in a `debian:bookworm-slim` Docker container; see `lkh_b3_experiment.py` and `results/kick_targeting/run_v5.ps1`.
- The task list `lkh_b3/tasks.txt` has CRLF line ends. Feed it through `tr -d '\r'`, as `results/kick_targeting/run_v5_s5_fix.ps1` does. See `results/kick_targeting/PROTOCOL_DEVIATIONS.log`, entry 2026-11-20 12:37.

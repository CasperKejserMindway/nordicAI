# Case 2 export — drone-flyby, currently-served configuration

This is a **copy**, extracted 2026-09-20 while the live version keeps running
on Hetzner. Nothing here has been removed from the original.

## What this is

The exact model + calibration currently serving the competition endpoint:
- `weights/drone_v5_v8.pt` — the primary detector (YOLO11n, fine-tuned on
  Helsinki + real crops + copy-paste augmentation)
- `weights/synth_v4_e4.pt` — a second model, used only for 6 classes it covers
  better (`condor,spacecraft,medium_plane,jammer,ta-ta,large_launcher`)
- a per-class box-size calibration string, tuned on the validation scene's
  appearance (see the honesty note below)

**Known performance**: 0.3526 on the validation board over 5 runs (spread
0.010), and a mean of 0.333 (sd 0.035, or 0.343 excluding one low outlier)
over 10 fresh runs on 2026-09-20. Latency p50 45ms, p95 70ms, max 214ms — well
inside the evaluator's 333ms-per-frame budget.

## Honesty note — read before trusting this number elsewhere

The box-scale calibration was tuned against the **validation** flight's
appearance. `FINDINGS.md` (in the source repo) documents that roughly a third
of this model's advantage over its simpler sibling (`n_synth1`, 0.3176, zero
scene-specific labels) is scene-specific and may not transfer to a genuinely
different flight — which is exactly what the competition's evaluation set is.
So: strong on the validation scene, moderately confident elsewhere, not
guaranteed.

## Running it

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./run.sh
```

Then check `curl http://localhost:9053/api` — it should report
`"weights":"drone_v5_v8.pt"` and non-zero `frames` once you send it traffic.

`dtos.py` and `utils.py` came from the official competition repo
(`Nordic-AI-Cup-2026/drone-flyby/`) — if that repo has moved on, re-pull
those two files from there rather than editing them here.

## Submitting a validation attempt from here

You'll need a public route to whatever port this serves on (ngrok, a cloud VM,
or a reverse tunnel), then:

```
curl -s -X POST -H "x-token: <the same key>" -H "Content-Type: application/json" \
  -d '{"url":"http://<your-public-host>:9053"}' \
  https://cases.nordicaicup.com/api/v1/usecases/drone-flyby/validate/queue
```

Validation is unlimited by the competition rules — safe to experiment.
**Never call the `/evaluate/queue` endpoint without being asked** — each case
gets exactly one evaluation attempt, ever.

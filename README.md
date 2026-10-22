# LongSWE SLO Bench

Search for the highest tested client concurrency satisfying **P90 TTFT < 10 s** and **mean decode TPS > 15 tokens/s**, using local LongSWE-Bench 128K data and an already running SGLang server.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
git clone https://github.com/hey-kong/longswe-slo-bench.git
cd longswe-slo-bench
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Configure and run

Replace the placeholders with your local paths and server URL:

```bash
DATASET="<local-LongSWE-Bench-128K-directory>"
TOKENIZER="<local-model-or-tokenizer-directory>"
BASE_URL="<SGLang-server-URL>"
OUTPUT_DIR="<new-results-directory>"
LABEL="hicache"

# Inspect local data before benchmarking.
python longswe_slo.py \
  --dataset "$DATASET" --tokenizer "$TOKENIZER" \
  --output-len 1024 --output-dir "$OUTPUT_DIR" --inspect-only

# Start the target server separately, then benchmark it.
python longswe_slo.py \
  --dataset "$DATASET" --tokenizer "$TOKENIZER" --base-url "$BASE_URL" \
  --label "$LABEL" --output-len 1024 --output-dir "$OUTPUT_DIR"
```

The dataset directory must contain the saved 128K dataset, including `dataset_dict.json` and its split directory. The server URL must include its scheme, host, and port.

For other server configurations, change `LABEL` to `hisparse`, `adaptive_hisparse`, or `kvduo`, and choose a new `OUTPUT_DIR`. Run one configuration at a time. Output directories must not already exist.

All data and tokenizer files are loaded locally. Use `--prompt-column` if field detection is ambiguous.

## Test settings

- A 1,200-second request-send window starts after local data preparation and warmup. No new generation requests are initiated after the deadline. Already-sent requests finish normally afterward, so total wall time can exceed 20 minutes.
- Concurrency starts at 8 and is capped at 64. Doubling and binary search use 180-second probes; `--confirm-seconds` reserves a target confirmation window, not a required minimum. Any remaining send time is used for confirmation, or to continue the last candidate if no probe passed.
- Requests drain completely between stages without resetting the global clock. Errors stop the run early; existing per-request timeouts still apply. The former `--drain-reserve` option has been removed.
- Fixed output length, temperature 0, and EOS ignored. Keep output length identical across configurations.
- Mean decode TPS is the arithmetic mean of `(output_tokens - 1) / (E2E - TTFT)` per request, excluding TTFT.
- Unique request prefixes prevent full-prefix reuse by default. Add `--allow-prefix-reuse` to replay unchanged prompts. Cache is flushed between stages unless `--keep-cache` is set.
- Full dataset prompts are used without truncation or a chat template. Actual token counts depend on the tokenizer.

## Results

See `summary.json` for `highest_passing_concurrency`, `confirmed_concurrency`, `smallest_observed_fail`, and per-stage metrics. A passing probe and a successful independent confirmation are reported separately; inconsistent PASS/FAIL observations are flagged. `requests.jsonl` contains individual request measurements; `config.json` records settings and dataset details.

Errors and insufficient samples do not count as passing. Results are observations within the send window, not proof of the global maximum or a full 20-minute run at one concurrency. Timing fields include `send_window_seconds`, `last_request_started_elapsed_s`, `final_drain_seconds`, and `elapsed_wall_seconds`.

Run `python longswe_slo.py --help` for all options.

## Timing checks

CPU-only mocked checks require no model, dataset, or running server:

```bash
python -m unittest -v test_longswe_slo.py
```

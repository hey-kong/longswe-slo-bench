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

- 1,200-second search budget; concurrency starts at 8 and is capped at 64.
- Doubling followed by binary search, with 180-second probes and at least 300 seconds reserved for confirmation. In-flight requests finish before each stage ends; draining may extend total wall time.
- Fixed output length, temperature 0, and EOS ignored. Keep output length identical across configurations.
- Mean decode TPS is the arithmetic mean of `(output_tokens - 1) / (E2E - TTFT)` per request, excluding TTFT.
- Unique request prefixes prevent full-prefix reuse by default. Add `--allow-prefix-reuse` to replay unchanged prompts. Cache is flushed between stages unless `--keep-cache` is set.
- Full dataset prompts are used without truncation or a chat template. Actual token counts depend on the tokenizer.

## Results

See `summary.json` for `confirmed_concurrency`, `smallest_observed_fail`, and per-stage metrics. `requests.jsonl` contains individual request measurements; `config.json` records settings and dataset details.

Errors and insufficient samples do not count as passing. Results identify the highest tested, reconfirmed candidate—not proof of the global maximum or a full 20-minute run at that concurrency.

Run `python longswe_slo.py --help` for all options.

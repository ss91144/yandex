"""
Interview task calculations
============================
Slide 1: OpenAI vs Anthropic GPU fleets, train/inference split
Slide 2: Sail Research GLM-5.2 margin (bottoms-up GPU-TCO model)

Data pulled: 2026-08-28. All external figures are estimates from public
reporting (companies do not disclose exact fleet sizes or infra costs).
Sources are listed inline and repeated on the slides.
"""

import json

# ---------------------------------------------------------------------
# SLIDE 1 — GPU fleets
# ---------------------------------------------------------------------

# OpenAI: Altman said "well over 1M GPUs" online by end of 2025 (own posts,
# reported widely, e.g. Tom's Hardware / Yahoo Finance, Jul 2025).
# Third-party datacenter tracker flopper.io (unofficial, extrapolated from
# public datacenter footprints) estimates ~1.4M physical GPUs / 5.7M
# H100-equivalents (H100e normalizes different chip generations to one
# comparable compute unit) as of mid-2026.
openai_gpus_physical_low = 1_000_000        # OpenAI's own "well over 1M" claim, EOY 2025
openai_gpus_physical_est = 1_400_000        # flopper.io tracker estimate, mid-2026
openai_h100e_est = 5_700_000                # flopper.io, H100-equivalent (accounts for newer/faster chips)

# Anthropic: does not disclose a GPU count; it runs on a mix of AWS
# Trainium, Google TPUs and Nvidia GPUs. Google deal: up to 1M TPUs,
# >1GW online in 2026 (Anthropic/Google press releases, Oct 2025 + Apr 2026).
# A second deal (Apr 2026) adds 3.5GW of TPU capacity from 2027.
# We convert power capacity to a rough GPU-equivalent using ~700W/accelerator
# (an analyst back-of-envelope conversion, TradingKey, Apr 2026) purely to
# make the two companies visually comparable on one slide — TPUs != GPUs,
# so this is explicitly flagged as an approximation on the slide.
anthropic_tpus_committed = 1_000_000        # "up to 1M TPUs" - Google/Anthropic Oct 2025 deal, contractual ceiling not necessarily deployed today
anthropic_gw_2026 = 1.0                     # >1GW TPU capacity online in 2026 (announced)
anthropic_gw_2027_add = 3.5                 # additional GW from 2027 (Apr 2026 deal)
watts_per_accelerator = 700                 # rough H100/H200-class TDP used for W->accelerator-count conversion
anthropic_accel_equiv_2026 = anthropic_gw_2026 * 1e9 / watts_per_accelerator
anthropic_accel_equiv_2027_total = (anthropic_gw_2026 + anthropic_gw_2027_add) * 1e9 / watts_per_accelerator

# Train vs inference split — no company publishes this. Two independent
# reference points, both flagged as estimates on the slide:
#  (a) Epoch AI / "Dean 2024"-style industry model: ~60-70% of *frontier lab*
#      compute went to training in 2025-2026, shifting toward inference
#      over the next few years (arXiv 2504.16138).
#  (b) Aggregator estimate (Lambda Finance, compiling SemiAnalysis/AWS/GCP
#      breadcrumbs) specifically for Anthropic: training 56%, inference 33%,
#      research 11% of compute *spend* (not compute-hours) — a single,
#      lower-confidence source, shown as a secondary data point only.
train_share_range = (0.60, 0.70)     # (a) industry-wide frontier-lab estimate, 2025-2026
inference_share_range = (0.30, 0.40)
anthropic_point_estimate = {"training": 0.56, "inference": 0.33, "research": 0.11}  # (b), single source, low confidence

# ---------------------------------------------------------------------
# SLIDE 2 — Sail Research GLM-5.2 margin model
# ---------------------------------------------------------------------

# --- Prices scraped from https://docs.sailresearch.com/pricing on 2026-08-28
# All figures: USD per 1,000,000 tokens.
glm_pricing = {
    "Default (ASAP)": {"input": 0.80, "cached": 0.16, "output": 3.00},
    "Balanced":        {"input": 0.50, "cached": 0.12, "output": 2.50},
    "Flex":            {"input": 0.40, "cached": 0.08, "output": 1.80},
}

# --- Model specs (public reporting on GLM-5.2, Zhipu/Z.ai, released 13 Jun 2026)
total_params = 744e9
active_params = 40e9          # active params per token (MoE) — consistent across sources
# Recommended deployment per vLLM's official recipe: FP8 checkpoint fits an
# 8xH200/8xH20 node (recipes.vllm.ai/zai-org/GLM-5.2). We model on 8xH200.
gpus_per_node = 8
h200_fp8_tflops_dense = 990   # NVIDIA published dense FP8 TFLOPS per H200 (no sparsity)

# --- Compute-bound cost model
# Standard rule of thumb used in industry cost models (e.g. SemiAnalysis):
#   FLOPs per generated token ~= 2 x active_parameters
# This gives a *lower-bound, compute-bound* estimate of GPU-time needed to
# serve one output token; real decode is often memory-bandwidth-bound at
# low batch sizes, but a commercial API provider batches heavily, which
# pushes realized throughput toward the compute-bound ceiling times an
# achieved "Model FLOPs Utilization" (MFU). We sweep MFU explicitly instead
# of guessing a single number.
flops_per_token = 2 * active_params

def node_tokens_per_sec(mfu):
    node_peak_flops = gpus_per_node * h200_fp8_tflops_dense * 1e12
    effective_flops = node_peak_flops * mfu
    return effective_flops / flops_per_token

def cost_per_million_output_tokens(mfu, usd_per_gpu_hour, overhead_multiplier=1.0):
    tps = node_tokens_per_sec(mfu)
    node_hours_per_million = 1e6 / tps / 3600
    gpu_hours_per_million = node_hours_per_million * gpus_per_node
    raw_cost = gpu_hours_per_million * usd_per_gpu_hour
    return raw_cost * overhead_multiplier

# --- Scenarios (bear / base / bull), each = (MFU, $/GPU-hr, overhead multiplier)
# GPU-hr rates from cloud-price surveys, Aug 2026 (Hyperbolic, GMI Cloud,
# Jarvislabs, getdeploying.com): on-demand H200 median ~$3.5-4.5/GPU-hr,
# specialised/reserved capacity as low as ~$1.5-2.5/GPU-hr.
# Overhead multiplier = fudge factor for real-world costs the pure FLOPs
# model ignores: sub-100% utilization, networking/storage, redundancy,
# non-serving staff & R&D allocated to inference infra. 1x = raw hardware
# only; 5x = generously loaded "fully-burdened" cost.
scenarios = {
    "Bear (thin/negative margin case)": dict(mfu=0.15, usd_per_gpu_hour=5.0, overhead_multiplier=5.0),
    "Base (reserved capacity, decent utilization)": dict(mfu=0.25, usd_per_gpu_hour=3.5, overhead_multiplier=2.0),
    "Bull (owned hardware, high utilization)": dict(mfu=0.40, usd_per_gpu_hour=2.0, overhead_multiplier=1.5),
}

# --- Sail Research — who they are (context for calibrating cost-model assumptions)
# Sail Research: VC-backed inference infra startup (Kleiner Perkins-led Series A +
# Sequoia-led seed, $80M total, $450M valuation, announced/emerged from stealth
# Jun 2026). Positions itself explicitly as throughput-over-latency infra for
# long-horizon agents, and makes specific efficiency claims in its own PR:
#   - "we carefully choose our chips, write custom inference engines, and run
#      a global controller that fully utilizes every computer in our fleet"
#     (Movva/CEO, via SiliconANGLE) -> a direct claim of high fleet utilization,
#     i.e. MFU near the top of our swept range, not the middle.
#   - claims "up to 10x lower cost per token than leading alternatives" and
#     topped the BrowseComp-Plus benchmark "at one-10th the inference cost of
#     rival services" (company PR / SiliconANGLE, Jun 2026).
# This doesn't prove a specific MFU number, but it's a reason to weight the
# Bull scenario as more representative of reality than Bear.
sail_context = {
    "funding_usd": 80_000_000,
    "valuation_usd": 450_000_000,
    "investors": ["Kleiner Perkins (Series A lead)", "Sequoia (Seed lead)", "Redpoint", "Theory Ventures", "Vine Ventures", "CRV"],
    "positioning": "throughput-over-latency inference infra for long-horizon AI agents",
    "self_claimed_advantage": "up to 10x lower cost per token than leading alternatives; custom inference engines; fleet run at high utilization by design",
}

# --- Cross-check: Z.ai's own first-party price for the same model, plus the
# cheapest observed third-party reseller rate, both checked 2026-08-28/09-01.
# Sources: VentureBeat (Jun 2026), layer3labs.io, orcarouter.ai, aipricing.guru
# (4 independent write-ups all agree on $1.40 / $4.40 for Z.ai's official API);
# OpenRouter model page for the cheapest third-party rate observed in the wild.
price_benchmarks_usd_per_1M = {
    "Z.ai (первоисточник модели, официальный API)": {"input": 1.40, "output": 4.40},
    "Sail Research, Default/ASAP":                   {"input": 0.80, "output": 3.00},
    "OpenRouter, самый дешёвый наблюдаемый reseller": {"input": 0.4875, "output": 1.56},
}
# Sail prices ~40% below the model creator's own list price, but well above
# the cheapest third-party reseller — i.e. squarely inside the competitive
# range for an open-weight model anyone can self-host, not an outlier discount.

results = {}
for name, params in scenarios.items():
    cost_1m_out = cost_per_million_output_tokens(**params)
    row = {"assumptions": params, "cost_per_1M_output_tokens_usd": round(cost_1m_out, 3), "margins": {}}
    for tier, prices in glm_pricing.items():
        margin = (prices["output"] - cost_1m_out) / prices["output"]
        row["margins"][tier] = round(margin * 100, 1)
    results[name] = row

if __name__ == "__main__":
    print("=== SLIDE 1 numbers ===")
    print("OpenAI physical GPUs (est.):", openai_gpus_physical_est)
    print("OpenAI H100-equivalents (est.):", openai_h100e_est)
    print("Anthropic accelerator-equivalent from committed power, 2026:", int(anthropic_accel_equiv_2026))
    print("Anthropic accelerator-equivalent from committed power, 2027+:", int(anthropic_accel_equiv_2027_total))
    print()
    print("=== SLIDE 2 numbers ===")
    print(json.dumps(results, indent=2))

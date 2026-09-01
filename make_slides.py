import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from calcs import (
    openai_gpus_physical_est, openai_h100e_est,
    anthropic_accel_equiv_2026, anthropic_accel_equiv_2027_total,
    train_share_range, inference_share_range, anthropic_point_estimate,
    glm_pricing, results, sail_context, price_benchmarks_usd_per_1M,
)

plt.rcParams["font.family"] = "DejaVu Sans"
FIGSIZE = (13.333, 7.5)  # 16:9
BG = "#0f1520"
CARD = "#161d2b"
TEXT = "#e8ecf3"
MUTED = "#8b96ab"
ACCENT = "#5fb3ff"
ACCENT2 = "#ff9a5f"
GREEN = "#4fd18a"
RED = "#ff6b6b"

pdf = PdfPages("slides.pdf")

# ======================================================================
# SLIDE 1 — GPU fleets
# ======================================================================
fig = plt.figure(figsize=FIGSIZE, facecolor=BG)
fig.patch.set_facecolor(BG)

fig.text(0.045, 0.94, "Сколько GPU у OpenAI и Anthropic и как они делятся между обучением и инференсом",
          fontsize=17, color=TEXT, fontweight="bold", va="top")
fig.text(0.045, 0.885, "Данные актуальны на 28 августа 2026 · компании не раскрывают точные цифры — везде оценки, диапазоны показывают неопределённость",
          fontsize=9.5, color=MUTED, va="top")

# --- Left: bar chart of fleet size (H100-equivalent / accelerator-equivalent)
ax1 = fig.add_axes([0.05, 0.24, 0.44, 0.55])
ax1.set_facecolor(BG)
labels = ["OpenAI\nфизич. GPU", "OpenAI\nH100-экв.", "Anthropic\nэкв., ~1 ГВт\n(2026)", "Anthropic\nэкв., ~4.5 ГВт\n(2027+)"]
values = [openai_gpus_physical_est/1e6, openai_h100e_est/1e6, anthropic_accel_equiv_2026/1e6, anthropic_accel_equiv_2027_total/1e6]
colors = [ACCENT, ACCENT, ACCENT2, ACCENT2]
bars = ax1.bar(labels, values, color=colors, width=0.6, edgecolor="none")
for b, v in zip(bars, values):
    ax1.text(b.get_x()+b.get_width()/2, v+0.08, f"{v:.1f}M", ha="center", color=TEXT, fontsize=11, fontweight="bold")
ax1.set_ylabel("млн ускорителей (оценка)", color=MUTED, fontsize=9)
ax1.tick_params(colors=MUTED, labelsize=7.8)
for spine in ax1.spines.values():
    spine.set_visible(False)
ax1.spines["bottom"].set_visible(True)
ax1.spines["bottom"].set_color(MUTED)
ax1.set_ylim(0, 7.5)
ax1.set_title("Размер парка (оценки третьих сторон)", color=TEXT, fontsize=11, pad=10)

# --- Right: train/inference split
ax2 = fig.add_axes([0.545, 0.18, 0.41, 0.62])
ax2.set_facecolor(BG)
ax2.axis("off")
ax2.set_title("Train vs Inference — оценки отрасли", color=TEXT, fontsize=11, loc="left", pad=10)

# stacked bar for industry-wide range
y0 = 0.72
train_mid = np.mean(train_share_range)
ax2.barh([y0], [train_mid], color=ACCENT, height=0.16)
ax2.barh([y0], [1-train_mid], left=[train_mid], color=ACCENT2, height=0.16)
ax2.text(0.02, y0, f"Training ~{train_share_range[0]*100:.0f}–{train_share_range[1]*100:.0f}%", va="center", ha="left", color="#06111f", fontsize=9.5, fontweight="bold")
ax2.text(train_mid+0.02, y0, f"Inference ~{inference_share_range[0]*100:.0f}–{inference_share_range[1]*100:.0f}%", va="center", ha="left", color="#06111f", fontsize=9.5, fontweight="bold")
ax2.text(0, y0+0.13, "Frontier labs в целом (Epoch AI / industry model)", color=MUTED, fontsize=8.7)

# Anthropic point estimate (single lower-confidence source)
y1 = 0.38
segs = [("training",0.56,ACCENT), ("inference",0.33,ACCENT2), ("research",0.11,"#c9a4ff")]
x = 0
for name, val, col in segs:
    ax2.barh([y1], [val], left=[x], color=col, height=0.16)
    if val > 0.08:
        ax2.text(x+val/2, y1, f"{val*100:.0f}%", va="center", ha="center", color="#06111f", fontsize=9, fontweight="bold")
    x += val
ax2.text(0, y1+0.13, "Anthropic, оценка расходов на compute (1 источник — Lambda Finance,\nагрегирует SemiAnalysis/AWS/GCP; низкая достоверность)", color=MUTED, fontsize=8.2)

ax2.set_xlim(0,1)
ax2.set_ylim(0,1)

legend_y = 0.05
ax2.text(0, legend_y, "■", color=ACCENT, fontsize=11)
ax2.text(0.02, legend_y, "training", color=MUTED, fontsize=8.5, va="center")
ax2.text(0.13, legend_y, "■", color=ACCENT2, fontsize=11)
ax2.text(0.15, legend_y, "inference", color=MUTED, fontsize=8.5, va="center")
ax2.text(0.27, legend_y, "■", color="#c9a4ff", fontsize=11)
ax2.text(0.29, legend_y, "research (Anthropic-оценка)", color=MUTED, fontsize=8.5, va="center")

# Bottom takeaway box
fig.patches.append(plt.Rectangle((0.045, 0.095), 0.91, 0.10, transform=fig.transFigure,
                                  facecolor=CARD, edgecolor=ACCENT, linewidth=1.2))
fig.text(0.065, 0.178, "Вывод:", color=ACCENT, fontsize=10.5, fontweight="bold", va="top")
fig.text(0.065, 0.150,
         "OpenAI, по собственным заявлениям и оценкам трекеров, оперирует ~1–1.4 млн GPU (~5.7 млн H100-экв.); Anthropic не раскрывает парк GPU/TPU\n"
         "напрямую, но законтрактовала мощность, эквивалентную ~1.4–6.4 млн ускорителей к 2027 г. Доля обучения в compute у обеих компаний,\n"
         "по независимым оценкам, сегодня выше доли инференса (~60–70% vs ~30–40%), но, по прогнозам, будет снижаться к 2027–2028.",
         color=TEXT, fontsize=9.0, va="top")

fig.text(0.045, 0.03, "Источники: OpenAI/Altman (X, июль 2025); flopper.io datacenter tracker (не офиц.); Anthropic/Google press releases (окт. 2025, апр. 2026);\n"
                       "TradingKey (конверсия ГВт→GPU); Epoch AI (arXiv 2504.16138); Lambda Finance (агрегатор, вторичный источник для Anthropic split).",
         color=MUTED, fontsize=6.8, va="top")

pdf.savefig(fig, facecolor=BG)
plt.close(fig)

# ======================================================================
# SLIDE 2 — Sail Research GLM-5.2 margin
# ======================================================================
fig = plt.figure(figsize=FIGSIZE, facecolor=BG)
fig.patch.set_facecolor(BG)

fig.text(0.045, 0.975, "С какой маржой Sail Research продаёт GLM 5.2",
          fontsize=17, color=TEXT, fontweight="bold", va="top")
fig.text(0.045, 0.93, "Цены — docs.sailresearch.com/pricing, 28 авг 2026 · маржа — оценка по bottom-up GPU-TCO модели (не по раскрытым данным Sail Research)",
          fontsize=8.8, color=MUTED, va="top")

# --- Col 1: pricing table
ax1 = fig.add_axes([0.045, 0.635, 0.285, 0.245])
ax1.set_facecolor(BG)
ax1.axis("off")
ax1.set_title("Цены GLM-5.2 (Sail), $/1M ток.", color=TEXT, fontsize=9.7, loc="left")
tiers = list(glm_pricing.keys())
col_labels = ["Tier", "In", "Cache", "Out"]
cell_text = [[t.replace(" (ASAP)","").replace("Default","Default"), f"${glm_pricing[t]['input']:.2f}", f"${glm_pricing[t]['cached']:.2f}", f"${glm_pricing[t]['output']:.2f}"] for t in tiers]
table = ax1.table(cellText=cell_text, colLabels=col_labels, loc="center", cellLoc="center",
                   colWidths=[0.36,0.22,0.22,0.22])
table.auto_set_font_size(False)
table.set_fontsize(8.6)
table.scale(1, 1.75)
for (r, c), cell in table.get_celld().items():
    cell.set_edgecolor("#2a3345")
    if r == 0:
        cell.set_facecolor("#1f2937")
        cell.set_text_props(color=TEXT, fontweight="bold")
    else:
        cell.set_facecolor(CARD)
        cell.set_text_props(color=TEXT)

# --- Col 2: who is Sail Research
ax2 = fig.add_axes([0.365, 0.635, 0.285, 0.245])
ax2.set_facecolor(BG)
ax2.axis("off")
ax2.set_title("Кто такой Sail Research", color=TEXT, fontsize=9.7, loc="left")
ax2.text(0, 0.95, "$80M (seed+A) @ $450M valuation,", color=TEXT, fontsize=8.2, va="top")
ax2.text(0, 0.82, "Kleiner Perkins / Sequoia, июнь 2026", color=TEXT, fontsize=8.2, va="top")
ax2.text(0, 0.64, "Позиционирование: throughput-over-\nlatency инфра для агентов", color=TEXT, fontsize=8.2, va="top")
ax2.text(0, 0.36, "Собственная заявка (PR компании):", color=MUTED, fontsize=7.8, va="top")
ax2.text(0, 0.24, "«fully utilizes every computer in\nour fleet» + «до 10× дешевле\nконкурентов» — т.е. заявляют\nвысокую загрузку GPU по дизайну", color=MUTED, fontsize=7.8, va="top", style="italic")
ax2.set_xlim(0,1); ax2.set_ylim(0,1)

# --- Col 3: price cross-check vs model creator and cheapest reseller
ax3b = fig.add_axes([0.685, 0.635, 0.285, 0.245])
ax3b.set_facecolor(BG)
ax3b.axis("off")
ax3b.set_title("Сверка цены: кто сколько берёт", color=TEXT, fontsize=9.7, loc="left")
bench_labels = ["Z.ai\n(создатель\nмодели)", "Sail\nDefault", "OpenRouter\n(дешевле\nвсех)"]
bench_out = [price_benchmarks_usd_per_1M[k]["output"] for k in price_benchmarks_usd_per_1M]
bench_colors = [ACCENT2, ACCENT, "#4fd18a"]
bx = np.arange(3)
bars = ax3b.bar(bx, bench_out, color=bench_colors, width=0.55)
for b, v in zip(bars, bench_out):
    ax3b.text(b.get_x()+b.get_width()/2, v+0.08, f"${v:.2f}", ha="center", color=TEXT, fontsize=8, fontweight="bold")
ax3b.set_xticks(bx)
ax3b.set_xticklabels(bench_labels, color=MUTED, fontsize=7.3)
ax3b.set_ylabel("$/1M output ток.", color=MUTED, fontsize=7.5)
ax3b.tick_params(colors=MUTED, labelsize=7)
for spine in ax3b.spines.values():
    spine.set_visible(False)
ax3b.set_ylim(0, 5.2)

# --- model spec + cost model, one compact line each
fig.text(0.045, 0.605, "Модель: 744B всего / ~40B активных на токен (MoE), FP8 на узле 8×H200 (recipes.vllm.ai/zai-org/GLM-5.2)  ·  "
                       "Себестоимость: FLOPs/ток. ≈ 2×активные_парам., cost = GPU-часы × $/GPU-час × MFU⁻¹ × overhead",
         color=MUTED, fontsize=8.0, va="top")
fig.text(0.045, 0.578, "Три сценария различаются по MFU (15–40%), цене GPU-часа ($2–5) и overhead-множителю (1.5×–5×) — раскладка допущений в notebook.",
         color=MUTED, fontsize=8.0, va="top")

# --- Bottom: margin bar chart across scenarios and tiers
ax3 = fig.add_axes([0.045, 0.185, 0.91, 0.175])
ax3.set_facecolor(BG)
scenario_names = list(results.keys())
short_names = ["Bear — рыночная аренда,\nнизкая утилизация", "Base — резервная ёмкость,\nумеренная утилизация", "Bull — своё железо,\nвысокая утилизация\n(ближе к заявкам Sail)"]
x = np.arange(len(scenario_names))
width = 0.25
tier_colors = {"Default (ASAP)": ACCENT, "Balanced": "#8ecff5", "Flex": ACCENT2}
for i, tier in enumerate(glm_pricing.keys()):
    vals = [results[s]["margins"][tier] for s in scenario_names]
    bars = ax3.bar(x + (i-1)*width, vals, width, label=tier, color=tier_colors[tier])
    for b, v in zip(bars, vals):
        ax3.text(b.get_x()+b.get_width()/2, v + (2 if v>=0 else -6), f"{v:.0f}%", ha="center",
                  color=TEXT, fontsize=8, fontweight="bold")
ax3.axhline(0, color=MUTED, linewidth=0.8)
ax3.set_xticks(x)
ax3.set_xticklabels(short_names, color=TEXT, fontsize=7.8)
ax3.set_ylabel("Валовая маржа, %", color=MUTED, fontsize=8.5)
ax3.tick_params(colors=MUTED, labelsize=8)
for spine in ax3.spines.values():
    spine.set_visible(False)
ax3.legend(loc="upper center", fontsize=8, facecolor=CARD, edgecolor="#2a3345", labelcolor=TEXT, ncol=3, framealpha=0.95)
ax3.set_ylim(-140, 175)
ax3.tick_params(axis="x", pad=10)

fig.patches.append(plt.Rectangle((0.045, 0.375), 0.91, 0.10, transform=fig.transFigure,
                                  facecolor=CARD, edgecolor=ACCENT, linewidth=1.2))
fig.text(0.065, 0.462, "Вывод:", color=ACCENT, fontsize=10.5, fontweight="bold", va="top")
fig.text(0.065, 0.436,
         "Даже в базовом сценарии маржа по всем тарифам положительная (~65–79%); в минус модель уходит только при совпадении худших предположений сразу по трём параметрам.\n"
         "Sail сама заявляет высокую загрузку фермы и специализированные inference-движки — это довод в пользу Base/Bull, а не Bear.\n"
         "Цена Sail ($3.00) на ~30% ниже официальной цены Z.ai ($4.40) за output, но выше минимальной рыночной ($1.56) — конкурентный reseller-прайсинг, не демпинг.",
         color=TEXT, fontsize=8.0, va="top")

fig.text(0.045, 0.03, "Источники: docs.sailresearch.com/pricing (28.08.2026); recipes.vllm.ai/zai-org/GLM-5.2; NVIDIA H200 datasheet (FP8 TFLOPS); Hyperbolic/GMI Cloud/Jarvislabs/getdeploying.com (аренда H200, авг. 2026);\n"
                       "SiliconANGLE / Fortune / AI Weekly (о Sail Research, июнь 2026); VentureBeat / layer3labs.io / orcarouter.ai / aipricing.guru (цена Z.ai); OpenRouter (минимальная цена reseller'ов). Полный расчёт — в notebook.",
         color=MUTED, fontsize=6.6, va="top")

pdf.savefig(fig, facecolor=BG)
plt.close(fig)

pdf.close()
print("saved slides.pdf")

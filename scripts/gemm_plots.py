#!/usr/bin/env python3
# Διαγράμματα για το Παραδοτέο ΙΙΙ (GEMM σε CUDA).
# Δεδομένα: ΠΡΑΓΜΑΤΙΚΕΣ μετρήσεις (ίδια συνεδρία) — T4 (Colab, sm_75) & RTX 3070 Ti (τοπικά, sm_86).
# Παράγει: fig1..fig5 PNG. Τρέξε:  python3 gemm_plots.py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C = {"naive": "#d62728", "tiled": "#ff7f0e", "register": "#2ca02c", "cublas": "#1f77b4"}

# ---------------------------------------------------------------- Σενάριο A
N = [128, 256, 512, 1024, 2048, 4096]
T4 = {
    "naive":    [194.6, 280.2, 307.8, 286.9, 438.6, 444.8],
    "tiled":    [409.6, 700.2, 826.9, 856.3, 696.8, 685.5],
    "register": [133.2, 518.6, 1962.2, 2852.5, 3337.0, 2524.7],
    "cublas":   [317.4, 862.3, 2666.4, 3173.6, 5597.7, 4000.1],
}
RTX = {
    "naive":    [602.4, 1129.9, 1304.2, 1364.6, 1339.0, 1414.7],
    "tiled":    [682.7, 1424.7, 1761.7, 1863.5, 1869.5, 1878.5],
    "register": [165.5, 694.2, 2617.0, 6405.5, 5958.2, 6431.1],
    "cublas":   [706.2, 3588.6, 8511.2, 12618.2, 13781.2, 13773.5],
}

def panel(ax, data, title):
    for k in ["naive", "tiled", "register", "cublas"]:
        ax.plot(N, data[k], "o-", color=C[k], label=k)
    ax.set_xscale("log", base=2); ax.set_xticks(N); ax.set_xticklabels(N)
    ax.set_xlabel("N (τετράγωνο)"); ax.set_ylabel("GFLOPS"); ax.set_title(title)
    ax.grid(True, ls=":", alpha=.5); ax.legend()

fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6), sharey=False)
panel(a, RTX, "RTX 3070 Ti (Ampere, sm_86)")
panel(b, T4,  "NVIDIA T4 (Turing, sm_75)")
fig.suptitle("Σενάριο A — Απόδοση (GFLOPS) συναρτήσει του N", fontweight="bold")
fig.tight_layout(); fig.savefig("fig1_gflops_vs_N.png", dpi=130); plt.close(fig)

# ---------------------------------------------------------------- fig2: register RTX vs T4
fig, ax = plt.subplots(figsize=(7, 4.6))
ax.plot(N, RTX["register"], "o-", color="#2ca02c", label="RTX 3070 Ti")
ax.plot(N, T4["register"],  "s--", color="#9467bd", label="T4 (Colab)")
ax.set_xscale("log", base=2); ax.set_xticks(N); ax.set_xticklabels(N)
ax.set_xlabel("N"); ax.set_ylabel("GFLOPS")
ax.set_title("Register blocking: RTX 3070 Ti vs T4")
ax.grid(True, ls=":", alpha=.5); ax.legend()
ax.annotate("το T4 υποχωρεί σε μεγάλο N\n(κορεσμός L2/bandwidth)", xy=(4096, T4["register"][-1]),
            xytext=(1100, 1500), arrowprops=dict(arrowstyle="->"))
fig.tight_layout(); fig.savefig("fig2_register_rtx_vs_t4.png", dpi=130); plt.close(fig)

# ---------------------------------------------------------------- Σενάριο C: GFLOPS vs K
K = [128, 256, 512, 1024, 2048, 4096]
reg_K_T4  = [3065.7, 3435.7, 3491.8, 2797.7, 2662.1, 2060.1]
reg_K_RTX = [5388.5, 5829.6, 5890.9, 6056.1, 6111.5, 6068.1]
fig, ax = plt.subplots(figsize=(7, 4.6))
ax.plot(K, reg_K_RTX, "o-", color="#2ca02c", label="RTX (αυξάνεται με K ↑)")
ax.plot(K, reg_K_T4,  "s--", color="#9467bd", label="T4 (μειώνεται με K ↓)")
ax.set_xscale("log", base=2); ax.set_xticks(K); ax.set_xticklabels(K)
ax.set_xlabel("K (εσωτερική διάσταση), M=N=2048"); ax.set_ylabel("Register GFLOPS")
ax.set_title("Σενάριο C — Εξάρτηση από το K (η αντίστροφη τάση)")
ax.grid(True, ls=":", alpha=.5); ax.legend()
fig.tight_layout(); fig.savefig("fig3_gflops_vs_K.png", dpi=130); plt.close(fig)

# ---------------------------------------------------------------- Σενάριο D: kernel vs total
shapes = ["Cube\n2048³", "Flat-K\n8192·128·8192", "Deep-K\n1024·8192·1024",
          "Tall\n4096·8192·256", "Skinny\n256·8192·4096"]
kT4, ttT4   = [7.87, 8.93, 12.73, 7.72, 7.06], [19.01, 71.36, 27.98, 40.25, 39.59]
kRTX, ttRTX = [2.97, 2.88, 2.50, 2.62, 2.65],  [7.94, 33.28, 16.76, 17.41, 17.15]
x = np.arange(len(shapes)); w = 0.38
fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.8))
for ax, k, tt, name in [(a, kRTX, ttRTX, "RTX 3070 Ti"), (b, kT4, ttT4, "T4 (Colab)")]:
    ax.bar(x - w/2, k,  w, label="kernel ms", color="#2ca02c")
    ax.bar(x + w/2, tt, w, label="total ms (με PCIe)", color="#1f77b4")
    ax.set_xticks(x); ax.set_xticklabels(shapes, fontsize=8)
    ax.set_ylabel("ms"); ax.set_title(name + " — ίδιες πράξεις (~1,72·10¹⁰)")
    ax.grid(True, axis="y", ls=":", alpha=.5); ax.legend()
fig.suptitle("Σενάριο D — kernel vs total ανά σχήμα (το flat-K κυριαρχείται από τη μεταφορά)",
             fontweight="bold")
fig.tight_layout(); fig.savefig("fig4_total_vs_shape.png", dpi=130); plt.close(fig)

# ---------------------------------------------------------------- Σενάριο E: φόρος ορίων
# Register GFLOPS κανονικοποιημένα ως % του N=1024 (ακριβές πολλαπλάσιο tile).
labels = ["1023", "1024", "1025"]
rtx_e = [6317.6, 6355.8, 4024.7]
t4_e  = [2571.3, 2708.9, 2646.7]
rtx_n = [v / rtx_e[1] * 100 for v in rtx_e]
t4_n  = [v / t4_e[1] * 100 for v in t4_e]
x = np.arange(len(labels)); w = 0.38
fig, ax = plt.subplots(figsize=(7, 4.6))
ax.bar(x - w/2, rtx_n, w, label="RTX 3070 Ti", color="#2ca02c")
ax.bar(x + w/2, t4_n,  w, label="T4 (Colab)",  color="#9467bd")
ax.axhline(100, color="gray", ls=":", lw=1)
ax.set_ylim(0, 116)
ax.set_xticks(x); ax.set_xticklabels([f"N={l}" for l in labels])
ax.set_ylabel("Register GFLOPS (% του N=1024)")
ax.set_title("Σενάριο E — «Φόρος ορίων» στο N=1025 (register)")
ax.grid(True, axis="y", ls=":", alpha=.5); ax.legend(loc="upper left")
ax.annotate("−37 %", xy=(2 - w/2, rtx_n[2]), xytext=(2 - w/2 - 0.12, rtx_n[2] - 18),
            fontweight="bold", color="#d62728")
ax.annotate("−2 %", xy=(2 + w/2, t4_n[2]), xytext=(2 + w/2 - 0.05, 103),
            fontweight="bold")
fig.tight_layout(); fig.savefig("fig5_boundary_tax.png", dpi=130); plt.close(fig)

print("OK: fig1..fig5 regenerated from real data.")

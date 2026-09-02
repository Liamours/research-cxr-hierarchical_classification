from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)

# ORPHANED as of 2026-08-28: this path was on a different machine/user profile
# (C:\Users\lulay\...) and analysis_export/ does not exist anywhere in the
# current project tree. This script plots a lambda/MC-dropout grid that was
# never the source of the manuscript's actual figures (fig1/fig2/fig4_calibration/
# fig5_gradcam) -- see revision_plan.md's D4 findings. Left as historical
# reference; point EXPORT at a real analysis_export/ directory before running.
EXPORT = Path(r"C:\Users\lulay\Desktop\research-cxr\analysis_export")
OUT = EXPORT / "figures"
OUT.mkdir(exist_ok=True)

RUNS = ["flat_uqoff", "flat_mc", "hier_uqoff", "hier_mc", "hier_lam0p1", "hier_lam1p0"]
LABELS = {
    "flat_uqoff": "Flat",
    "flat_mc": "Flat, MC-Dropout",
    "hier_uqoff": "Hierarchical, λ=0.5",
    "hier_mc": "Hierarchical, MC-Dropout, λ=0.5",
    "hier_lam0p1": "Hierarchical, λ=0.1",
    "hier_lam1p0": "Hierarchical, λ=1.0",
}
TICKS = {
    "flat_uqoff": "Flat",
    "flat_mc": "Flat, MC",
    "hier_uqoff": "Hier, λ=0.5",
    "hier_mc": "Hier, MC, λ=0.5",
    "hier_lam0p1": "Hier, λ=0.1",
    "hier_lam1p0": "Hier, λ=1.0",
}
COLORS = {r: plt.cm.viridis(x) for r, x in zip(RUNS, np.linspace(0.05, 0.9, len(RUNS)))}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.weight": "normal",
    "axes.titleweight": "normal",
    "axes.labelweight": "normal",
    "font.size": 11,
})


def fig1_headline_metrics():
    df = pd.read_csv(EXPORT / "results_all_metrics.csv", index_col="metric")
    df = df[RUNS]
    metrics = [
        ("auroc_macro", "AUROC (macro)"),
        ("map_macro", "mAP (macro)"),
        ("aurc_macro", "AURC (macro)"),
        ("ece", "ECE"),
        ("hcv_rate", "HCV (%)"),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.2))
    for ax, (key, title) in zip(axes, metrics):
        vals = df.loc[key].values
        if key == "hcv_rate":
            vals = vals * 100
        bars = ax.bar(range(len(RUNS)), vals, color=[COLORS[r] for r in RUNS])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=7)
        ax.set_title(title)
        ax.set_xticks(range(len(RUNS)))
        ax.set_xticklabels([TICKS[r] for r in RUNS], rotation=45, ha="right", fontsize=8)
        ax.margins(y=0.15)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "1_headline_metrics.png", dpi=200)
    plt.close(fig)


def fig2_calibration_vs_lambda():
    df = pd.read_csv(EXPORT / "results_all_metrics.csv", index_col="metric")
    lam_runs = ["flat_uqoff", "hier_lam0p1", "hier_uqoff", "hier_lam1p0"]
    lam_vals = [0.0, 0.1, 0.5, 1.0]
    ece = df.loc["ece", lam_runs].values
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(lam_vals, ece, marker="o", color=plt.cm.viridis(0.4))
    for x, y in zip(lam_vals, ece):
        ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8)
    ax.set_title("ECE vs HBCE penalty weight")
    ax.set_xlabel("lambda")
    ax.set_ylabel("ECE")
    ax.margins(y=0.15)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "2_calibration_vs_lambda.png", dpi=200)
    plt.close(fig)


def fig3_hcv_vs_lambda():
    df = pd.read_csv(EXPORT / "results_all_metrics.csv", index_col="metric")
    lam_runs = ["flat_uqoff", "hier_lam0p1", "hier_uqoff", "hier_lam1p0"]
    lam_vals = [0.0, 0.1, 0.5, 1.0]
    hcv = df.loc["hcv_rate", lam_runs].values * 100
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(lam_vals, hcv, marker="o", color=plt.cm.viridis(0.7))
    for x, y in zip(lam_vals, hcv):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8)
    ax.set_title("Hierarchy violation rate vs lambda")
    ax.set_xlabel("lambda")
    ax.set_ylabel("HCV (%)")
    ax.margins(y=0.15)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "3_hcv_vs_lambda.png", dpi=200)
    plt.close(fig)


def fig4_training_curves():
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for r in RUNS:
        tl = pd.read_csv(EXPORT / "runs" / r / "train_log.csv")
        ax.plot(tl["epoch"], tl["val_auroc_macro"], color=COLORS[r], label=LABELS[r], linewidth=2)
    ax.set_title("Validation AUROC by epoch")
    ax.set_xlabel("epoch")
    ax.set_ylabel("val AUROC (macro)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "4_training_curves.png", dpi=200)
    plt.close(fig)


def fig5_per_class_auroc_heatmap():
    labels = None
    data = {}
    for r in RUNS:
        js = json.loads((EXPORT / "runs" / r / "eval_metrics_test.json").read_text())
        pc = js["report"]["auroc"]["per_class"]
        if labels is None:
            labels = [c for c, v in pc.items() if v == v]
        data[r] = [pc[c] for c in labels]
    mat = np.array([data[r] for r in RUNS]).T
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(mat, cmap="viridis", aspect="auto")
    vmin, vmax = np.nanmin(mat), np.nanmax(mat)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            txt_color = "white" if (v - vmin) / (vmax - vmin) < 0.5 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color=txt_color)
    ax.set_xticks(range(len(RUNS)))
    ax.set_xticklabels([TICKS[r] for r in RUNS], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("Per-class AUROC (test)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(OUT / "5_per_class_auroc_heatmap.png", dpi=200)
    plt.close(fig)


def fig5b_per_class_auroc_3d():
    labels = None
    data = {}
    for r in RUNS:
        js = json.loads((EXPORT / "runs" / r / "eval_metrics_test.json").read_text())
        pc = js["report"]["auroc"]["per_class"]
        if labels is None:
            labels = [c for c, v in pc.items() if v == v]
        data[r] = [pc[c] for c in labels]
    mat = np.array([data[r] for r in RUNS]).T  # rows=labels, cols=runs

    n_labels, n_runs = mat.shape
    x_spacing, y_spacing = 1.6, 2.4
    xpos, ypos = np.meshgrid(np.arange(n_runs) * x_spacing, np.arange(n_labels) * y_spacing)
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = np.zeros_like(xpos, dtype=float)
    dx, dy = 1.3, 1.8
    dz = mat.ravel()

    vmin, vmax = np.nanmin(dz), np.nanmax(dz)
    colors = plt.cm.viridis((dz - vmin) / (vmax - vmin))

    fig = plt.figure(figsize=(18, 15))
    ax = fig.add_subplot(projection="3d")
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, shade=True)

    ax.set_xticks(np.arange(n_runs) * x_spacing + dx / 2)
    ax.set_xticklabels([TICKS[r] for r in RUNS], rotation=-25, ha="left", fontsize=11)
    ax.set_yticks(np.arange(n_labels) * y_spacing + dy / 2)
    ax.set_yticklabels(labels, rotation=-10, ha="left", fontsize=10)
    ax.set_zlabel("AUROC", fontsize=13, labelpad=14)
    ax.set_title("Per-class AUROC (test)", fontsize=16, pad=20)
    ax.tick_params(axis="x", pad=0)
    ax.tick_params(axis="y", pad=0)
    ax.tick_params(axis="z", pad=8)
    ax.view_init(elev=16, azim=-62)
    ax.set_box_aspect((1.1, 2.0, 1.0))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.05)
    fig.savefig(OUT / "5b_per_class_auroc_3d.png", dpi=220)
    plt.close(fig)


def fig8_hierarchy_diagram():
    PARENT_COLOR = "#BFDCF2"
    CHILD_COLOR = "#c9c9c9"
    EDGE_COLOR = "#c9c9c9"

    tree = [
        ("Pneumonia", [
            "Coronavirus Disease 2019 Pneumonia (COVID-19)",
            "Aspiration Pneumonia",
            "Other Viral Pneumonia",
        ]),
        ("Interstitial Lung Disease (ILD)", [
            "Idiopathic Pulmonary Fibrosis (IPF)",
            "Cryptogenic Organizing Pneumonia (COP)",
            "Hypersensitivity Pneumonitis",
            "Silicosis",
            "Asbestosis",
            "Other Pneumoconiosis",
            "Sarcoidosis",
        ]),
        ("Tuberculosis", ["Post-Tuberculosis Obstructive Syndrome"]),
        ("Pleural Effusion", ["Pleural Empyema"]),
        ("Pulmonary Hypertension", ["Cor Pulmonale"]),
    ]

    fig, ax = plt.subplots(figsize=(13, 11))
    y = 0
    row_h = 1.0
    parent_x, child_x = 0.0, 2.9

    for parent, children in tree:
        ax.text(parent_x, -y, parent, fontsize=12, va="center", ha="left",
                bbox=dict(boxstyle="round,pad=0.35", facecolor=PARENT_COLOR, edgecolor="none"))
        p_y = y
        for child in children:
            y += row_h
            ax.plot([parent_x + 0.05, child_x - 0.05], [-p_y + 0.02, -y],
                    color=EDGE_COLOR, linewidth=1.5, zorder=1)
            ax.text(child_x, -y, child, fontsize=10.5, va="center", ha="left",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=CHILD_COLOR, edgecolor="none"))
        y += row_h * 0.6

    ax.set_xlim(-0.3, 8.8)
    ax.set_ylim(-y + 0.5, 1.0)
    ax.axis("off")
    ax.set_title("Label hierarchy: 13 edges, 5 parents (18 of 51 canonical labels)")

    legend_items = [
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=PARENT_COLOR, markersize=12, label="parent condition"),
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=CHILD_COLOR, markersize=12, label="child condition"),
    ]
    ax.legend(handles=legend_items, loc="lower right", frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT / "8_hierarchy_diagram.png", dpi=220)
    plt.close(fig)


def fig6_reliability_diagram():
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    for r in ["flat_uqoff", "hier_lam1p0"]:
        js = json.loads((EXPORT / "runs" / r / "eval_metrics_test.json").read_text())
        bins = js["report"]["calibration"]["reliability"]["bins"]
        conf = [b["avg_conf"] for b in bins if b["count"] > 0]
        acc = [b["avg_acc"] for b in bins if b["count"] > 0]
        ax.plot(conf, acc, marker="o", markersize=4, color=COLORS[r], label=LABELS[r])
    ax.set_title("Reliability diagram (test)")
    ax.set_xlabel("predicted confidence")
    ax.set_ylabel("empirical accuracy")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "6_reliability_diagram.png", dpi=200)
    plt.close(fig)


def fig7_test_coverage_by_dataset():
    df = pd.read_csv(EXPORT / "test_set_composition.csv")
    single = df[df["test_datasets"].notna() & ~df["test_datasets"].str.contains(",", na=False) & (df["test_datasets"] != "")]
    totals = single.groupby("test_datasets")["n_test_applicable"].max()
    all_datasets = ["nih-cxr14", "covidx-cxr4", "vindr-pcxr", "chexpert", "vindr-cxr", "tbx11k"]
    totals = totals.reindex(all_datasets, fill_value=0)
    norm = totals / totals.max()
    colors = [plt.cm.viridis(v) for v in norm]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(range(len(all_datasets)), totals.values, color=colors)
    for b, v in zip(bars, totals.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,.0f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(all_datasets)))
    ax.set_xticklabels(all_datasets, rotation=45, ha="right", fontsize=9)
    ax.set_title("Test images per dataset")
    ax.set_ylabel("n test images")
    ax.margins(y=0.12)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "7_test_coverage_by_dataset.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fig1_headline_metrics()
    fig2_calibration_vs_lambda()
    fig3_hcv_vs_lambda()
    fig4_training_curves()
    fig5_per_class_auroc_heatmap()
    fig5b_per_class_auroc_3d()
    fig6_reliability_diagram()
    fig7_test_coverage_by_dataset()
    fig8_hierarchy_diagram()
    print("wrote 7 figures to", OUT)

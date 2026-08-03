"""Read-only post-acceptance analysis for an accepted W3 package.

Public seam: ``analyze_accepted_w3_package`` reads sealed acceptance artifacts
and writes a separate analysis directory. It never mutates the package, never
copies checkpoints, and never changes acceptance state.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
# Reproducible vector exports (CreationDate / ModDate).
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from lap.verifiers.cover.w3_contracts import content_hash
from lap.verifiers.cover.w3_contracts import write_canonical_json

ANALYSIS_SCHEMA = "osx_cover_w3_post_acceptance_analysis_v1"
REQUIRED_CONDITIONS = (
    "circular:+x",
    "circular:+y",
    "circular:-x",
    "circular:-y",
    "square:+x",
    "square:+y",
    "square:-x",
    "square:-y",
)

# Publication styling (academic-plotting Workflow 2 — Ocean Dusk).
_COLORS = ["#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51", "#0072B2", "#56B4E9", "#8C8C8C"]
_OUR = "#E76F51"
_BASELINE = "#B0BEC5"


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "legend.frameon": False,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.15,
            "grid.linestyle": "-",
            "lines.linewidth": 1.8,
            "lines.markersize": 5,
            "svg.hashsalt": "w3-post-acceptance-analysis-v1",
        }
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_file(path: Path, *, label: str) -> Path:
    if not path.is_file():
        raise ValueError(f"incomplete package: missing {label} ({path})")
    return path


def _resolve_and_guard(*, package_root: Path, output_root: Path) -> tuple[Path, Path]:
    package = package_root.resolve()
    output = output_root.resolve()
    if not package.is_dir():
        raise ValueError(f"package_root is not a directory: {package}")
    try:
        output.relative_to(package)
    except ValueError:
        return package, output
    raise ValueError("output_root must not be inside package_root")


def _validate_acceptance(acceptance: dict[str, Any]) -> None:
    if "training" not in acceptance or not isinstance(acceptance["training"], list):
        raise ValueError("incomplete package: acceptance.json missing training epochs")
    if not acceptance["training"]:
        raise ValueError("incomplete package: acceptance.json training is empty")
    evaluation = acceptance.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError("incomplete package: acceptance.json missing evaluation")
    for key in ("pool", "retrieval", "margins", "conditions"):
        if key not in evaluation:
            raise ValueError(f"incomplete package: acceptance.json missing evaluation.{key}")
    conditions = evaluation["conditions"]
    missing = [name for name in REQUIRED_CONDITIONS if name not in conditions]
    if missing:
        raise ValueError(f"incomplete package: missing conditions {missing}")
    if "authority" not in acceptance:
        raise ValueError("incomplete package: acceptance.json missing authority")


def _validate_ablation(ablation_doc: dict[str, Any]) -> dict[str, Any]:
    ablation = ablation_doc.get("ablation")
    if not isinstance(ablation, dict):
        raise ValueError("incomplete package: paired_ablation.json missing ablation")
    if "wrist_benefit_established" not in ablation:
        raise ValueError("incomplete package: ablation missing wrist_benefit_established")
    if "two_view_minus_base_only" not in ablation or "paired_ci95" not in ablation:
        raise ValueError("incomplete package: ablation missing paired comparison fields")
    return ablation


def _extract_summary(
    *,
    package: Path,
    acceptance: dict[str, Any],
    ablation: dict[str, Any],
) -> dict[str, Any]:
    training = acceptance["training"]
    evaluation = acceptance["evaluation"]
    pool = evaluation["pool"]
    retrieval = evaluation["retrieval"]
    margins = evaluation["margins"]
    conditions = {
        name: {
            "count": int(conditions_row["count"]),
            "action_to_semantic_top1": float(conditions_row["action_to_semantic_top1"]),
            "semantic_to_action_top1": float(conditions_row["semantic_to_action_top1"]),
            "aligned_minus_shuffled_mean": float(conditions_row["aligned_minus_shuffled_mean"]),
            "aligned_minus_nearby_mean": float(conditions_row["aligned_minus_nearby_mean"]),
        }
        for name, conditions_row in (
            (name, evaluation["conditions"][name]) for name in REQUIRED_CONDITIONS
        )
    }
    summary: dict[str, Any] = {
        "schema": ANALYSIS_SCHEMA,
        "package_root": str(package),
        "authority": acceptance["authority"],
        "offline_only_authority": True,
        "wrist_benefit_established": bool(ablation["wrist_benefit_established"]),
        "training": {
            "epochs": [int(row["epoch"]) for row in training],
            "loss": [float(row["loss"]) for row in training],
            "validation_loss": [float(row["validation_loss"]) for row in training],
        },
        "chance_baselines": {
            "pool_count": int(pool["count"]),
            "top1": float(pool["top1_chance"]),
            "top5": float(pool["top5_chance"]),
        },
        "retrieval": {
            "action_to_semantic_top1": float(retrieval["action_to_semantic_top1"]),
            "action_to_semantic_top1_ci95": [float(x) for x in retrieval["action_to_semantic_top1_ci95"]],
            "action_to_semantic_top5": float(retrieval["action_to_semantic_top5"]),
            "semantic_to_action_top1": float(retrieval["semantic_to_action_top1"]),
            "semantic_to_action_top1_ci95": [float(x) for x in retrieval["semantic_to_action_top1_ci95"]],
            "semantic_to_action_top5": float(retrieval["semantic_to_action_top5"]),
        },
        "margins": {
            "aligned_minus_shuffled": {
                "mean": float(margins["aligned_minus_shuffled"]["mean"]),
                "ci95": [float(x) for x in margins["aligned_minus_shuffled"]["ci95"]],
                "fraction_gt_zero": float(margins["aligned_minus_shuffled"]["fraction_gt_zero"]),
            },
            "aligned_minus_nearby": {
                "mean": float(margins["aligned_minus_nearby"]["mean"]),
                "ci95": [float(x) for x in margins["aligned_minus_nearby"]["ci95"]],
                "fraction_gt_zero": float(margins["aligned_minus_nearby"]["fraction_gt_zero"]),
            },
        },
        "conditions": conditions,
        "paired_ablation": {
            "wrist_benefit_established": bool(ablation["wrist_benefit_established"]),
            "two_view_minus_base_only": {
                key: float(ablation["two_view_minus_base_only"][key])
                for key in (
                    "action_to_semantic_top1",
                    "semantic_to_action_top1",
                    "aligned_minus_shuffled",
                    "aligned_minus_nearby",
                )
            },
            "paired_ci95": {
                key: [float(x) for x in ablation["paired_ci95"][key]]
                for key in (
                    "action_to_semantic_top1",
                    "semantic_to_action_top1",
                    "aligned_minus_shuffled",
                    "aligned_minus_nearby",
                )
            },
        },
        "figures": [
            "figures/fig_loss_curves.png",
            "figures/fig_loss_curves.pdf",
            "figures/fig_retrieval_vs_chance.png",
            "figures/fig_retrieval_vs_chance.pdf",
            "figures/fig_margins_ci95.png",
            "figures/fig_margins_ci95.pdf",
            "figures/fig_conditions_heatmap.png",
            "figures/fig_conditions_heatmap.pdf",
            "figures/fig_paired_ablation.png",
            "figures/fig_paired_ablation.pdf",
        ],
        "caveats": [
            "Authority is offline-only: recorded-data integration, not robot or candidate-selection authority.",
            (
                "Wrist benefit was established."
                if ablation["wrist_benefit_established"]
                else "Wrist benefit was not established (paired CI includes zero or fails the seal rule)."
            ),
            "This analysis is read-only visualization of the accepted package; it does not retrain or retune thresholds.",
            "Checkpoints were not copied into the analysis directory.",
        ],
    }
    summary["content_hash"] = content_hash(summary)
    return summary


def _save_fig(fig: plt.Figure, figures_dir: Path, stem: str) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / f"{stem}.pdf")
    fig.savefig(figures_dir / f"{stem}.png", dpi=300)
    plt.close(fig)


def _plot_loss_curves(summary: dict[str, Any], figures_dir: Path) -> None:
    epochs = summary["training"]["epochs"]
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.plot(epochs, summary["training"]["loss"], color=_COLORS[0], marker="o", markevery=max(1, len(epochs) // 8), label="train loss")
    ax.plot(
        epochs,
        summary["training"]["validation_loss"],
        color=_OUR,
        marker="s",
        markevery=max(1, len(epochs) // 8),
        label="validation loss",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and validation loss")
    ax.legend(loc="upper right")
    _save_fig(fig, figures_dir, "fig_loss_curves")


def _plot_retrieval_vs_chance(summary: dict[str, Any], figures_dir: Path) -> None:
    labels = ["A→S top-1", "A→S top-5", "S→A top-1", "S→A top-5"]
    observed = [
        summary["retrieval"]["action_to_semantic_top1"],
        summary["retrieval"]["action_to_semantic_top5"],
        summary["retrieval"]["semantic_to_action_top1"],
        summary["retrieval"]["semantic_to_action_top5"],
    ]
    chance = [
        summary["chance_baselines"]["top1"],
        summary["chance_baselines"]["top5"],
        summary["chance_baselines"]["top1"],
        summary["chance_baselines"]["top5"],
    ]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    ax.bar(x - width / 2, observed, width, color=_OUR, label="observed")
    ax.bar(x + width / 2, chance, width, color=_BASELINE, label="exact chance")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Hit rate")
    ax.set_title("Retrieval versus exact chance")
    ax.legend(loc="upper left")
    _save_fig(fig, figures_dir, "fig_retrieval_vs_chance")


def _plot_margins(summary: dict[str, Any], figures_dir: Path) -> None:
    names = ["shuffled margin", "nearby margin"]
    means = [
        summary["margins"]["aligned_minus_shuffled"]["mean"],
        summary["margins"]["aligned_minus_nearby"]["mean"],
    ]
    cis = [
        summary["margins"]["aligned_minus_shuffled"]["ci95"],
        summary["margins"]["aligned_minus_nearby"]["ci95"],
    ]
    yerr = np.array([[m - lo, hi - m] for m, (lo, hi) in zip(means, cis, strict=True)]).T
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    ax.bar(names, means, color=[_COLORS[1], _COLORS[4]], yerr=yerr, capsize=4, error_kw={"elinewidth": 1.2})
    ax.axhline(0.0, color="#8C8C8C", linewidth=1.0)
    ax.set_ylabel("Aligned − negative (mean)")
    ax.set_title("Margins with 95% CI")
    _save_fig(fig, figures_dir, "fig_margins_ci95")


def _plot_conditions_heatmap(summary: dict[str, Any], figures_dir: Path) -> None:
    # 2 shapes × 4 directions; metric = action→semantic top-1.
    shapes = ("circular", "square")
    dirs = ("+x", "+y", "-x", "-y")
    matrix = np.zeros((2, 4), dtype=np.float64)
    for i, shape in enumerate(shapes):
        for j, direction in enumerate(dirs):
            key = f"{shape}:{direction}"
            matrix[i, j] = summary["conditions"][key]["action_to_semantic_top1"]
    fig, ax = plt.subplots(figsize=(4.2, 2.4))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0.0)
    ax.set_xticks(range(4), dirs)
    ax.set_yticks(range(2), shapes)
    ax.set_title("Eight-condition A→S top-1")
    for i in range(2):
        for j in range(4):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save_fig(fig, figures_dir, "fig_conditions_heatmap")


def _plot_paired_ablation(summary: dict[str, Any], figures_dir: Path) -> None:
    labels = ["A→S top-1", "S→A top-1", "shuffled margin", "nearby margin"]
    keys = (
        "action_to_semantic_top1",
        "semantic_to_action_top1",
        "aligned_minus_shuffled",
        "aligned_minus_nearby",
    )
    deltas = [summary["paired_ablation"]["two_view_minus_base_only"][k] for k in keys]
    cis = [summary["paired_ablation"]["paired_ci95"][k] for k in keys]
    yerr = np.array([[d - lo, hi - d] for d, (lo, hi) in zip(deltas, cis, strict=True)]).T
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    ax.bar(labels, deltas, color=_COLORS[1], yerr=yerr, capsize=4, error_kw={"elinewidth": 1.2})
    ax.axhline(0.0, color="#8C8C8C", linewidth=1.0)
    ax.set_ylabel("Two-view − base-only")
    title = "Paired ablation (wrist benefit established)" if summary["wrist_benefit_established"] else "Paired ablation (wrist benefit not established)"
    ax.set_title(title)
    ax.tick_params(axis="x", labelrotation=15)
    _save_fig(fig, figures_dir, "fig_paired_ablation")


def _render_report(summary: dict[str, Any]) -> str:
    r = summary["retrieval"]
    m_s = summary["margins"]["aligned_minus_shuffled"]
    m_n = summary["margins"]["aligned_minus_nearby"]
    chance = summary["chance_baselines"]
    wrist = summary["wrist_benefit_established"]
    train = summary["training"]
    lines = [
        "# W3 accepted package — plain-language analysis",
        "",
        "This report looks at an **already accepted** training package. It does not retrain anything.",
        "It only reads the sealed numbers and draws pictures so a beginner can see what those numbers mean.",
        "",
        "## The big picture in one minute",
        "",
        "Think of the verifier as a matching game: given a robot motion snippet, can it find the right language",
        "description (and the other way around)? Random guessing would only succeed at the **exact chance** rates",
        f"below (pool size {chance['pool_count']}). Anything clearly above chance is the model doing real work.",
        "",
        f"- Authority boundary: **offline only** (`{summary['authority']}`). Not robot control. Not live candidate picking.",
        f"- Wrist / two-view benefit established? **{'yes' if wrist else 'no'}**.",
        "",
        "## Did training settle down?",
        "",
        "Training loss is how wrong the model is on the data it sees while learning. Validation loss is the same idea",
        "on held-out data. Both should generally fall as epochs go on.",
        "",
        f"- Epochs: {train['epochs'][0]} → {train['epochs'][-1]}",
        f"- Train loss: {train['loss'][0]:.4f} → {train['loss'][-1]:.4f}",
        f"- Validation loss: {train['validation_loss'][0]:.4f} → {train['validation_loss'][-1]:.4f}",
        "",
        "Figure: `figures/fig_loss_curves.png`",
        "",
        "## Matching better than chance?",
        "",
        "Top-1 means “the correct match is ranked first.” Top-5 means “it is in the first five.”",
        "We compare each score to exact chance for this pool.",
        "",
        f"| Direction | Top-1 | Top-5 | Chance top-1 | Chance top-5 |",
        f"|---|---:|---:|---:|---:|",
        f"| Action → language | {r['action_to_semantic_top1']:.4f} | {r['action_to_semantic_top5']:.4f} | {chance['top1']:.6f} | {chance['top5']:.6f} |",
        f"| Language → action | {r['semantic_to_action_top1']:.4f} | {r['semantic_to_action_top5']:.4f} | {chance['top1']:.6f} | {chance['top5']:.6f} |",
        "",
        f"Action→language top-1 95% CI: [{r['action_to_semantic_top1_ci95'][0]:.4f}, {r['action_to_semantic_top1_ci95'][1]:.4f}]",
        f"",
        f"Language→action top-1 95% CI: [{r['semantic_to_action_top1_ci95'][0]:.4f}, {r['semantic_to_action_top1_ci95'][1]:.4f}]",
        "",
        "Figure: `figures/fig_retrieval_vs_chance.png`",
        "",
        "## Are good pairs stronger than bad pairs?",
        "",
        "A **margin** asks: does the correct pairing score higher than a bad pairing?",
        "Shuffled = random wrong pairs. Nearby = hard near-miss pairs.",
        "Positive mean with a 95% interval above zero means the advantage is statistically on the right side of zero.",
        "",
        f"- Shuffled margin mean {m_s['mean']:.4f}, 95% CI [{m_s['ci95'][0]:.4f}, {m_s['ci95'][1]:.4f}] (fraction > 0: {m_s['fraction_gt_zero']:.3f})",
        f"- Nearby margin mean {m_n['mean']:.4f}, 95% CI [{m_n['ci95'][0]:.4f}, {m_n['ci95'][1]:.4f}] (fraction > 0: {m_n['fraction_gt_zero']:.3f})",
        "",
        "Figure: `figures/fig_margins_ci95.png`",
        "",
        "## Eight conditions at a glance",
        "",
        "The eval splits into eight shape×direction buckets. The heatmap shows action→language top-1 in each bucket.",
        "Uneven cells mean some motion settings are harder than others — useful context, not a retune knob.",
        "",
        "Figure: `figures/fig_conditions_heatmap.png`",
        "",
        "## Two-view versus base-only (wrist camera)",
        "",
        "Paired ablation asks: does adding the wrist view help **the same** protocol?",
        "Bars show two-view minus base-only. Error bars are the paired 95% intervals.",
        "",
        (
            "Verdict: wrist benefit **was established** by the sealed ablation rule."
            if wrist
            else "Verdict: wrist benefit **was not established**. The sealed package still accepts offline deployment, but this particular gain is not proven."
        ),
        "",
        "Figure: `figures/fig_paired_ablation.png`",
        "",
        "## Failures and caveats (read this before trusting a headline)",
        "",
    ]
    for caveat in summary["caveats"]:
        lines.append(f"- {caveat}")
    lines.extend(
        [
            "",
            f"Machine-readable twin: `analysis_summary.json` (schema `{ANALYSIS_SCHEMA}`, content_hash `{summary['content_hash']}`).",
            "",
        ]
    )
    return "\n".join(lines)


def analyze_accepted_w3_package(*, package_root: Path, output_root: Path) -> dict[str, Any]:
    """Read an accepted W3 package and write a separate analysis directory.

    Returns a small receipt with resolved paths. Never mutates ``package_root``.
    """
    package, output = _resolve_and_guard(package_root=package_root, output_root=output_root)
    _require_file(package / "deployment" / "ACCEPTED_W3_DEPLOYMENT", label="ACCEPTED_W3_DEPLOYMENT")
    acceptance_path = _require_file(package / "acceptance.json", label="acceptance.json")
    ablation_path = _require_file(
        package / "base_only" / "paired_ablation" / "paired_ablation.json",
        label="paired_ablation.json",
    )

    acceptance = _load_json(acceptance_path)
    _validate_acceptance(acceptance)
    ablation_doc = _load_json(ablation_path)
    ablation = _validate_ablation(ablation_doc)

    summary = _extract_summary(package=package, acceptance=acceptance, ablation=ablation)

    if output.exists():
        # Replace only the analysis tree we own; never touch the package.
        for child in list(output.iterdir()):
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)
    output.mkdir(parents=True, exist_ok=True)

    _apply_style()
    figures_dir = output / "figures"
    _plot_loss_curves(summary, figures_dir)
    _plot_retrieval_vs_chance(summary, figures_dir)
    _plot_margins(summary, figures_dir)
    _plot_conditions_heatmap(summary, figures_dir)
    _plot_paired_ablation(summary, figures_dir)

    write_canonical_json(output / "analysis_summary.json", summary)
    (output / "report.md").write_text(_render_report(summary), encoding="utf-8")

    # Hard guard: analysis output must not contain checkpoints.
    leaked = list(output.rglob("*.pt"))
    if leaked:
        raise RuntimeError(f"analysis output unexpectedly contains checkpoints: {leaked}")

    return {
        "schema": ANALYSIS_SCHEMA,
        "package_root": str(package),
        "output_root": str(output),
        "content_hash": summary["content_hash"],
        "wrist_benefit_established": summary["wrist_benefit_established"],
        "authority": summary["authority"],
    }

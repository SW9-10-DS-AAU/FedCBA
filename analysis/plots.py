from pathlib import Path

from .uuid_extractor import extract_uuid_from_filename

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.rcParams.update({"figure.dpi": 200})


def _ci(std: pd.Series, n: pd.Series, confidence: float = 0.95) -> pd.Series:
    """Half-width of a t-based confidence interval: t_{alpha/2, n-1} * std / sqrt(n)."""
    alpha = 1 - confidence
    df = (n - 1).clip(lower=1)
    t_vals = pd.Series([stats.t.ppf(1 - alpha / 2, d) for d in df], index=std.index)
    return t_vals * std / np.sqrt(n)


def _band(std: pd.Series, n: pd.Series | None, mode: str) -> pd.Series | None:
    """Return half-width of the error band. Returns None if CI requested but n is missing."""
    if mode == "ci":
        return _ci(std, n) if n is not None else None
    return std

ROLE_LABELS = {
    "good": "Honest",
    "bad": "Malicious",
    "freerider": "Freerider",
    "inactive": "Inactive",
}

BEHAVIOR_COLORS = {
    "good":      "#2196F3",
    "bad":       "#d62728",
    "freerider": "#9467bd",
    "inactive":  "#7f7f7f",
}

STRATEGY_COLORS = {
    "dotproduct":    "#2196F3",
    "naive":         "#FF9800",
    "accuracy_loss": "#E91E63",
    "accuracy_only": "#4CAF50",
    "loss_only":     "#9C27B0",
}


def plot_accuracy_loss_over_rounds(agg_global: pd.DataFrame, error_band: str = "ci") -> plt.Figure:
    """
    Dual-axis line chart: accuracy (left y-axis) + loss (right y-axis)
    with error band (95% CI by default, or ±std).

    Expects columns: round, accuracy_mean, accuracy_std, loss_mean, loss_std.
    """
    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax2 = ax1.twinx()

    rounds = agg_global["round"]
    band_label = "95% CI" if error_band == "ci" else "±std"
    n = agg_global["n"] if "n" in agg_global.columns else None
    ci_drawn = False

    # Accuracy
    ax1.plot(rounds, agg_global["accuracy_mean"], color="#2196F3",
             linewidth=2, label="Accuracy")
    if "accuracy_std" in agg_global.columns:
        b = _band(agg_global["accuracy_std"], n, error_band)
        if b is not None:
            ax1.fill_between(
                rounds,
                agg_global["accuracy_mean"] - b,
                agg_global["accuracy_mean"] + b,
                alpha=0.2, color="#2196F3",
            )
            ci_drawn = True

    # Loss
    ax2.plot(rounds, agg_global["loss_mean"], color="#FF5722",
             linewidth=2, linestyle="--", label="Loss")
    if "loss_std" in agg_global.columns:
        b = _band(agg_global["loss_std"], n, error_band)
        if b is not None:
            ax2.fill_between(
                rounds,
                agg_global["loss_mean"] - b,
                agg_global["loss_mean"] + b,
                alpha=0.2, color="#FF5722",
            )
            ci_drawn = True

    ax1.set_xlabel("Round")
    ax1.set_ylabel("Global Accuracy", color="#2196F3")
    ax2.set_ylabel("Global Loss", color="#FF5722")
    ax1.tick_params(axis="y", labelcolor="#2196F3")
    ax2.tick_params(axis="y", labelcolor="#FF5722")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    handles = lines1 + lines2
    labels = labels1 + labels2
    if ci_drawn:
        handles.append(Patch(facecolor="gray", alpha=0.3))
        labels.append(band_label)
    ax1.legend(handles, labels, loc="lower right")

    ax1.grid(True, alpha=0.3)
    fig._plot_name = "accuracy_loss_over_rounds"
    fig._uuids = agg_global.attrs.get("experiment_ids", [])
    fig.tight_layout()
    return fig


def plot_strategy_comparison_lines(agg_by_strategy: pd.DataFrame, error_band: str = "ci") -> plt.Figure:
    """
    One line per strategy, mean accuracy over rounds with error band (95% CI by default, or ±std).

    Expects columns: contribution_score_strategy, round, accuracy_mean,
    accuracy_std.
    """
    fig, ax = plt.subplots(figsize=(9, 4))

    band_label = "95% CI" if error_band == "ci" else "±std"
    ci_in_legend = False
    for strategy, group in agg_by_strategy.groupby("contribution_score_strategy"):
        color = STRATEGY_COLORS.get(strategy, None)
        group = group.sort_values("round")
        ax.plot(group["round"], group["accuracy_mean"],
                label=strategy, color=color, linewidth=2)
        if "accuracy_std" in group.columns:
            n = group["n"] if "n" in group.columns else None
            b = _band(group["accuracy_std"], n, error_band)
            if b is not None:
                ax.fill_between(
                    group["round"],
                    group["accuracy_mean"] - b,
                    group["accuracy_mean"] + b,
                    alpha=0.15, color=color, label="_nolegend_",
                )
                ci_in_legend = True

    handles, labels = ax.get_legend_handles_labels()
    if ci_in_legend:
        handles.append(Patch(facecolor="gray", alpha=0.3))
        labels.append(band_label)
    ax.set_xlabel("Round")
    ax.set_ylabel("Global Accuracy (%)")
    ax.legend(handles, labels, title="Strategy")
    ax.grid(True, alpha=0.3)
    fig._plot_name = "strategy_comparison_lines"
    fig._uuids = agg_by_strategy.attrs.get("experiment_ids", [])
    fig.tight_layout()
    return fig


def plot_strategy_comparison_boxplot(agg_final: pd.DataFrame) -> plt.Figure:
    """
    One box per strategy showing final-round accuracy distribution.

    Expects columns: contribution_score_strategy, final_accuracy.
    """
    fig, ax = plt.subplots(figsize=(9, 4))

    strategies = sorted(agg_final["contribution_score_strategy"].unique())
    data = [
        agg_final.loc[
            agg_final["contribution_score_strategy"] == s, "final_accuracy"
        ].values
        for s in strategies
    ]
    colors = [STRATEGY_COLORS.get(s, "#888888") for s in strategies]

    bp = ax.boxplot(data, patch_artist=True, labels=strategies)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel("Strategy")
    ax.set_ylabel("Final-Round Accuracy (%)")
    ax.grid(True, alpha=0.3, axis="y")
    fig._plot_name = "strategy_comparison_boxplot"
    fig._uuids = agg_final.attrs.get("experiment_ids", [])
    fig.tight_layout()
    return fig


def plot_grs_by_role(agg_grs: pd.DataFrame, error_band: str = "ci") -> plt.Figure:
    """
    One line per role (eventual user type), GRS over rounds with error band (95% CI by default, or ±std).

    Expects columns: role, round, grs_mean, grs_std.
    """
    fig, ax = plt.subplots(figsize=(9, 4))

    band_label = "95% CI" if error_band == "ci" else "±std"
    ci_in_legend = False
    for role, group in agg_grs.groupby("role"):
        color = BEHAVIOR_COLORS.get(role, None)
        group = group.sort_values("round")
        ax.plot(group["round"], group["grs_mean"],
                label=ROLE_LABELS[role], color=color, linewidth=2)
        if "grs_std" in group.columns:
            n = group["n"] if "n" in group.columns else None
            b = _band(group["grs_std"], n, error_band)
            if b is not None:
                ax.fill_between(
                    group["round"],
                    group["grs_mean"] - b,
                    group["grs_mean"] + b,
                    alpha=0.15, color=color, label="_nolegend_",
                )
                ci_in_legend = True

    handles, labels = ax.get_legend_handles_labels()
    if ci_in_legend:
        handles.append(Patch(facecolor="gray", alpha=0.3))
        labels.append(band_label)
    ax.set_xlabel("Round")
    ax.set_ylabel("Global Reputation Score (ETH)")
    ax.legend(handles, labels, title="Role:")
    ax.grid(True, alpha=0.3)
    fig._plot_name = "grs_by_role"
    fig._uuids = agg_grs.attrs.get("experiment_ids", [])
    fig.tight_layout()
    return fig


def plot_contribution_score_by_role(agg_scores: pd.DataFrame, error_band: str = "ci") -> plt.Figure:
    """
    One line per role (eventual user type), contribution score over rounds
    with error band (95% CI by default, or ±std).

    Expects columns: role, round, score_mean, score_std.
    """
    fig, ax = plt.subplots(figsize=(9, 4))

    band_label = "95% CI" if error_band == "ci" else "±std"
    ci_in_legend = False
    for role, group in agg_scores.groupby("role"):
        color = BEHAVIOR_COLORS.get(role, None)
        group = group.sort_values("round")
        ax.plot(group["round"], group["score_mean"],
                label=ROLE_LABELS[role], color=color, linewidth=2)
        if "score_std" in group.columns:
            n = group["n"] if "n" in group.columns else None
            b = _band(group["score_std"], n, error_band)
            if b is not None:
                ax.fill_between(
                    group["round"],
                    group["score_mean"] - b,
                    group["score_mean"] + b,
                    alpha=0.15, color=color, label="_nolegend_",
                )
                ci_in_legend = True

    handles, labels = ax.get_legend_handles_labels()
    if ci_in_legend:
        handles.append(Patch(facecolor="gray", alpha=0.3))
        labels.append(band_label)
    ax.set_xlabel("Round")
    ax.set_ylabel("Contribution Score")
    ax.legend(handles, labels, title="Role")
    ax.grid(True, alpha=0.3)
    fig._plot_name = "contribution_score_by_role"
    fig._uuids = agg_scores.attrs.get("experiment_ids", [])
    fig.tight_layout()
    return fig


def plot_grs_by_role_relative(agg_grs: pd.DataFrame, error_band: str = "ci") -> plt.Figure:
    """
    One line per role, GRS over rounds-since-activation with error band (95% CI by default, or ±std).
    A vertical dashed line at x=0 marks the activation moment.

    Expects columns: role, relative_round, grs_mean, grs_std.
    """
    fig, ax = plt.subplots(figsize=(9, 4))

    band_label = "95% CI" if error_band == "ci" else "±std"
    ci_in_legend = False
    for role, group in agg_grs.groupby("role"):
        color = BEHAVIOR_COLORS.get(role, None)
        group = group.sort_values("relative_round")
        ax.plot(group["relative_round"], group["grs_mean"],
                label=ROLE_LABELS[role], color=color, linewidth=2)
        if "grs_std" in group.columns:
            n = group["n"] if "n" in group.columns else None
            b = _band(group["grs_std"], n, error_band)
            if b is not None:
                ax.fill_between(
                    group["relative_round"],
                    group["grs_mean"] - b,
                    group["grs_mean"] + b,
                    alpha=0.15, color=color, label="_nolegend_",
                )
                ci_in_legend = True

    ax.axvline(0, color="yellow", linestyle="--", linewidth=1, alpha=0.5, label="Activation")
    handles, labels = ax.get_legend_handles_labels()
    if ci_in_legend:
        handles.append(Patch(facecolor="gray", alpha=0.3))
        labels.append(band_label)
    ax.set_xlabel("Rounds since activation")
    ax.set_ylabel("Global Reputation Score (ETH)")
    ax.legend(handles, labels, title="Role:")
    ax.grid(True, alpha=0.3)
    fig._plot_name = "grs_by_role_relative"
    fig._uuids = agg_grs.attrs.get("experiment_ids", [])
    fig.tight_layout()
    return fig


def plot_contribution_score_by_role_relative(agg_scores: pd.DataFrame, error_band: str = "ci") -> plt.Figure:
    """
    One line per role, contribution score over rounds-since-activation with error band (95% CI by default, or ±std).
    A vertical dashed line at x=0 marks the activation moment.

    Expects columns: role, relative_round, score_mean, score_std, n.
    """
    fig, ax = plt.subplots(figsize=(9, 4))

    band_label = "95% CI" if error_band == "ci" else "±std"
    ci_in_legend = False
    for role, group in agg_scores.groupby("role"):
        color = BEHAVIOR_COLORS.get(role, None)
        group = group.sort_values("relative_round")
        ax.plot(group["relative_round"], group["score_mean"],
                label=ROLE_LABELS[role], color=color, linewidth=2)
        if "score_std" in group.columns:
            n = group["n"] if "n" in group.columns else None
            b = _band(group["score_std"], n, error_band)
            if b is not None:
                ax.fill_between(
                    group["relative_round"],
                    group["score_mean"] - b,
                    group["score_mean"] + b,
                    alpha=0.15, color=color, label="_nolegend_",
                )
                ci_in_legend = True

    ax.axvline(0, color="yellow", linestyle="--", linewidth=1, alpha=0.5, label="Activation")
    handles, labels = ax.get_legend_handles_labels()
    if ci_in_legend:
        handles.append(Patch(facecolor="gray", alpha=0.3))
        labels.append(band_label)
    ax.set_xlabel("Rounds since activation")
    ax.set_ylabel("Contribution Score")
    ax.legend(handles, labels, title="Role:")
    ax.grid(True, alpha=0.3)
    fig._plot_name = "contribution_score_by_role_relative"
    fig._uuids = agg_scores.attrs.get("experiment_ids", [])
    fig.tight_layout()
    return fig


def plot_grs_by_user(
    grs_users: pd.DataFrame,
    metadata: pd.DataFrame | None = None,
) -> plt.Figure:
    """
    One line per user, GRS over rounds.

    metadata: optional full metadata DataFrame. If provided, vertical dashed lines
    are drawn at malicious_start_round / freerider_start_round for roles present
    in the data, extracted automatically from the matching experiment row.
    """
    fig, ax = plt.subplots(figsize=(9, 4))

    for (user_id, behavior), group in grs_users.groupby(["user_id", "role"]):
        group = group.sort_values("round")

        if "state" in group.columns:
            # Find first exit/disqualification event
            terminal = group[group["state"].isin(["disqualified", "exited"])]

            if not terminal.empty:
                first_terminal_round = terminal["round"].min()

                # Keep all rounds up to and including the first terminal round
                group = group[group["round"] <= first_terminal_round]

        ax.plot(
            group["round"],
            group["grs"],
            label=f"User {user_id} ({ROLE_LABELS[behavior]})",
            alpha=0.5,
        )
    if metadata is not None:
        experiment_id = grs_users["experiment_id"].iloc[0]
        meta = metadata[metadata["experiment_id"] == experiment_id].iloc[0]
        roles_in_data = grs_users["role"].unique()
        for role, col in (("bad", "malicious_start_round"), ("freerider", "freerider_start_round")):
            if role not in roles_in_data or col not in meta.index:
                continue
            activation_round = meta[col]
            if pd.isna(activation_round):
                continue
            color = BEHAVIOR_COLORS.get(role, "black")
            ax.axvline(
                int(activation_round),
                linestyle="--",
                color=color,
                linewidth=2,
                alpha=1.0,
                label=f"{ROLE_LABELS.get(role, role)} activation",
            )

    ax.set_xlabel("Round")
    ax.set_ylabel("Global Reputation Score (ETH)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(title="Users", loc="lower left")
    ax.grid(True, alpha=0.3) # alpha: makes the grid subtle/faint so it doesn't compete with the data
    fig._plot_name = "grs_by_user"
    fig._uuids = list(grs_users["experiment_id"].unique())
    fig.tight_layout()
    return fig


def plot_global_acc_by_aggregation_strategy(acc_by_strategy: pd.DataFrame, error_band: str = "ci") -> plt.Figure:
    """
    One line per aggregation rule, mean accuracy over rounds with error band (95% CI by default, or ±std).

    Expects columns: aggregation_rule, round, accuracy_mean, accuracy_std.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    band_label = "95% CI" if error_band == "ci" else "±std"
    ci_in_legend = False
    for strategy, group in acc_by_strategy.groupby("aggregation_rule"):
        color = STRATEGY_COLORS.get(strategy)
        group = group.sort_values("round")
        ax.plot(group["round"], group["accuracy_mean"], label=strategy, color=color, linewidth=2)
        if "accuracy_std" in group.columns:
            n = group["n"] if "n" in group.columns else None
            b = _band(group["accuracy_std"], n, error_band)
            if b is not None:
                ax.fill_between(
                    group["round"],
                    group["accuracy_mean"] - b,
                    group["accuracy_mean"] + b,
                    alpha=0.15, color=color, label="_nolegend_",
                )
                ci_in_legend = True

    handles, labels = ax.get_legend_handles_labels()
    if ci_in_legend:
        handles.append(Patch(facecolor="gray", alpha=0.3))
        labels.append(band_label)
    ax.set_xlabel("Round")
    ax.set_ylabel("Global Accuracy (%)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(handles, labels, title="Agg. Strategy")
    ax.grid(True, alpha=0.3)
    fig._plot_name = "global_acc_by_aggregation_strategy"
    fig._uuids = acc_by_strategy.attrs.get("experiment_ids", [])
    fig.tight_layout()

    return fig



def plot_global_loss_by_aggregation_strategy(loss_by_strategy: pd.DataFrame, error_band: str = "ci") -> plt.Figure:
    """
    One line per aggregation rule, mean loss over rounds with error band (95% CI by default, or ±std).

    Expects columns: aggregation_rule, round, loss_mean, loss_std.
    """
    fig, ax = plt.subplots(figsize=(9, 4))

    band_label = "95% CI" if error_band == "ci" else "±std"
    ci_in_legend = False
    for strategy, group in loss_by_strategy.groupby("aggregation_rule"):
        color = STRATEGY_COLORS.get(strategy)
        group = group.sort_values("round")
        ax.plot(group["round"], group["loss_mean"], label=strategy, color=color, linewidth=2)
        if "loss_std" in group.columns:
            n = group["n"] if "n" in group.columns else None
            b = _band(group["loss_std"], n, error_band)
            if b is not None:
                ax.fill_between(
                    group["round"],
                    group["loss_mean"] - b,
                    group["loss_mean"] + b,
                    alpha=0.15, color=color, label="_nolegend_",
                )
                ci_in_legend = True

    handles, labels = ax.get_legend_handles_labels()
    if ci_in_legend:
        handles.append(Patch(facecolor="gray", alpha=0.3))
        labels.append(band_label)
    ax.set_xlabel("Round")
    ax.set_ylabel("Global Loss")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(handles, labels, title="Agg. Strategy:")
    ax.grid(True, alpha=0.3)
    fig._plot_name = "global_loss_by_aggregation_strategy"
    fig._uuids = loss_by_strategy.attrs.get("experiment_ids", [])
    fig.tight_layout()

    return fig




def plot_gas_cost_by_tx_type(agg_gas: pd.DataFrame) -> plt.Figure:
    """
    Grouped bar chart of mean gas used per transaction type, one bar group per
    tx_type and one bar per contribution_score_strategy, with 95% CI error bars.

    Expects columns: tx_type, contribution_score_strategy, gas_mean, gas_std, n.
    """
    fig, ax = plt.subplots(figsize=(9, 4))

    tx_types = sorted(agg_gas["tx_type"].unique())
    strategies = sorted(agg_gas["contribution_score_strategy"].unique())
    n_tx = len(tx_types)
    n_strategies = len(strategies)
    width = 0.8 / n_strategies
    x = range(n_tx)

    for i, strategy in enumerate(strategies):
        group = agg_gas[agg_gas["contribution_score_strategy"] == strategy]
        means = []
        errors = []
        for tx in tx_types:
            row = group[group["tx_type"] == tx]
            if row.empty:
                means.append(float("nan"))
                errors.append(0)
            else:
                means.append(row["gas_mean"].iloc[0])
                if "gas_std" in row.columns and "n" in row.columns:
                    ci_val = _ci(row["gas_std"], row["n"])
                    errors.append(float(ci_val.iloc[0]))
                else:
                    errors.append(0)

        xpos = [xi - 0.4 + i * width + width / 2 for xi in x]
        color = STRATEGY_COLORS.get(strategy, "#607c8a")
        ax.bar(xpos, means, width, yerr=errors, capsize=4,
               color=color, alpha=0.8, edgecolor="black", linewidth=0.8,
               label=strategy)

    ax.set_xticks(list(x))
    ax.set_xticklabels(tx_types, rotation=10, ha="right")
    ax.set_xlabel("Transaction Type")
    ax.set_ylabel("Mean Gas Used")
    fig._plot_name = "gas_cost_by_tx_type"
    fig._uuids = agg_gas.attrs.get("experiment_ids", [])
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="black", linewidth=0, marker="|", markersize=10, markeredgewidth=1.5))
    labels.append("95% CI")
    ax.legend(handles, labels, title="Strategy")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig



def plot_round_kicked_by_strategy(
    agg_kicked: pd.DataFrame,
    title: str = "Effectiveness of Strategies in Removing Dishonest Participants",
    max_rounds: int | None = None,
) -> plt.Figure:
    """
    Grouped bar chart: for each contribution score strategy, show at which
    round each user role was disqualified (lower = removed sooner = better).
    Asymmetric error bars show min/max range across runs.

    Inspired by kickedGraph() in scripts/processData.py.

    Expects columns: contribution_score_strategy, role,
                     mean_round_kicked, low_err, high_err.
    """
    if agg_kicked.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No disqualified users", ha="center", va="center", transform=ax.transAxes)
        return fig

    strategies = sorted(agg_kicked["contribution_score_strategy"].unique())
    roles = sorted(agg_kicked["role"].unique())

    n_strategies = len(strategies)
    n_roles = len(roles)
    x = range(n_strategies)
    width = 0.8 / n_roles

    fig, ax = plt.subplots(figsize=(max(7, n_strategies * 1.8), 5))

    for role_idx, role in enumerate(roles):
        role_data = agg_kicked[agg_kicked["role"] == role]
        color = BEHAVIOR_COLORS.get(role, "#888888")

        means   = []
        low_err = []
        high_err = []
        missing = []

        for strategy in strategies:
            row = role_data[role_data["contribution_score_strategy"] == strategy]
            if row.empty:
                means.append(float("nan"))
                low_err.append(0)
                high_err.append(0)
                missing.append(True)
            else:
                means.append(row["mean_round_kicked"].iloc[0])
                low_err.append(row["low_err"].iloc[0])
                high_err.append(row["high_err"].iloc[0])
                missing.append(False)

        xpos = [xi - 0.4 + role_idx * width + width / 2 for xi in x]

        bar_means = [m if not missing[i] else float("nan") for i, m in enumerate(means)]
        show_err = any(l != 0 or h != 0 for l, h in zip(low_err, high_err))

        ax.bar(
            xpos,
            bar_means,
            width,
            yerr=[low_err, high_err] if show_err else None,
            capsize=4,
            color=color,
            edgecolor="black",
            linewidth=0.8,
            alpha=0.8,
            label=ROLE_LABELS[role],
        )

        y_top = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else (max_rounds or 1)
        for xi, is_missing in zip(xpos, missing):
            if is_missing:
                ax.text(
                    xi, y_top * 0.02, "N/A",
                    ha="center", va="bottom",
                    fontsize=8, color="gray", rotation=90,
                )

    ax.set_xticks(list(x))
    ax.set_xticklabels(strategies, rotation=10, ha="right")
    ax.set_ylabel("Round Kicked (lower = removed sooner)")
    ax.set_title(title)
    fig._plot_name = "round_kicked_by_strategy"
    fig._uuids = agg_kicked.attrs.get("experiment_ids", [])
    ax.legend(title="Role")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig



def plot_merge_weights_by_behavior(agg_weights: pd.DataFrame, stats: pd.DataFrame | None = None, error_band: str = "ci") -> plt.Figure:
    """
    One line per behavior, average merge weight over rounds with error band (95% CI by default, or ±std).
    Rounds where a behavior was never merged will have no point (NaN weight_mean).

    Expects agg_weights columns: behavior, round, weight_mean, weight_std.
    Expects stats columns: behavior, total_rounds, rounds_merged, pct_merged, users_merged.
    """
    fig, ax = plt.subplots(figsize=(9, 4))

    band_label = "95% CI" if error_band == "ci" else "±std"
    ci_in_legend = False
    for behavior, group in agg_weights.groupby("behavior"):
        color = BEHAVIOR_COLORS.get(behavior, None)
        group = group.sort_values("round")
        ax.plot(group["round"], group["weight_mean"],
                label=ROLE_LABELS.get(behavior, behavior), color=color, linewidth=2)
        if "weight_std" in group.columns:
            n = group["n"] if "n" in group.columns else None
            b = _band(group["weight_std"], n, error_band)
            if b is not None:
                ax.fill_between(
                    group["round"],
                    group["weight_mean"] - b,
                    group["weight_mean"] + b,
                    alpha=0.15, color=color, label="_nolegend_",
                )
                ci_in_legend = True

    ax.set_xlabel("Round")
    ax.set_ylabel("Merge Weight")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)

    if stats is not None:
        stats_by_behavior = stats.set_index("behavior")
        handles, labels = [], []
        for behavior in agg_weights["behavior"].unique():
            color = BEHAVIOR_COLORS.get(behavior, "black")
            role  = ROLE_LABELS.get(behavior, behavior)
            handle = Line2D([0], [0], color=color, linewidth=2)
            if behavior in stats_by_behavior.index:
                row    = stats_by_behavior.loc[behavior]
                merged = int(row["rounds_merged"]) if pd.notna(row["rounds_merged"]) else 0
                total  = int(row["total_rounds"])
                users  = int(row["user_count"])
                label  = f"{role:<14}  {merged:>2}/{total:<2} rounds  ·  {users} user(s)"
            else:
                label = role
            handles.append(handle)
            labels.append(label)
        if ci_in_legend:
            handles.append(Patch(facecolor="gray", alpha=0.3))
            labels.append(band_label)
        ax.legend(
            handles, labels,
            title="Not-merged by behavior",
            loc="lower right",
            fontsize=8,
            prop={"family": "monospace", "size": 8},
            framealpha=0.9,
            edgecolor="#cccccc",
        )
    else:
        handles, labels = ax.get_legend_handles_labels()
        if ci_in_legend:
            handles.append(Patch(facecolor="gray", alpha=0.3))
            labels.append(band_label)
        ax.legend(handles, labels, title="Behavior")
    fig._plot_name = "merge_weights_by_behavior"
    fig._uuids = agg_weights.attrs.get("experiment_ids", [])
    fig.tight_layout()
    return fig



def _next_graph_id(directory: Path) -> str:
    existing = [p.name for p in directory.glob("*.svg")]
    ids = []
    for name in existing:
        part = name.split("-")[0]
        if part.isdigit():
            ids.append(int(part))
    next_id = max(ids) + 1 if ids else 1
    return f"{next_id:03d}"


def delete_figure(directory: str | Path, graph_id: str) -> None:
    directory = Path(directory)
    matches = list(directory.glob(f"{graph_id}-*.svg"))
    if not matches:
        raise FileNotFoundError(f"No SVG with graph_id '{graph_id}' in {directory}")

    for svg in matches:
        svg.unlink()

    mappings_path = directory / "mappings.txt"

    if mappings_path.exists():
        lines = mappings_path.read_text().splitlines(keepends=True)
        kept = [l for l in lines if not l.startswith(f"{graph_id}:") and l.strip()]
        if kept:
            mappings_path.write_text("".join(kept))
        else:
            mappings_path.unlink()


def save_figure(fig: plt.Figure, base_dir, experiment_name=None, suffix: str = "", dpi: int = 300):
    plot_name = getattr(fig, "_plot_name", "figure")
    uuids = getattr(fig, "_uuids", [])
    directory = Path(base_dir) / experiment_name if experiment_name is not None else Path(base_dir)
    directory.mkdir(parents=True, exist_ok=True)

    graph_id = _next_graph_id(directory)
    stem = f"{graph_id}-{plot_name}{f'-{suffix}' if suffix else ''}"
    fig.savefig(directory / f"{stem}.svg", dpi=dpi, bbox_inches="tight")

    if uuids:
        with open(directory / "mappings.txt", "a") as f:
            for uid in uuids:
                try:
                    uid = extract_uuid_from_filename(uid)
                except ValueError:
                    pass  # already a bare UUID or unrecognised format — write as-is
                f.write(f"{graph_id}: {uid}\n")

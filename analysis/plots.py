from pathlib import Path

from .uuid_extractor import extract_uuid_from_filename
from .aggregations import agg_grs_by_role, compute_state_percentages

import scienceplots  # noqa: F401 — registers "science" style with matplotlib
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import numpy as np
import pandas as pd
from scipy import stats
import re

matplotlib.rcdefaults()
plt.style.use(["science", "high-vis"])
matplotlib.rcParams.update({
    "figure.dpi": 300,
    "pgf.rcfonts": False,
    # "text.usetex": False, # Disable this if you do not have local latex installed. Uses latex for rendering fonts.
})
figure_file_extensions = ["pgf", "svg", "png", "pdf"]
figure_file_extension = figure_file_extensions[3]


ROLE_LABELS = {
    "good": "Honest",
    "bad": "Malicious",
    "freerider": "Freerider",
    "inactive": "Inactive",
}

ROLE_ORDER = ["good", "freerider", "bad"]

BEHAVIOR_COLORS = {
    "good":      "#2196F3",
    "bad":       "#d62728",
    "freerider": "#9467bd",
    "inactive":  "#90EE90",
    "good_exited": "#44b348",
}

CONTRIBUTION_SCORE_COLORS = {
    "dotproduct":    "#2196F3",
    "naive":         "#FF9800",
    "accuracy_loss": "#E91E63",
    "accuracy_only": "#4CAF50",
    "loss_only":     "#9C27B0",
}

CONTRIBUTION_SCORE_PAPER_NAME = {
    "dotproduct":    "Dot-Product",
    "naive":         "Naive",
    "accuracy_loss": "Accuracy & Loss",
    "accuracy_only": "Accuracy",
    "loss_only":     "Loss",
}

AGGREGATION_STRATEGY_STYLES = {
    "positives_only":     {"color": "#0d49fb", "linestyle": "-",  "linewidth": 2},
    "plus_one_normalize": {"color": "#e6091c", "linestyle": "-",  "linewidth": 2},
    "GRS_aggregation":    {"color": "#26eb47", "linestyle": "-.",  "linewidth": 2},
    "binary_switch":      {"color": "#8936df", "linestyle": "--", "linewidth": 2},
    "partial_switch":     {"color": "#fec32d", "linestyle": "--", "linewidth": 2},
    "FedAVG":             {"color": "#25d7fd", "linestyle": ":",  "linewidth": 2},
}

AGGREGATION_STRATEGY_PAPER_NAME = {
    "positives_only": "Positives-Only",
    "plus_one_normalize": "Offset-Normalized",
    "GRS_aggregation": "Reputation-Based",
    "binary_switch": "Binary Switch",
    "partial_switch": "Partial Switch",
    "FedAVG": "FedAvg",
}

ACTIVATION_COLOR = "#666666"

def _strategy_style(key: str) -> dict:
    if key in AGGREGATION_STRATEGY_STYLES:
        return AGGREGATION_STRATEGY_STYLES[key]
    for prefix, style in AGGREGATION_STRATEGY_STYLES.items():
        if key.startswith(prefix):
            return style
    return {}


def _strategy_color(key: str) -> str | None:
    return _strategy_style(key).get("color")



def _strategy_label(key: str) -> str:
    if key in AGGREGATION_STRATEGY_PAPER_NAME:
        return AGGREGATION_STRATEGY_PAPER_NAME[key]

    elif key.startswith("binary_switch"):
        m = re.search(r"\[([^,]+),\s*([^]]+)]", key)
        func1, func2 = (m.group(1).strip(), m.group(2).strip()) if m else (None, None)
        return (key
                .replace("binary_switch", "Binary Switch")
                .replace("[", "(")
                .replace("]", ")")
                .replace(func1, _strategy_label(func1))
                .replace(func2, _strategy_label(func2)))

    elif key.startswith("partial_switch"):
        m = re.search(r"\[([^,]+),\s*([^,]+),\s*([^]]+)]", key)
        variant, func1, func2 = (m.group(1).strip(), m.group(2).strip(), m.group(3).strip()) if m else (None, None, None)
        return (key
                .replace("partial_switch", "Partial Switch")
                .replace(variant + ", ", "")
                .replace("[", "(")
                .replace("]", ")")
                .replace(func1, _strategy_label(func1))
                .replace(func2, _strategy_label(func2)))
    else:
        raise KeyError(f"Unknown strategy key: {key!r}")


def _ordered_strategies(keys):
    result = []
    for pattern in AGGREGATION_STRATEGY_PAPER_NAME:
        result.extend(k for k in keys if k.startswith(pattern))
    return result


def _x_tick_interval(max_round: int) -> int:
    if max_round > 20:
        return 5
    if max_round > 12:
        return 2
    return 1


def _inset_layout(y0: float) -> dict:
    """Return bbox/size kwargs for _add_zoom_inset."""
    return dict(bbox_to_anchor=(0.45, y0, 0.53, 0.95), width="45%", height="40%")


def _add_zoom_inset(
    ax: plt.Axes,
    data: pd.DataFrame,
    x_range: tuple[float, float],
    bbox_to_anchor: tuple,
    width: str,
    height: str,
    loc: str,
    loc1: int,
    loc2: int,
    y_col: str,
) -> None:
    zoom_data = data[data["round"] >= x_range[0]]
    y1 = zoom_data[y_col].min()
    y2 = zoom_data[y_col].max()
    pad = (y2 - y1) * 0.2 or 0.01

    axins = inset_axes(ax, width=width, height=height, loc=loc,
                       bbox_to_anchor=bbox_to_anchor, bbox_transform=ax.transAxes)
    strategy_groups = {s: grp for s, grp in data.groupby("aggregation_rule")}
    for strategy in _ordered_strategies(strategy_groups):
        group = strategy_groups[strategy].sort_values("round")
        axins.plot(group["round"], group[y_col], **{**_strategy_style(strategy), "linewidth": 1.5})
    axins.set_xlim(x_range[0] - 0.1, x_range[1] + 0.1)
    axins.set_ylim(y1 - pad, y2 + pad)
    axins.xaxis.set_major_locator(MultipleLocator(1))
    axins.tick_params(labelsize=7)
    axins.add_patch(Rectangle((0, 0), 1, 1, transform=axins.transAxes, facecolor="white", edgecolor="none", zorder=0))
    axins.grid(True, alpha=0.3)
    mark_inset(ax, axins, loc1=loc1, loc2=loc2, fc="none", ec="0.5", linewidth=0.8)



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


def plot_accuracy_loss_over_rounds(agg_global: pd.DataFrame, error_band: str = "ci") -> plt.Figure:
    """
    Dual-axis line chart: accuracy (left y-axis) + loss (right y-axis)
    with error band (95% CI by default, or ±std).

    Expects columns: round, accuracy_mean, accuracy_std, loss_mean, loss_std.
    """
    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax2 = ax1.twinx()

    rounds = agg_global["round"]
    band_label = r"95\% CI" if error_band == "ci" else r"±std"
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


# def plot_strategy_comparison_lines(agg_by_strategy: pd.DataFrame, error_band: str = "ci") -> plt.Figure:
#     """
#     One line per strategy, mean accuracy over rounds with error band (95% CI by default, or ±std).
#
#     Expects columns: contribution_score_strategy, round, accuracy_mean,
#     accuracy_std.
#     """
#     fig, ax = plt.subplots(figsize=(9, 4))
#
#     band_label = r"95\% CI" if error_band == "ci" else r"±std"
#     ci_in_legend = False
#     for contrib_score, group in agg_by_strategy.groupby("contribution_score_strategy"):
#         color = CONTRIBUTION_SCORE_COLORS.get(contrib_score)
#         group = group.sort_values("round")
#         ax.plot(group["round"], group["accuracy_mean"],
#                 label=contrib_score, color=color, linewidth=2)
#         if "accuracy_std" in group.columns:
#             n = group["n"] if "n" in group.columns else None
#             b = _band(group["accuracy_std"], n, error_band)
#             if b is not None:
#                 ax.fill_between(
#                     group["round"],
#                     group["accuracy_mean"] - b,
#                     group["accuracy_mean"] + b,
#                     alpha=0.15, color=color, label="_nolegend_",
#                 )
#                 ci_in_legend = True
#
#     handles, labels = ax.get_legend_handles_labels()
#     if ci_in_legend:
#         handles.append(Patch(facecolor="gray", alpha=0.3))
#         labels.append(band_label)
#     ax.set_xlabel("Round")
#     ax.set_ylabel(r"Global Accuracy (\%)")
#     ax.legend(handles, labels, title="Strategy")
#     ax.grid(True, alpha=0.3)
#     fig._plot_name = "strategy_comparison_lines"
#     fig._uuids = agg_by_strategy.attrs.get("experiment_ids", [])
#     fig.tight_layout()
#     return fig


# def plot_strategy_comparison_boxplot(agg_final: pd.DataFrame) -> plt.Figure:
#     """
#     One box per strategy showing final-round accuracy distribution.
#
#     Expects columns: contribution_score_strategy, final_accuracy.
#     """
#     fig, ax = plt.subplots(figsize=(9, 4))
#
#     strategies = sorted(agg_final["contribution_score_strategy"].unique())
#     data = [
#         agg_final.loc[
#             agg_final["contribution_score_strategy"] == s, "final_accuracy"
#         ].values
#         for s in strategies
#     ]
#     colors = [CONTRIBUTION_SCORE_COLORS.get(s, "#888888") for s in strategies]
#
#     bp = ax.boxplot(data, patch_artist=True, labels=strategies)
#     for patch, color in zip(bp["boxes"], colors):
#         patch.set_facecolor(color)
#         patch.set_alpha(0.7)
#
#     ax.set_xlabel("Strategy")
#     ax.set_ylabel("Final-Round Accuracy (%)")
#     ax.grid(True, alpha=0.3, axis="y")
#     fig._plot_name = "strategy_comparison_boxplot"
#     fig._uuids = agg_final.attrs.get("experiment_ids", [])
#     fig.tight_layout()
#     return fig


def plot_grs_by_role(
    agg_grs: pd.DataFrame,
    error_band: str = "ci",
) -> plt.Figure:
    """
    One line per role (eventual user type), GRS over rounds with error band (95% CI by default, or ±std).

    Expects columns: role, round, grs_mean, grs_std.
    activation_round: if provided, draws a vertical line at activation_round - 1 labeled "Pre-Attack Round".
    """
    fig, ax = plt.subplots(figsize=(9, 4))

    band_label = r"95\% CI" if error_band == "ci" else r"±std"
    ci_in_legend = False

    role_groups = {role: grp for role, grp in agg_grs.groupby("role")}
    for role in ROLE_ORDER:
        if role not in role_groups:
            continue
        group = role_groups[role]
        color = BEHAVIOR_COLORS.get(role, None)
        group = group.sort_values("round")
        ax.plot(group["round"], group["grs_mean"],
                label=ROLE_LABELS[role], color=color, linewidth=2, alpha=0.7)
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

    # if activation_rounds:
    #     for role in ROLE_ORDER:
    #         if role not in activation_rounds:
    #             continue
    #         color = BEHAVIOR_COLORS.get(role, "black")
    #         ax.axvline(
    #             activation_rounds[role],
    #             color=color, linestyle="--", linewidth=1.5, alpha=0.6,
    #             label=f"{ROLE_LABELS.get(role, role)} activation",
    #         )

    act = agg_grs.attrs.get("activation_round")
    if act is not None:
        ax.axvline(act - 1, color=ACTIVATION_COLOR, linestyle="--", linewidth=1.5, label="Pre-Attack Round")

    handles, labels = ax.get_legend_handles_labels()
    if ci_in_legend:
        handles.append(Patch(facecolor="gray", alpha=0.3))
        labels.append(band_label)
    ax.set_xlabel("Round")
    ax.set_ylabel("Global Reputation Score (ETH)")
    ax.xaxis.set_major_locator(MultipleLocator(_x_tick_interval(int(agg_grs["round"].max()))))
    ax.set_xlim(-0.2, int(agg_grs["round"].max()) + 0.5)
    ax.legend(handles, labels, title="Role:")
    ax.grid(True, alpha=0.3)
    fig._legend_handles = handles
    fig._legend_labels = labels
    fig._plot_name = "grs_by_role"
    fig._uuids = agg_grs.attrs.get("experiment_ids", [])
    fig.tight_layout()
    return fig


# def plot_contribution_score_by_role(agg_scores: pd.DataFrame, error_band: str = "ci") -> plt.Figure:
#     """
#     One line per role (eventual user type), contribution score over rounds
#     with error band (95% CI by default, or ±std).
#
#     Expects columns: role, round, score_mean, score_std.
#     """
#     fig, ax = plt.subplots(figsize=(9, 4))
#
#     band_label = r"95\% CI" if error_band == "ci" else r"±std"
#     ci_in_legend = False
#     for role, group in agg_scores.groupby("role"):
#         color = BEHAVIOR_COLORS.get(role, None)
#         group = group.sort_values("round")
#         ax.plot(group["round"], group["score_mean"],
#                 label=ROLE_LABELS[role], color=color, linewidth=2)
#         if "score_std" in group.columns:
#             n = group["n"] if "n" in group.columns else None
#             b = _band(group["score_std"], n, error_band)
#             if b is not None:
#                 ax.fill_between(
#                     group["round"],
#                     group["score_mean"] - b,
#                     group["score_mean"] + b,
#                     alpha=0.15, color=color, label="_nolegend_",
#                 )
#                 ci_in_legend = True
#
#     handles, labels = ax.get_legend_handles_labels()
#     if ci_in_legend:
#         handles.append(Patch(facecolor="gray", alpha=0.3))
#         labels.append(band_label)
#     ax.set_xlabel("Round")
#     ax.set_ylabel("Contribution Score")
#     ax.legend(handles, labels, title="Role")
#     ax.grid(True, alpha=0.3)
#     fig._plot_name = "contribution_score_by_role"
#     fig._uuids = agg_scores.attrs.get("experiment_ids", [])
#     fig.tight_layout()
#     return fig


def plot_grs_by_role_relative(agg_grs: pd.DataFrame, error_band: str = "ci") -> plt.Figure:
    """
    One line per role, GRS over rounds-since-activation with error band (95% CI by default, or ±std).
    A vertical dashed line at x=0 marks the activation moment.

    Expects columns: role, relative_round, grs_mean, grs_std.
    """
    fig, ax = plt.subplots(figsize=(9, 4))

    band_label = r"95\% CI" if error_band == "ci" else r"±std"
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

    ax.axvline(0, color=ACTIVATION_COLOR, linestyle="--", linewidth=1.5, label="Activation Round")
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

    band_label = r"95\% CI" if error_band == "ci" else r"±std"
    ci_in_legend = False
    role_groups = {role: grp for role, grp in agg_scores.groupby("role")}
    for role in ROLE_ORDER:
        if role not in role_groups:
            continue
        group = role_groups[role]
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

    ax.axvline(0, color=ACTIVATION_COLOR, linestyle="--", linewidth=1.5, label="Activation Round")
    handles, labels = ax.get_legend_handles_labels()
    if ci_in_legend:
        handles.append(Patch(facecolor="gray", alpha=0.3))
        labels.append(band_label)
    ax.set_xlabel("Rounds since activation")
    ax.set_ylabel("Contribution Score")
    ax.legend(handles, labels)
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
        act_round = None
        for role, col in (("bad", "malicious_start_round"), ("freerider", "freerider_start_round")):
            if role not in roles_in_data or col not in meta.index:
                continue
            val = meta[col]
            if not pd.isna(val):
                act_round = int(val)
                break
        if act_round is not None:
            ax.axvline(act_round - 1, color=ACTIVATION_COLOR, linestyle="--", linewidth=1.5, label="Pre-Attack Round")

    ax.set_xlabel("Round")
    ax.set_ylabel("Global Reputation Score (ETH)")
    ax.xaxis.set_major_locator(MultipleLocator(_x_tick_interval(int(grs_users["round"].max()))))
    ax.set_xlim(-0.2, int(grs_users["round"].max()) + 0.5)
    ax.legend(title="Users", loc="lower left")
    ax.grid(True, alpha=0.3) # alpha: makes the grid subtle/faint so it doesn't compete with the data
    fig._plot_name = "grs_by_user"
    fig._uuids = list(grs_users["experiment_id"].unique())
    fig.tight_layout()
    return fig

def plot_grs_by_role_by_aggregation_strategy(data, metadata, res):
    figs = {}

    strategy_groups = {strategy: grp for strategy, grp in metadata.groupby("aggregation_rule")}
    for strategy in _ordered_strategies(strategy_groups):
        metadata_group = strategy_groups[strategy]
        experiment_ids = metadata_group["experiment_id"].unique()

        data_group = data[
            data["experiment_id"].isin(experiment_ids)
        ]

        aggregated = agg_grs_by_role(
            data_group,
            metadata_group
        )

        # Extract activation rounds only for roles present in the data
        # roles_in_data = set(aggregated["role"].unique())
        # act = {}
        # for role, col in [("bad", "malicious_start_round"), ("freerider", "freerider_start_round")]:
        #     if role not in roles_in_data:
        #         continue
        #     if col in metadata_group.columns:
        #         val = metadata_group[col].dropna().mode()
        #         if not val.empty:
        #             act[role] = int(val.iloc[0])

        fig = plot_grs_by_role(aggregated)
        ax = fig.axes[0]
        role_handles = fig._legend_handles
        role_labels = fig._legend_labels

        # users dataframe from your experiment results
        users_group = res["users"][
            res["users"]["experiment_id"].isin(experiment_ids)
        ]

        pct = compute_state_percentages(users_group)
        ax2 = ax.twinx()
        width = 0.1

        # Build bar entries in ROLE_ORDER, only for roles present in pct
        roles_in_pct = set(pct["role"].unique())
        _bar_entries = {
            "good":      [("disqualified_pct", f"{ROLE_LABELS['good']}: Disq.",      BEHAVIOR_COLORS["good"]),
                          ("exited_pct",        "Honest: Exited",                    BEHAVIOR_COLORS["good_exited"])],
            "freerider": [("disqualified_pct", f"{ROLE_LABELS['freerider']}: Disq.", BEHAVIOR_COLORS["freerider"])],
            "bad":       [("disqualified_pct", f"{ROLE_LABELS['bad']}: Disq.",       BEHAVIOR_COLORS["bad"])],
        }
        flat = [(role, m, lbl, c) for role in ROLE_ORDER if role in roles_in_pct for m, lbl, c in _bar_entries.get(role, [])]
        n_bars = len(flat)
        bar_specs = [
            (role, m, lbl, (-n_bars / 2 + 0.5 + i) * width, c)
            for i, (role, m, lbl, c) in enumerate(flat)
        ]

        for role, metric, label, offset, color in bar_specs:
            tmp = pct[pct["role"] == role]
            ax2.bar(
                tmp["round"] + offset,
                tmp[metric],
                width=width,
                alpha=0.6,
                label=label,
                color=color,
            )

        ax2.set_ylim(0, 160)
        ax2.set_yticks([0, 20, 40, 60, 80, 100])
        ax2.set_ylabel(r"Exited or Disqualified (\%)")

        ax.legend(role_handles, role_labels, loc="upper left")
        ax2.legend(loc="upper right", bbox_to_anchor=(1, 0.97))

        fig.suptitle(f"Agg. Strategy: {_strategy_label(strategy)}", y=1.05)

        figs[strategy] = fig
    return figs


def plot_global_acc_by_aggregation_strategy(
    acc_by_strategy: pd.DataFrame,
    error_band: str = "ci",
    show_ci: bool = False,
) -> plt.Figure:
    """
    One line per aggregation rule, mean accuracy over rounds with error band (95% CI by default, or ±std).

    Expects columns: aggregation_rule, round, accuracy_mean, accuracy_std.
    """
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)

    band_label = r"95\% CI" if error_band == "ci" else r"±std"
    ci_in_legend = False
    strategy_groups = {s: grp for s, grp in acc_by_strategy.groupby("aggregation_rule")}
    for strategy in _ordered_strategies(strategy_groups):
        group = strategy_groups[strategy].sort_values("round")
        ax.plot(group["round"], group["accuracy_mean"], label=_strategy_label(strategy), **_strategy_style(strategy))

        if show_ci and "accuracy_std" in group.columns:
            n = group["n"] if "n" in group.columns else None
            b = _band(group["accuracy_std"], n, error_band)
            if b is not None:
                ax.fill_between(
                    group["round"],
                    group["accuracy_mean"] - b,
                    group["accuracy_mean"] + b,
                    alpha=0.15, color=_strategy_color(strategy), label="_nolegend_",
                )
                ci_in_legend = True

    act = acc_by_strategy.attrs.get("activation_round")
    if act is not None:
        ax.axvline(act - 1, color=ACTIVATION_COLOR, linestyle="--", linewidth=1.5, label="Pre-Attack Round")

    handles, labels = ax.get_legend_handles_labels()
    if ci_in_legend:
        handles.append(Patch(facecolor="gray", alpha=0.3))
        labels.append(band_label)
    ax.set_xlabel("Round")
    ax.set_ylabel(r"Global Accuracy (\%)")
    ax.xaxis.set_major_locator(MultipleLocator(_x_tick_interval(int(acc_by_strategy["round"].max()))))
    ax.set_xlim(-0.2, int(acc_by_strategy["round"].max()) + 0.5)
    ax.legend(handles, labels, title="Agg. Strategy", fontsize=8)
    ax.grid(True, alpha=0.3)

    max_round = int(acc_by_strategy["round"].max())
    _add_zoom_inset(ax, acc_by_strategy,
                    x_range=(max_round - 2, max_round),
                    **_inset_layout(y0=0.40),
                    loc="lower right",
                    loc1=1, loc2=2,
                    y_col="accuracy_mean")

    fig._plot_name = "global_acc_by_aggregation_strategy"
    fig._uuids = acc_by_strategy.attrs.get("experiment_ids", [])

    return fig



def plot_global_loss_by_aggregation_strategy(
    loss_by_strategy: pd.DataFrame,
    error_band: str = "ci",
    show_ci: bool = False,
) -> plt.Figure:
    """
    One line per aggregation rule, mean loss over rounds with error band (95% CI by default, or ±std).

    Expects columns: aggregation_rule, round, loss_mean, loss_std.
    """
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)

    band_label = r"95\% CI" if error_band == "ci" else r"±std"
    ci_in_legend = False

    strategy_groups = {s: grp for s, grp in loss_by_strategy.groupby("aggregation_rule")}
    for strategy in _ordered_strategies(strategy_groups):
        group = strategy_groups[strategy].sort_values("round")

        ax.plot(group["round"], group["loss_mean"], label=_strategy_label(strategy), **_strategy_style(strategy))
        if show_ci and "loss_std" in group.columns:
            n = group["n"] if "n" in group.columns else None
            b = _band(group["loss_std"], n, error_band)
            if b is not None:
                ax.fill_between(
                    group["round"],
                    group["loss_mean"] - b,
                    group["loss_mean"] + b,
                    alpha=0.15, color=_strategy_color(strategy), label="_nolegend_",
                )
                ci_in_legend = True

    act = loss_by_strategy.attrs.get("activation_round")
    if act is not None:
        ax.axvline(act - 1, color=ACTIVATION_COLOR, linestyle="--", linewidth=1.5, label="Pre-Attack Round")

    handles, labels = ax.get_legend_handles_labels()
    if ci_in_legend:
        handles.append(Patch(facecolor="gray", alpha=0.3))
        labels.append(band_label)
    ax.set_xlabel("Round")
    ax.set_ylabel("Global Loss")
    ax.xaxis.set_major_locator(MultipleLocator(_x_tick_interval(int(loss_by_strategy["round"].max()))))
    ax.set_xlim(-0.2, int(loss_by_strategy["round"].max()) + 0.5)
    ax.legend(handles, labels, title="Agg. Strategy:", fontsize=8)
    ax.grid(True, alpha=0.3)

    max_round = int(loss_by_strategy["round"].max())
    _add_zoom_inset(ax, loss_by_strategy,
                    x_range=(max_round - 2, max_round),
                    **_inset_layout(y0=0.20),
                    loc="lower right",
                    loc1=3, loc2=4,
                    y_col="loss_mean")

    fig._plot_name = "global_loss_by_aggregation_strategy"
    fig._uuids = loss_by_strategy.attrs.get("experiment_ids", [])

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
        color = CONTRIBUTION_SCORE_COLORS.get(strategy, "#607c8a")
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
    labels.append(r"95\% CI")
    ax.legend(handles, labels, title="Strategy")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig



# def plot_round_kicked_by_strategy(
#     agg_kicked: pd.DataFrame,
#     max_rounds: int | None = None,
# ) -> plt.Figure:
#     """
#     Grouped bar chart: for each contribution score strategy, show at which
#     round each user role was disqualified (lower = removed sooner = better).
#     Asymmetric error bars show min/max range across runs.
#
#     Inspired by kickedGraph() in scripts/processData.py.
#
#     Expects columns: contribution_score_strategy, role,
#                      mean_round_kicked, low_err, high_err.
#     """
#     if agg_kicked.empty:
#         fig, ax = plt.subplots()
#         ax.text(0.5, 0.5, "No disqualified users", ha="center", va="center", transform=ax.transAxes)
#         return fig
#
#     contrib_scores = sorted(agg_kicked["contribution_score_strategy"].unique())
#     roles = sorted(agg_kicked["role"].unique())
#
#     n_scores = len(contrib_scores)
#     n_roles = len(roles)
#     x = range(n_scores)
#     width = 0.8 / n_roles
#
#     fig, ax = plt.subplots(figsize=(max(7, n_scores * 1.8), 5))
#
#     for role_idx, role in enumerate(roles):
#         role_data = agg_kicked[agg_kicked["role"] == role]
#         color = BEHAVIOR_COLORS.get(role, "#888888")
#
#         means   = []
#         low_err = []
#         high_err = []
#         missing = []
#
#         for contrib_score in contrib_scores:
#             row = role_data[role_data["contribution_score_strategy"] == contrib_score]
#             if row.empty:
#                 means.append(float("nan"))
#                 low_err.append(0)
#                 high_err.append(0)
#                 missing.append(True)
#             else:
#                 means.append(row["mean_round_kicked"].iloc[0])
#                 low_err.append(row["low_err"].iloc[0])
#                 high_err.append(row["high_err"].iloc[0])
#                 missing.append(False)
#
#         xpos = [xi - 0.4 + role_idx * width + width / 2 for xi in x]
#
#         bar_means = [m if not missing[i] else float("nan") for i, m in enumerate(means)]
#         show_err = any(l != 0 or h != 0 for l, h in zip(low_err, high_err))
#
#         ax.bar(
#             xpos,
#             bar_means,
#             width,
#             yerr=[low_err, high_err] if show_err else None,
#             capsize=4,
#             color=color,
#             edgecolor="black",
#             linewidth=0.8,
#             alpha=0.8,
#             label=ROLE_LABELS[role],
#         )
#
#         y_top = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else (max_rounds or 1)
#         for xi, is_missing in zip(xpos, missing):
#             if is_missing:
#                 ax.text(
#                     xi, y_top * 0.02, "N/A",
#                     ha="center", va="bottom",
#                     fontsize=8, color="gray", rotation=90,
#                 )
#
#     ax.set_xticks(list(x))
#     ax.set_xticklabels(contrib_scores, rotation=10, ha="right")
#     ax.set_ylabel("Round Kicked (lower = removed sooner)")
#     fig._plot_name = "round_kicked_by_strategy"
#     fig._uuids = agg_kicked.attrs.get("experiment_ids", [])
#     ax.legend(title="Role")
#     ax.grid(axis="y", linestyle="--", alpha=0.5)
#     ax.set_axisbelow(True)
#     fig.tight_layout()
#     return fig



def plot_merge_weights_by_behavior(agg_weights: pd.DataFrame, stats: pd.DataFrame | None = None, error_band: str = "ci") -> plt.Figure:
    """
    One line per behavior, average merge weight over rounds with error band (95% CI by default, or ±std).
    Rounds where a behavior was never merged will have no point (NaN weight_mean).

    Expects agg_weights columns: behavior, round, weight_mean, weight_std.
    Expects stats columns: behavior, total_rounds, rounds_merged, pct_merged, users_merged.
    """
    fig, ax = plt.subplots(figsize=(9, 4))

    band_label = r"95\% CI" if error_band == "ci" else r"±std"
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
    # behavior_groups = {b: grp for b, grp in agg_weights.groupby("behavior")}
    # for behavior in ROLE_ORDER:
    #     if behavior not in behavior_groups:
    #         continue
    #     group = behavior_groups[behavior]
    #     color = BEHAVIOR_COLORS.get(behavior, None)
    #     group = group.sort_values("round")
    #     ax.plot(group["round"], group["weight_mean"],
    #             label=ROLE_LABELS.get(behavior, behavior), color=color, linewidth=2)
    #     if "weight_std" in group.columns:
    #         n = group["n"] if "n" in group.columns else None
    #         b = _band(group["weight_std"], n, error_band)
    #         if b is not None:
    #             ax.fill_between(
    #                 group["round"],
    #                 group["weight_mean"] - b,
    #                 group["weight_mean"] + b,
    #                 alpha=0.15, color=color, label="_nolegend_",
    #             )
    #             ci_in_legend = True

    ax.set_xlabel("Round")
    ax.set_ylabel("Merge Weight")
    ax.xaxis.set_major_locator(MultipleLocator(_x_tick_interval(int(agg_weights["round"].max()))))
    ax.set_xlim(-0.2, int(agg_weights["round"].max()) + 0.5)
    ax.grid(True, alpha=0.3)

    if stats is not None:
        stats_by_behavior = stats.set_index("behavior")
        handles, labels = [], []
        # for behavior in [b for b in ROLE_ORDER if b in behavior_groups]:
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


def save_dataframe(df: "pd.DataFrame", base_dir, experiment_name=None, suffix: str = "") -> None:
    """Save a summary DataFrame as CSV, using the same sequential ID namespace as save_figure.

    The DataFrame must carry provenance via attrs:
      df.attrs["name"]           — used as the filename stem (e.g. "malicious_summary")
      df.attrs["experiment_ids"] — list of experiment_id strings; UUIDs are extracted and
                                   written to mappings.txt so the source runs are traceable.
    """
    name = df.attrs.get("name", "dataframe")
    experiment_ids = df.attrs.get("experiment_ids", [])

    directory = Path(base_dir) / experiment_name if experiment_name is not None else Path(base_dir)
    directory.mkdir(parents=True, exist_ok=True)

    data_id = _next_graph_id(directory)
    stem = f"{data_id}-{name}{f'-{suffix}' if suffix else ''}"
    df.to_csv(directory / f"{stem}.csv", index=False)

    if experiment_ids:
        with open(directory / "mappings.txt", "a") as f:
            for eid in experiment_ids:
                try:
                    uid = extract_uuid_from_filename(eid)
                except ValueError:
                    uid = eid
                f.write(f"{data_id}: {uid}\n")


def _next_graph_id(directory: Path) -> str:
    existing = [p.name for ext in figure_file_extensions for p in directory.glob(f"*.{ext}")] + [p.name for p in directory.glob("*.csv")]
    ids = []
    for name in existing:
        part = name.split("-")[0]
        if part.isdigit():
            ids.append(int(part))
    next_id = max(ids) + 1 if ids else 1
    return f"{next_id:03d}"


def delete_figure(directory: str | Path, graph_id: str) -> None:
    directory = Path(directory)
    matches = [p for ext in figure_file_extensions for p in directory.glob(f"{graph_id}-*.{ext}")]
    if not matches:
        raise FileNotFoundError(f"No figure with graph_id '{graph_id}' in {directory}")

    for file in matches:
        file.unlink()

    mappings_path = directory / "mappings.txt"

    if mappings_path.exists():
        lines = mappings_path.read_text().splitlines(keepends=True)
        kept = [l for l in lines if not l.startswith(f"{graph_id}:") and l.strip()]
        if kept:
            mappings_path.write_text("".join(kept))
        else:
            mappings_path.unlink()


def plot_eval_reward_diff_by_role(
    agg_rewards: pd.DataFrame,
    error_band: str = "ci",
) -> plt.Figure:
    """
    Grouped bar chart of mean evaluation reward gain (rewarded − staked) per role per round.
    Bars are centered on zero; y-axis is clamped to the physical bounds [-1/3, +1/3] ETH.

    Expects columns: role, round, reward_diff_mean, reward_diff_std, n.
    """
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)

    rounds = sorted(agg_rewards["round"].unique())
    roles = [r for r in ROLE_ORDER if r in agg_rewards["role"].values]
    n_roles = len(roles)
    bar_width = 0.65 / n_roles
    offsets = np.arange(n_roles) * bar_width - (n_roles - 1) * bar_width / 2

    # Each round a user stakes staking_min_grs = min_collateral / punishfactorContrib = 1/3 ETH.
    # The worst they can do is get back 0 (reward_diff = -1/3), and we cap the display at +1/3.
    # Without this, the CI whiskers can overshoot the physical bounds — especially in later rounds
    # where few experiments still have that role active, making the t-CI blow up from small n.
    EVAL_REWARD_MIN = -1 / 3
    EVAL_REWARD_MAX =  1 / 3

    handles, labels = [], []
    for i, role in enumerate(roles):
        grp = agg_rewards[agg_rewards["role"] == role].sort_values("round")
        color = BEHAVIOR_COLORS.get(role)
        xpos = [r + offsets[i] for r in rounds if r in grp["round"].values]
        grp = grp[grp["round"].isin(rounds)]
        y = grp["reward_diff_mean"].values

        yerr = None
        if "reward_diff_std" in grp.columns:
            b = _band(grp["reward_diff_std"], grp["n"] if "n" in grp.columns else None, error_band)
            if b is not None:
                half = b.values
                # Don't let whiskers (lines) exceed physical bounds (reward_diff can't leave [-1/3, 1/3]).
                # y - lower >= EVAL_REWARD_MIN  →  lower <= y - EVAL_REWARD_MIN
                # y + upper <= EVAL_REWARD_MAX  →  upper <= EVAL_REWARD_MAX - y
                lower = np.clip(half, 0, y - EVAL_REWARD_MIN)
                upper = np.clip(half, 0, EVAL_REWARD_MAX - y)
                yerr = np.array([lower, upper])

        bars = ax.bar(xpos, y, width=bar_width, color=color, alpha=0.85,
                      yerr=yerr, capsize=3, error_kw={"linewidth": 0.8})
        handles.append(bars)
        labels.append(ROLE_LABELS.get(role, role))

    act = agg_rewards.attrs.get("activation_round")
    if act is not None:
        ax.axvspan(act - 1.5, act - 0.5, color=ACTIVATION_COLOR, alpha=0.2, zorder=0)
        handles.append(Patch(facecolor=ACTIVATION_COLOR, alpha=0.4))
        labels.append("Pre-Attack Round")

    band_label = r"95\% CI" if error_band == "ci" else r"±std"
    handles.append(Line2D([0], [0], color="black", linewidth=0, marker="|", markersize=10, markeredgewidth=1.5))
    labels.append(band_label)

    for r in rounds[:-1]:
        ax.axvline(r + 0.5, color="gray", linewidth=0.5, alpha=0.4, zorder=0)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylim(-0.35, 0.35)
    ax.set_xlabel("Round")
    ax.set_ylabel("Evaluation Voting Reward Gain")
    ax.set_xlim(rounds[0] - 0.4, rounds[-1] + 0.4)
    ax.set_xticks(rounds)
    ax.set_xticklabels(rounds)
    ax.legend(handles, labels, fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_axisbelow(True)

    fig._plot_name = "eval_reward_diff_by_role"
    fig._uuids = agg_rewards.attrs.get("experiment_ids", [])

    return fig


def save_figure(fig: plt.Figure, base_dir, experiment_name=None, suffix: str = "", dpi: int = 300):
    plot_name = getattr(fig, "_plot_name", "figure")
    uuids = getattr(fig, "_uuids", [])
    directory = Path(base_dir) / experiment_name if experiment_name is not None else Path(base_dir)
    directory.mkdir(parents=True, exist_ok=True)

    graph_id = _next_graph_id(directory)
    stem = f"{graph_id}-{plot_name}{f'-{suffix}' if suffix else ''}"
    fig.savefig(directory / f"{stem}.{figure_file_extension}", dpi=dpi, bbox_inches="tight", transparent=True)

    if uuids:
        with open(directory / "mappings.txt", "a") as f:
            for uid in uuids:
                try:
                    uid = extract_uuid_from_filename(uid)
                except ValueError:
                    pass  # already a bare UUID or unrecognised format — write as-is
                f.write(f"{graph_id}: {uid}\n")
            f.write("\n")

"""
Analysis code for manuscript titled "Micro-offline gains do not drive implicit motor sequence learning" 
Written by: Tharan Suresh
Date: 2026/07/03

The code here reproduces the figures in the manuscript, using the provided data.

Builds three multi-panel figures as PDFs in the current directory:

  Figure_2.pdf       A: Mean RT across blocks      B: Skill comparison (RT)
                     C: Speed across blocks        D: Skill comparison (speed)
  Figure_3.pdf       A/B: Cumulative micro-learning (Acquisition, per group)
                     C/D: Summed micro-learning (Acquisition / Retention)
                     E/F: Gain-vs-skill correlations (Acquisition / Retention skill)
  Supplemental_Figure_1.pdf
                     Within-block changes in speed during Acquisition and Retention             
  Supplemental_Figure_2.pdf
                     A/B: Cumulative micro-learning (Retention, per group)
                     C:   PDP explicit knowledge
                     D:   Micro-offline vs micro-online gains (pooled)

"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# ==========================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = BASE_DIR
FIG_DIR = BASE_DIR
os.makedirs(FIG_DIR, exist_ok=True)

COLOR_SEQ = "#7570b3"
COLOR_RAN = "#1b9e77"

METRIC_PALETTE = {
    "Micro-online": "#1F77B4",
    "Micro-offline": "#D62728",
    "Total": "#000000",
}
METRIC_LINESTYLES = {
    "Micro-online": "-",
    "Micro-offline": "-",
    "Total": "--",
}

# Process Dissociation Procedure (PDP): a participant is flagged with explicit
# sequence knowledge when they exceed both component thresholds.
PDP_GROUP_MAP = {1: "Sequence", 2: "Random"} # Manuscript uses "Sequence" and "No-sequence" labels, but the PDP data uses 1/2 
INCLUSION_THRESHOLD = 4
EXCLUSION_THRESHOLD = 8

TASK_CONFIG = {
    "Acquisition": {
        "random_blocks": [1, 2, 8, 14],
        "n_blocks": 14,
        "skill_sequence_block": 13,
        "skill_random_first": 1,
        "skill_random_last": 14,
        "skill_sequence_random_ref": 14,  # Sequence-group skill: LAST random block
    },
    "Retention": {
        "random_blocks": [1, 7],
        "n_blocks": 7,
        "skill_sequence_block": 2,
        "skill_random_first": 1,
        "skill_random_last": 7,
        "skill_sequence_random_ref": 1,  # Sequence-group skill: FIRST random block
    },
}


def sem(series):
    return series.sem()


def style_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)


def unify_ylims(axes):
    low = min(ax.get_ylim()[0] for ax in axes)
    high = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(low, high)


def load_master_data():
    master = pd.read_csv(os.path.join(CSV_DIR, "SRTT_master_data.csv"))
    master["RT"] = pd.to_numeric(master["RT"], errors="coerce")
    master["ACC"] = pd.to_numeric(master["ACC"], errors="coerce")
    master["Block"] = master["Block"].astype(int)
    master["TrialInBlock"] = master["TrialInBlock"].astype(int)
    return master


def add_common_columns(df, task):
    cfg = TASK_CONFIG[task]
    out = df[df["Task"] == task].copy()
    out["block_label"] = np.where(
        out["Block"].isin(cfg["random_blocks"]), "Random", "Sequence"
    )
    out["sequence_number"] = ((out["TrialInBlock"] - 1) // 12) + 1
    return out


def compute_block_rt(task_df):
    per_subject = (
        task_df.groupby(["SubjectID", "Group", "Block"], as_index=False)["RT"]
        .mean()
        .rename(columns={"RT": "mean_rt"})
    )
    summary = (
        per_subject.groupby(["Group", "Block"])["mean_rt"]
        .agg(["mean", sem])
        .reset_index()
        .rename(columns={"mean": "group_mean", "sem": "group_sem"})
    )
    return per_subject, summary


def compute_speed(task_df):
    df_iki = task_df.copy()
    df_iki["trial_in_seq"] = ((df_iki["TrialInBlock"] - 1) % 12) + 1 # loop through 12 trials per sequence
    df_iki = df_iki[df_iki["trial_in_seq"] >= 2].copy() 

    seq_speed = (
        df_iki.groupby(["SubjectID", "Group", "Block", "block_label", "sequence_number"], as_index=False)["RT"]
        .mean()
        .rename(columns={"RT": "mean_iki"})
    )
    seq_speed["speed"] = 1000.0 / seq_speed["mean_iki"]

    block_speed = (
        seq_speed.groupby(["SubjectID", "Group", "Block"], as_index=False)["speed"]
        .mean()
        .rename(columns={"speed": "mean_speed"})
    )
    block_speed_summary = (
        block_speed.groupby(["Group", "Block"])["mean_speed"]
        .agg(["mean", sem])
        .reset_index()
        .rename(columns={"mean": "group_mean", "sem": "group_sem"})
    )
    return seq_speed, block_speed, block_speed_summary


def compute_seq2seq_speed_gains(seq_speed, task):
    rows = []
    for subject_id, subj_data in seq_speed.groupby("SubjectID"):
        subj_data = subj_data.sort_values(["Block", "sequence_number"])
        group = subj_data["Group"].iloc[0]
        blocks = sorted(subj_data["Block"].unique())

        for index, block in enumerate(blocks[:-1]):
            next_block = blocks[index + 1]
            source_block = subj_data[subj_data["Block"] == block]
            target_block = subj_data[subj_data["Block"] == next_block]

            if source_block["block_label"].iloc[0] != "Sequence":
                continue
            if target_block["block_label"].iloc[0] != "Sequence":
                continue

            first_speed = source_block.loc[source_block["sequence_number"] == 1, "speed"].mean()
            last_speed = source_block.loc[source_block["sequence_number"] == 10, "speed"].mean()
            next_first_speed = target_block.loc[target_block["sequence_number"] == 1, "speed"].mean()

            if np.isnan(first_speed) or np.isnan(last_speed) or np.isnan(next_first_speed):
                print('Warning: Missing speed data for SubjectID {}, Block {} or {}'.format(subject_id, block, next_block))
                continue

            micro_online = last_speed - first_speed
            micro_offline = next_first_speed - last_speed
            rows.append(
                {
                    "SubjectID": subject_id,
                    "Task": task,
                    "Group": group,
                    "source_block": block,
                    "target_block": next_block,
                    "micro_online": micro_online,
                    "micro_offline": micro_offline,
                    "total": micro_online + micro_offline,
                }
            )

    seq2seq = pd.DataFrame(rows)
    if seq2seq.empty:
        return seq2seq, pd.DataFrame(
            columns=["Group", "transition_index", "group_mean", "group_sem"]
        )

    seq2seq = seq2seq.sort_values(["SubjectID", "source_block"]).copy()
    seq2seq["transition_index"] = seq2seq.groupby("SubjectID").cumcount() + 1
    seq2seq["cum_online"] = seq2seq.groupby("SubjectID")["micro_online"].cumsum()
    seq2seq["cum_offline"] = seq2seq.groupby("SubjectID")["micro_offline"].cumsum()
    seq2seq["cum_speed"] = seq2seq.groupby("SubjectID")["total"].cumsum()
    cumulative_summary = (
        seq2seq.groupby(["Group", "transition_index"])["cum_speed"]
        .agg(["mean", sem])
        .reset_index()
        .rename(columns={"mean": "group_mean", "sem": "group_sem"})
    )
    return seq2seq, cumulative_summary


def compute_skill(task_df, task):
    cfg = TASK_CONFIG[task]
    rows = []
    for subject_id, subj_data in task_df.groupby("SubjectID"):
        group = subj_data["Group"].iloc[0]
        rt_random_first = subj_data.loc[subj_data["Block"] == cfg["skill_random_first"], "RT"].mean()
        rt_random_last = subj_data.loc[subj_data["Block"] == cfg["skill_random_last"], "RT"].mean()
        rt_random_ref = subj_data.loc[subj_data["Block"] == cfg["skill_sequence_random_ref"], "RT"].mean()
        rt_sequence_block = subj_data.loc[subj_data["Block"] == cfg["skill_sequence_block"], "RT"].mean()
        skill = (
            rt_random_ref - rt_sequence_block
            if group == "Sequence"
            else rt_random_first - rt_random_last
        )
        rows.append(
            {
                "SubjectID": subject_id,
                "Task": task,
                "Group": group,
                "skill": skill,
                "rt_random_first": rt_random_first,
                "rt_random_last": rt_random_last,
                "rt_sequence_block": rt_sequence_block,
            }
        )
    return pd.DataFrame(rows)


def compute_speed_skill(block_speed, task):
    cfg = TASK_CONFIG[task]
    rows = []
    for subject_id, subj_data in block_speed.groupby("SubjectID"):
        group = subj_data["Group"].iloc[0]
        speed_random_first = subj_data.loc[subj_data["Block"] == cfg["skill_random_first"], "mean_speed"].mean()
        speed_random_last = subj_data.loc[subj_data["Block"] == cfg["skill_random_last"], "mean_speed"].mean()
        speed_random_ref = subj_data.loc[subj_data["Block"] == cfg["skill_sequence_random_ref"], "mean_speed"].mean()
        speed_sequence_block = subj_data.loc[subj_data["Block"] == cfg["skill_sequence_block"], "mean_speed"].mean()
        # speed is inverse to RT, so the subtraction order is flipped relative
        # to compute_skill to keep positive skill = better performance
        skill_speed = (
            speed_sequence_block - speed_random_ref
            if group == "Sequence"
            else speed_random_last - speed_random_first
        )
        rows.append(
            {
                "SubjectID": subject_id,
                "Task": task,
                "Group": group,
                "skill_speed": skill_speed,
                "speed_random_first": speed_random_first,
                "speed_random_last": speed_random_last,
                "speed_sequence_block": speed_sequence_block,
            }
        )
    return pd.DataFrame(rows)


def sum_seq2seq_gains(seq2seq, task):
    if seq2seq.empty:
        return pd.DataFrame(
            columns=[
                "SubjectID",
                "Task",
                "Group",
                "summed_micro_online",
                "summed_micro_offline",
                "summed_total_speed",
            ]
        )

    summary = (
        seq2seq.groupby(["SubjectID", "Group"], as_index=False)[
            ["micro_online", "micro_offline", "total"]
        ]
        .sum()
        .rename(
            columns={
                "micro_online": "summed_micro_online",
                "micro_offline": "summed_micro_offline",
                "total": "summed_total_speed",
            }
        )
    )
    summary["Task"] = task
    return summary


def load_pdp():
    pdp = pd.read_csv(os.path.join(CSV_DIR, "SRTT_PDP.csv"))
    pdp["Group"] = pdp["group"].map(PDP_GROUP_MAP)
    pdp = pdp.dropna(subset=["inclusion", "exclusion"])
    pdp["explicit_knowledge"] = (
        (pdp["inclusion"] > INCLUSION_THRESHOLD) & (pdp["exclusion"] > EXCLUSION_THRESHOLD)
    )
    return pdp

# Here again, the control group is labelled "No-sequence" in the manuscript while being coded "Random" in the data files.
GROUP_RENAME = {"Random": "No-sequence"}
GROUP_ORDER = ["Sequence", "No-sequence"]
GROUP_PALETTE = {"Sequence": COLOR_SEQ, "No-sequence": COLOR_RAN}
GROUP_MARKERS = {"Sequence": "o", "No-sequence": "s"}
GROUP_COLORS = GROUP_PALETTE
# Short display labels used in legends / group axis ticks
GROUP_LABEL = {"Sequence": "SEQ", "No-sequence": "NO-SEQ"}

SUMMED_METRICS = {
    "summed_micro_online": "Micro-online",
    "summed_micro_offline": "Micro-offline",
    "summed_total_speed": "Total",
}

# --- Global style ---------------------------------------------------------
plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica"],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.labelweight": "bold",
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "axes.grid": False,
})


# ==========================================================================
# Helpers
# ==========================================================================
def pfmt(p):
    """Format a p-value, using 'p<0.001' rather than the misleading 'p=0.000'."""
    return "p<0.001" if p < 0.001 else f"p={p:.3f}"

def panel_letter(ax, letter):
    ax.annotate(letter, xy=(-0.18, 1.05), xycoords="axes fraction",
                fontsize=12, fontweight="bold", va="bottom", ha="right")


def style(ax):
    style_axis(ax)


MEAN_COLOR = "#d62728"   # red mean marker


def raincloud(ax, values, center, color, width_half=0.2, box_width=0.05,
              point_size=10, draw_points=True):
    """Draw one raincloud at ``center``.
    Half-violin ("cloud") on the right, jittered raw points ("rain") on the
    left, and a thin boxplot in the middle whose red dot marks the mean.
    """
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return

    # cloud: half violin, clipped to the right of the centre line
    vp = ax.violinplot([values], positions=[center], widths=width_half * 2,
                        showextrema=False)
    for body in vp["bodies"]:
        verts = body.get_paths()[0].vertices
        verts[:, 0] = np.clip(verts[:, 0], center, np.inf)
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.45)
        body.set_linewidth(0.8)

    # rain: jittered individual points on the left
    if draw_points:
        jit = np.random.uniform(-width_half * 0.95, -width_half * 0.15, len(values))
        ax.scatter(center + jit, values, s=point_size, color=color, alpha=0.75,
                   edgecolor="white", linewidth=0.2, zorder=5)

    # inbuilt boxplot + mean marker at the centre
    ax.boxplot([values], positions=[center], widths=box_width, patch_artist=True,
               showfliers=False, showmeans=True,
               medianprops=dict(color="black", linewidth=1.0),
               boxprops=dict(facecolor="white", edgecolor="black", linewidth=0.8),
               whiskerprops=dict(color="black", linewidth=0.8),
               capprops=dict(color="black", linewidth=0.8),
               meanprops=dict(marker="o", markerfacecolor=MEAN_COLOR,
                              markeredgecolor="black", markersize=4, zorder=8))


def rainclouds(ax, df, xcol, ycol, order, palette, hue=None, hue_order=None,
               offset=0.25, width_half=0.2, xticklabels=None):
    """Raincloud per category, split into two groups by hue."""
    if hue is None:
        for i, cat in enumerate(order):
            raincloud(ax, df.loc[df[xcol] == cat, ycol], i, palette[cat],
                      width_half=width_half)
    else:
        k = len(hue_order)
        for i, cat in enumerate(order):
            for j, lev in enumerate(hue_order):
                c = i + (j - (k - 1) / 2) * 2 * offset
                raincloud(ax, df.loc[(df[xcol] == cat) & (df[hue] == lev), ycol],
                          c, palette[lev], width_half=width_half)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order if xticklabels is None else xticklabels)


# ==========================================================================
# Reusable panel builders
# ==========================================================================
def group_lineplot(ax, data, x_col, y_col):
    """Group mean +/- SE line plot."""
    sns.lineplot(data=data, x=x_col, y=y_col, hue="Group", style="Group",
                 hue_order=GROUP_ORDER, style_order=GROUP_ORDER, palette=GROUP_PALETTE,
                 markers=GROUP_MARKERS, dashes=False, errorbar="se",
                 err_kws={"alpha": 0.18, "linewidth": 0}, linewidth=1.8,
                 markersize=3, ax=ax)


def blocks_line_panel(fig, cell, per_subject_by_task, value_col, ylabel,
                      letter, add_legend=False):
    """Two-subplot (Acquisition wide / Retention narrow) across-block line plot."""
    inner = cell.subgridspec(1, 2, width_ratios=[2, 1], wspace=0.05)
    axes = []
    for i, task in enumerate(["Acquisition", "Retention"]):
        ax = fig.add_subplot(inner[0, i])
        data = per_subject_by_task[task]
        group_lineplot(ax, data, "Block", value_col)
        cfg = TASK_CONFIG[task]
        for b in cfg["random_blocks"]:
            ax.axvspan(b - 0.4, b + 0.4, color="gray", alpha=0.08)
            ax.text(b, 0.02, "random", rotation=90, transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=6, color="gray")
        ax.set_xticks(range(1, cfg["n_blocks"] + 1))
        ax.set_xlabel("Block")
        ax.set_title(task, fontweight="bold")
        style(ax)
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
        axes.append(ax)

    unify_ylims(axes)
    axes[0].set_ylabel(ylabel)
    axes[1].set_ylabel("")
    axes[1].tick_params(labelleft=False)          # shared Y -> drop duplicate ticks
    panel_letter(axes[0], letter)
    if add_legend:
        handles = [mlines.Line2D([], [], color=GROUP_PALETTE[g],
                                 marker=GROUP_MARKERS[g], label=GROUP_LABEL[g])
                   for g in GROUP_ORDER]
        axes[0].legend(handles=handles, frameon=False, loc="upper right")
    return axes


def skill_violin_panel(fig, cell, data_all, value_col, ylabel, letter):
    """Two-subplot (Acquisition / Retention) Sequence-vs-No-sequence comparison."""
    inner = cell.subgridspec(1, 2, wspace=0.25)
    axes = []
    for task in ["Acquisition", "Retention"]:
        ax = fig.add_subplot(inner[0, 0] if task == "Acquisition" else inner[0, 1])
        d = data_all[data_all["Task"] == task]
        rainclouds(ax, d, "Group", value_col, GROUP_ORDER, palette=GROUP_PALETTE,
                   width_half=0.28, xticklabels=[GROUP_LABEL[g] for g in GROUP_ORDER])
        ax.set_title(task, fontweight="bold")
        style(ax)
        axes.append((ax, task, d))

    # shared Y across the two tasks
    lo = min(ax.get_ylim()[0] for ax, _, _ in axes)
    hi = max(max(d[value_col]) for _, _, d in axes)
    rng = hi - lo
    for ax, task, d in axes:
        ax.set_ylim(lo - 0.05 * rng, hi + 0.22 * rng)
        ax.set_xlim(-0.6, 1.6)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8, zorder=0)

    axes[0][0].set_ylabel(ylabel)
    axes[1][0].set_ylabel("")
    axes[1][0].tick_params(labelleft=False)
    panel_letter(axes[0][0], letter)
    return [ax for ax, _, _ in axes]


def cumulative_micro_panel(ax, seq2seq, group, letter, title,
                           ylabel_text=None, legend=False):
    """Single-group cumulative micro-online / -offline / total line plot."""
    d = seq2seq[seq2seq["Group"] == group]
    for label, col in [("Micro-online", "cum_online"),
                       ("Micro-offline", "cum_offline"),
                       ("Total", "cum_speed")]:
        sns.lineplot(data=d, x="transition_index", y=col,
                     color=METRIC_PALETTE[label], linestyle=METRIC_LINESTYLES[label],
                     errorbar="se", err_style="bars",
                     err_kws={"capsize": 0, "elinewidth": 1.0},
                     marker="o", linewidth=1.6, markersize=4, label=label, ax=ax)
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("Seq→Seq transition")
    ax.set_ylabel(ylabel_text if ylabel_text else "")
    ax.set_xticks(sorted(d["transition_index"].unique()))
    style(ax)
    if legend:
        ax.legend(frameon=False, title=None)
    else:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
    panel_letter(ax, letter)


def summed_violin_panel(ax, summed_task, letter, title, ylabel_text=None,
                        add_legend=False, ylim=(-20, 25)):
    """Grouped rainclouds of the three summed metrics, per group."""
    long = summed_task.melt(id_vars=["SubjectID", "Task", "Group"],
                            value_vars=list(SUMMED_METRICS),
                            var_name="metric_raw", value_name="gain")
    long["metric"] = long["metric_raw"].map(SUMMED_METRICS)
    metric_order = list(SUMMED_METRICS.values())

    offset = 0.25
    rainclouds(ax, long, "metric", "gain", metric_order, palette=GROUP_PALETTE,
               hue="Group", hue_order=GROUP_ORDER, offset=offset, width_half=0.15)

    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_ylim(*ylim)
    ax.set_xlim(-0.6, len(metric_order) - 0.4)
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel(ylabel_text if ylabel_text else "")
    if not ylabel_text:
        ax.tick_params(labelleft=False)
    if add_legend:
        handles = [mpatches.Patch(facecolor=GROUP_PALETTE[g], label=GROUP_LABEL[g])
                   for g in GROUP_ORDER]
        ax.legend(handles=handles, frameon=False, loc="lower center", fontsize=8)
    style(ax)
    panel_letter(ax, letter)


def corr_square(ax, df, xcol, ycol, xlabel, ylabel, pooled=False, letter=None,
                hide_yticks=False):
    """Square scatter + regression. Per-group lines unless ``pooled``."""
    if pooled:
        groups = [("All", "#444444")]
    else:
        groups = [(g, GROUP_PALETTE[g]) for g in GROUP_ORDER]

    y_top = 0.97
    for g, color in groups:
        grp = df if g == "All" else df[df["Group"] == g]
        grp = grp.dropna(subset=[xcol, ycol])
        if grp.empty:
            continue
        ax.scatter(grp[xcol], grp[ycol], color=color, s=22, alpha=0.85,
                   edgecolor="none")
        x, y = grp[xcol], grp[ycol]
        if len(grp) >= 3 and x.nunique() > 1 and y.nunique() > 1:
            slope, intercept, r, p, _ = stats.linregress(x, y)
            xs = np.array([x.min(), x.max()])
            ax.plot(xs, intercept + slope * xs, color=color, linewidth=1.6)
            prefix = "" if pooled else f"{GROUP_LABEL[g]}: "
            ax.annotate(f"{prefix}r={r:.2f}, {pfmt(p)}",
                        xy=(0.04, y_top), xycoords="axes fraction", color=color,
                        fontsize=7.5, va="top",
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                                  edgecolor="none", alpha=0.75))
            y_top -= 0.10

    ax.axhline(0, color="black", linestyle="--", linewidth=0.6)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if hide_yticks:
        ax.tick_params(labelleft=False)
    ax.set_box_aspect(1)      # square plotting box
    style(ax)
    if letter is not None:
        panel_letter(ax, letter)


# ==========================================================================
# Data assembly (recomputed from the master data)
# ==========================================================================
def build_data():
    master = load_master_data()
    per_task = {}
    for task in ["Acquisition", "Retention"]:
        task_df = add_common_columns(master, task)
        block_rt, _ = compute_block_rt(task_df)
        seq_speed, block_speed, _ = compute_speed(task_df)
        seq2seq, _ = compute_seq2seq_speed_gains(seq_speed, task)
        per_task[task] = {
            "block_rt": block_rt,
            "block_speed": block_speed,
            "seq_speed": seq_speed,
            "seq2seq": seq2seq,
            "skill": compute_skill(task_df, task),
            "speed_skill": compute_speed_skill(block_speed, task),
            "summed": sum_seq2seq_gains(seq2seq, task),
        }

    # Rename the control group everywhere: "Random" -> "No-sequence"
    for frames in per_task.values():
        for frame in frames.values():
            if "Group" in frame.columns:
                frame["Group"] = frame["Group"].replace(GROUP_RENAME)

    def cat(key):
        return pd.concat([per_task["Acquisition"][key], per_task["Retention"][key]],
                         ignore_index=True)

    # Acquisition gains + skill, joined with retention skill (as in main())
    acq = pd.merge(per_task["Acquisition"]["skill"], per_task["Acquisition"]["summed"],
                   on=["SubjectID", "Task", "Group"], how="left").rename(columns={
        "skill": "skill_acquisition",
        "summed_micro_online": "summed_micro_online_acquisition",
        "summed_micro_offline": "summed_micro_offline_acquisition",
        "summed_total_speed": "summed_total_speed_acquisition",
    })
    ret_skill = per_task["Retention"]["skill"][["SubjectID", "skill"]].rename(
        columns={"skill": "skill_retention"})
    cross = pd.merge(acq, ret_skill, on="SubjectID", how="inner")

    return {
        "per_task": per_task,
        "block_rt": {t: per_task[t]["block_rt"] for t in per_task},
        "block_speed": {t: per_task[t]["block_speed"] for t in per_task},
        "skill_all": cat("skill"),
        "speed_skill_all": cat("speed_skill"),
        "seq2seq_all": cat("seq2seq"),
        "cross": cross,
        "pdp": load_pdp().assign(
            Group=lambda d: d["Group"].replace(GROUP_RENAME)),
    }


# ==========================================================================
# Figures
# ==========================================================================
def figure_2(data):
    fig = plt.figure(figsize=(8, 7.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1.25, 1],
                          hspace=0.4, wspace=0.35)

    blocks_line_panel(fig, gs[0, 0], data["block_rt"], "mean_rt",
                      "Mean RT (ms)", "A", add_legend=True)          # A top-left
    blocks_line_panel(fig, gs[1, 0], data["block_speed"], "mean_speed",
                      "Speed (kp/s)", "C")                            # C bottom-left
    skill_violin_panel(fig, gs[0, 1], data["skill_all"], "skill",
                       "Skill (ms)", "B")                             # B top-right
    skill_violin_panel(fig, gs[1, 1], data["speed_skill_all"], "skill_speed",
                       "Skill (kp/s)", "D")                           # D bottom-right

    fig.savefig(os.path.join(FIG_DIR, "Figure_2.pdf"), bbox_inches="tight")
    plt.close(fig)


def figure_3(data):
    seq2seq_acq = data["seq2seq_all"][data["seq2seq_all"]["Task"] == "Acquisition"]
    summed_acq = data["per_task"]["Acquisition"]["summed"]
    summed_ret = data["per_task"]["Retention"]["summed"]
    cross = data["cross"]

    fig = plt.figure(figsize=(8, 10.5))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.15, 1.15, 1.6, 1.6], hspace=0.4)

    # Row 1: cumulative micro-learning (Acquisition), per group
    r0 = gs[0].subgridspec(1, 2, wspace=0.3)
    axA = fig.add_subplot(r0[0, 0])
    cumulative_micro_panel(axA, seq2seq_acq, "Sequence", "A", title="Sequence",
                           ylabel_text="Cumulative speed (kp/s)", legend=True)
    axB = fig.add_subplot(r0[0, 1])
    cumulative_micro_panel(axB, seq2seq_acq, "No-sequence", "B", title="No-Sequence")
    axB.set_ylim(-3, 3)
    axB.set_yticks(range(-3, 4))

    # Row 2: summed micro-learning comparison (Acquisition / Retention)
    r1 = gs[1].subgridspec(1, 2, wspace=0.3)
    summed_violin_panel(fig.add_subplot(r1[0, 0]), summed_acq, "C", title="Acquisition",
                        ylabel_text="Σ gain (kp/s)", add_legend=True)
    summed_violin_panel(fig.add_subplot(r1[0, 1]), summed_ret, "D", title="Retention")

    # Row 3: E (online) and F (offline) vs Acquisition skill
    rEF = gs[2].subgridspec(1, 2, wspace=0.3)
    axE = fig.add_subplot(rEF[0, 0])
    corr_square(axE, cross, "summed_micro_online_acquisition", "skill_acquisition",
                "Micro-online gain (kp/s)", "Acquisition skill\nΔRT (ms)", letter="E")
    axF = fig.add_subplot(rEF[0, 1])
    corr_square(axF, cross, "summed_micro_offline_acquisition", "skill_acquisition",
                "Micro-offline gain (kp/s)", "Acquisition skill\nΔRT (ms)", letter="F")

    # Row 4: G (online) and H (offline) vs Retention skill
    rGH = gs[3].subgridspec(1, 2, wspace=0.3)
    axG = fig.add_subplot(rGH[0, 0])
    corr_square(axG, cross, "summed_micro_online_acquisition", "skill_retention",
                "Micro-online gain (kp/s)", "Retention skill\nΔRT (ms)", letter="G")
    axH = fig.add_subplot(rGH[0, 1])
    corr_square(axH, cross, "summed_micro_offline_acquisition", "skill_retention",
                "Micro-offline gain (kp/s)", "Retention skill\nΔRT (ms)", letter="H")

    # Keep the skill (y) axes on a common 100 ms tick interval across E-H
    for a in (axE, axF, axG, axH):
        a.yaxis.set_major_locator(mticker.MultipleLocator(100))

    fig.savefig(os.path.join(FIG_DIR, "Figure_3.pdf"), bbox_inches="tight")
    plt.close(fig)


def pdp_violin_panel(ax, pdp, letter):
    long = pdp.melt(id_vars=["subID", "Group", "explicit_knowledge"],
                    value_vars=["inclusion", "exclusion"],
                    var_name="Condition", value_name="score")
    long["Condition"] = long["Condition"].str.capitalize()
    order = ["Inclusion", "Exclusion"]

    offset, wh = 0.25, 0.15
    for i, cond in enumerate(order):
        for j, g in enumerate(GROUP_ORDER):
            c = i + (j - 0.5) * 2 * offset
            sel = (long["Condition"] == cond) & (long["Group"] == g)
            # cloud + box, but supply the rain points ourselves (implicit/explicit)
            raincloud(ax, long.loc[sel, "score"], c, GROUP_PALETTE[g],
                      width_half=wh, draw_points=False)
            # implicit subjects: small filled points on the left
            impl = long.loc[sel & ~long["explicit_knowledge"], "score"]
            ax.scatter(c + np.random.uniform(-wh * 0.95, -wh * 0.15, len(impl)), impl,
                       facecolor=GROUP_COLORS[g], edgecolor="white", linewidth=0.3,
                       s=14, alpha=0.9, zorder=6)
            # explicit-knowledge subjects: same size as the rest, but dark edge
            expl = long.loc[sel & long["explicit_knowledge"], "score"]
            if not expl.empty:
                ax.scatter(c + np.random.uniform(-wh * 0.95, -wh * 0.15, len(expl)),
                           expl, facecolor=GROUP_COLORS[g], edgecolor="black",
                           linewidth=0.8, s=14, marker="o", zorder=7)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_xlim(-0.6, 1.6)

    ax.axhline(INCLUSION_THRESHOLD, ls="--", color="gray", lw=1.0)
    ax.axhline(EXCLUSION_THRESHOLD, ls="--", color="gray", lw=1.0)
    ax.set_xlabel("PDP component")
    ax.set_ylabel("Fragment completion accuracy")
    handles = [mpatches.Patch(facecolor=GROUP_PALETTE[g], label=GROUP_LABEL[g])
               for g in GROUP_ORDER]
    ax.legend(handles=handles, frameon=False, loc="upper right")
    style(ax)
    panel_letter(ax, letter)


def supplemental_figure_1(data):
    """S1: performance speed per sequence within each block (Acq wide / Ret narrow)"""
    gap = 2
    fig, axes = plt.subplots(1, 2, figsize=(8, 5), gridspec_kw={"width_ratios": [2, 1]})

    for ax, task in zip(axes, ["Acquisition", "Retention"]):
        seq_speed = data["per_task"][task]["seq_speed"].copy()
        seq_speed["x"] = (seq_speed["Block"] - 1) * (10 + gap) + seq_speed["sequence_number"]
        for group in GROUP_ORDER:
            grp = seq_speed[seq_speed["Group"] == group]
            for block in sorted(grp["Block"].unique()):
                sns.lineplot(
                    data=grp[grp["Block"] == block], x="x", y="speed",
                    color=GROUP_PALETTE[group], marker=GROUP_MARKERS[group],
                    errorbar="se", err_kws={"alpha": 0.15, "linewidth": 0},
                    linewidth=1.2, markersize=3, ax=ax,
                )

    unify_ylims(axes)

    for ax, task in zip(axes, ["Acquisition", "Retention"]):
        cfg = TASK_CONFIG[task]
        tick_pos = []
        for block in range(1, cfg["n_blocks"] + 1):
            x_start = (block - 1) * (10 + gap) + 0.5
            x_end = (block - 1) * (10 + gap) + 10.5
            if block in cfg["random_blocks"]:
                ax.axvspan(x_start, x_end, alpha=0.1, color="gray")
            tick_pos.append((x_start + x_end) / 2)
        ax.set_title(task, fontweight="bold")
        ax.set_xlabel("Block")
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(range(1, cfg["n_blocks"] + 1))
        style(ax)

    # shared Y across the two tasks -> keep the label/ticks only on the left panel
    axes[0].set_ylabel("Speed (kp/s)")
    axes[1].set_ylabel("")
    axes[1].tick_params(labelleft=False)

    axes[0].legend(
        handles=[mlines.Line2D([], [], color=GROUP_PALETTE[g],
                               marker=GROUP_MARKERS[g], label=GROUP_LABEL[g])
                 for g in GROUP_ORDER],
        frameon=False,
    )
    fig.savefig(os.path.join(FIG_DIR, "Supplemental_Figure_1.pdf"), bbox_inches="tight")
    plt.close(fig)


def supplemental_figure_2(data):
    seq2seq_ret = data["seq2seq_all"][data["seq2seq_all"]["Task"] == "Retention"]
    cross = data["cross"]

    fig = plt.figure(figsize=(8, 7.5))
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35)

    # Row 1: cumulative micro-learning (Retention), per group
    axA = fig.add_subplot(gs[0, 0])
    cumulative_micro_panel(axA, seq2seq_ret, "Sequence", "A", title="Sequence",
                           ylabel_text="Cumulative speed gain (kp/s)", legend=True)
    axB = fig.add_subplot(gs[0, 1])
    cumulative_micro_panel(axB, seq2seq_ret, "No-sequence", "B", title="No-Sequence")
    axB.set_ylim(-2, 2)                       # own scale, not shared with panel A
    axB.set_yticks(range(-2, 3))

    # Row 2: PDP explicit knowledge + pooled offline-vs-online correlation
    pdp_violin_panel(fig.add_subplot(gs[1, 0]), data["pdp"], "C")
    corr_square(fig.add_subplot(gs[1, 1]), cross,
                "summed_micro_online_acquisition", "summed_micro_offline_acquisition",
                "Micro-online gain (kp/s)", "Micro-offline gain (kp/s)", pooled=True,
                letter="D")

    fig.savefig(os.path.join(FIG_DIR, "Supplemental_Figure_2.pdf"), bbox_inches="tight")
    plt.close(fig)


def write_analysis_csvs():
    """Regenerate the tidy CSVs that iSRTT_stats.Rmd reads."""
    master = load_master_data()
    task_data = {}
    speed_per_subject = {}
    for task in ["Acquisition", "Retention"]:
        task_df = add_common_columns(master, task)
        seq_speed, block_speed, _ = compute_speed(task_df)
        seq2seq, _ = compute_seq2seq_speed_gains(seq_speed, task)
        task_data[task] = {
            "skill": compute_skill(task_df, task),
            "speed_skill": compute_speed_skill(block_speed, task),
            "summed_gains": sum_seq2seq_gains(seq2seq, task),
            "seq2seq": seq2seq,
        }
        speed_per_subject[task] = block_speed

    skill_all = pd.concat(
        [task_data["Acquisition"]["skill"], task_data["Retention"]["skill"]],
        ignore_index=True,
    )
    speed_skill_all = pd.concat(
        [task_data["Acquisition"]["speed_skill"], task_data["Retention"]["speed_skill"]],
        ignore_index=True,
    )
    summed_gains_all = pd.concat(
        [task_data["Acquisition"]["summed_gains"], task_data["Retention"]["summed_gains"]],
        ignore_index=True,
    )
    seq2seq_all = pd.concat(
        [task_data["Acquisition"]["seq2seq"], task_data["Retention"]["seq2seq"]],
        ignore_index=True,
    )

    skill_all.to_csv(os.path.join(CSV_DIR, "SRTT_skill_metric_all.csv"), index=False)
    speed_skill_all.to_csv(os.path.join(CSV_DIR, "SRTT_skill_speed_metric_all.csv"), index=False)
    summed_gains_all.to_csv(os.path.join(CSV_DIR, "SRTT_seq2seq_summed_gains_all.csv"), index=False)
    seq2seq_all.to_csv(os.path.join(CSV_DIR, "SRTT_bonstrup_speed_seq2seq_all.csv"), index=False)

    speed_across_blocks_long = pd.concat(
        [speed_per_subject[task].assign(Task=task) for task in ["Acquisition", "Retention"]],
        ignore_index=True,
    )[["SubjectID", "Task", "Block", "Group", "mean_speed"]]
    speed_across_blocks_long.to_csv(
        os.path.join(CSV_DIR, "summary_SRTT_speed_data.csv"), index=False
    )

    acq_summary = pd.merge(
        task_data["Acquisition"]["skill"],
        task_data["Acquisition"]["summed_gains"],
        on=["SubjectID", "Task", "Group"],
        how="left",
    ).rename(
        columns={
            "skill": "skill_acquisition",
            "summed_micro_online": "summed_micro_online_acquisition",
            "summed_micro_offline": "summed_micro_offline_acquisition",
            "summed_total_speed": "summed_total_speed_acquisition",
        }
    )
    ret_skill = task_data["Retention"]["skill"][["SubjectID", "skill"]].rename(
        columns={"skill": "skill_retention"}
    )
    cross_df = pd.merge(acq_summary, ret_skill, on="SubjectID", how="inner")
    cross_df.to_csv(
        os.path.join(CSV_DIR, "SRTT_acquisition_gain_skill_correlations.csv"), index=False
    )


def main():
    np.random.seed(0)   
    write_analysis_csvs()
    data = build_data()
    figure_2(data)
    figure_3(data)
    supplemental_figure_1(data)
    supplemental_figure_2(data)
    print("Wrote Figure_2.pdf, Figure_3.pdf, Supplemental_Figure_1.pdf, "
          "Supplemental_Figure_2.pdf to", FIG_DIR)


if __name__ == "__main__":
    main()

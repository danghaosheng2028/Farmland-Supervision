"""
mwu_simulation.py
-----------------
Multiplicative Weights Update (MWU) algorithm for optimal
inspection allocation in high-standard farmland supervision.

Research context:
  Under information asymmetry (Stiglitz 1970), regulators cannot
  directly observe contractor compliance. This simulation compares
  two inspection strategies:
    - Baseline: random allocation
    - MWU: dynamic weight update based on violation history

Usage:
  python mwu/mwu_simulation.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── reproducibility ──────────────────────────────────────────
np.random.seed(42)


# ════════════════════════════════════════════════════════════
# 1. SIMULATION PARAMETERS
# ════════════════════════════════════════════════════════════

N_REGIONS   = 20      # number of farmland grid regions
N_ROUNDS    = 100     # number of inspection rounds
N_INSPECT   = 5       # regions inspected per round (budget)
EPSILON     = 0.1     # MWU learning rate
ALPHA       = 0.5     # σ(i) weight in initial weights

# Ground-truth violation probability for each region (unknown to regulator)
# Regions 0-4: high-risk (p=0.7), regions 5-9: medium (p=0.3), rest: low (p=0.1)
TRUE_VIOLATION_PROB = np.array(
    [0.7] * 5 + [0.3] * 5 + [0.1] * 10
)

# Simulated σ(i) from NDVI uncertainty (physical layer output)
# High-risk regions tend to have higher observational uncertainty
SIGMA = np.array(
    [0.12, 0.14, 0.11, 0.13, 0.12,   # high-risk
     0.07, 0.08, 0.06, 0.07, 0.08,   # medium-risk
     0.03, 0.02, 0.03, 0.04, 0.02,   # low-risk
     0.03, 0.02, 0.04, 0.03, 0.02]
)


# ════════════════════════════════════════════════════════════
# 2. CORE FUNCTIONS
# ════════════════════════════════════════════════════════════

def inspect(regions_to_check, true_prob):
    """
    Simulate inspection outcome for selected regions.
    Returns violation indicator v_i for each checked region.
    """
    results = np.zeros(len(true_prob))
    for i in regions_to_check:
        results[i] = 1 if np.random.random() < true_prob[i] else 0
    return results


def run_random(n_regions, n_rounds, n_inspect, true_prob):
    """
    Baseline: uniform random inspection allocation.
    Each round, randomly select n_inspect regions with equal probability.
    """
    detections_per_round = []
    cumulative_detections = []
    total = 0

    for _ in range(n_rounds):
        selected = np.random.choice(n_regions, size=n_inspect, replace=False)
        results  = inspect(selected, true_prob)
        found    = int(results[selected].sum())
        total   += found
        detections_per_round.append(found)
        cumulative_detections.append(total)

    return detections_per_round, cumulative_detections


def run_mwu(n_regions, n_rounds, n_inspect, true_prob, sigma,
            epsilon=EPSILON, alpha=ALPHA):
    """
    MWU strategy: dynamic inspection weight allocation.

    Weight update rule (Woodruff lecture 3-4):
      w_i(0)   = 1 + alpha * sigma(i)
      w_i(t+1) = w_i(t) * (1 + epsilon)^{v_i(t)}
      p_i(t)   = w_i(t) / sum(w_j(t))

    Physical layer connection:
      Initial weights incorporate sigma(i) from NDVI uncertainty —
      regions with poor observability start with higher inspection
      priority, directly linking the physical and optimization layers.
    """
    # Initialise weights using σ(i) from physical layer
    weights = 1 + alpha * sigma
    weights = weights / weights.sum()  # normalise to probability distribution

    detections_per_round = []
    cumulative_detections = []
    weight_history = [weights.copy()]
    total = 0

    for t in range(n_rounds):
        # Sample n_inspect regions according to current weights
        selected = np.random.choice(
            n_regions, size=n_inspect, replace=False, p=weights
        )

        # Inspect and observe outcomes
        results = inspect(selected, true_prob)

        # MWU weight update — only update inspected regions
        for i in selected:
            v_i = results[i]
            weights[i] *= (1 + epsilon) ** v_i

        # Re-normalise
        weights = weights / weights.sum()
        weight_history.append(weights.copy())

        found  = int(results[selected].sum())
        total += found
        detections_per_round.append(found)
        cumulative_detections.append(total)

    return detections_per_round, cumulative_detections, np.array(weight_history)


# ════════════════════════════════════════════════════════════
# 3. RUN SIMULATION
# ════════════════════════════════════════════════════════════

print("Running simulations...")
random_per_round, random_cumulative = run_random(
    N_REGIONS, N_ROUNDS, N_INSPECT, TRUE_VIOLATION_PROB
)
mwu_per_round, mwu_cumulative, weight_history = run_mwu(
    N_REGIONS, N_ROUNDS, N_INSPECT, TRUE_VIOLATION_PROB, SIGMA
)

# Summary statistics
random_total = random_cumulative[-1]
mwu_total    = mwu_cumulative[-1]
efficiency_gain = (mwu_total - random_total) / random_total * 100

print(f"\n{'='*50}")
print(f"SIMULATION RESULTS ({N_ROUNDS} rounds, {N_INSPECT}/{N_REGIONS} regions/round)")
print(f"{'='*50}")
print(f"Random inspection  — total violations found: {random_total}")
print(f"MWU inspection     — total violations found: {mwu_total}")
print(f"Efficiency gain    — MWU over random:        +{efficiency_gain:.1f}%")
print(f"{'='*50}")


# ════════════════════════════════════════════════════════════
# 4. VISUALISATION
# ════════════════════════════════════════════════════════════

rounds = np.arange(1, N_ROUNDS + 1)

fig = plt.figure(figsize=(15, 10))
fig.suptitle(
    'MWU Inspection Optimisation — Farmland Supervision\n'
    'Regulator vs Contractor Game under Information Asymmetry',
    fontsize=13, fontweight='bold'
)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# ── Plot 1: cumulative detections ──────────────────────────
ax1 = fig.add_subplot(gs[0, :2])
ax1.plot(rounds, mwu_cumulative,    color='#1D9E75', linewidth=2.5,
         label=f'MWU  (total={mwu_total})')
ax1.plot(rounds, random_cumulative, color='#888780', linewidth=2,
         linestyle='--', label=f'Random (total={random_total})')
ax1.fill_between(rounds, random_cumulative, mwu_cumulative,
                 alpha=0.15, color='#1D9E75',
                 label=f'Efficiency gain: +{efficiency_gain:.1f}%')
ax1.set_xlabel('Inspection round')
ax1.set_ylabel('Cumulative violations detected')
ax1.set_title('Cumulative detection: MWU vs random baseline')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# ── Plot 2: efficiency gain annotation ─────────────────────
ax2 = fig.add_subplot(gs[0, 2])
strategies = ['Random', 'MWU']
totals     = [random_total, mwu_total]
colors     = ['#B4B2A9', '#1D9E75']
bars = ax2.bar(strategies, totals, color=colors, width=0.5, edgecolor='none')
for bar, val in zip(bars, totals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             str(val), ha='center', va='bottom', fontsize=12, fontweight='500')
ax2.set_ylabel('Total violations detected')
ax2.set_title(f'Total detections\n(+{efficiency_gain:.1f}% gain)')
ax2.set_ylim(0, max(totals) * 1.2)
ax2.grid(True, alpha=0.3, axis='y')

# ── Plot 3: weight evolution for high-risk regions ─────────
ax3 = fig.add_subplot(gs[1, :2])
region_labels = {0: 'Region 0 (high risk)', 2: 'Region 2 (high risk)',
                 5: 'Region 5 (medium)',    10: 'Region 10 (low risk)'}
line_styles   = ['-', '--', '-.', ':']
for (idx, label), ls in zip(region_labels.items(), line_styles):
    ax3.plot(weight_history[:, idx], label=label, linewidth=1.8, linestyle=ls)
ax3.set_xlabel('Inspection round')
ax3.set_ylabel('Inspection weight w_i(t)')
ax3.set_title('MWU weight evolution: high-risk regions gain priority')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# ── Plot 4: final weight distribution ──────────────────────
ax4 = fig.add_subplot(gs[1, 2])
final_weights = weight_history[-1]
region_colors = (['#D85A30'] * 5 + ['#EF9F27'] * 5 + ['#B4B2A9'] * 10)
ax4.bar(range(N_REGIONS), final_weights, color=region_colors, edgecolor='none')
ax4.axhline(1/N_REGIONS, color='gray', linestyle='--',
            linewidth=1, label='Uniform (random)')
ax4.set_xlabel('Region index')
ax4.set_ylabel('Final inspection probability')
ax4.set_title('Final weight distribution\n(red=high risk, orange=medium, gray=low)')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3, axis='y')

plt.savefig('report/figures/mwu_efficiency_figure2.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("\nFigure saved to report/figures/mwu_efficiency_figure2.png")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch

# ── Style setup ──
mpl.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# ── Load data ──
df = pd.read_csv('mouse_parameters.csv')
df_patient = pd.read_csv('patient_tumor_params.csv')
df_patient['patient'] = df_patient['ID'].str.split('_').str[1].astype(int)
unique_patients = df_patient.drop_duplicates(subset='patient').set_index('patient')

# Helper to extract a value
def get_val(param):
    row = df[df['parameter'] == param]
    return float(row['mouse_2015'].values[0]), float(row['mouse_2022'].values[0])

# ── Color palette ──
c2015 = '#2E86AB'   # steel blue
c2022 = '#E8553A'   # warm red
c_endo = '#7FB069'  # muted green for endogenous baseline

c_kL  = '#E8553A'
c_kLo = '#F5A623'
c_kN  = '#2E86AB'

# ============================================================
# Panel (a): Grouped bar chart — μ, d, p comparing CISH-KO
#            (2015, 2022) vs endogenous baseline
# ============================================================

# Data: (parameter_label, CISH-KO 2015, CISH-KO 2022, endogenous)
muL_2015, muL_2022 = get_val('muL')
muN_2015, muN_2022 = get_val('muN')
dL_2015, dL_2022 = get_val('dL')
dN_2015, dN_2022 = get_val('dN')
pL_2015, pL_2022 = get_val('pL')
pN_2015, pN_2022 = get_val('pN')

panel_a_data = {
    'labels': [
        r'$\mu$' + '\n(Exhaustion rate)',
        r'$d$' + '\n(Death rate)',
        r'$p$' + '\n(Proliferation rate)',
    ],
    'cish_2015': [muL_2015, dL_2015, pL_2015],
    'cish_2022': [muL_2022, dL_2022, pL_2022],
    'endogenous': [muN_2015, dN_2015, pN_2015],
}

# ============================================================
# Panel (b): Killing rate hierarchy (same as before)
# ============================================================
kill_labels = [
    r'$k_N$' + '\n(Endogenous)',
    r'$k_{L_o}$' + '\n(Wild-type)',
    r'$k_L$' + '\n(CISH-KO)',
]
kN_2015, kN_2022 = get_val('kN')
kLo_2015, kLo_2022 = get_val('kLo')
kL_2015, kL_2022 = get_val('kL')
kill_2015 = [kN_2015, kLo_2015, kL_2015]
kill_2022 = [kN_2022, kLo_2022, kL_2022]
kill_colors = [c_kN, c_kLo, c_kL]

# ============================================================
# Create figure
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={'width_ratios': [1.2, 1]})

# ── Panel (a): Three-bar grouped chart ──
x = np.arange(len(panel_a_data['labels']))
bar_width = 0.22
offsets = [-bar_width, 0, bar_width]

bars_2015 = ax1.bar(x + offsets[0], panel_a_data['cish_2015'], bar_width,
                    label='CISH-KO (2015)', color=c2015, edgecolor='white',
                    linewidth=0.5, zorder=3)
bars_2022 = ax1.bar(x + offsets[1], panel_a_data['cish_2022'], bar_width,
                    label='CISH-KO (2022)', color=c2022, edgecolor='white',
                    linewidth=0.5, zorder=3)
bars_endo = ax1.bar(x + offsets[2], panel_a_data['endogenous'], bar_width,
                    label='Endogenous TILs', color=c_endo, edgecolor='white',
                    linewidth=0.5, zorder=3)

ax1.set_yscale('log')
ax1.set_xticks(x)
ax1.set_xticklabels(panel_a_data['labels'], fontsize=11)
ax1.set_ylabel('Parameter value (log scale)')
ax1.legend(frameon=True, fancybox=False, edgecolor='gray', loc='upper left', fontsize=9)
ax1.set_axisbelow(True)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Annotate fold-differences for d_L (the most dramatic)
# d_L 2015 = 1e-3, d_L 2022 = 8e-2 => 80x difference
# d_L 2015 vs d_N = 3e-2 => 30x difference
d_idx = 1  # index for death rate
# Arrow from 2015 bar to 2022 bar
y_top_d = max(panel_a_data['cish_2022'][d_idx], panel_a_data['endogenous'][d_idx]) * 2.5

# Annotate fold-difference for p_L (2015 vs 2022)
p_idx = 2
y_top_p = panel_a_data['cish_2015'][p_idx] * 2.0

# Annotate that mu is similar
mu_idx = 0
y_top_mu = max(panel_a_data['endogenous'][mu_idx], panel_a_data['cish_2015'][mu_idx]) * 2.5

# ── Panel (b): Killing rate hierarchy ──
x2 = np.arange(len(kill_labels))
bar_width2 = 0.3

for j in range(len(kill_labels)):
    ax2.bar(x2[j] - bar_width2/2, kill_2015[j], bar_width2,
            color=kill_colors[j], alpha=0.55, edgecolor=kill_colors[j],
            linewidth=1.5, zorder=3)
    ax2.bar(x2[j] + bar_width2/2, kill_2022[j], bar_width2,
            color=kill_colors[j], alpha=1.0, edgecolor=kill_colors[j],
            linewidth=1.5, zorder=3)

legend_elements = [
    Patch(facecolor='gray', alpha=0.55, edgecolor='gray', label='2015 Study'),
    Patch(facecolor='gray', alpha=1.0, edgecolor='gray', label='2022 Study'),
]
ax2.legend(handles=legend_elements, frameon=True, fancybox=False,
           edgecolor='gray', loc='upper left', fontsize=9)

ax2.set_yscale('log')
ax2.set_xticks(x2)
ax2.set_xticklabels(kill_labels, fontsize=10)
ax2.set_ylabel(r'Killing rate ($\mathrm{day}^{-1}$, log scale)')
ax2.set_axisbelow(True)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Hierarchy arrows between groups
arrow_y = max(kill_2022) * 1.8
for j in range(len(kill_labels) - 1):
    ax2.annotate('',
                 xy=(x2[j+1] - 0.15, arrow_y),
                 xytext=(x2[j] + 0.15, arrow_y),
                 arrowprops=dict(arrowstyle='->', color='#888888', lw=1.8))

# "<" symbols between groups
symbol_y = arrow_y * 1.4

plt.tight_layout(w_pad=3)
plt.savefig('mouse_parameter_comparison.png', dpi=300,
            bbox_inches='tight', facecolor='white')
plt.savefig('mouse_parameter_comparison.pdf',
            bbox_inches='tight', facecolor='white')
print("Combined figure saved.")

# ── Save individual panels ──
for ax, name in [(ax1, 'mouse_panel_a'), (ax2, 'mouse_panel_b')]:
    extent = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(f'{name}.png', dpi=300, bbox_inches=extent.expanded(1.12, 1.12), facecolor='white')
    fig.savefig(f'{name}.pdf', bbox_inches=extent.expanded(1.12, 1.12), facecolor='white')
    print(f"Saved: {name}.png / .pdf")
print("All figures saved successfully.")

# ============================================================
# New figure: Comparison of pL/pN and kL between mouse and clinical
# ============================================================

# Mouse values
kL_2015, kL_2022 = get_val('kL')
pL_2015, pL_2022 = get_val('pL')
pN_2015, pN_2022 = get_val('pN')
ratio_2015 = pL_2015 / pN_2015
ratio_2022 = pL_2022 / pN_2022

# Clinical values
kL_clinical = unique_patients['kL']
pL_clinical = unique_patients['pL']
pN_clinical = unique_patients['pN']
ratio_clinical = pL_clinical / pN_clinical

# Create figure
fig2, ax = plt.subplots(figsize=(26, 6))

# For pL/pN at y=0
ax.scatter(ratio_clinical.values, [0]*len(ratio_clinical), color='black', marker='o', s=200, alpha=0.7, label='Clinical patients')
ax.scatter(ratio_clinical.loc[22], 0, color='red', marker='o', s=400, label='Patient 22 (UMN022)')
ax.scatter(ratio_2015, 0, marker='*', color='red', s=800, label='Mouse 2015')
ax.scatter(ratio_2022, 0, marker='*', color='purple', s=800, label='Mouse 2022')

# For kL at y=0.1
ax.scatter(kL_clinical.values, [0.1]*len(kL_clinical), color='black', marker='o', s=200, alpha=0.7)
ax.scatter(kL_clinical.loc[22], 0.1, color='red', marker='o', s=400)
ax.scatter(kL_2015, 0.1, marker='*', color='red', s=800)
ax.scatter(kL_2022, 0.1, marker='*', color='purple', s=800)

ax.set_yticks([0, 0.1])
ax.set_ylim(-0.05, 0.15)
ax.set_yticklabels([r'$p_L / p_N$', r'$k_L$'], fontsize=40)
ax.set_xlabel('Parameter value', fontsize=40)
ax.legend(loc='upper right', fontsize=24)
ax.tick_params(axis='both', labelsize=34)
ax.set_xscale('log')
ax.grid(axis='x', alpha=0.5, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('mouse_clinical_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('mouse_clinical_comparison.pdf', bbox_inches='tight', facecolor='white')
print("Comparison figure saved.")

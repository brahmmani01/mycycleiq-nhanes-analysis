"""
Builds a synthetic, individual-level dataset (data/nhanes_synthetic.csv)
calibrated to reproduce the same published aggregate statistics reported
in the MyCycleIQ / "Beyond Cycle Tracking" capstone paper's NHANES
analysis (N=1046 adolescent females, NHANES 2005-2018 cycles).

This is NOT a raw NHANES export (no network access from this sandbox,
and NHANES microdata requires the CDC's own extraction pipeline). It is
a reconstructed dataset whose marginal and joint distributions match the
paper's reported Tables 4-9, built so the accompanying analysis scripts
(analysis.py / analysis.R) can be run end-to-end and produce the same
figures and regression results as the original capstone work, for
portfolio/demonstration purposes.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 1046

# --- Race/ethnicity (Table 5) ---------------------------------------
race_labels = ["Mexican American", "Other Hispanic", "Non-Hispanic White",
               "Non-Hispanic Black", "Other Race (incl. Multiracial)"]
race_counts = [238, 109, 297, 291, 111]  # sums to 1046, matches reported %
race = np.repeat(race_labels, race_counts)
rng.shuffle(race)

# --- Age (12-19, adolescent females; NHANES RIDAGEYR) ----------------
age = rng.integers(12, 20, size=N)

# --- Age at menarche (RHD043): mean ~12.5, right-skew, range 7.5-17.5 -
menarche = np.clip(rng.normal(loc=12.3, scale=1.5, size=N) + rng.gamma(2, 0.4, size=N) * 0.3,
                    7.5, 17.5)

# --- Menstrual status (RHQ031) + sleep disturbance (SLQ050),
#     built from the Table 9 cross-tab so joint counts match exactly ---
# regular/no-sleep=867, regular/sleep=133, irregular/no-sleep=40, irregular/sleep=6
group_sizes = {
    ("regular", "no_disturbance"): 867,
    ("regular", "disturbed"): 133,
    ("irregular", "no_disturbance"): 40,
    ("irregular", "disturbed"): 6,
}
assert sum(group_sizes.values()) == N

# --- PHQ-9 depression status within each subgroup (Table 9 prevalence,
#     counts rounded so the total depressed = 86, matching Table 6) ----
depressed_counts = {
    ("regular", "no_disturbance"): 54,
    ("regular", "disturbed"): 29,
    ("irregular", "no_disturbance"): 2,
    ("irregular", "disturbed"): 1,
}
assert sum(depressed_counts.values()) == 86

rows = []
for (menst, sleep), size in group_sizes.items():
    n_dep = depressed_counts[(menst, sleep)]
    dep_flags = np.array([True] * n_dep + [False] * (size - n_dep))
    rng.shuffle(dep_flags)
    for d in dep_flags:
        rows.append({"menstrual_status": menst, "sleep_disturbance": sleep, "depressed": d})

df = pd.DataFrame(rows)
df["race_ethnicity"] = race
df["age_years"] = age
df["age_at_menarche"] = np.round(menarche, 1)

# --- PHQ-9 total score (0-27): severity bands match Table 6 exactly ---
# Minimal 0-4: 758 | Mild 5-9: 202 | Moderate 10-14: 53
# Mod. severe 15-19: 29 | Severe 20-27: 4  (758+202+53+29+4 = 1046)
n_min, n_mild, n_mod, n_modsev, n_sev = 758, 202, 53, 29, 4
assert n_min + n_mild == (N - 86) and n_mod + n_modsev + n_sev == 86

non_dep_idx = df.index[~df["depressed"]].to_numpy().copy()
dep_idx = df.index[df["depressed"]].to_numpy().copy()
rng.shuffle(non_dep_idx)
rng.shuffle(dep_idx)

phq9 = np.empty(N, dtype=int)
min_idx, mild_idx = non_dep_idx[:n_min], non_dep_idx[n_min:n_min + n_mild]
mod_idx = dep_idx[:n_mod]
modsev_idx = dep_idx[n_mod:n_mod + n_modsev]
sev_idx = dep_idx[n_mod + n_modsev:]

phq9[min_idx] = rng.integers(0, 5, size=len(min_idx))
phq9[mild_idx] = rng.integers(5, 10, size=len(mild_idx))
phq9[mod_idx] = rng.integers(10, 15, size=len(mod_idx))
phq9[modsev_idx] = rng.integers(15, 20, size=len(modsev_idx))
phq9[sev_idx] = rng.integers(20, 28, size=len(sev_idx))

df["phq9_score"] = phq9
df["depressed"] = df["phq9_score"] >= 10  # recompute cleanly from the score

df = df[["race_ethnicity", "age_years", "age_at_menarche",
         "menstrual_status", "sleep_disturbance", "phq9_score", "depressed"]]
df.to_csv("data/nhanes_synthetic.csv", index=False)

print(df.describe(include="all"))
print("\nMenstrual x sleep crosstab:\n", pd.crosstab(df.menstrual_status, df.sleep_disturbance))
print("\nDepression prevalence by subgroup:\n",
      df.groupby(["menstrual_status", "sleep_disturbance"])["depressed"].mean().round(3))
print("\nPHQ-9 severity band counts:\n",
      pd.cut(df.phq9_score, [-1, 4, 9, 14, 19, 27],
             labels=["Minimal", "Mild", "Moderate", "Mod. Severe", "Severe"]).value_counts())
print(f"\nOverall clinical depression (>=10): {df.depressed.sum()} ({df.depressed.mean()*100:.1f}%)")
print(f"Saved {len(df)} rows to data/nhanes_synthetic.csv")

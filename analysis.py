"""
Reproduces the statistical analysis and figures from the MyCycleIQ /
"Beyond Cycle Tracking" capstone (NHANES adolescent menstrual health +
depression analysis), run against the synthetic dataset in
data/nhanes_synthetic.csv (see generate_dataset.py for provenance).

No external stats package (statsmodels) is available in this
environment, so logistic regression is implemented directly via
Newton-Raphson IRLS, which additionally makes the method transparent
rather than a black-box call.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams["figure.dpi"] = 110
df = pd.read_csv("data/nhanes_synthetic.csv")


# =====================================================================
# Manual logistic regression (IRLS / Newton-Raphson), statsmodels-style
# output: coefficients, SE, z, p-value, and odds ratios with 95% CI.
# =====================================================================
def logistic_regression(X, y, max_iter=50, tol=1e-8):
    X = np.column_stack([np.ones(len(X)), X])
    n, k = X.shape
    beta = np.zeros(k)
    for _ in range(max_iter):
        eta = X @ beta
        p = 1 / (1 + np.exp(-eta))
        W = p * (1 - p)
        W = np.clip(W, 1e-6, None)
        gradient = X.T @ (y - p)
        H = -(X.T * W) @ X
        step = np.linalg.solve(H, gradient)
        beta_new = beta - step
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    cov = np.linalg.inv(-H)
    se = np.sqrt(np.diag(cov))
    z = beta / se
    pvals = 2 * (1 - stats.norm.cdf(np.abs(z)))
    return beta, se, z, pvals


def report_model(name, X, y, colnames):
    beta, se, z, p = logistic_regression(X, y)
    print(f"\n=== {name} ===")
    print(f"{'term':28s} {'OR':>8s} {'95% CI':>18s} {'p-value':>10s}")
    for i, cname in enumerate(["(intercept)"] + colnames):
        if cname == "(intercept)":
            continue
        or_ = np.exp(beta[i])
        lo = np.exp(beta[i] - 1.96 * se[i])
        hi = np.exp(beta[i] + 1.96 * se[i])
        print(f"{cname:28s} {or_:8.2f} [{lo:6.2f}, {hi:6.2f}] {p[i]:10.3g}")
    return beta, se, p


# --- encode predictors -----------------------------------------------
df["irregular"] = (df.menstrual_status == "irregular").astype(int)
df["sleep_disturbed"] = (df.sleep_disturbance == "disturbed").astype(int)
y = df["depressed"].astype(int).to_numpy()

# Model 1: depression ~ menstrual irregularity (unadjusted)
X1 = df[["irregular"]].to_numpy(dtype=float)
report_model("Model 1: unadjusted (menstrual irregularity only)", X1, y, ["irregular"])

# Model 2: depression ~ irregularity + sleep disturbance + age
#          (+ race/ethnicity dummies, reference = Non-Hispanic White)
race_dummies = pd.get_dummies(df["race_ethnicity"], drop_first=False)
race_dummies = race_dummies.drop(columns=["Non-Hispanic White"])
X2_df = pd.concat([
    df[["irregular", "sleep_disturbed", "age_years"]].reset_index(drop=True),
    race_dummies.reset_index(drop=True).astype(float),
], axis=1)
X2 = X2_df.to_numpy(dtype=float)
report_model("Model 2: adjusted (+ sleep disturbance, age, race/ethnicity)",
             X2, y, list(X2_df.columns))


# =====================================================================
# Figures
# =====================================================================

# Figure 1: Age at menarche distribution
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(df["age_at_menarche"], bins=20, color="#4C72B0", edgecolor="white")
ax.axvline(df["age_at_menarche"].mean(), color="black", linestyle="--",
           label=f"mean = {df['age_at_menarche'].mean():.1f}")
ax.set_xlabel("Age at menarche (years)")
ax.set_ylabel("Count")
ax.set_title("Figure 1: Age at Menarche Distribution")
ax.legend()
fig.tight_layout()
fig.savefig("figures/fig1_age_at_menarche.png")
plt.close(fig)

# Figure 2: PHQ-9 severity by menstrual regularity
bins = [-1, 4, 9, 14, 19, 27]
labels = ["Minimal", "Mild", "Moderate", "Mod. Severe", "Severe"]
df["phq9_band"] = pd.cut(df["phq9_score"], bins=bins, labels=labels)
ct = pd.crosstab(df["phq9_band"], df["menstrual_status"], normalize="columns") * 100
fig, ax = plt.subplots(figsize=(7, 4.5))
ct.plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452"])
ax.set_ylabel("% of subgroup")
ax.set_xlabel("PHQ-9 severity category")
ax.set_title("Figure 2: PHQ-9 Severity by Menstrual Regularity")
ax.legend(title="Menstrual status")
fig.tight_layout()
fig.savefig("figures/fig2_phq9_by_regularity.png")
plt.close(fig)

# Figure 3: PHQ-9 score distribution, late-adolescent subgroup (15-19)
late = df[df.age_years >= 15]
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(late["phq9_score"], bins=range(0, 29), color="#55A868", edgecolor="white")
ax.set_xlabel("PHQ-9 total score")
ax.set_ylabel("Count")
ax.set_title("Figure 3: PHQ-9 Score Distribution (ages 15-19)")
fig.tight_layout()
fig.savefig("figures/fig3_phq9_late_adolescent.png")
plt.close(fig)

# Figure 4: Depression prevalence by menstrual status x sleep disturbance
prev = df.groupby(["menstrual_status", "sleep_disturbance"])["depressed"].mean() * 100
prev = prev.unstack()
fig, ax = plt.subplots(figsize=(6, 4))
prev.plot(kind="bar", ax=ax, color=["#4C72B0", "#C44E52"])
ax.set_ylabel("Depression prevalence (%)")
ax.set_xlabel("Menstrual status")
ax.set_title("Figure 4: Depression Prevalence by Menstrual Status & Sleep")
ax.legend(title="Sleep disturbance")
fig.tight_layout()
fig.savefig("figures/fig4_depression_prevalence.png")
plt.close(fig)

# Figure 5: App evaluation scores (Table 3 in the paper)
apps = pd.read_csv("data/app_evaluation_scores.csv")
fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(apps))
width = 0.25
ax.bar(x - width, apps["gamification"], width, label="Gamification", color="#4C72B0")
ax.bar(x, apps["mental_health"], width, label="Mental Health", color="#DD8452")
ax.bar(x + width, apps["privacy"], width, label="Privacy", color="#55A868")
ax.set_xticks(x)
ax.set_xticklabels(apps["application"], rotation=30, ha="right")
ax.set_ylabel("Score (out of 10)")
ax.set_title("Figure 6: Menstrual Health App Evaluation (30-point rubric)")
ax.legend()
fig.tight_layout()
fig.savefig("figures/fig5_app_evaluation_scores.png")
plt.close(fig)

print("\nFigures written to figures/")

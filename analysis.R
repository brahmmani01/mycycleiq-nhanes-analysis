# =====================================================================
# Beyond Cycle Tracking — NHANES adolescent menstrual health & depression
# R version of the original capstone analysis pipeline
# (R 4.3.2, tidyverse, gtsummary, ggplot2 — as used in the paper)
# =====================================================================
library(tidyverse)
library(gtsummary)

df <- read_csv("data/nhanes_synthetic.csv") %>%
  mutate(
    irregular        = as.integer(menstrual_status == "irregular"),
    sleep_disturbed  = as.integer(sleep_disturbance == "disturbed"),
    depressed        = as.integer(depressed),
    phq9_band        = cut(phq9_score,
                            breaks = c(-1, 4, 9, 14, 19, 27),
                            labels = c("Minimal", "Mild", "Moderate",
                                       "Mod. Severe", "Severe"))
  )

# ---- Table: sample characteristics (mirrors Table 5 in the paper) ---
df %>%
  select(race_ethnicity, menstrual_status, sleep_disturbance, age_at_menarche) %>%
  tbl_summary()

# ---- Figure 1: age at menarche ---------------------------------------
ggplot(df, aes(age_at_menarche)) +
  geom_histogram(bins = 20, fill = "#4C72B0", color = "white") +
  geom_vline(aes(xintercept = mean(age_at_menarche)), linetype = "dashed") +
  labs(title = "Age at Menarche Distribution", x = "Age at menarche (years)", y = "Count") +
  theme_minimal()
ggsave("figures/r_fig1_age_at_menarche.png", width = 6, height = 4)

# ---- Figure 2: PHQ-9 severity by menstrual regularity ----------------
df %>%
  count(menstrual_status, phq9_band) %>%
  group_by(menstrual_status) %>%
  mutate(pct = n / sum(n) * 100) %>%
  ggplot(aes(phq9_band, pct, fill = menstrual_status)) +
  geom_col(position = "dodge") +
  labs(title = "PHQ-9 Severity by Menstrual Regularity",
       x = "PHQ-9 severity category", y = "% of subgroup", fill = "Menstrual status") +
  theme_minimal()
ggsave("figures/r_fig2_phq9_by_regularity.png", width = 7, height = 4.5)

# ---- Model 1: unadjusted logistic regression -------------------------
model1 <- glm(depressed ~ irregular, data = df, family = binomial)
tbl_regression(model1, exponentiate = TRUE)

# ---- Model 2: adjusted for sleep disturbance, age, race/ethnicity ----
model2 <- glm(depressed ~ irregular + sleep_disturbed + age_years + race_ethnicity,
              data = df, family = binomial)
tbl_regression(model2, exponentiate = TRUE)

# ---- Figure 5: forest plot of adjusted odds ratios -------------------
broom::tidy(model2, exponentiate = TRUE, conf.int = TRUE) %>%
  filter(term != "(Intercept)") %>%
  ggplot(aes(x = estimate, y = term)) +
  geom_point() +
  geom_errorbarh(aes(xmin = conf.low, xmax = conf.high), height = 0.2) +
  geom_vline(xintercept = 1, linetype = "dashed") +
  scale_x_log10() +
  labs(title = "Adjusted Odds Ratios — Model 2", x = "Odds ratio (log scale)", y = NULL) +
  theme_minimal()
ggsave("figures/r_fig5_forest_plot.png", width = 7, height = 4.5)

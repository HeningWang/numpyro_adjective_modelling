# =============================================================================
#  Plotting script for main paper — Slider-rating study
#  Run from: 10-writing/
# =============================================================================

library(tidyverse)
library(aida)

# ── Theme ────────────────────────────────────────────────────────────────────
CSP_colors <- c(
  "#7581B3", "#99C2C2", "#C65353", "#E2BA78", "#5C7457", "#575463",
  "#B0B7D4", "#66A3A3", "#DB9494", "#D49735", "#9BB096", "#D4D3D9",
  "#414C76", "#993333"
)

scale_colour_discrete <- function(...) {
  scale_colour_manual(..., values = CSP_colors)
}
scale_fill_discrete <- function(...) {
  scale_fill_manual(..., values = CSP_colors)
}
theme_set(theme_aida())
theme_model <- function() {
  theme_aida() +
    theme(
      axis.text.y  = element_text(size = 14),
      axis.text.x  = element_text(size = 14),
      axis.title.y = element_text(size = 16),
      axis.title.x = element_text(size = 16),
      legend.text  = element_text(size = 14),
      legend.title = element_text(size = 14)
    )
}

# ── Output directory ─────────────────────────────────────────────────────────
fig_dir <- "figures"
dir.create(fig_dir, showWarnings = FALSE)

# ── Load data ────────────────────────────────────────────────────────────────
df_emp   <- read_csv("data/slider_empirical.csv")
df_pred  <- read_csv("data/slider_predictions.csv")
df_cond  <- read_csv("data/slider_condition_summary.csv")
df_loo   <- read_csv("data/slider_loo_comparison.csv") %>%
  rename(model = 1)

# ── Nicer labels ─────────────────────────────────────────────────────────────
rp_labels <- c(
  "both"   = "Both necessary",
  "first"  = "Size sufficient",
  "second" = "Colour sufficient"
)
sharp_labels <- c(
  "sharp"   = "High",
  "blurred" = "Low"
)

df_emp <- df_emp %>%
  mutate(
    relevant_property = factor(relevant_property, levels = names(rp_labels),
                               labels = rp_labels),
    sharpness = factor(sharpness, levels = names(sharp_labels),
                       labels = sharp_labels),
    # Center slider at 0 (0.5 = no preference)
    slider_centered = human_slider - 0.5
  )

# =============================================================================
#  FIGURE 1 — Empirical slider ratings
# =============================================================================
fig1 <- df_emp %>%
  group_by(relevant_property, sharpness) %>%
  summarise(
    mean_slider = mean(slider_centered),
    ci_lo = mean_slider - qt(0.975, n() - 1) * sd(slider_centered) / sqrt(n()),
    ci_hi = mean_slider + qt(0.975, n() - 1) * sd(slider_centered) / sqrt(n()),
    .groups = "drop"
  ) %>%
  ggplot(aes(x = relevant_property, y = mean_slider, fill = sharpness)) +
  geom_col(position = position_dodge(0.8), width = 0.7, alpha = 0.85) +
  geom_errorbar(
    aes(ymin = ci_lo, ymax = ci_hi),
    position = position_dodge(0.8), width = 0.2, linewidth = 0.6
  ) +
  geom_hline(yintercept = 0, linetype = "dashed", colour = "grey40") +
  scale_fill_manual(values = CSP_colors[1:2], name = "Size discriminability") +
  labs(
    x = "Referential context",
    y = "Mean slider rating (centered)"
  ) +
  theme_model() +
  theme(legend.position = "top")

ggsave(file.path(fig_dir, "slider_empirical.pdf"), fig1,
       width = 7, height = 4.5, dpi = 300)
cat("[✓] slider_empirical.pdf\n")


# =============================================================================
#  Slider lmer analysis: referential-context × size-discriminability interaction
# =============================================================================
library(lme4)
library(lmerTest)

df_lmer <- df_emp %>%
  mutate(
    rp = factor(relevant_property,
                levels = c("Size sufficient", "Both necessary", "Colour sufficient")),
    sh = factor(sharpness, levels = c("Low", "High"))
  )

m_full <- lmer(slider_centered ~ rp * sh + (1 | id) + (1 | item), data = df_lmer)
m_add  <- lmer(slider_centered ~ rp + sh + (1 | id) + (1 | item), data = df_lmer)
m_rel  <- lmer(slider_centered ~ rp      + (1 | id) + (1 | item), data = df_lmer)
m_null <- lmer(slider_centered ~ 1       + (1 | id) + (1 | item), data = df_lmer)

cat("\n=== ANOVA (Type III) on full model ===\n")
print(anova(m_full))

cat("\n=== Model comparison: interaction (additive vs full) ===\n")
print(anova(m_add, m_full))

cat("\n=== Model comparison: size-discriminability main effect ===\n")
print(anova(m_rel, m_add))

cat("\n=== Model comparison: referential-context main effect ===\n")
print(anova(m_null, m_rel))

# Save summary to text file
sink(file.path(fig_dir, "slider_lmer_summary.txt"))
cat("=== Full model summary ===\n")
print(summary(m_full))
cat("\n=== ANOVA (Type III) ===\n")
print(anova(m_full))
cat("\n=== Model comparison: interaction ===\n")
print(anova(m_add, m_full))
cat("\n=== Model comparison: size-discriminability main effect ===\n")
print(anova(m_rel, m_add))
cat("\n=== Model comparison: referential-context main effect ===\n")
print(anova(m_null, m_rel))
sink()
cat("[✓] slider_lmer_summary.txt\n")


# =============================================================================
#  FIGURE 2 — Correlation: inc_hier model predictions vs. empirical
# =============================================================================

# Condition-level data for the inc_hier model
df_corr <- df_cond %>%
  mutate(
    relevant_property = factor(relevant_property, levels = names(rp_labels),
                               labels = rp_labels),
    sharpness = factor(sharpness, levels = names(sharp_labels),
                       labels = sharp_labels),
    cond_label = paste(relevant_property, sharpness, sep = " / ")
  )

# Compute R²
r_sq <- cor(df_corr$emp_mean, df_corr$pred_mean_incremental_recursive)^2

fig2 <- df_corr %>%
  ggplot(aes(x = emp_mean, y = pred_mean_incremental_recursive)) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", colour = "grey50") +
  geom_errorbar(
    aes(ymin = pred_lo_incremental_recursive, ymax = pred_hi_incremental_recursive),
    width = 0.005, linewidth = 0.5, colour = "grey60"
  ) +
  geom_point(aes(colour = cond_label), size = 4) +
  scale_colour_manual(values = CSP_colors[1:6], name = "Condition") +
  annotate(
    "text", x = min(df_corr$emp_mean), y = max(df_corr$pred_hi_incremental_recursive),
    label = paste0("italic(R)^2 == ", formatC(r_sq, format = "f", digits = 3)),
    parse = TRUE, hjust = 0, vjust = 1, size = 5, colour = "grey30"
  ) +
  labs(
    x = "Empirical mean slider rating",
    y = "Predicted mean (inc. context-updating, 95% CI)"
  ) +
  coord_fixed() +
  theme_model() +
  theme(legend.position = "right")

ggsave(file.path(fig_dir, "slider_correlation_inc_hier.pdf"), fig2,
       width = 7, height = 5.5, dpi = 300)
cat("[✓] slider_correlation_inc_hier.pdf\n")


# =============================================================================
#  FIGURE 3 — Model comparison (LOO ELPD) — 2×2 speaker × semantics
# =============================================================================

model_labels <- c(
  "incremental_recursive" = "Incremental, context-updating",
  "incremental_static"    = "Incremental, context-fixed",
  "global_recursive"      = "Global, context-updating",
  "global_static"         = "Global, context-fixed"
)

# Best model in red, others in blue
df_loo_plot <- df_loo %>%
  mutate(
    model_label = factor(model_labels[model],
                         levels = rev(model_labels[order(match(names(model_labels), model))])),
    is_best = rank == 0
  )

fig3 <- df_loo_plot %>%
  ggplot(aes(x = -elpd_diff, y = model_label, colour = is_best)) +
  geom_vline(xintercept = 0, linetype = "dashed", colour = "grey60") +
  geom_pointrange(
    aes(xmin = -elpd_diff - dse, xmax = -elpd_diff + dse),
    size = 0.8, linewidth = 0.9
  ) +
  scale_colour_manual(
    values = c("TRUE" = CSP_colors[3], "FALSE" = CSP_colors[1]),
    guide = "none"
  ) +
  scale_x_continuous(expand = expansion(mult = c(0.02, 0.05))) +
  labs(
    x = expression(Delta * "ELPD (LOO) relative to best model"),
    y = NULL
  ) +
  theme_model() +
  theme(axis.text.y = element_text(size = 13))

ggsave(file.path(fig_dir, "slider_model_comparison.pdf"), fig3,
       width = 7.5, height = 3.5, dpi = 300)
cat("[✓] slider_model_comparison.pdf\n")

cat("\n[Done] All slider figures saved to", fig_dir, "\n")


# =============================================================================
# =============================================================================
#  PRODUCTION STUDY FIGURES
# =============================================================================
# =============================================================================

# ── Load production data ─────────────────────────────────────────────────────
df_prod_emp  <- read_csv("data/production_empirical.csv")
df_prod_pred <- read_csv("data/production_predictions.csv")
df_prod_corr <- read_csv("data/production_correlation.csv")
df_prod_loo  <- read_csv("data/production_loo_comparison.csv") %>%
  rename(model = 1)

# ── Remap utterance labels: D → S ───────────────────────────────────────────
rename_D_to_S <- function(x) gsub("D", "S", x)
df_prod_emp  <- df_prod_emp  %>% mutate(utterance_label = rename_D_to_S(utterance_label))
df_prod_corr <- df_prod_corr %>% mutate(utterance_label = rename_D_to_S(utterance_label))

# ── Nicer condition labels ───────────────────────────────────────────────────
df_prod_emp <- df_prod_emp %>%
  mutate(
    relevant_property = factor(relevant_property, levels = names(rp_labels),
                               labels = rp_labels),
    sharpness = factor(sharpness, levels = names(sharp_labels),
                       labels = sharp_labels),
    condition = paste(relevant_property, sharpness, sep = " / ")
  )

df_prod_corr <- df_prod_corr %>%
  mutate(
    relevant_property = factor(relevant_property, levels = names(rp_labels),
                               labels = rp_labels),
    sharpness = factor(sharpness, levels = names(sharp_labels),
                       labels = sharp_labels),
    condition = paste(relevant_property, sharpness, sep = " / ")
  )

# =============================================================================
#  FIGURE 4 — Production empirical: utterance-type distributions by condition
# =============================================================================

# Keep only the top utterance types (>= 2% in at least one condition)
top_utts <- df_prod_emp %>%
  group_by(utterance_label) %>%
  summarise(max_prop = max(human_mean), .groups = "drop") %>%
  filter(max_prop >= 0.02) %>%
  pull(utterance_label)

df_prod_emp_top <- df_prod_emp %>%
  filter(utterance_label %in% top_utts) %>%
  mutate(utterance_label = factor(utterance_label, levels = rev(top_utts[order(
    match(top_utts, c("S", "SC", "SCF", "SF", "SFC", "C", "CS", "CSF",
                       "CF", "CFS", "F", "FS", "FSC", "FC", "FCS")))])))

fig4 <- df_prod_emp_top %>%
  ggplot(aes(x = human_mean, y = utterance_label, fill = sharpness)) +
  geom_col(position = position_dodge(0.7), width = 0.6, alpha = 0.85) +
  geom_errorbarh(
    aes(xmin = human_lo, xmax = human_hi),
    position = position_dodge(0.7), height = 0.25, linewidth = 0.5
  ) +
  facet_wrap(~ relevant_property, ncol = 3) +
  scale_fill_manual(values = CSP_colors[1:2], name = "Size discriminability") +
  labs(
    x = "Proportion",
    y = "Utterance type"
  ) +
  theme_model() +
  theme(
    legend.position   = "top",
    strip.text        = element_text(size = 13),
    panel.spacing     = unit(1, "lines")
  )

ggsave(file.path(fig_dir, "production_empirical.pdf"), fig4,
       width = 12, height = 5.5, dpi = 300)
cat("[✓] production_empirical.pdf\n")


# =============================================================================
#  FIGURE 4b — PPC barplot: empirical vs model predictions side by side
# =============================================================================

# Prepare model predictions with nicer labels
df_prod_pred_nice <- df_prod_pred %>%
  mutate(utterance_label = rename_D_to_S(utterance_label)) %>%
  mutate(
    relevant_property = factor(relevant_property, levels = names(rp_labels),
                               labels = rp_labels),
    sharpness = factor(sharpness, levels = names(sharp_labels),
                       labels = sharp_labels)
  )

# Select the best model (incremental_recursive) for the PPC comparison
df_ppc_model <- df_prod_pred_nice %>%
  filter(model == "incremental_recursive") %>%
  select(relevant_property, sharpness, utterance_label,
         mean = model_mean, lo = model_lo, hi = model_hi) %>%
  mutate(source = "Incremental, context-updating")

df_ppc_human <- df_prod_emp %>%
  select(relevant_property, sharpness, utterance_label,
         mean = human_mean, lo = human_lo, hi = human_hi) %>%
  mutate(source = "Empirical")

df_ppc <- bind_rows(df_ppc_human, df_ppc_model)

# Filter to utterance types >= 5% in at least one cell
top_utts_ppc <- df_ppc %>%
  group_by(utterance_label) %>%
  summarise(max_prop = max(mean), .groups = "drop") %>%
  filter(max_prop >= 0.05) %>%
  pull(utterance_label)

# Canonical ordering (S-initial, C-initial, F-initial)
utt_order <- c(
  "S", "SC", "SCF", "SF", "SFC",
  "C", "CS", "CSF", "CF", "CFS",
  "F", "FS", "FSC", "FC", "FCS"
)

# Facet label for sharpness rows
sharp_facet_labels <- c(
  "High" = "Size discrim.: High",
  "Low"  = "Size discrim.: Low"
)

df_ppc_top <- df_ppc %>%
  filter(utterance_label %in% top_utts_ppc) %>%
  mutate(
    utterance_label = factor(
      utterance_label,
      levels = rev(intersect(utt_order, top_utts_ppc))
    ),
    source = factor(
      source,
      levels = c("Empirical",
                  "Incremental, context-updating")
    )
  )

fig4b <- df_ppc_top %>%
  ggplot(aes(x = mean, y = utterance_label,
             fill = source)) +
  geom_col(
    position = position_dodge(0.7),
    width = 0.6, alpha = 0.85
  ) +
  geom_errorbarh(
    aes(xmin = lo, xmax = hi),
    position = position_dodge(0.7),
    height = 0.25, linewidth = 0.5
  ) +
  facet_grid(
    sharpness ~ relevant_property,
    labeller = labeller(
      sharpness = sharp_facet_labels
    )
  ) +
  scale_fill_manual(
    values = CSP_colors[c(1, 3)], name = NULL
  ) +
  scale_x_continuous(
    labels = scales::percent_format(accuracy = 1)
  ) +
  labs(x = "Proportion", y = "Utterance type") +
  theme_model() +
  theme(
    legend.position = "top",
    strip.text = element_text(size = 13),
    panel.spacing = unit(1, "lines")
  )

ggsave(file.path(fig_dir, "production_ppc_barplot.pdf"), fig4b,
       width = 12, height = 7, dpi = 300)
cat("[✓] production_ppc_barplot.pdf\n")


# =============================================================================
#  FIGURE 5 — Correlation: inc_hier model vs. empirical (production)
# =============================================================================

# Filter out zero-proportion points for cleaner plot
df_prod_corr_nz <- df_prod_corr %>%
  filter(human_mean > 0 | model_mean > 0)

r_sq_prod <- cor(df_prod_corr_nz$human_mean, df_prod_corr_nz$model_mean)^2

fig5 <- df_prod_corr_nz %>%
  ggplot(aes(x = human_mean, y = model_mean)) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", colour = "grey50") +
  geom_errorbar(
    aes(ymin = model_lo, ymax = model_hi),
    width = 0.005, linewidth = 0.4, colour = "grey60"
  ) +
  geom_errorbarh(
    aes(xmin = human_lo, xmax = human_hi),
    height = 0.005, linewidth = 0.4, colour = "grey60"
  ) +
  geom_point(aes(colour = condition), size = 2.5, alpha = 0.8) +
  scale_colour_manual(values = CSP_colors[1:6], name = "Condition") +
  annotate(
    "text", x = 0.02, y = max(df_prod_corr_nz$model_hi, na.rm = TRUE),
    label = paste0("italic(R)^2 == ", formatC(r_sq_prod, format = "f", digits = 3)),
    parse = TRUE, hjust = 0, vjust = 1, size = 5, colour = "grey30"
  ) +
  labs(
    x = "Empirical proportion",
    y = "Predicted proportion (inc. context-updating, 95% CI)"
  ) +
  coord_fixed() +
  theme_model() +
  theme(legend.position = "right")

ggsave(file.path(fig_dir, "production_correlation_inc_hier.pdf"), fig5,
       width = 7, height = 5.5, dpi = 300)
cat("[✓] production_correlation_inc_hier.pdf\n")


# =============================================================================
#  FIGURE 6 — Model comparison (LOO ELPD) — Production, 2×2 speaker × semantics
# =============================================================================

df_prod_loo_plot <- df_prod_loo %>%
  mutate(
    model_label = factor(model_labels[model],
                         levels = rev(model_labels[order(match(names(model_labels), model))])),
    is_best = rank == 0
  )

fig6 <- df_prod_loo_plot %>%
  ggplot(aes(x = -elpd_diff, y = model_label, colour = is_best)) +
  geom_vline(xintercept = 0, linetype = "dashed", colour = "grey60") +
  geom_pointrange(
    aes(xmin = -elpd_diff - dse, xmax = -elpd_diff + dse),
    size = 0.8, linewidth = 0.9
  ) +
  scale_colour_manual(
    values = c("TRUE" = CSP_colors[3], "FALSE" = CSP_colors[1]),
    guide = "none"
  ) +
  scale_x_continuous(expand = expansion(mult = c(0.02, 0.05))) +
  labs(
    x = expression(Delta * "ELPD (LOO) relative to best model"),
    y = NULL
  ) +
  theme_model() +
  theme(axis.text.y = element_text(size = 13))

ggsave(file.path(fig_dir, "production_model_comparison.pdf"), fig6,
       width = 7.5, height = 3.5, dpi = 300)
cat("[✓] production_model_comparison.pdf\n")


# =============================================================================
#  Production glmer analyses
# =============================================================================

# Load trial-level production data
df_prod_raw <- read_csv("../01-dataset/01-production-data-preprocessed.csv",
                        show_col_types = FALSE) %>%
  filter(conditions %in% c("zrdc", "erdc", "brdc")) %>%
  filter(annotation != "") %>%
  mutate(
    relevant_property = factor(relevant_property,
                               levels = c("first", "both", "second")),
    sharpness = factor(sharpness, levels = c("blurred", "sharp"),
                       labels = c("Low", "High"))
  )

cat("\nProduction trial-level N:", nrow(df_prod_raw), "\n")

# --- Analysis 1: Size-first advantage (D-initial = 1) ---
df_prod_raw <- df_prod_raw %>%
  mutate(
    size_first = as.integer(grepl("^D", annotation))
  )

cat("\nSize-first coding:\n")
print(table(df_prod_raw$annotation, df_prod_raw$size_first))

m_sf_full <- glmer(size_first ~ relevant_property * sharpness + (1 | id) + (1 | item),
                   data = df_prod_raw, family = binomial)
m_sf_add  <- glmer(size_first ~ relevant_property + sharpness + (1 | id) + (1 | item),
                   data = df_prod_raw, family = binomial)
m_sf_rel  <- glmer(size_first ~ relevant_property + (1 | id) + (1 | item),
                   data = df_prod_raw, family = binomial)
m_sf_null <- glmer(size_first ~ 1 + (1 | id) + (1 | item),
                   data = df_prod_raw, family = binomial)

cat("\n=== Size-first: Full model summary ===\n")
print(summary(m_sf_full))
cat("\n=== Size-first: interaction (additive vs full) ===\n")
print(anova(m_sf_add, m_sf_full))
cat("\n=== Size-first: size-discriminability main effect ===\n")
print(anova(m_sf_rel, m_sf_add))
cat("\n=== Size-first: referential-context main effect ===\n")
print(anova(m_sf_null, m_sf_rel))

# --- Analysis 2: Over-informative utterance ---
# Coding:
#   both-necessary:    3-adjective = over-informative (DCF, DFC, CDF, CFD, FDC, FCD)
#   size-sufficient:   multi-word besides D = over-informative
#   colour-sufficient: multi-word besides C = over-informative
df_prod_raw <- df_prod_raw %>%
  mutate(
    n_adj = nchar(annotation),
    over_informative = case_when(
      relevant_property == "both"   & n_adj == 3 ~ 1L,
      relevant_property == "first"  & n_adj > 1  ~ 1L,
      relevant_property == "second" & n_adj > 1  ~ 1L,
      TRUE ~ 0L
    )
  )

cat("\nOver-informative coding by condition:\n")
print(df_prod_raw %>%
  group_by(relevant_property) %>%
  summarise(n = n(), n_oi = sum(over_informative),
            prop_oi = mean(over_informative), .groups = "drop"))

m_oi_full <- glmer(over_informative ~ relevant_property * sharpness + (1 | id) + (1 | item),
                   data = df_prod_raw, family = binomial)
m_oi_add  <- glmer(over_informative ~ relevant_property + sharpness + (1 | id) + (1 | item),
                   data = df_prod_raw, family = binomial)
m_oi_rel  <- glmer(over_informative ~ relevant_property + (1 | id) + (1 | item),
                   data = df_prod_raw, family = binomial)
m_oi_null <- glmer(over_informative ~ 1 + (1 | id) + (1 | item),
                   data = df_prod_raw, family = binomial)

cat("\n=== Over-informative: Full model summary ===\n")
print(summary(m_oi_full))
cat("\n=== Over-informative: interaction (additive vs full) ===\n")
print(anova(m_oi_add, m_oi_full))
cat("\n=== Over-informative: size-discriminability main effect ===\n")
print(anova(m_oi_rel, m_oi_add))
cat("\n=== Over-informative: referential-context main effect ===\n")
print(anova(m_oi_null, m_oi_rel))

# Save to text file
sink(file.path(fig_dir, "production_glmer_summary.txt"))
cat("=== SIZE-FIRST ADVANTAGE ===\n\n")
cat("--- Full model summary ---\n")
print(summary(m_sf_full))
cat("\n--- Interaction (additive vs full) ---\n")
print(anova(m_sf_add, m_sf_full))
cat("\n--- Size-discriminability main effect ---\n")
print(anova(m_sf_rel, m_sf_add))
cat("\n--- Referential-context main effect ---\n")
print(anova(m_sf_null, m_sf_rel))
cat("\n\n=== OVER-INFORMATIVE UTTERANCES ===\n\n")
cat("--- Full model summary ---\n")
print(summary(m_oi_full))
cat("\n--- Interaction (additive vs full) ---\n")
print(anova(m_oi_add, m_oi_full))
cat("\n--- Size-discriminability main effect ---\n")
print(anova(m_oi_rel, m_oi_add))
cat("\n--- Referential-context main effect ---\n")
print(anova(m_oi_null, m_oi_rel))
sink()
cat("[✓] production_glmer_summary.txt\n")

cat("\n[Done] All production figures saved to", fig_dir, "\n")


# =============================================================================
# =============================================================================
#  SIMULATION FIGURES
# =============================================================================
# =============================================================================

# ── Load simulation data ─────────────────────────────────────────────────────
df_sim <- read_csv("../04-simulation-w-randomstates/simulation_full_run.csv",
                   show_col_types = FALSE)

df_sim <- df_sim %>%
  mutate(
    speaker_label = recode(speaker,
      "incremental_speaker" = "Incremental",
      "global_speaker"      = "Global"
    ),
    semantics_label = recode(semantics,
      "recursive" = "Recursive",
      "static"    = "Static"
    )
  )

cat("Simulation rows:", nrow(df_sim), "\n")

# =============================================================================
#  FIGURE 7 — Effect of wf (perceptual blur), faceted by speaker
# =============================================================================

df_sim_wf <- df_sim %>%
  group_by(speaker_label, wf) %>%
  summarise(
    mean_bb = mean(probs_big_blue),
    se_bb   = sd(probs_big_blue) / sqrt(n()),
    mean_bB = mean(probs_blue_big),
    se_bB   = sd(probs_blue_big) / sqrt(n()),
    .groups = "drop"
  ) %>%
  pivot_longer(
    cols      = c(mean_bb, mean_bB, se_bb, se_bB),
    names_to  = c(".value", "utterance"),
    names_pattern = "(mean|se)_(.*)"
  ) %>%
  mutate(utterance = recode(utterance, bb = "big blue", bB = "blue big"))

fig7 <- df_sim_wf %>%
  ggplot(aes(x = wf, y = mean, colour = utterance)) +
  geom_point(size = 3) +
  geom_line(linewidth = 0.8) +
  geom_ribbon(aes(ymin = mean - 2 * se, ymax = mean + 2 * se,
                  fill = utterance), alpha = 0.15, colour = NA) +
  facet_wrap(~ speaker_label) +
  scale_colour_manual(values = CSP_colors[1:2], name = "Utterance order") +
  scale_fill_manual(values = CSP_colors[1:2], guide = "none") +
  labs(
    x = expression(italic(w)[f] ~ "(perceptual blur)"),
    y = "P(referent | utterance)"
  ) +
  theme_model() +
  theme(
    legend.position = "top",
    strip.text      = element_text(size = 14)
  )

ggsave(file.path(fig_dir, "sim_wf.pdf"), fig7,
       width = 8, height = 4.5, dpi = 300)
cat("[✓] sim_wf.pdf\n")


# =============================================================================
#  FIGURE 8 — Effect of k (threshold), faceted by speaker
# =============================================================================

df_sim_k <- df_sim %>%
  group_by(speaker_label, k) %>%
  summarise(
    mean_bb = mean(probs_big_blue),
    se_bb   = sd(probs_big_blue) / sqrt(n()),
    mean_bB = mean(probs_blue_big),
    se_bB   = sd(probs_blue_big) / sqrt(n()),
    .groups = "drop"
  ) %>%
  pivot_longer(
    cols      = c(mean_bb, mean_bB, se_bb, se_bB),
    names_to  = c(".value", "utterance"),
    names_pattern = "(mean|se)_(.*)"
  ) %>%
  mutate(utterance = recode(utterance, bb = "big blue", bB = "blue big"))

fig8 <- df_sim_k %>%
  ggplot(aes(x = k, y = mean, colour = utterance)) +
  geom_point(size = 3) +
  geom_line(linewidth = 0.8) +
  geom_ribbon(aes(ymin = mean - 2 * se, ymax = mean + 2 * se,
                  fill = utterance), alpha = 0.15, colour = NA) +
  facet_wrap(~ speaker_label) +
  scale_colour_manual(values = CSP_colors[1:2], name = "Utterance order") +
  scale_fill_manual(values = CSP_colors[1:2], guide = "none") +
  labs(
    x = expression(italic(k) ~ "(threshold)"),
    y = "P(referent | utterance)"
  ) +
  theme_model() +
  theme(
    legend.position = "top",
    strip.text      = element_text(size = 14)
  )

ggsave(file.path(fig_dir, "sim_k.pdf"), fig8,
       width = 8, height = 4.5, dpi = 300)
cat("[✓] sim_k.pdf\n")


# =============================================================================
#  FIGURE 9 — Effect of color_semvalue (nu), faceted by speaker
# =============================================================================

df_sim_csv <- df_sim %>%
  group_by(speaker_label, color_semvalue) %>%
  summarise(
    mean_bb = mean(probs_big_blue),
    se_bb   = sd(probs_big_blue) / sqrt(n()),
    mean_bB = mean(probs_blue_big),
    se_bB   = sd(probs_blue_big) / sqrt(n()),
    .groups = "drop"
  ) %>%
  pivot_longer(
    cols      = c(mean_bb, mean_bB, se_bb, se_bB),
    names_to  = c(".value", "utterance"),
    names_pattern = "(mean|se)_(.*)"
  ) %>%
  mutate(utterance = recode(utterance, bb = "big blue", bB = "blue big"))

fig9 <- df_sim_csv %>%
  ggplot(aes(x = color_semvalue, y = mean, colour = utterance)) +
  geom_point(size = 3) +
  geom_line(linewidth = 0.8) +
  geom_ribbon(aes(ymin = mean - 2 * se, ymax = mean + 2 * se,
                  fill = utterance), alpha = 0.15, colour = NA) +
  facet_wrap(~ speaker_label) +
  scale_colour_manual(values = CSP_colors[1:2], name = "Utterance order") +
  scale_fill_manual(values = CSP_colors[1:2], guide = "none") +
  labs(
    x = expression(nu ~ "(colour semantic value)"),
    y = "P(referent | utterance)"
  ) +
  theme_model() +
  theme(
    legend.position = "top",
    strip.text      = element_text(size = 14)
  )

ggsave(file.path(fig_dir, "sim_colorsemval.pdf"), fig9,
       width = 8, height = 4.5, dpi = 300)
cat("[✓] sim_colorsemval.pdf\n")

# =============================================================================
#  FIGURE 10 — Size-first advantage by nobj × size discriminability, faceted by speaker
# =============================================================================

df_sim <- df_sim %>%
  mutate(
    size_first_advantage = probs_big_blue - probs_blue_big
  )

# Use simple factor labels; expressions are set in scale_colour_manual
df_sim <- df_sim %>%
  mutate(
    context_label = factor(
      sd_spread,
      levels = c(2.0, 7.75, 15.0),
      labels = c("Blurred", "Baseline", "Sharp")
    )
  )

df_sim_nobj <- df_sim %>%
  group_by(speaker_label, semantics_label, nobj, context_label) %>%
  summarise(
    mean_adv = mean(size_first_advantage),
    se_adv   = sd(size_first_advantage) / sqrt(n()),
    .groups  = "drop"
  )

fig10 <- df_sim_nobj %>%
  ggplot(aes(x = nobj, y = mean_adv, colour = context_label, fill = context_label)) +
  geom_point(size = 2.5) +
  geom_line(linewidth = 0.8) +
  geom_ribbon(aes(ymin = mean_adv - 2 * se_adv,
                  ymax = mean_adv + 2 * se_adv),
              alpha = 0.15, colour = NA) +
  geom_hline(yintercept = 0, linetype = "dashed", colour = "grey30") +
  facet_grid(semantics_label ~ speaker_label) +
  scale_colour_manual(
    values = CSP_colors[c(3, 2, 1)],
    name = "Context",
    labels = c(
      "Blurred"  = expression(paste("Blurred (", sigma[spread], " = 2)")),
      "Baseline" = expression(paste("Baseline (", sigma[spread], " = 7.75)")),
      "Sharp"    = expression(paste("Sharp (", sigma[spread], " = 15)"))
    )
  ) +
  scale_fill_manual(values = CSP_colors[c(3, 2, 1)], guide = "none") +
  scale_x_continuous(breaks = unique(df_sim$nobj)) +
  labs(
    x = "N objects",
    y = "Advantage (big blue \u2212 blue big)"
  ) +
  theme_model() +
  theme(
    legend.position = "right",
    strip.text      = element_text(size = 14)
  )

ggsave(file.path(fig_dir, "sim_advantage_nobj.pdf"), fig10,
       width = 9, height = 7, dpi = 300)
cat("[\u2713] sim_advantage_nobj.pdf\n")

cat("\n[Done] All simulation figures saved to", fig_dir, "\n")

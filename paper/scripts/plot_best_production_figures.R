# =============================================================================
#  CSP-styled figures for the advocated principled production model
#  Produces (matching plotting_main_paper.R style exactly):
#    figures/production_ppc_barplot_best.pdf
#    figures/production_correlation_best.pdf
#  Run from: paper/
#      Rscript scripts/plot_best_production_figures.R
# =============================================================================

library(tidyverse)
library(aida)

# ── Theme (identical to plotting_main_paper.R) ───────────────────────────────
CSP_colors <- c(
  "#7581B3", "#99C2C2", "#C65353", "#E2BA78", "#5C7457", "#575463",
  "#B0B7D4", "#66A3A3", "#DB9494", "#D49735", "#9BB096", "#D4D3D9",
  "#414C76", "#993333"
)

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

fig_dir <- "figures"

rp_labels <- c(
  "both"   = "Both necessary",
  "first"  = "Size sufficient",
  "second" = "Colour sufficient"
)
sharp_labels <- c(
  "sharp"   = "High",
  "blurred" = "Low"
)
rename_D_to_S <- function(x) gsub("D", "S", x)
best_model_id <- "principled_salience_stop_regularized_responsepolicy_boundedform_sizesharp_2x2_inc_static_fixedeps"
model_display <- "Incremental, context-fixed"

# ── Load best-model data ─────────────────────────────────────────────────────
df_prod_emp  <- read_csv("data/production_empirical.csv")
df_prod_pred <- read_csv("data/production_predictions_best.csv")
df_prod_corr <- read_csv("data/production_correlation_best.csv")

df_prod_emp <- df_prod_emp %>%
  mutate(utterance_label = rename_D_to_S(utterance_label))
df_prod_corr <- df_prod_corr %>%
  mutate(utterance_label = rename_D_to_S(utterance_label))

# =============================================================================
#  PPC barplot — empirical vs advocated principled model, by condition
# =============================================================================
df_prod_pred_nice <- df_prod_pred %>%
  filter(model == best_model_id) %>%
  mutate(utterance_label = rename_D_to_S(utterance_label)) %>%
  mutate(
    relevant_property = factor(relevant_property, levels = names(rp_labels),
                               labels = rp_labels),
    sharpness = factor(sharpness, levels = names(sharp_labels),
                       labels = sharp_labels)
  )

df_prod_emp_nice <- df_prod_emp %>%
  mutate(
    relevant_property = factor(relevant_property, levels = names(rp_labels),
                               labels = rp_labels),
    sharpness = factor(sharpness, levels = names(sharp_labels),
                       labels = sharp_labels)
  )

df_ppc_model <- df_prod_pred_nice %>%
  select(relevant_property, sharpness, utterance_label,
         mean = model_mean, lo = model_lo, hi = model_hi) %>%
  mutate(source = model_display)

df_ppc_human <- df_prod_emp_nice %>%
  select(relevant_property, sharpness, utterance_label,
         mean = human_mean, lo = human_lo, hi = human_hi) %>%
  mutate(source = "Empirical")

df_ppc <- bind_rows(df_ppc_human, df_ppc_model)

top_utts_ppc <- df_ppc %>%
  group_by(utterance_label) %>%
  summarise(max_prop = max(mean), .groups = "drop") %>%
  filter(max_prop >= 0.05) %>%
  pull(utterance_label)

utt_order <- c(
  "S", "SC", "SCF", "SF", "SFC",
  "C", "CS", "CSF", "CF", "CFS",
  "F", "FS", "FSC", "FC", "FCS"
)
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
    source = factor(source, levels = c("Empirical", model_display))
  )

fig_ppc <- df_ppc_top %>%
  ggplot(aes(x = mean, y = utterance_label, fill = source)) +
  geom_col(position = position_dodge(0.7), width = 0.6, alpha = 0.85) +
  geom_errorbarh(
    aes(xmin = lo, xmax = hi),
    position = position_dodge(0.7), height = 0.25, linewidth = 0.5
  ) +
  facet_grid(
    sharpness ~ relevant_property,
    labeller = labeller(sharpness = sharp_facet_labels)
  ) +
  scale_fill_manual(values = CSP_colors[c(1, 3)], name = NULL) +
  scale_x_continuous(labels = scales::percent_format(accuracy = 1)) +
  labs(x = "Proportion", y = "Utterance type") +
  theme_model() +
  theme(
    legend.position = "top",
    strip.text = element_text(size = 13),
    panel.spacing = unit(1, "lines")
  )

ggsave(file.path(fig_dir, "production_ppc_barplot_best.pdf"), fig_ppc,
       width = 12, height = 7, dpi = 300)
cat("[✓] production_ppc_barplot_best.pdf\n")

# =============================================================================
#  Correlation — advocated principled model vs. empirical
# =============================================================================
df_prod_corr <- df_prod_corr %>%
  { if ("model" %in% names(.)) filter(., model == best_model_id) else . } %>%
  mutate(
    relevant_property = factor(relevant_property, levels = names(rp_labels),
                               labels = rp_labels),
    sharpness = factor(sharpness, levels = names(sharp_labels),
                       labels = sharp_labels),
    condition = paste(relevant_property, sharpness, sep = " / ")
  )

df_prod_corr_nz <- df_prod_corr %>%
  filter(human_mean > 0 | model_mean > 0)

r_sq_prod <- cor(df_prod_corr_nz$human_mean, df_prod_corr_nz$model_mean)^2

fig_corr <- df_prod_corr_nz %>%
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
    label = paste0("italic(R)^2 == ",
                   formatC(r_sq_prod, format = "f", digits = 3)),
    parse = TRUE, hjust = 0, vjust = 1, size = 5, colour = "grey30"
  ) +
  labs(
    x = "Empirical proportion",
    y = "Predicted proportion (incremental, context-fixed; 95% CI)"
  ) +
  coord_fixed() +
  theme_model() +
  theme(legend.position = "right")

ggsave(file.path(fig_dir, "production_correlation_best.pdf"), fig_corr,
       width = 7, height = 5.5, dpi = 300)
cat("[✓] production_correlation_best.pdf\n")

# =============================================================================
#  ELPD comparison — principled parameter-matched 2x2
# =============================================================================
model_labels <- c(
  "principled_salience_stop_regularized_responsepolicy_boundedform_sizesharp_2x2_inc_static_fixedeps"  = "Incremental, context-fixed",
  "principled_salience_stop_regularized_responsepolicy_boundedform_sizesharp_2x2_inc_rec_fixedeps"     = "Incremental, context-updating",
  "principled_salience_stop_regularized_responsepolicy_boundedform_sizesharp_2x2_glob_static_fixedeps" = "Global, context-fixed",
  "principled_salience_stop_regularized_responsepolicy_boundedform_sizesharp_2x2_glob_rec_fixedeps"    = "Global, context-updating"
)

df_prod_loo <- read_csv("data/production_loo_comparison_best.csv") %>%
  rename(model = 1)

df_prod_loo_plot <- df_prod_loo %>%
  mutate(
    model_label = factor(
      model_labels[model],
      levels = rev(model_labels[order(match(names(model_labels), model))])
    ),
    is_best = rank == 0
  )

fig_loo <- df_prod_loo_plot %>%
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

ggsave(file.path(fig_dir, "production_model_comparison_best.pdf"), fig_loo,
       width = 7.5, height = 3.5, dpi = 300)
cat("[✓] production_model_comparison_best.pdf\n")

# =============================================================================
#  Slider heldout comparison and PPC
# =============================================================================
slider_model_labels <- c(
  "planned_usefulness_signed_order_static" = "Planned usefulness, context-fixed",
  "planned_usefulness_order_static" = "Planned usefulness, context-fixed (unconstrained)",
  "planned_usefulness_order" = "Planned usefulness, context-updating",
  "incremental_recursive" = "Greedy, context-updating",
  "incremental_static" = "Greedy, context-fixed"
)

df_slider_heldout <- read_csv("data/slider_heldout_elpd_model_summary.csv") %>%
  filter(model %in% names(slider_model_labels)) %>%
  mutate(
    delta_elpd = total_heldout_elpd -
      total_heldout_elpd[model == "planned_usefulness_signed_order_static"],
    model_label = factor(
      slider_model_labels[model],
      levels = rev(slider_model_labels[names(slider_model_labels) %in% model])
    ),
    diagnostics_ok = diagnostics_ok %in% TRUE
  )

fig_slider_heldout <- df_slider_heldout %>%
  ggplot(aes(x = delta_elpd, y = model_label, colour = diagnostics_ok)) +
  geom_vline(xintercept = 0, linetype = "dashed", colour = "grey60") +
  geom_point(size = 2.8) +
  scale_colour_manual(
    values = c("TRUE" = CSP_colors[3], "FALSE" = CSP_colors[1]),
    labels = c("TRUE" = "Diagnostics pass/warn", "FALSE" = "Diagnostics fail"),
    name = NULL
  ) +
  labs(
    x = expression(Delta * " heldout ELPD relative to selected model"),
    y = NULL
  ) +
  theme_model() +
  theme(
    axis.text.y = element_text(size = 13),
    legend.position = "top"
  )

ggsave(file.path(fig_dir, "slider_model_comparison.pdf"), fig_slider_heldout,
       width = 7.5, height = 3.5, dpi = 300)
cat("[✓] slider_model_comparison.pdf\n")

df_sl_emp <- read_csv("data/slider_empirical.csv") %>%
  group_by(relevant_property, sharpness) %>%
  summarise(
    mean = mean(human_slider),
    lo = mean - qt(0.975, n() - 1) * sd(human_slider) / sqrt(n()),
    hi = mean + qt(0.975, n() - 1) * sd(human_slider) / sqrt(n()),
    .groups = "drop"
  ) %>%
  mutate(source = "Empirical")

df_sl_pred <- read_csv("data/slider_condition_summary.csv") %>%
  transmute(
    relevant_property, sharpness,
    mean = pred_mean_planned_usefulness_signed_order_static,
    lo = pred_lo_planned_usefulness_signed_order_static,
    hi = pred_hi_planned_usefulness_signed_order_static,
    source = "Planned usefulness, context-fixed"
  )

df_sl_ppc <- bind_rows(df_sl_emp, df_sl_pred) %>%
  mutate(
    relevant_property = factor(relevant_property, levels = names(rp_labels),
                               labels = rp_labels),
    sharpness = factor(sharpness, levels = names(sharp_labels),
                       labels = sharp_labels),
    source = factor(source,
                    levels = c("Empirical", "Planned usefulness, context-fixed"))
  )

sharp_facet_labels <- c(
  "High" = "Size discrim.: High",
  "Low"  = "Size discrim.: Low"
)

fig_sl_ppc <- df_sl_ppc %>%
  ggplot(aes(x = mean, y = relevant_property, fill = source)) +
  geom_col(position = position_dodge(0.7), width = 0.6, alpha = 0.85) +
  geom_errorbarh(
    aes(xmin = lo, xmax = hi),
    position = position_dodge(0.7), height = 0.25, linewidth = 0.5
  ) +
  geom_vline(xintercept = 0.5, linetype = "dashed", colour = "grey50") +
  facet_wrap(~ sharpness, ncol = 1,
             labeller = labeller(sharpness = sharp_facet_labels)) +
  scale_fill_manual(values = CSP_colors[c(1, 3)], name = NULL) +
  scale_x_continuous(limits = c(0, 1)) +
  labs(x = "Mean slider rating (size-first preference)",
       y = "Referential context") +
  theme_model() +
  theme(
    legend.position = "top",
    strip.text = element_text(size = 13),
    panel.spacing = unit(1, "lines")
  )

ggsave(file.path(fig_dir, "slider_ppc_best.pdf"), fig_sl_ppc,
       width = 7.5, height = 5.5, dpi = 300)
cat("[✓] slider_ppc_best.pdf\n")

df_sl_corr <- read_csv("data/slider_condition_summary.csv") %>%
  transmute(
    relevant_property, sharpness,
    human_mean = emp_mean,
    model_mean = pred_mean_planned_usefulness_signed_order_static,
    model_lo = pred_lo_planned_usefulness_signed_order_static,
    model_hi = pred_hi_planned_usefulness_signed_order_static
  ) %>%
  mutate(
    relevant_property = factor(relevant_property, levels = names(rp_labels),
                               labels = rp_labels),
    sharpness = factor(sharpness, levels = names(sharp_labels),
                       labels = sharp_labels),
    condition = paste(relevant_property, sharpness, sep = " / ")
  )

r_sq_slider <- cor(df_sl_corr$human_mean, df_sl_corr$model_mean)^2

fig_sl_corr <- df_sl_corr %>%
  ggplot(aes(x = human_mean, y = model_mean)) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", colour = "grey50") +
  geom_errorbar(
    aes(ymin = model_lo, ymax = model_hi),
    width = 0.005, linewidth = 0.4, colour = "grey60"
  ) +
  geom_point(aes(colour = condition), size = 2.8, alpha = 0.9) +
  scale_colour_manual(values = CSP_colors[1:6], name = "Condition") +
  annotate(
    "text", x = min(df_sl_corr$human_mean), y = max(df_sl_corr$model_hi, na.rm = TRUE),
    label = paste0("italic(R)^2 == ",
                   formatC(r_sq_slider, format = "f", digits = 3)),
    parse = TRUE, hjust = 0, vjust = 1, size = 5, colour = "grey30"
  ) +
  labs(
    x = "Empirical mean rating",
    y = "Predicted mean rating"
  ) +
  coord_fixed(xlim = c(0.55, 0.9), ylim = c(0.55, 0.9)) +
  theme_model() +
  theme(legend.position = "right")

ggsave(file.path(fig_dir, "slider_correlation_inc_hier.pdf"), fig_sl_corr,
       width = 6.5, height = 5.2, dpi = 300)
cat("[✓] slider_correlation_inc_hier.pdf\n")

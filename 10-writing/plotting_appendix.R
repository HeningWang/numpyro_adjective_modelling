# =============================================================================
#  Plotting script for appendix — Production ablation model comparison
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


# =============================================================================
#  FIGURE A1 — Production ablation LOO comparison (4 main + 2 ablations)
# =============================================================================

df_abl_loo <- read_csv("data/production_ablation_loo_comparison.csv") %>%
  rename(model = 1)

ablation_labels <- c(
  "incremental_recursive"  = "Incremental, context-updating",
  "incremental_static"     = "Incremental, context-fixed",
  "global_recursive"       = "Global, context-updating",
  "global_static"          = "Global, context-fixed",
  "incremental_lookahead"  = "Incremental, lookahead",
  "incremental_lm_only"    = "Incremental, LM only"
)

df_abl_plot <- df_abl_loo %>%
  filter(model %in% names(ablation_labels)) %>%
  mutate(
    model_label = factor(ablation_labels[model],
                         levels = rev(ablation_labels[order(match(names(ablation_labels), model))])),
    is_best = rank == 0
  )

fig_a1 <- df_abl_plot %>%
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

ggsave(file.path(fig_dir, "production_ablation_loo.pdf"), fig_a1,
       width = 8, height = 4.5, dpi = 300)
cat("[✓] production_ablation_loo.pdf\n")

cat("\n[Done] All appendix figures saved to", fig_dir, "\n")

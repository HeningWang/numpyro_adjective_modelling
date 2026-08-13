# Participant and outcome localization of the K-UPD-HKO hierarchy gain.
# Run from paper/: Rscript scripts/plot_kappa_hierarchy_diagnostics.R

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(patchwork)
  library(readr)
})

source("scripts/csp_figure_style.R")

participant_gain <- read_csv(
  "data/production_kappa_hierarchy_participant_gain.csv",
  show_col_types = FALSE
)
outcome_gain <- read_csv(
  "data/production_kappa_hierarchy_outcome_gain.csv",
  show_col_types = FALSE
)

p_participant_gain <- ggplot(
  participant_gain,
  aes(x = kappa_mean, y = elpd_gain)
) +
  geom_hline(yintercept = 0, colour = CSP_COLORS[["grey"]], linewidth = .7) +
  geom_vline(
    xintercept = unique(participant_gain$shared_kappa),
    colour = CSP_COLORS[["text"]], linetype = "dashed", linewidth = .7
  ) +
  geom_smooth(
    method = "loess", formula = y ~ x, se = FALSE,
    colour = CSP_COLORS[["green"]], linewidth = 1.0, span = .7
  ) +
  geom_point(
    colour = CSP_COLORS[["main"]], fill = "white",
    shape = 21, size = 2.7, stroke = .8, alpha = .9
  ) +
  scale_x_continuous(
    limits = c(-.04, 1.06), breaks = c(0, .5, 1),
    labels = c("0\nGlobal", ".5", "1\nFully incremental")
  ) +
  labs(
    x = expression("Participant successive-choice weight " * kappa[i]),
    y = expression(Delta * "ELPD by participant")
  ) +
  theme_csp() +
  theme(
    panel.grid.minor = element_blank(),
    plot.margin = margin(8, 20, 8, 12)
  )

outcome_plot_data <- outcome_gain %>%
  filter(grouping == "Response length") %>%
  mutate(
    response_length = factor(
      response_length,
      levels = c(1, 2, 3),
      labels = c("One adjective", "Two adjectives", "Three adjectives")
    )
  )

p_outcome_gain <- ggplot(
  outcome_plot_data,
  aes(x = estimate_elpd, y = response_length)
) +
  geom_vline(xintercept = 0, colour = CSP_COLORS[["grey"]], linewidth = .7) +
  geom_errorbar(
    aes(xmin = credible_lower_95, xmax = credible_upper_95),
    orientation = "y", width = .15, linewidth = .75,
    colour = CSP_COLORS[["green"]]
  ) +
  geom_point(colour = CSP_COLORS[["green"]], size = 3.4) +
  scale_x_continuous(
    breaks = c(-50, 0, 50, 100, 150, 200, 250, 300),
    expand = expansion(mult = c(.03, .05))
  ) +
  labs(
    x = expression(Delta * "ELPD (participant-specific " * kappa[i] * " - shared " * kappa * ")"),
    y = NULL
  ) +
  theme_csp() +
  theme(
    panel.grid.major.y = element_blank(),
    axis.ticks.y = element_blank(),
    plot.margin = margin(8, 20, 8, 12)
  )

hierarchy_diagnostics <- (p_participant_gain + labs(tag = "A")) /
  (p_outcome_gain + labs(tag = "B")) +
  plot_layout(heights = c(1.35, .65))

save_csp_pdf(
  hierarchy_diagnostics,
  "figures/production_kappa_hierarchy_diagnostics.pdf",
  9.4,
  6.6
)

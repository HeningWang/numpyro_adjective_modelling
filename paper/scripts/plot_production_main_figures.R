# Main production-model figures and the supplementary size-colour distribution.
# Run from paper/ using the frozen summaries already exported to data/.

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(patchwork)
  library(readr)
  library(scales)
  library(tidyr)
})

source("scripts/csp_figure_style.R")

dir.create("figures", showWarnings = FALSE)

model_levels <- c("Observed", "Global", "Plan-guided", "Fully incremental")
model_palette <- c(
  "Observed" = CSP_COLORS[["text"]],
  "Global" = CSP_COLORS[["main"]],
  "Plan-guided" = CSP_COLORS[["green"]],
  "Fully incremental" = CSP_COLORS[["emphasis"]]
)
model_shapes <- c("Observed" = 21, "Global" = 16, "Plan-guided" = 17, "Fully incremental" = 15)

if (!("--architecture-only" %in% commandArgs(trailingOnly = TRUE))) {
family_labels <- c(
  dimension_color = "Size-colour",
  dimension_form = "Size-form",
  color_form = "Colour-form"
)
response_labels <- c(
  D = "S", DC = "SC", DCF = "SCF", DF = "SF", DFC = "SFC",
  C = "C", CD = "CS", CDF = "CSF", CF = "CF", CFD = "CFS",
  F = "F", FD = "FS", FDC = "FSC", FC = "FC", FCD = "FCS"
)

ppc_architecture <- read_csv(
  "data/production_v10_ppc_summary.csv",
  show_col_types = FALSE
)
ppc_best <- read_csv(
  "data/production_v18_best_ppc_summary.csv",
  show_col_types = FALSE
)

# -----------------------------------------------------------------------------
# Full-distribution posterior-predictive adequacy.
# -----------------------------------------------------------------------------
ppc_exact <- ppc_best %>%
  filter(summary_type == "Exact response") %>%
  mutate(
    family = recode(combination, !!!family_labels),
    context = case_when(
      combination == "dimension_color" & relevant_property == "first" ~ "Size sufficient",
      combination == "dimension_color" & relevant_property == "second" ~ "Colour sufficient",
      combination == "dimension_form" & relevant_property == "first" ~ "Size sufficient",
      combination == "dimension_form" & relevant_property == "second" ~ "Form sufficient",
      combination == "color_form" & relevant_property == "first" ~ "Colour sufficient",
      combination == "color_form" & relevant_property == "second" ~ "Form sufficient",
      TRUE ~ "Both necessary"
    ),
    context_order = recode(relevant_property, first = 1L, both = 2L, second = 3L),
    discriminability = recode(sharpness, blurred = "Low", sharp = "High"),
    condition = paste(context, discriminability, sep = " · "),
    response = factor(unname(response_labels[category]), levels = unname(response_labels))
  )

empirical_intervals <- read_csv(
  "data/production_v10_ppc_empirical_intervals.csv",
  show_col_types = FALSE
)
empirical_intervals_all <- empirical_intervals %>%
  filter(summary_type == "Exact response")

correlation_input <- ppc_exact %>%
  filter(model == "Plan-guided") %>%
  left_join(
    empirical_intervals_all %>%
      select(combination, relevant_property, sharpness, category,
             observed_lower, observed_upper),
    by = c("combination", "relevant_property", "sharpness", "category")
  )
correlation_stats <- read_csv("data/production_v10_ppc_correlation_stats.csv", show_col_types = FALSE)

ppc_bar_data <- read_csv(
  "data/production_v10_ppc_size_colour_bars.csv",
  show_col_types = FALSE
) %>%
  mutate(
    context = factor(
      context,
      levels = c("Size sufficient", "Both necessary", "Colour sufficient")
    ),
    discriminability = factor(discriminability, levels = c("High", "Low")),
    response = factor(response, levels = rev(intersect(unname(response_labels), unique(response)))),
    source = factor(source, levels = c("Plan-guided", "Observed"))
  )

p_full_distribution <- ggplot(
  ppc_bar_data,
  aes(x = proportion, y = response, fill = source)
) +
  geom_col(position = position_dodge(.72), width = .62, alpha = .88) +
  geom_errorbar(
    aes(xmin = lower, xmax = upper),
    orientation = "y", position = position_dodge(.72),
    width = .26, linewidth = .5
  ) +
  facet_grid(
    discriminability ~ context,
    labeller = labeller(
      discriminability = c(High = "High", Low = "Low")
    )
  ) +
  scale_fill_manual(
    values = c(
      "Observed" = CSP_COLORS[["main"]],
      "Plan-guided" = CSP_COLORS[["green"]]
    ),
    breaks = c("Observed", "Plan-guided"),
    labels = c("Observed", "Plan-guided (best model)"),
    name = NULL
  ) +
  scale_x_continuous(labels = percent_format(accuracy = 1), limits = c(0, .86)) +
  labs(x = "Proportion", y = "Completed utterance") +
  theme_csp() +
  theme(
    legend.position = "top",
    panel.grid.major.y = element_blank(),
    panel.spacing = unit(.8, "lines")
  )

p_correlation <- ggplot(correlation_input, aes(x = observed_proportion, y = predicted_mean)) +
  geom_abline(
    intercept = 0, slope = 1, linetype = "dashed",
    colour = CSP_COLORS[["grey"]], linewidth = .7
  ) +
  geom_errorbar(
    aes(ymin = predicted_q025, ymax = predicted_q975),
    width = 0, linewidth = .32, alpha = .32,
    colour = CSP_COLORS[["green"]]
  ) +
  geom_errorbar(
    aes(xmin = observed_lower, xmax = observed_upper),
    orientation = "y", width = 0, linewidth = .32, alpha = .32,
    colour = CSP_COLORS[["main"]]
  ) +
  geom_point(size = 2.15, alpha = .7, colour = CSP_COLORS[["green"]]) +
  scale_x_continuous(labels = percent_format(accuracy = 1), limits = c(0, .9)) +
  scale_y_continuous(labels = percent_format(accuracy = 1), limits = c(0, .9)) +
  annotate(
    "text", x = .035, y = .865, hjust = 0, vjust = 1, size = 4.2,
    colour = CSP_COLORS[["text"]],
    label = sprintf(
      "r = %.3f",
      correlation_stats$pearson_r
    )
  ) +
  labs(x = "Observed proportion", y = "Predicted proportion") +
  coord_fixed() +
  theme_csp() +
  theme(legend.position = "none")

theory_fit <- read_csv(
  "data/production_v18_ppc_theory_outcomes.csv",
  show_col_types = FALSE
) %>%
  mutate(
    source = factor(source, levels = c("Plan-guided", "Observed")),
    outcome = factor(
      outcome,
      levels = c("Size-initial responses", "Redundant adjective use")
    ),
    context = factor(
      context,
      levels = c("Colour sufficient", "Both necessary", "Size sufficient")
    ),
    discriminability = factor(discriminability, levels = c("High", "Low"))
  )

p_theory_fit <- ggplot(
  theory_fit,
  aes(x = proportion, y = context, fill = source)
) +
  geom_col(position = position_dodge(.7), width = .58, alpha = .88) +
  geom_errorbar(
    aes(xmin = lower, xmax = upper),
    orientation = "y",
    position = position_dodge(.7),
    width = .20,
    linewidth = .45
  ) +
  facet_grid(
    discriminability ~ outcome,
    labeller = labeller(
      outcome = c(
        "Size-initial responses" = "Size-initial\nresponses",
        "Redundant adjective use" = "Redundant\nadjective use"
      )
    )
  ) +
  scale_fill_manual(
    values = c(
      "Observed" = CSP_COLORS[["main"]],
      "Plan-guided" = CSP_COLORS[["green"]]
    ),
    name = NULL
  ) +
  scale_x_continuous(
    labels = percent_format(accuracy = 1),
    limits = c(0, 1),
    breaks = seq(0, 1, .5),
    expand = expansion(mult = c(.03, .08))
  ) +
  labs(x = "Proportion", y = NULL) +
  theme_csp() +
  theme(
    legend.position = "top",
    panel.grid.major.y = element_blank(),
    axis.text.x = element_text(size = 13),
    axis.text.y = element_text(size = 13),
    strip.text = element_text(size = 14, face = "bold"),
    panel.spacing.x = unit(1.35, "lines"),
    panel.spacing.y = unit(.65, "lines")
  )

# The behavioural recodings lead the main figure; the complete size-colour
# distribution is retained separately using the same frozen values and intervals.
figure_ppc <- (p_theory_fit | p_correlation) +
  plot_layout(widths = c(1.48, 1)) +
  plot_annotation(tag_levels = "A")
save_csp_pdf(figure_ppc, "figures/production_architecture_ppc.pdf", 9.4, 4.7)
save_csp_pdf(
  p_full_distribution,
  "figures/production_architecture_ppc_full_responses.pdf",
  9.4,
  5.5
)

}

# The exported comparison preserves its original reference model and intervals.
architecture_plot_data <- read_csv("data/production_architecture_figure_data.csv", show_col_types = FALSE)

p_architecture <- architecture_plot_data %>%
  filter(panel == "Factorial predictive loss") %>%
  mutate(
    label = factor(
      label,
      levels = c("Fully incremental", "Global", "Plan-guided")
    ),
    semantic_regime = factor(
      semantic_regime,
      levels = c("Context-fixed", "Sequential context updating")
    )
  ) %>%
  ggplot(aes(
    x = estimate, y = label,
    colour = label, shape = semantic_regime,
    group = semantic_regime
  )) +
  geom_vline(xintercept = 0, colour = CSP_COLORS[["grey"]], linewidth = .6) +
  geom_errorbar(
    aes(xmin = lower, xmax = upper),
    orientation = "y",
    position = position_dodge(width = .42),
    width = .12,
    linewidth = .7
  ) +
  geom_point(position = position_dodge(width = .42), size = 3.7, stroke = 1.1) +
  scale_colour_manual(
    values = c(
      "Plan-guided" = CSP_COLORS[["green"]],
      "Global" = CSP_COLORS[["text"]],
      "Fully incremental" = CSP_COLORS[["text"]]
    ),
    guide = "none"
  ) +
  scale_shape_manual(
    values = c("Context-fixed" = 17, "Sequential context updating" = 15),
    name = NULL
  ) +
  scale_x_continuous(breaks = scales::breaks_pretty(5)) +
  labs(
    x = "ELPD loss relative to\nplan-guided updating\n(lower is better)",
    y = NULL
  ) +
  theme_csp() +
  theme(
    legend.position = "top",
    legend.direction = "vertical",
    legend.justification = "left",
    legend.margin = margin(0, 0, 0, 0),
    legend.box.spacing = unit(2, "pt"),
    panel.grid.major.y = element_blank(),
    axis.ticks.y = element_blank()
  )

p_kappa <- architecture_plot_data %>%
  filter(panel == "Successive-choice contribution") %>%
  ggplot(aes(x = estimate, y = 1)) +
  geom_segment(
    aes(x = 0, xend = 1, yend = 1), linewidth = 2.5,
    colour = CSP_COLORS[["grey"]], lineend = "round"
  ) +
  geom_errorbar(
    aes(xmin = lower, xmax = upper), orientation = "y", width = .15,
    linewidth = 1.1, colour = CSP_COLORS[["green"]]
  ) +
  geom_point(size = 4.5, colour = CSP_COLORS[["green"]]) +
  geom_text(aes(x = estimate, y = 1.24, label = sprintf("kappa == %.3f", estimate)), size = 5, parse = TRUE) +
  scale_x_continuous(
    limits = c(-.12, 1.12), breaks = c(0, 1),
    labels = c("Global", "Fully\nincremental")
  ) +
  scale_y_continuous(NULL, breaks = NULL, limits = c(.72, 1.38)) +
  labs(x = "Incremental-choice\nweight") +
  coord_cartesian(clip = "off") +
  theme_csp() +
  theme(
    panel.grid = element_blank(),
    plot.margin = margin(18, 12, 8, 12)
  )

p_semantics <- architecture_plot_data %>%
  filter(panel == "Semantic architecture") %>%
  mutate(label = factor(
    label,
    levels = c(
      "Fully incremental", "Plan-guided", "Global", "Average across architectures"
    )
  )) %>%
  ggplot(aes(x = estimate, y = label, colour = label)) +
  geom_vline(xintercept = 0, colour = CSP_COLORS[["grey"]], linewidth = .8) +
  geom_errorbar(
    aes(xmin = lower, xmax = upper), orientation = "y", width = .14,
    linewidth = 1.0
  ) +
  geom_point(size = 3.8) +
  scale_colour_manual(
    values = c(
      "Global" = CSP_COLORS[["text"]],
      "Plan-guided" = CSP_COLORS[["green"]],
      "Fully incremental" = CSP_COLORS[["text"]],
      "Average across architectures" = CSP_COLORS[["emphasis"]]
    ),
    guide = "none"
  ) +
  scale_x_continuous(
    breaks = scales::breaks_pretty(4),
    expand = expansion(mult = c(.01, .01))
  ) +
  labs(
    x = "ELPD difference\n(updating - fixed)",
    y = NULL
  ) +
  theme_csp() +
  theme(
    panel.grid.major.y = element_blank(),
    axis.ticks.y = element_blank(),
    axis.text.y = element_text(size = 10.5)
  )

architecture_comparison_figure <- (
  (p_architecture + labs(tag = "A")) |
    ((p_kappa + labs(tag = "B")) / (p_semantics + labs(tag = "C")) +
       plot_layout(heights = c(1, 1.05)))
) +
  plot_layout(widths = c(1.2, 1), guides = "keep")
save_csp_pdf(
  architecture_comparison_figure,
  "figures/production_architecture_comparison.pdf",
  9.4,
  5.6
)

# Figures 7--10: architecture comparison, overall PPC, residuals, and participant variation.
# Run from paper/ after export_v10_ppc.py and export_participant_figure_data.py.

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
dir.create("data", showWarnings = FALSE)

model_levels <- c("Observed", "Global", "Plan-guided", "Fully incremental")
model_palette <- c(
  "Observed" = CSP_COLORS[["text"]],
  "Global" = CSP_COLORS[["main"]],
  "Plan-guided" = CSP_COLORS[["green"]],
  "Fully incremental" = CSP_COLORS[["emphasis"]]
)
model_shapes <- c("Observed" = 21, "Global" = 16, "Plan-guided" = 17, "Fully incremental" = 15)

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
correlation_test <- cor.test(
  correlation_input$observed_proportion,
  correlation_input$predicted_mean,
  method = "pearson"
)
correlation_stats <- correlation_input %>%
  summarise(
    n_cells = n(),
    pearson_r = unname(correlation_test$estimate),
    r_squared = pearson_r^2,
    p_value = correlation_test$p.value,
    rmse = sqrt(mean((observed_proportion - predicted_mean)^2)),
    mae = mean(abs(observed_proportion - predicted_mean)),
    interval_coverage = mean(
      observed_proportion >= predicted_q025 & observed_proportion <= predicted_q975
    )
  )
write_csv(correlation_stats, "data/production_v10_ppc_correlation_stats.csv")
correlation_stats <- read_csv("data/production_v10_ppc_correlation_stats.csv", show_col_types = FALSE)

ppc_size_colour <- correlation_input %>%
  filter(combination == "dimension_color") %>%
  mutate(
    context = factor(
      context,
      levels = c("Size sufficient", "Both necessary", "Colour sufficient")
    ),
    discriminability = factor(discriminability, levels = c("High", "Low"))
  )

retained_responses <- ppc_size_colour %>%
  group_by(response) %>%
  summarise(max_proportion = max(c(observed_proportion, predicted_mean)), .groups = "drop") %>%
  filter(max_proportion >= .05) %>%
  pull(response) %>%
  as.character()

ppc_bar_data <- bind_rows(
  ppc_size_colour %>%
    transmute(
      context, discriminability, response,
      source = "Observed", proportion = observed_proportion,
      lower = observed_lower, upper = observed_upper
    ),
  ppc_size_colour %>%
    transmute(
      context, discriminability, response,
      source = "Plan-guided", proportion = predicted_mean,
      lower = predicted_q025, upper = predicted_q975
    )
  ) %>%
  filter(as.character(response) %in% retained_responses) %>%
  mutate(
    response = factor(
      as.character(response),
      levels = rev(intersect(unname(response_labels), retained_responses))
    ),
    source = factor(source, levels = c("Plan-guided", "Observed"))
  )
write_csv(ppc_bar_data, "data/production_v10_ppc_size_colour_bars.csv")
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
    response = factor(response, levels = rev(intersect(unname(response_labels), retained_responses))),
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
      "r = %.3f\np < .001",
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
    legend.position = "none",
    panel.grid.major.y = element_blank(),
    axis.text.x = element_text(size = 11.5),
    axis.text.y = element_text(size = 10.5),
    strip.text = element_text(size = 14, face = "bold"),
    panel.spacing.x = unit(1.35, "lines"),
    panel.spacing.y = unit(.65, "lines")
  )

figure_7_bottom <- (p_correlation | p_theory_fit) +
  plot_layout(widths = c(.85, 1.25))

figure_7 <- (p_full_distribution / figure_7_bottom) +
  plot_layout(heights = c(1.45, 1.05)) +
  plot_annotation(tag_levels = "A")
save_csp_pdf(figure_7, "figures/production_architecture_ppc.pdf", 9.4, 8.1)

# -----------------------------------------------------------------------------
# Architecture comparison and theory-linked endpoint residuals.
# -----------------------------------------------------------------------------
matched_plan_guided <- read_csv(
  "../analysis_revision/14_final_architecture/ho_architecture_model_table_v10.csv",
  show_col_types = FALSE
)
factorial_statistics <- read_csv(
  "data/production_architecture_factorial_statistics_v16.csv",
  show_col_types = FALSE
)
architecture_plot_data <- bind_rows(
  factorial_statistics %>%
    filter(row_type == "factorial_cell") %>%
    transmute(
      panel = "Factorial predictive loss",
      label = architecture,
      semantic_regime,
      estimate = estimate_elpd,
      lower = credible_lower_95,
      upper = credible_upper_95
    ),
  matched_plan_guided %>%
    filter(model == "K-HO") %>%
    transmute(
      panel = "Successive-choice contribution", label = "kappa",
      semantic_regime = NA_character_,
      estimate = kappa_mean, lower = kappa_q025, upper = kappa_q975
    ),
  factorial_statistics %>%
    filter(estimand %in% c(
      "semantic_regime_main_effect",
      "semantic_regime_simple_effect"
    )) %>%
    transmute(
      panel = "Semantic architecture",
      label = recode(
        label,
        `Average across production architectures` = "Average across architectures"
      ),
      semantic_regime = NA_character_,
      estimate = estimate_elpd,
      lower = credible_lower_95,
      upper = credible_upper_95
    )
)
write_csv(architecture_plot_data, "data/production_architecture_figure_data.csv")
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
    values = c("Context-fixed" = 1, "Sequential context updating" = 16),
    name = NULL
  ) +
  scale_x_continuous(limits = c(-20, 480), breaks = seq(0, 400, 100)) +
  labs(
    x = "Effect of incremental\npragmatic production\n(ELPD, lower is better)",
    y = NULL
  ) +
  theme_csp() +
  theme(
    legend.position = "top",
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
  annotate("text", x = .4472, y = 1.24, label = expression(kappa == .447), size = 5) +
  scale_x_continuous(
    limits = c(-.12, 1.12), breaks = c(0, 1),
    labels = c("Global", "Fully\nincremental")
  ) +
  scale_y_continuous(NULL, breaks = NULL, limits = c(.72, 1.38)) +
  labs(x = "Successive-choice weight") +
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
  geom_vline(
    xintercept = c(-4, 4), linetype = "dashed",
    colour = CSP_COLORS[["grey"]], linewidth = .65
  ) +
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
    limits = c(-4.5, 4.5),
    breaks = c(-4, -2, 0, 2, 4),
    expand = expansion(mult = c(.01, .01))
  ) +
  labs(
    x = "Effect of hierarchical\nsemantic composition\n(ELPD, updating - fixed)",
    y = NULL
  ) +
  theme_csp() +
  theme(
    panel.grid.major.y = element_blank(),
    axis.ticks.y = element_blank(),
    axis.text.y = element_text(size = 10.5)
  )

architecture_source_palette <- c(
  "Observed" = CSP_COLORS[["main"]],
  "Global" = CSP_COLORS[["text"]],
  "Plan-guided" = CSP_COLORS[["green"]],
  "Fully incremental" = CSP_COLORS[["emphasis"]]
)
architecture_source_shapes <- c(
  "Observed" = 16,
  "Global" = 1,
  "Plan-guided" = 17,
  "Fully incremental" = 15
)
initial_spread_threshold <- .067
utterance_spread_threshold <- .05
architecture_dodge <- 1.0
condition_diagnostics <- read_csv(
  "data/production_architecture_condition_tv_diagnostics.csv",
  show_col_types = FALSE
)
cell_diagnostics <- read_csv(
  "data/production_architecture_cell_spread_diagnostics.csv",
  show_col_types = FALSE
)
initial_focus_cells <- cell_diagnostics %>%
  filter(
    summary_type == "Initial adjective",
    architecture_spread >= initial_spread_threshold
  ) %>%
  select(combination, relevant_property, sharpness, category)

initial_predictions <- ppc_architecture %>%
  filter(summary_type == "Initial adjective") %>%
  semi_join(
    initial_focus_cells,
    by = c("combination", "relevant_property", "sharpness", "category")
  ) %>%
  transmute(
    combination, relevant_property, sharpness, category,
    source = model,
    proportion = predicted_mean,
    lower = predicted_q025,
    upper = predicted_q975
  )
initial_observed <- empirical_intervals %>%
  filter(summary_type == "Initial adjective") %>%
  semi_join(
    initial_focus_cells,
    by = c("combination", "relevant_property", "sharpness", "category")
  ) %>%
  transmute(
    combination, relevant_property, sharpness, category,
    source = "Observed",
    proportion = observed_mean,
    lower = observed_lower,
    upper = observed_upper
  )
initial_architecture_data <- bind_rows(initial_observed, initial_predictions) %>%
  mutate(
    context = case_when(
      combination == "dimension_color" & relevant_property == "first" ~ "Size sufficient",
      combination == "dimension_color" & relevant_property == "second" ~ "Colour sufficient",
      combination == "dimension_form" & relevant_property == "first" ~ "Size sufficient",
      combination == "dimension_form" & relevant_property == "second" ~ "Form sufficient",
      combination == "color_form" & relevant_property == "first" ~ "Colour sufficient",
      combination == "color_form" & relevant_property == "second" ~ "Form sufficient",
      combination == "dimension_color" & relevant_property == "both" ~ "Both size and colour necessary",
      combination == "dimension_form" & relevant_property == "both" ~ "Both size and form necessary",
      combination == "color_form" & relevant_property == "both" ~ "Both colour and form necessary"
    ),
    initial_adjective = recode(category, D = "Size", C = "Colour", F = "Form"),
    discriminability = recode(
      sharpness,
      sharp = "High size discrim.",
      blurred = "Low size discrim."
    ),
    source = factor(source, levels = model_levels),
    panel = factor(
      paste(context, discriminability, sep = ",\n"),
      levels = c(
        "Both size and colour necessary,\nHigh size discrim.",
        "Both size and form necessary,\nHigh size discrim.",
        "Both colour and form necessary,\nHigh size discrim.",
        "Size sufficient,\nLow size discrim.",
        "Both size and form necessary,\nLow size discrim.",
        "Both colour and form necessary,\nLow size discrim."
      )
    ),
    row_label = category,
    row_id = paste(panel, row_label, sep = "|")
  )
initial_row_levels <- initial_architecture_data %>%
  distinct(panel, relevant_property, category, row_id) %>%
  mutate(
    context_order = match(relevant_property, c("second", "both", "first")),
    adjective_order = match(category, c("F", "C", "D"))
  ) %>%
  arrange(panel, context_order, adjective_order) %>%
  pull(row_id)
initial_architecture_data <- initial_architecture_data %>%
  mutate(row_id = factor(row_id, levels = unique(initial_row_levels))) %>%
  group_by(panel) %>%
  mutate(panel_rows = n_distinct(category)) %>%
  ungroup()
write_csv(
  initial_architecture_data,
  "data/production_architecture_initial_adjective_predictions.csv"
)

make_initial_architecture_panel <- function(plot_data, x_title = NULL, y_expand = .55) {
  ggplot(
    plot_data,
    aes(x = proportion, y = row_id, colour = source, shape = source)
  ) +
    geom_errorbar(
      aes(xmin = lower, xmax = upper),
      orientation = "y",
      position = position_dodge(width = architecture_dodge),
      width = .12,
      linewidth = .55
    ) +
    geom_point(position = position_dodge(width = architecture_dodge), size = 3.2, stroke = 1.0) +
    facet_wrap(~panel, nrow = 1, scales = "free") +
    scale_y_discrete(
      labels = function(values) sub("^.*\\|", "", values),
      expand = expansion(add = y_expand)
    ) +
    scale_colour_manual(values = architecture_source_palette, name = NULL) +
    scale_shape_manual(values = architecture_source_shapes, name = NULL) +
    scale_x_continuous(
      labels = percent_format(accuracy = 1),
      breaks = scales::breaks_pretty(n = 4),
      expand = expansion(mult = c(.02, .04))
    ) +
    labs(x = x_title, y = "Initial adjective") +
    theme_csp() +
    theme(
      legend.position = "top",
      panel.grid.major.y = element_blank(),
      axis.ticks.y = element_blank(),
      axis.text.y = element_text(size = 11),
      strip.text = element_text(size = 10.5, face = "bold"),
      panel.spacing.x = unit(1.1, "lines")
    )
}

p_initial_single <- make_initial_architecture_panel(
  filter(initial_architecture_data, panel_rows == 1),
  y_expand = .55
)
p_initial_dual <- make_initial_architecture_panel(
  filter(initial_architecture_data, panel_rows == 2),
  x_title = "Proportion",
  y_expand = .12
) +
  theme(legend.position = "none")
p_initial_architectures <- wrap_plots(
  p_initial_single + labs(tag = "A"),
  p_initial_dual,
  ncol = 1,
  heights = c(.62, 1.5),
  axis_titles = "collect_y"
)

focus_conditions <- condition_diagnostics %>%
  slice_max(maximum_pairwise_tv, n = 3, with_ties = FALSE) %>%
  select(all_of(c("combination", "relevant_property", "sharpness")))
focus_responses <- cell_diagnostics %>%
  filter(
    summary_type == "Exact response",
    architecture_spread >= utterance_spread_threshold
  ) %>%
  semi_join(focus_conditions, by = c("combination", "relevant_property", "sharpness")) %>%
  select(combination, relevant_property, sharpness, category)

focus_predictions <- ppc_architecture %>%
  filter(summary_type == "Exact response") %>%
  semi_join(
    focus_responses,
    by = c("combination", "relevant_property", "sharpness", "category")
  ) %>%
  transmute(
    combination, relevant_property, sharpness, category,
    source = model,
    proportion = predicted_mean,
    lower = predicted_q025,
    upper = predicted_q975
  )
focus_observed <- empirical_intervals %>%
  filter(summary_type == "Exact response") %>%
  semi_join(
    focus_responses,
    by = c("combination", "relevant_property", "sharpness", "category")
  ) %>%
  transmute(
    combination, relevant_property, sharpness, category,
    source = "Observed",
    proportion = observed_mean,
    lower = observed_lower,
    upper = observed_upper
  )
focus_architecture_data <- bind_rows(focus_observed, focus_predictions) %>%
  mutate(
    context = case_when(
      combination == "dimension_form" & relevant_property == "first" ~ "Size sufficient",
      combination == "color_form" & relevant_property == "both" ~ "Both colour and form necessary"
    ),
    discriminability = recode(
      sharpness,
      sharp = "High size discrim.",
      blurred = "Low size discrim."
    ),
    panel = factor(
      paste(context, discriminability, sep = ",\n"),
      levels = c(
        "Size sufficient,\nLow size discrim.",
        "Both colour and form necessary,\nHigh size discrim.",
        "Both colour and form necessary,\nLow size discrim."
      )
    ),
    response = recode(category, !!!response_labels),
    row_id = paste(panel, response, sep = "|"),
    source = factor(source, levels = model_levels)
  )
focus_row_levels <- focus_architecture_data %>%
  distinct(panel, response, row_id) %>%
  group_by(panel) %>%
  arrange(response, .by_group = TRUE) %>%
  ungroup() %>%
  pull(row_id)
focus_architecture_data <- focus_architecture_data %>%
  mutate(row_id = factor(row_id, levels = rev(unique(focus_row_levels))))
write_csv(
  focus_architecture_data,
  "data/production_architecture_focus_utterance_predictions.csv"
)

p_focus_utterances <- ggplot(
  focus_architecture_data,
  aes(x = proportion, y = row_id, colour = source, shape = source)
) +
  geom_errorbar(
    aes(xmin = lower, xmax = upper),
    orientation = "y",
    position = position_dodge(width = architecture_dodge),
    width = .12,
    linewidth = .55
  ) +
  geom_point(position = position_dodge(width = architecture_dodge), size = 3.2, stroke = 1.0) +
  facet_wrap(~panel, nrow = 1, scales = "free") +
  scale_y_discrete(
    labels = function(values) sub("^.*\\|", "", values),
    expand = expansion(add = .65)
  ) +
  scale_colour_manual(values = architecture_source_palette, name = NULL) +
  scale_shape_manual(values = architecture_source_shapes, name = NULL) +
  scale_x_continuous(
    labels = percent_format(accuracy = 1),
    breaks = scales::breaks_pretty(n = 4),
    expand = expansion(mult = c(.02, .04))
  ) +
  labs(x = "Proportion", y = "Completed utterance") +
  theme_csp() +
  theme(
    legend.position = "none",
    panel.grid.major.y = element_blank(),
    axis.ticks.y = element_blank(),
    strip.text = element_text(size = 10.5, face = "bold"),
    panel.spacing.x = unit(1.0, "lines")
  )

p_architecture_predictions <- (
  wrap_elements(full = p_initial_architectures) /
    (p_focus_utterances + labs(tag = "B"))
) +
  plot_layout(heights = c(1.55, .72))

architecture_comparison_figure <- (
  (p_architecture + labs(tag = "A")) |
    ((p_kappa + labs(tag = "B")) / (p_semantics + labs(tag = "C")) +
       plot_layout(heights = c(1, 1.05)))
) +
  plot_layout(widths = c(1.2, 1), guides = "collect") &
  theme(legend.position = "top")
save_csp_pdf(
  architecture_comparison_figure,
  "figures/production_architecture_comparison.pdf",
  9.4,
  5.6
)

save_csp_pdf(
  p_architecture_predictions,
  "figures/production_architecture_residuals.pdf",
  9.4,
  7.8
)

# -----------------------------------------------------------------------------
# Participant heterogeneity and its unresolved source.
# -----------------------------------------------------------------------------
parameter_path <- "data/production_participant_parameter_intervals.csv"
if (!file.exists(parameter_path)) {
  stop("Run analysis_revision/13_form_identification/v9_hierarchy/export_participant_figure_data.py first")
}
participant_parameters <- read_csv(parameter_path, show_col_types = FALSE) %>%
  mutate(
    parameter = recode(
      parameter,
      `Pragmatic optimality` = "atop('Pragmatic optimality', alpha[i])",
      `Successive-choice contribution` = "atop('Successive-choice weight', kappa[i])",
      `Successive-choice kappa` = "atop('Successive-choice weight', kappa[i])",
      `Stable-order weighting` = "atop('Stable-order weight', beta[i])",
      `Stable-order weight` = "atop('Stable-order weight', beta[i])"
    ),
    parameter = factor(
      parameter,
      levels = c(
        "atop('Pragmatic optimality', alpha[i])",
        "atop('Successive-choice weight', kappa[i])",
        "atop('Stable-order weight', beta[i])"
      )
    )
  ) %>%
  group_by(parameter) %>%
  arrange(posterior_mean, .by_group = TRUE) %>%
  mutate(participant_rank = row_number()) %>%
  ungroup()
write_csv(
  participant_parameters,
  "data/production_participant_parameter_intervals_ranked.csv"
)

participant_prediction_threshold <- .20
participant_prediction_summary <- read_csv(
  "data/production_participant_prediction_summary.csv",
  show_col_types = FALSE
)
participant_prediction_gaps <- read_csv(
  "data/production_participant_prediction_gaps.csv",
  show_col_types = FALSE
)
selected_participant_conditions <- participant_prediction_gaps %>%
  filter(absolute_gap >= participant_prediction_threshold) %>%
  group_by(parameter, combination, relevant_property, sharpness) %>%
  summarise(condition_maximum_gap = max(absolute_gap), .groups = "drop") %>%
  group_by(parameter) %>%
  slice_max(condition_maximum_gap, n = 3, with_ties = FALSE) %>%
  ungroup() %>%
  select(parameter, combination, relevant_property, sharpness)
selected_participant_cells <- participant_prediction_gaps %>%
  filter(absolute_gap >= participant_prediction_threshold) %>%
  semi_join(
    selected_participant_conditions,
    by = c("parameter", "combination", "relevant_property", "sharpness")
  ) %>%
  select(parameter, combination, relevant_property, sharpness, category)

participant_prediction_data <- participant_prediction_summary %>%
  filter(parameter_group %in% c("Lower 15%", "Upper 15%")) %>%
  semi_join(
    selected_participant_cells,
    by = c("parameter", "combination", "relevant_property", "sharpness", "category")
  ) %>%
  mutate(
    context = case_when(
      combination == "dimension_color" & relevant_property == "first" ~ "Size sufficient",
      combination == "dimension_color" & relevant_property == "second" ~ "Colour sufficient",
      combination == "dimension_color" & relevant_property == "both" ~ "Both size and colour necessary",
      combination == "dimension_form" & relevant_property == "first" ~ "Size sufficient",
      combination == "dimension_form" & relevant_property == "second" ~ "Form sufficient",
      combination == "dimension_form" & relevant_property == "both" ~ "Both size and form necessary",
      combination == "color_form" & relevant_property == "first" ~ "Colour sufficient",
      combination == "color_form" & relevant_property == "second" ~ "Form sufficient",
      combination == "color_form" & relevant_property == "both" ~ "Both colour and form necessary"
    ),
    discriminability = recode(
      sharpness,
      sharp = "High size discrim.",
      blurred = "Low size discrim."
    ),
    panel_id = paste(combination, relevant_property, sharpness, sep = "|"),
    panel_label = paste(context, discriminability, sep = "\n"),
    response = recode(category, !!!response_labels),
    row_id = paste(panel_id, response, sep = "|"),
    parameter_group = factor(parameter_group, levels = c("Lower 15%", "Upper 15%"))
  )
participant_gap_segments <- participant_prediction_gaps %>%
  filter(absolute_gap >= participant_prediction_threshold) %>%
  semi_join(
    selected_participant_conditions,
    by = c("parameter", "combination", "relevant_property", "sharpness")
  ) %>%
  inner_join(
    participant_prediction_data %>%
      distinct(parameter, combination, relevant_property, sharpness, category, panel_id, panel_label, row_id),
    by = c("parameter", "combination", "relevant_property", "sharpness", "category")
  )

p_parameter_intervals <- ggplot(
  participant_parameters,
  aes(x = participant_rank, y = posterior_mean)
) +
  geom_linerange(
    aes(ymin = posterior_q025, ymax = posterior_q975),
    colour = CSP_COLORS[["grey"]], linewidth = .35
  ) +
  geom_point(colour = CSP_COLORS[["green"]], size = 1.25) +
  facet_wrap(
    ~ parameter,
    scales = "free_y",
    ncol = 3,
    labeller = labeller(parameter = label_parsed)
  ) +
  labs(x = "Participant rank", y = "Posterior estimate") +
  theme_csp() +
  theme(
    panel.grid.minor = element_blank(),
    panel.spacing.x = unit(1.1, "lines"),
    strip.text = element_text(size = 9.5, face = "bold")
  )

make_participant_prediction_plot <- function(parameter_name, model_title, ncol) {
  plot_data <- participant_prediction_data %>%
    filter(parameter == parameter_name)
  panel_levels <- plot_data %>%
    distinct(combination, relevant_property, sharpness, panel_id) %>%
    mutate(
      family_order = match(combination, c("dimension_color", "dimension_form", "color_form")),
      context_order = match(relevant_property, c("first", "both", "second")),
      sharpness_order = match(sharpness, c("sharp", "blurred"))
    ) %>%
    arrange(family_order, context_order, sharpness_order) %>%
    pull(panel_id)
  panel_labels <- plot_data %>%
    distinct(panel_id, panel_label) %>%
    {setNames(.$panel_label, .$panel_id)}
  row_levels <- plot_data %>%
    distinct(panel_id, category, row_id) %>%
    mutate(
      panel_order = match(panel_id, panel_levels),
      response_order = match(category, names(response_labels))
    ) %>%
    arrange(panel_order, response_order) %>%
    pull(row_id)
  plot_data <- plot_data %>%
    mutate(
      panel_id = factor(panel_id, levels = panel_levels),
      row_id = factor(row_id, levels = rev(row_levels))
    )
  segment_data <- participant_gap_segments %>%
    filter(parameter == parameter_name) %>%
    mutate(
      panel_id = factor(panel_id, levels = panel_levels),
      row_id = factor(row_id, levels = rev(row_levels))
    )
  observed_data <- plot_data %>%
    distinct(parameter_group, panel_id, row_id, observed_proportion)

  ggplot(
    plot_data,
    aes(
      x = predicted_mean,
      y = row_id,
      colour = parameter_group,
      group = parameter_group
    )
  ) +
    geom_errorbar(
      aes(xmin = predicted_q025, xmax = predicted_q975),
      orientation = "y",
      position = position_dodge(width = .52),
      width = .12,
      linewidth = .5
    ) +
    geom_point(
      aes(shape = "Model prediction"),
      position = position_dodge(width = .52),
      size = 2.8
    ) +
    geom_point(
      data = observed_data,
      aes(
        x = observed_proportion,
        y = row_id,
        colour = parameter_group,
        group = parameter_group,
        shape = "Observed"
      ),
      inherit.aes = FALSE,
      position = position_dodge(width = .52),
      size = 2.8,
      stroke = 1.0
    ) +
    facet_wrap(
      ~panel_id,
      ncol = ncol,
      scales = "free",
      labeller = as_labeller(panel_labels)
    ) +
    scale_y_discrete(labels = function(values) sub("^.*\\|", "", values)) +
    scale_colour_manual(
      values = c(
        "Lower 15%" = CSP_COLORS[["main"]],
        "Upper 15%" = CSP_COLORS[["emphasis"]]
      ),
      labels = c("Lower 15%" = "Lower 15%", "Upper 15%" = "Upper 15%"),
      name = "Participant group"
    ) +
    scale_shape_manual(
      values = c("Model prediction" = 15, "Observed" = 1),
      name = "Estimate"
    ) +
    scale_x_continuous(
      labels = percent_format(accuracy = 1),
      breaks = scales::breaks_pretty(n = 4),
      expand = expansion(mult = c(.03, .06))
    ) +
    labs(
      title = model_title,
      x = "Response proportion",
      y = "Completed utterance"
    ) +
    theme_csp() +
    theme(
      plot.title = element_text(size = 12.5, face = "bold", hjust = 0),
      legend.position = "top",
      panel.grid.major.y = element_blank(),
      axis.ticks.y = element_blank(),
      axis.text.y = element_text(size = 10.5),
      strip.text = element_text(size = 9.2, face = "bold"),
      panel.spacing.x = unit(.9, "lines"),
      panel.spacing.y = unit(.75, "lines")
    )
}

p_kappa_prediction_gaps <- make_participant_prediction_plot(
  "Successive-choice contribution",
  expression("Successive-choice weight" ~ (kappa[i])),
  ncol = 3
)
p_alpha_prediction_gaps <- make_participant_prediction_plot(
  "Pragmatic optimality",
  expression("Pragmatic optimality" ~ (alpha[i])),
  ncol = 3
)
p_beta_prediction_gaps <- make_participant_prediction_plot(
  "Stable-order weighting",
  expression("Stable-order weight" ~ (beta[i])),
  ncol = 3
)
p_participant_prediction_gaps <- wrap_plots(
  p_alpha_prediction_gaps,
  p_kappa_prediction_gaps,
  p_beta_prediction_gaps,
  ncol = 1,
  heights = c(1, 1, 1),
  guides = "collect"
) & theme(legend.position = "top")

figure_10 <- (
  (p_parameter_intervals + labs(tag = "A")) /
    wrap_elements(full = p_participant_prediction_gaps) + labs(tag = "B")
) +
  plot_layout(heights = c(.72, 2.55))
save_csp_pdf(figure_10, "figures/production_participant_variation.pdf", 9.4, 12.4)

cat("Wrote architecture comparison, PPC, residual, and participant figures with supporting CSVs.\n")

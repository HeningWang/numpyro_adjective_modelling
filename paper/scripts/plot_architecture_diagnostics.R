# Supplementary architecture residuals and participant variation.
# Run from paper/ after the frozen figure data have been exported.

source("scripts/plot_production_main_figures.R")

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
      relevant_property == "both" ~ "Both necessary"
    ),
    initial_adjective = recode(category, D = "Size", C = "Colour", F = "Form"),
    discriminability = recode(
      sharpness,
      sharp = "High size discrim.",
      blurred = "Low size discrim."
    ),
    source = factor(source, levels = model_levels),
    panel = factor(paste(unname(family_labels[combination]), context, discriminability, sep = "\n")),
    row_label = unname(response_labels[category]),
    row_id = paste(panel, row_label, sep = "|")
  )
stopifnot(!anyNA(initial_architecture_data$panel))
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
      combination == "dimension_color" & relevant_property == "first" ~ "Size sufficient",
      combination == "dimension_color" & relevant_property == "second" ~ "Colour sufficient",
      combination == "dimension_form" & relevant_property == "first" ~ "Size sufficient",
      combination == "dimension_form" & relevant_property == "second" ~ "Form sufficient",
      combination == "color_form" & relevant_property == "first" ~ "Colour sufficient",
      combination == "color_form" & relevant_property == "second" ~ "Form sufficient",
      relevant_property == "both" ~ "Both necessary"
    ),
    discriminability = recode(sharpness, sharp = "High size discrim.", blurred = "Low size discrim."),
    panel = factor(paste(unname(family_labels[combination]), context, discriminability, sep = "\n")),
    response = recode(category, !!!response_labels),
    row_id = paste(panel, response, sep = "|"),
    source = factor(source, levels = model_levels)
  )
stopifnot(!anyNA(focus_architecture_data$panel))
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

save_csp_pdf(
  p_architecture_predictions,
  "figures/production_architecture_residuals.pdf",
  9.4,
  7.8
)

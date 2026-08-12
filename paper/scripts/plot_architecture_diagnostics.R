# Figures 7--9: overall PPC, architecture residuals, and participant variation.
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

ppc <- read_csv("data/production_v10_ppc_summary.csv", show_col_types = FALSE)

# -----------------------------------------------------------------------------
# Figure 7: full-distribution posterior-predictive adequacy.
# -----------------------------------------------------------------------------
ppc_exact <- ppc %>%
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

empirical_intervals_all <- read_csv(
  "data/production_v10_ppc_empirical_intervals.csv",
  show_col_types = FALSE
) %>%
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
    source = factor(source, levels = c("Observed", "Plan-guided"))
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
    source = factor(source, levels = c("Observed", "Plan-guided"))
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

ordering_fit <- ppc %>%
  filter(
    model == "Plan-guided",
    summary_type == "Exact response",
    combination == "dimension_color",
    relevant_property == "both",
    category %in% c("DC", "CD")
  ) %>%
  summarise(
    observed = sum(
      observed_proportion[category == "DC"] *
        observation_count[category == "DC"]
    ) / sum(observed_proportion * observation_count),
    predicted = sum(
      predicted_mean[category == "DC"] *
        observation_count[category == "DC"]
    ) / sum(predicted_mean * observation_count),
    .groups = "drop"
  ) %>%
  mutate(
    outcome = "Conditional size-first order",
    condition = "Both necessary"
  ) %>%
  select(outcome, condition, observed, predicted)

redundancy_fit <- ppc %>%
  filter(
    model == "Plan-guided",
    summary_type == "Response length",
    combination == "dimension_color",
    (relevant_property %in% c("first", "second") & category %in% c("2", "3")) |
      (relevant_property == "both" & category == "3")
  ) %>%
  group_by(relevant_property, sharpness) %>%
  summarise(
    observed = sum(observed_proportion),
    predicted = sum(predicted_mean),
    .groups = "drop"
  ) %>%
  mutate(
    outcome = "Redundant adjective use",
    condition = paste(
      recode(
        relevant_property,
        first = "Size sufficient",
        both = "Both necessary",
        second = "Colour sufficient"
      ),
      recode(sharpness, blurred = "Low", sharp = "High"),
      sep = " · "
    )
  ) %>%
  select(outcome, condition, observed, predicted)

theory_fit <- bind_rows(ordering_fit, redundancy_fit) %>%
  pivot_longer(
    cols = c(observed, predicted),
    names_to = "source",
    values_to = "proportion"
  ) %>%
  mutate(
    source = recode(source, observed = "Observed", predicted = "Plan-guided"),
    source = factor(source, levels = c("Plan-guided", "Observed")),
    outcome = factor(
      outcome,
      levels = c("Conditional size-first order", "Redundant adjective use")
    ),
    condition = factor(
      condition,
      levels = c(
        "Both necessary",
        "Colour sufficient · Low", "Colour sufficient · High",
        "Both necessary · Low", "Both necessary · High",
        "Size sufficient · Low", "Size sufficient · High"
      )
    )
  )
write_csv(theory_fit, "data/production_v10_ppc_theory_outcomes.csv")
theory_fit <- read_csv(
  "data/production_v10_ppc_theory_outcomes.csv",
  show_col_types = FALSE
) %>%
  mutate(
    source = factor(source, levels = c("Plan-guided", "Observed")),
    outcome = factor(
      outcome,
      levels = c("Conditional size-first order", "Redundant adjective use")
    ),
    condition = factor(
      condition,
      levels = c(
        "Both necessary",
        "Colour sufficient · Low", "Colour sufficient · High",
        "Both necessary · Low", "Both necessary · High",
        "Size sufficient · Low", "Size sufficient · High"
      )
    )
  )

theory_outcome_panel <- function(data, title, show_x_title = TRUE) {
  ggplot(data, aes(x = proportion, y = condition, fill = source)) +
    geom_col(position = position_dodge(.7), width = .58, alpha = .88) +
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
      breaks = seq(0, 1, .25),
      expand = expansion(mult = c(0, .02))
    ) +
    labs(
      title = title,
      x = if (show_x_title) "Proportion" else NULL,
      y = NULL
    ) +
    theme_csp() +
    theme(
      legend.position = "none",
      panel.grid.major.y = element_blank(),
      axis.text.y = element_text(size = 11.5),
      plot.title = element_text(size = 14, hjust = .5),
      plot.margin = margin(2, 6, 2, 6)
    )
}

p_ordering_fit <- theory_outcome_panel(
  theory_fit %>% filter(outcome == "Conditional size-first order"),
  "Conditional size-first order",
  FALSE
)
p_redundancy_fit <- theory_outcome_panel(
  theory_fit %>% filter(outcome == "Redundant adjective use"),
  "Redundant adjective use"
)
p_theory_fit <- wrap_elements(
  full = p_ordering_fit / p_redundancy_fit +
    plot_layout(heights = c(.58, 1.72))
)

figure_7 <- p_full_distribution / (p_correlation | p_theory_fit) +
  plot_layout(heights = c(1.45, 1.05), widths = c(1, 1.18)) +
  plot_annotation(tag_levels = "A")
save_csp_pdf(figure_7, "figures/production_architecture_ppc.pdf", 9.4, 8.1)

# -----------------------------------------------------------------------------
# Figure 8: predictive comparison and magnified architecture residuals.
# -----------------------------------------------------------------------------
architecture_models <- read_csv(
  "../analysis_revision/14_final_architecture/ho_architecture_model_table_v10.csv",
  show_col_types = FALSE
)
architecture_contrasts <- read_csv(
  "../analysis_revision/14_final_architecture/ho_architecture_contrasts_v10.csv",
  show_col_types = FALSE
)
architecture_plot_data <- bind_rows(
  tibble(
    panel = "Production architecture",
    label = c("Plan-guided", "Global", "Fully incremental"),
    estimate = c(
      0,
      architecture_contrasts$delta_elpd_loo[architecture_contrasts$comparison == "K-HO_minus_G-HO"],
      architecture_contrasts$delta_elpd_loo[architecture_contrasts$comparison == "K-HO_minus_I-HO"]
    ),
    lower = c(
      0,
      architecture_contrasts$participant_bootstrap_total_q025[architecture_contrasts$comparison == "K-HO_minus_G-HO"],
      architecture_contrasts$participant_bootstrap_total_q025[architecture_contrasts$comparison == "K-HO_minus_I-HO"]
    ),
    upper = c(
      0,
      architecture_contrasts$participant_bootstrap_total_q975[architecture_contrasts$comparison == "K-HO_minus_G-HO"],
      architecture_contrasts$participant_bootstrap_total_q975[architecture_contrasts$comparison == "K-HO_minus_I-HO"]
    )
  ),
  architecture_models %>%
    filter(model == "K-HO") %>%
    transmute(
      panel = "Successive-choice contribution", label = "kappa",
      estimate = kappa_mean, lower = kappa_q025, upper = kappa_q975
    )
)
write_csv(architecture_plot_data, "data/production_architecture_figure_data.csv")
architecture_plot_data <- read_csv("data/production_architecture_figure_data.csv", show_col_types = FALSE)

p_architecture <- architecture_plot_data %>%
  filter(panel == "Production architecture") %>%
  mutate(label = factor(label, levels = rev(c("Plan-guided", "Global", "Fully incremental")))) %>%
  ggplot(aes(x = estimate, y = label)) +
  geom_vline(xintercept = 0, colour = CSP_COLORS[["grey"]], linewidth = .6) +
  geom_errorbar(aes(xmin = lower, xmax = upper), orientation = "y", width = .14, linewidth = .75) +
  geom_point(aes(colour = label), size = 3.7) +
  scale_colour_manual(
    values = c(
      "Plan-guided" = CSP_COLORS[["green"]],
      "Global" = CSP_COLORS[["text"]],
      "Fully incremental" = CSP_COLORS[["text"]]
    ),
    guide = "none"
  ) +
  scale_x_continuous(limits = c(-20, 680), breaks = seq(0, 600, 200)) +
  labs(x = "Predictive loss", y = NULL) +
  theme_csp() +
  theme(panel.grid.major.y = element_blank(), axis.ticks.y = element_blank())

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
  annotate("text", x = .4472, y = 1.22, label = expression(kappa == .447), size = 5) +
  scale_x_continuous(
    limits = c(-.12, 1.12), breaks = c(0, 1),
    labels = c("Global", "Fully\nincremental")
  ) +
  scale_y_continuous(NULL, breaks = NULL) +
  labs(x = "Successive-choice weight") +
  theme_csp() +
  theme(panel.grid = element_blank())

residual_examples <- ppc %>%
  filter(
    (summary_type == "Initial adjective" & combination == "dimension_color" &
       relevant_property == "both" & sharpness == "blurred" & category == "C") |
      (summary_type == "Response length" & combination == "color_form" &
         relevant_property == "second" & sharpness == "blurred" & category == "2") |
      (summary_type == "Exact response" & combination == "color_form" &
         relevant_property == "both" & sharpness == "blurred" & category == "FC")
  ) %>%
  mutate(
    outcome = case_when(
      summary_type == "Initial adjective" ~ "Colour initial\nSize-colour, both necessary",
      summary_type == "Response length" ~ "Two adjectives\nColour-form, form sufficient",
      TRUE ~ "Form-colour utterance\nColour-form, both necessary"
    ),
    outcome = factor(
      outcome,
      levels = c(
        "Colour initial\nSize-colour, both necessary",
        "Two adjectives\nColour-form, form sufficient",
        "Form-colour utterance\nColour-form, both necessary"
      )
    ),
    model = factor(model, levels = model_levels[-1]),
    residual_percentage_points = 100 * (predicted_mean - observed_proportion)
  ) %>%
  select(outcome, model, observed_proportion, predicted_mean, residual_percentage_points)
write_csv(residual_examples, "data/production_architecture_residual_examples.csv")
residual_examples <- read_csv("data/production_architecture_residual_examples.csv", show_col_types = FALSE) %>%
  mutate(
    outcome = factor(outcome, levels = levels(residual_examples$outcome)),
    model = factor(model, levels = model_levels[-1])
  )

p_residual_examples <- ggplot(
  residual_examples,
  aes(x = residual_percentage_points, y = outcome, colour = model, shape = model)
) +
  geom_vline(xintercept = 0, colour = CSP_COLORS[["grey"]], linewidth = .7) +
  geom_segment(
    aes(x = 0, xend = residual_percentage_points, yend = outcome),
    position = position_dodge(width = .55), linewidth = .55, alpha = .7
  ) +
  geom_point(position = position_dodge(width = .55), size = 3.2) +
  scale_colour_manual(values = model_palette[-1], name = NULL) +
  scale_shape_manual(values = model_shapes[-1], name = NULL) +
  scale_x_continuous(
    limits = c(-5.5, 5.5), breaks = seq(-5, 5, 2.5),
    labels = label_number(suffix = " pp")
  ) +
  labs(x = "Prediction minus observation", y = NULL) +
  theme_csp() +
  theme(
    legend.position = "top",
    panel.grid.major.y = element_blank(),
    axis.ticks.y = element_blank()
  )

figure_8 <- (p_architecture | p_kappa) / p_residual_examples +
  plot_layout(heights = c(.9, 1.2), widths = c(1.2, 1)) +
  plot_annotation(tag_levels = "A")
save_csp_pdf(figure_8, "figures/production_architecture_results.pdf", 9.4, 7.8)

# -----------------------------------------------------------------------------
# Figure 9: participant heterogeneity and its unresolved source.
# -----------------------------------------------------------------------------
parameter_path <- "data/production_participant_parameter_intervals.csv"
if (!file.exists(parameter_path)) {
  stop("Run analysis_revision/13_form_identification/v9_hierarchy/export_participant_figure_data.py first")
}
participant_parameters <- read_csv(parameter_path, show_col_types = FALSE) %>%
  mutate(
    parameter = recode(
      parameter,
      `Successive-choice contribution` = "Participant kappa",
      `Successive-choice kappa` = "Participant kappa",
      `Stable-order weighting` = "Stable-order weight",
      `Stable-order weight` = "Stable-order weight"
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

participant_raw <- read_csv(
  "../analysis_revision/12_deterministic_encoding/model_input_raw_observed_9100.csv",
  show_col_types = FALSE
)
participant_loo <- read_csv(
  "../analysis_revision/13_form_identification/v9_hierarchy/participant_hierarchy_pointwise_loo_v9.csv",
  show_col_types = FALSE
)

participant_trials <- participant_raw %>%
  left_join(participant_loo, by = "canonical_row_position") %>%
  mutate(
    response_length = nchar(annotation),
    initial = substr(annotation, 1, 1),
    canonical_order = vapply(
      strsplit(annotation, ""),
      function(value) all(diff(match(value, c("D", "C", "F"))) > 0),
      logical(1)
    ),
    gain_hk = HK - H0,
    gain_ho = HO - H0,
    hk_minus_ho = HK - HO
  )

participant_profiles <- participant_trials %>%
  group_by(id) %>%
  summarise(
    gain_participant_kappa = sum(gain_hk),
    gain_stable_order = sum(gain_ho),
    kappa_minus_order = sum(hk_minus_ho),
    one_adjective = mean(response_length == 1),
    three_adjectives = mean(response_length == 3),
    size_initial_multi = mean(initial[response_length > 1] == "D"),
    canonical_multi = mean(canonical_order[response_length > 1]),
    .groups = "drop"
  ) %>%
  mutate(
    pattern = case_when(
      kappa_minus_order <= quantile(kappa_minus_order, .15) ~ "Stable order favoured",
      kappa_minus_order >= quantile(kappa_minus_order, .85) ~ "Successive choice favoured",
      TRUE ~ "Middle 70%"
    )
  )
write_csv(participant_profiles, "data/production_participant_predictive_profiles.csv")
participant_profiles <- read_csv(
  "data/production_participant_predictive_profiles.csv",
  show_col_types = FALSE
)

profile_summary <- participant_profiles %>%
  select(pattern, one_adjective, three_adjectives, size_initial_multi, canonical_multi) %>%
  pivot_longer(-pattern, names_to = "measure", values_to = "value") %>%
  group_by(pattern, measure) %>%
  summarise(mean = mean(value), .groups = "drop") %>%
  mutate(
    pattern = factor(
      pattern,
      levels = c("Stable order favoured", "Middle 70%", "Successive choice favoured")
    ),
    measure = recode(
      measure,
      one_adjective = "One adjective",
      three_adjectives = "Three adjectives",
      size_initial_multi = "Size initial | multi-adjective",
      canonical_multi = "Canonical order | multi-adjective"
    )
  )
write_csv(profile_summary, "data/production_participant_pattern_summary.csv")
profile_summary <- read_csv("data/production_participant_pattern_summary.csv", show_col_types = FALSE) %>%
  mutate(pattern = factor(pattern, levels = levels(profile_summary$pattern)))

participant_profile_observations <- participant_profiles %>%
  select(id, pattern, one_adjective, three_adjectives, size_initial_multi, canonical_multi) %>%
  pivot_longer(
    c(one_adjective, three_adjectives, size_initial_multi, canonical_multi),
    names_to = "measure",
    values_to = "value"
  ) %>%
  mutate(
    pattern = factor(
      pattern,
      levels = c("Stable order favoured", "Middle 70%", "Successive choice favoured")
    ),
    measure = recode(
      measure,
      one_adjective = "One adjective",
      three_adjectives = "Three adjectives",
      size_initial_multi = "Size initial",
      canonical_multi = "Canonical order"
    )
  )
write_csv(
  participant_profile_observations,
  "data/production_participant_profile_observations.csv"
)
participant_profile_observations <- read_csv(
  "data/production_participant_profile_observations.csv",
  show_col_types = FALSE
) %>%
  mutate(
    pattern = factor(
      pattern,
      levels = c("Stable order favoured", "Middle 70%", "Successive choice favoured")
    )
  )

participant_residual_observations <- participant_trials %>%
  inner_join(participant_profiles %>% select(id, pattern), by = "id") %>%
  transmute(
    id,
    pattern,
    response_length = paste(response_length, "adjective"),
    initial = recode(initial, D = "Size initial", C = "Colour initial", F = "Form initial"),
    hk_minus_ho
  ) %>%
  pivot_longer(c(response_length, initial), names_to = "dimension", values_to = "cell") %>%
  group_by(id, pattern, dimension, cell) %>%
  summarise(participant_mean_difference = mean(hk_minus_ho), .groups = "drop")

participant_residuals <- participant_residual_observations %>%
  group_by(pattern, dimension, cell) %>%
  summarise(
    mean_log_score_difference = mean(participant_mean_difference),
    se = sd(participant_mean_difference) / sqrt(n()),
    lower = mean_log_score_difference - qt(.975, df = n() - 1) * se,
    upper = mean_log_score_difference + qt(.975, df = n() - 1) * se,
    participants = n(),
    .groups = "drop"
  ) %>%
  mutate(
    pattern = factor(
      pattern,
      levels = c("Stable order favoured", "Middle 70%", "Successive choice favoured")
    ),
    dimension = recode(dimension, response_length = "Utterance length", initial = "Initial adjective")
  )
write_csv(participant_residuals, "data/production_participant_residual_localisation.csv")
participant_residuals <- read_csv(
  "data/production_participant_residual_localisation.csv",
  show_col_types = FALSE
) %>%
  mutate(
    pattern = factor(pattern, levels = levels(participant_residuals$pattern)),
    dimension = factor(dimension, levels = c("Utterance length", "Initial adjective"))
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
  facet_wrap(~ parameter, scales = "free_y", ncol = 1) +
  labs(x = "Participant rank", y = "Posterior estimate") +
  theme_csp() +
  theme(
    panel.grid.minor = element_blank(),
    panel.spacing.y = unit(.65, "lines")
  )

p_participant_gains <- ggplot(
  participant_profiles %>%
    mutate(
      pattern_display = recode(
        pattern,
        `Stable order favoured` = "Stable order",
        `Middle 70%` = "Middle 70%",
        `Successive choice favoured` = "Successive choice"
      )
    ),
  aes(x = gain_participant_kappa, y = gain_stable_order)
) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", colour = CSP_COLORS[["grey"]]) +
  geom_hline(yintercept = 0, colour = CSP_COLORS[["grey"]], linewidth = .4) +
  geom_vline(xintercept = 0, colour = CSP_COLORS[["grey"]], linewidth = .4) +
  geom_point(aes(colour = pattern_display), size = 2.5, alpha = .8) +
  scale_colour_manual(
    values = c(
      "Stable order" = CSP_COLORS[["main"]],
      "Middle 70%" = CSP_COLORS[["grey"]],
      "Successive choice" = CSP_COLORS[["emphasis"]]
    ),
    name = NULL
  ) +
  labs(
    x = "Gain: successive choice",
    y = "Gain: stable order"
  ) +
  theme_csp() +
  theme(legend.position = "right") +
  guides(colour = guide_legend(ncol = 1, byrow = TRUE))

p_profile <- ggplot(
  participant_profile_observations,
  aes(x = value, y = measure, colour = pattern)
) +
  geom_point(
    position = position_jitter(height = .12, width = 0),
    size = 1.35,
    alpha = .22,
    show.legend = FALSE
  ) +
  geom_point(
    data = profile_summary %>%
    mutate(
      measure = recode(
        measure,
        `Size initial | multi-adjective` = "Size initial",
        `Canonical order | multi-adjective` = "Canonical order"
      )
    ),
    aes(x = mean, y = measure, colour = pattern, shape = pattern),
    position = position_dodge(width = .55),
    size = 3.2,
    inherit.aes = FALSE
  ) +
  scale_colour_manual(
    values = c(
      "Stable order favoured" = CSP_COLORS[["main"]],
      "Middle 70%" = CSP_COLORS[["grey"]],
      "Successive choice favoured" = CSP_COLORS[["emphasis"]]
    ),
    name = NULL
  ) +
  scale_shape_manual(values = c(16, 17, 15), name = NULL) +
  scale_x_continuous(
    labels = percent_format(accuracy = 1), limits = c(0, 1), breaks = c(0, .5, 1)
  ) +
  labs(x = "Observed response proportion", y = NULL) +
  theme_csp() +
  theme(
    legend.position = "top",
    panel.grid.major.y = element_blank(),
    axis.ticks.y = element_blank(),
    axis.text.y = element_text(size = 12)
  ) +
  guides(
    colour = guide_legend(nrow = 2, byrow = TRUE),
    shape = guide_legend(nrow = 2, byrow = TRUE)
  )

p_residual_localisation <- ggplot(
  participant_residuals %>%
    mutate(
      pattern_display = recode(
        as.character(pattern),
        `Stable order favoured` = "Stable order",
        `Middle 70%` = "Middle 70%",
        `Successive choice favoured` = "Successive choice"
      ),
      pattern_display = factor(pattern_display, levels = c("Stable order", "Middle 70%", "Successive choice")),
      dimension = recode(
        as.character(dimension),
        `Utterance length` = "Length",
        `Initial adjective` = "Initial"
      ),
      cell = recode(
        cell,
        `1 adjective` = "1 adj.",
        `2 adjective` = "2 adj.",
        `3 adjective` = "3 adj.",
        `Colour initial` = "Colour",
        `Form initial` = "Form",
        `Size initial` = "Size"
      ),
      dimension = factor(dimension, levels = c("Length", "Initial")),
      response = paste(dimension, cell, sep = " - "),
      response = factor(
        response,
        levels = rev(c(
          "Length - 1 adj.", "Length - 2 adj.", "Length - 3 adj.",
          "Initial - Colour", "Initial - Form", "Initial - Size"
        ))
      )
    ),
  aes(
    x = mean_log_score_difference,
    y = response,
    colour = pattern_display,
    shape = pattern_display
  )
) +
  geom_vline(xintercept = 0, colour = CSP_COLORS[["grey"]], linewidth = .7) +
  geom_errorbar(
    aes(xmin = lower, xmax = upper),
    orientation = "y",
    position = position_dodge(width = .55),
    width = .12,
    linewidth = .55
  ) +
  geom_point(position = position_dodge(width = .55), size = 2.7) +
  scale_colour_manual(
    values = c(
      "Stable order" = CSP_COLORS[["main"]],
      "Middle 70%" = CSP_COLORS[["grey"]],
      "Successive choice" = CSP_COLORS[["emphasis"]]
    ),
    name = NULL
  ) +
  scale_shape_manual(values = c(16, 17, 15), name = NULL) +
  scale_x_continuous(
    breaks = c(-.5, 0, .25),
    limits = c(-.65, .35)
  ) +
  labs(
    x = "Mean log-score difference\n(successive choice - stable order)",
    y = NULL
  ) +
  theme_csp() +
  theme(
    panel.grid.major.y = element_blank(),
    axis.ticks.y = element_blank(),
    axis.text.y = element_text(size = 11),
    legend.position = "none"
  )

figure_9 <- (p_parameter_intervals | p_participant_gains) /
  (p_profile | p_residual_localisation) +
  plot_layout(heights = c(1.1, .9), widths = c(.95, 1.25)) +
  plot_annotation(tag_levels = "A")
save_csp_pdf(figure_9, "figures/production_participant_variation.pdf", 9.4, 8.6)

cat("Wrote Figures 7--9 and their supporting CSVs.\n")

# CSP-style figures for the production and model-comparison revisions.
# Run from paper/: Rscript scripts/plot_revision_figures.R

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

participant_interval <- function(data, value_name) {
  value <- data[[value_name]]
  value <- value[is.finite(value)]
  n <- length(value)
  estimate <- mean(value)
  se <- if (n > 1) stats::sd(value) / sqrt(n) else 0
  critical <- if (n > 1) stats::qt(.975, df = n - 1) else 0
  tibble(
    estimate = estimate,
    lower = estimate - critical * se,
    upper = estimate + critical * se,
    n_participants = n
  )
}

summarise_participant_rates <- function(data, grouping) {
  data %>%
    group_by(across(all_of(grouping))) %>%
    group_modify(~ participant_interval(.x, "proportion")) %>%
    ungroup()
}

family_labels <- c(
  dimension_color = "Size-colour",
  dimension_form = "Size-form",
  color_form = "Colour-form"
)

context_levels <- c(
  "Size sufficient",
  "Colour sufficient",
  "Form sufficient",
  "Both necessary"
)

set_levels <- c(
  "Size",
  "Colour",
  "Form",
  "Size + colour",
  "Size + form",
  "Colour + form",
  "All three"
)

set_labels <- c(
  C = "Colour",
  D = "Size",
  F = "Form",
  CD = "Size + colour",
  DF = "Size + form",
  CF = "Colour + form",
  CDF = "All three"
)

initial_labels <- c(D = "Size", C = "Colour", F = "Form")

production <- read_csv(
  "../analysis_revision/12_deterministic_encoding/model_input_raw_observed_9100.csv",
  show_col_types = FALSE
) %>%
  transmute(
    canonical_row_position,
    id,
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
    context_order = case_when(
      relevant_property == "first" ~ 1L,
      relevant_property == "both" ~ 2L,
      TRUE ~ 3L
    ),
    discriminability = recode(sharpness, blurred = "Low", sharp = "High"),
    annotation,
    adjective_set_code = vapply(
      strsplit(annotation, ""),
      function(value) paste(sort(value), collapse = ""),
      character(1)
    ),
    response_length = nchar(annotation),
    initial = recode(substr(annotation, 1, 1), !!!initial_labels),
    target_set = recode(
      combination,
      dimension_color = "CD",
      dimension_form = "DF",
      color_form = "CF"
    ),
    canonical_first = case_when(
      combination == "dimension_color" & annotation == "DC" ~ 1,
      combination == "dimension_color" & annotation == "CD" ~ 0,
      combination == "dimension_form" & annotation == "DF" ~ 1,
      combination == "dimension_form" & annotation == "FD" ~ 0,
      combination == "color_form" & annotation == "CF" ~ 1,
      combination == "color_form" & annotation == "FC" ~ 0,
      TRUE ~ NA_real_
    )
  ) %>%
  mutate(
    family = factor(family, levels = unname(family_labels)),
    context = factor(context, levels = context_levels),
    discriminability = factor(discriminability, levels = c("Low", "High")),
    adjective_set = factor(unname(set_labels[adjective_set_code]), levels = set_levels),
    response_length = factor(
      response_length,
      levels = 1:3,
      labels = c("One adjective", "Two adjectives", "Three adjectives")
    ),
    initial = factor(initial, levels = c("Size", "Colour", "Form"))
  )

stopifnot(nrow(production) == 9100L)

# -----------------------------------------------------------------------------
# Behavioural production figures.
# Figure 5 restores the v0.3 overview logic with the current canonical data.
# Figure 6 shows the two recodings used in the targeted hypothesis tests.
# -----------------------------------------------------------------------------
response_labels <- c(
  D = "Size",
  DC = "Size-colour",
  DCF = "Size-colour-form",
  DF = "Size-form",
  DFC = "Size-form-colour",
  C = "Colour",
  CD = "Colour-size",
  CDF = "Colour-size-form",
  CF = "Colour-form",
  CFD = "Colour-form-size",
  F = "Form",
  FD = "Form-size",
  FDC = "Form-size-colour",
  FC = "Form-colour",
  FCD = "Form-colour-size"
)

size_colour_distribution_participant <- production %>%
  filter(family == "Size-colour") %>%
  count(id, context, context_order, discriminability, annotation, name = "n") %>%
  group_by(id, context, context_order, discriminability) %>%
  complete(annotation = names(response_labels), fill = list(n = 0L)) %>%
  mutate(proportion = n / sum(n)) %>%
  ungroup()

size_colour_distribution_summary <- summarise_participant_rates(
  size_colour_distribution_participant,
  c("context", "context_order", "discriminability", "annotation")
) %>%
  mutate(response = unname(response_labels[annotation])) %>%
  arrange(context_order, discriminability, match(annotation, names(response_labels)))

write_csv(
  size_colour_distribution_summary,
  "data/production_size_colour_distribution_summary.csv"
)
size_colour_distribution_summary <- read_csv(
  "data/production_size_colour_distribution_summary.csv",
  show_col_types = FALSE
) %>%
  mutate(
    context = factor(
      context,
      levels = c("Size sufficient", "Both necessary", "Colour sufficient")
    ),
    discriminability = factor(discriminability, levels = c("Low", "High")),
    annotation = factor(annotation, levels = names(response_labels)),
    response = factor(response, levels = unname(response_labels))
  )

displayed_responses <- size_colour_distribution_summary %>%
  group_by(response) %>%
  summarise(maximum = max(estimate), .groups = "drop") %>%
  filter(maximum >= .02) %>%
  pull(response)

discriminability_palette <- c(
  "Low" = CSP_COLORS[["accent_dark"]],
  "High" = CSP_COLORS[["main"]]
)

p_distribution <- size_colour_distribution_summary %>%
  filter(response %in% displayed_responses) %>%
  mutate(response = factor(response, levels = rev(intersect(unname(response_labels), displayed_responses)))) %>%
  ggplot(aes(x = estimate, y = response, fill = discriminability)) +
  geom_col(
    position = position_dodge(width = .72),
    width = .62,
    alpha = .9
  ) +
  geom_errorbar(
    aes(xmin = pmax(0, lower), xmax = pmin(1, upper)),
    orientation = "y",
    position = position_dodge(width = .72),
    width = .24,
    linewidth = .55
  ) +
  facet_wrap(~ context, nrow = 1) +
  scale_fill_manual(values = discriminability_palette, name = "Size discriminability") +
  scale_x_continuous(
    labels = c("", "25%", "50%", "75%"),
    limits = c(0, .75),
    breaks = seq(0, .75, .25),
    expand = expansion(mult = c(0, .02))
  ) +
  labs(x = "Proportion of responses", y = "Produced adjective string") +
  theme_csp() +
  theme(
    legend.position = "top",
    panel.grid.major.y = element_blank(),
    strip.text = element_text(size = 14, face = "plain")
  )

save_csp_pdf(
  p_distribution,
  "figures/production_distribution.pdf",
  8.4,
  5.4
)

hypothesis_recodings <- read_csv(
  "../analysis/results_behavioral_bayes/csp_711af39/summary/behavioral_bayesian_predictions.csv",
  show_col_types = FALSE
) %>%
  filter(study %in% c("production_overinformative", "production_size_initial")) %>%
  transmute(
    outcome = recode(
      study,
      production_overinformative = "Over-informative responses",
      production_size_initial = "Size-initial responses"
    ),
    context = recode(
      context,
      size_sufficient = "Size sufficient",
      both_necessary = "Both necessary",
      colour_sufficient = "Colour sufficient"
    ),
    discriminability = recode(discriminability, low = "Low", high = "High"),
    observed_proportion = observed_mean,
    posterior_median = median,
    lower,
    upper,
    n_observations
  )

write_csv(
  hypothesis_recodings,
  "data/production_hypothesis_recodings_summary.csv"
)
hypothesis_recodings <- read_csv(
  "data/production_hypothesis_recodings_summary.csv",
  show_col_types = FALSE
) %>%
  mutate(
    context = factor(
      context,
      levels = c("Size sufficient", "Both necessary", "Colour sufficient")
    ),
    discriminability = factor(discriminability, levels = c("Low", "High"))
  )

recode_panel <- function(data, y_label, tag) {
  ggplot(data, aes(x = context, y = posterior_median, fill = discriminability)) +
    geom_col(
      position = position_dodge(width = .72),
      width = .62,
      alpha = .9
    ) +
    geom_errorbar(
      aes(ymin = lower, ymax = upper),
      position = position_dodge(width = .72),
      width = .18,
      linewidth = .65
    ) +
    scale_fill_manual(values = discriminability_palette, name = "Size discriminability") +
    scale_x_discrete(
      labels = c(
        "Size sufficient" = "Size\nsufficient",
        "Both necessary" = "Both\nnecessary",
        "Colour sufficient" = "Colour\nsufficient"
      )
    ) +
    scale_y_continuous(
      labels = percent_format(accuracy = 1),
      limits = c(0, 1),
      breaks = seq(0, 1, .25),
      expand = expansion(mult = c(0, .02))
    ) +
    labs(x = "Referential context", y = y_label, tag = tag) +
    theme_csp() +
    theme(
      legend.position = "top",
      panel.grid.major.x = element_blank()
    )
}

p_overinformative <- recode_panel(
  hypothesis_recodings %>% filter(outcome == "Over-informative responses"),
  "Over-informative responses",
  "A"
)
p_size_initial <- recode_panel(
  hypothesis_recodings %>% filter(outcome == "Size-initial responses"),
  "Size-initial responses",
  "B"
)

production_hypothesis_recodings <- (p_overinformative | p_size_initial) +
  plot_layout(guides = "collect") &
  theme(legend.position = "top")

save_csp_pdf(
  production_hypothesis_recodings,
  "figures/production_hypothesis_recodings.pdf",
  8.4,
  4.7
)

# -----------------------------------------------------------------------------
# Study 1a/1b replication figure, redrawn at the Figure 3 typography standard.
# -----------------------------------------------------------------------------
slider_replication <- read_csv(
  "data/slider_replication_figure_summary.csv",
  show_col_types = FALSE
) %>%
  mutate(
    context = factor(
      context,
      levels = c("Size sufficient", "Both necessary", "Colour sufficient")
    ),
    study = factor(study, levels = c("Study 1a", "Study 1b"))
  )

p_replication <- ggplot(
  slider_replication,
  aes(x = context, y = estimate, colour = study, group = study)
) +
  geom_hline(yintercept = 0, colour = CSP_COLORS[["grey"]], linewidth = .6) +
  geom_line(linewidth = .8, position = position_dodge(width = .10)) +
  geom_errorbar(
    aes(ymin = lower, ymax = upper),
    width = .10,
    linewidth = .65,
    position = position_dodge(width = .10)
  ) +
  geom_point(size = 3.2, position = position_dodge(width = .10)) +
  scale_colour_manual(
    values = c("Study 1a" = CSP_COLORS[["main"]], "Study 1b" = CSP_COLORS[["emphasis"]]),
    name = NULL
  ) +
  scale_y_continuous(limits = c(-.1, .4), breaks = seq(-.1, .4, .1)) +
  labs(x = "Referential context", y = "Centred size-first rating") +
  theme_csp() +
  theme(legend.position = "top", panel.grid.minor = element_blank())

save_csp_pdf(p_replication, "figures/study1_slider_replication.pdf", 7.0, 4.5)

# -----------------------------------------------------------------------------
# Section 5: population architecture and successive-choice contribution.
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
      panel = "Successive-choice contribution",
      label = "kappa",
      estimate = kappa_mean,
      lower = kappa_q025,
      upper = kappa_q975
    )
)

write_csv(architecture_plot_data, "data/production_architecture_figure_data.csv")
architecture_plot_data <- read_csv(
  "data/production_architecture_figure_data.csv",
  show_col_types = FALSE
)

p_architecture <- architecture_plot_data %>%
  filter(panel == "Production architecture") %>%
  mutate(label = factor(label, levels = rev(c("Plan-guided", "Global", "Fully incremental")))) %>%
  ggplot(aes(x = estimate, y = label)) +
  geom_vline(xintercept = 0, colour = CSP_COLORS[["grey"]], linewidth = .6) +
  geom_errorbar(aes(xmin = lower, xmax = upper), orientation = "y", width = .14, linewidth = .75) +
  geom_point(
    aes(colour = label),
    size = 3.7
  ) +
  scale_colour_manual(
    values = c(
      "Plan-guided" = CSP_COLORS[["green"]],
      "Global" = CSP_COLORS[["text"]],
      "Fully incremental" = CSP_COLORS[["text"]]
    ),
    guide = "none"
  ) +
  scale_x_continuous(limits = c(-20, 680), breaks = seq(0, 600, 200)) +
  labs(x = "Held-out predictive loss (lower is better)", y = NULL, tag = "A") +
  theme_csp() +
  theme(
    panel.grid.major.y = element_blank(),
    axis.ticks.y = element_blank()
  )

p_kappa <- architecture_plot_data %>%
  filter(panel == "Successive-choice contribution") %>%
  ggplot(aes(x = estimate, y = 1)) +
  geom_segment(
    aes(x = 0, xend = 1, yend = 1),
    linewidth = 2.5,
    colour = CSP_COLORS[["grey"]],
    lineend = "round"
  ) +
  geom_errorbar(
    aes(xmin = lower, xmax = upper),
    orientation = "y",
    width = .15,
    linewidth = 1.1,
    colour = CSP_COLORS[["green"]]
  ) +
  geom_point(size = 4.5, colour = CSP_COLORS[["green"]]) +
  annotate("text", x = .4472, y = 1.22, label = expression(kappa == .447), size = 5) +
  scale_x_continuous(
    limits = c(-.12, 1.12),
    breaks = c(0, 1),
    labels = c("Global", "Fully incremental")
  ) +
  scale_y_continuous(NULL, breaks = NULL) +
  labs(x = "Contribution of successive adjective choices", tag = "B") +
  theme_csp() +
  theme(panel.grid = element_blank())

architecture_figure <- p_architecture / p_kappa +
  plot_layout(heights = c(1.05, .82))

# The final architecture comparison is exported by
# scripts/plot_architecture_diagnostics.R alongside its endpoint diagnostics.

# -----------------------------------------------------------------------------
# Section 5: posterior-predictive interpretation of plan-guided production.
# -----------------------------------------------------------------------------
production_ppc <- read_csv(
  "data/production_v10_ppc_summary.csv",
  show_col_types = FALSE
)

production_ppc_raw <- read_csv(
  "../analysis_revision/12_deterministic_encoding/model_input_raw_observed_9100.csv",
  show_col_types = FALSE
)

bootstrap_empirical_rate <- function(data, grouping, reps = 4000) {
  participant_cells <- data %>%
    group_by(across(all_of(c(grouping, "id")))) %>%
    summarise(successes = sum(success), trials = n(), .groups = "drop")
  participant_cells %>%
    group_by(across(all_of(grouping))) %>%
    group_modify(~ {
      participant_rows <- seq_len(nrow(.x))
      bootstrap_rate <- replicate(reps, {
        sampled_rows <- sample(participant_rows, length(participant_rows), replace = TRUE)
        sum(.x$successes[sampled_rows]) / sum(.x$trials[sampled_rows])
      })
      tibble(
        observed_mean = sum(.x$successes) / sum(.x$trials),
        observed_lower = quantile(bootstrap_rate, .025),
        observed_upper = quantile(bootstrap_rate, .975)
      )
    }) %>%
    ungroup()
}

exact_categories <- sort(unique(production_ppc_raw$annotation))
initial_categories <- c("D", "C", "F")
set.seed(20260812)
empirical_exact <- tidyr::crossing(
  production_ppc_raw,
  category = exact_categories
) %>%
  mutate(success = annotation == category) %>%
  bootstrap_empirical_rate(
    c("combination", "relevant_property", "sharpness", "category")
  ) %>%
  mutate(summary_type = "Exact response")

set.seed(20260812)
empirical_initial <- tidyr::crossing(
  production_ppc_raw,
  category = initial_categories
) %>%
  mutate(success = substr(annotation, 1, 1) == category) %>%
  bootstrap_empirical_rate(
    c("combination", "relevant_property", "sharpness", "category")
  ) %>%
  mutate(summary_type = "Initial adjective")

empirical_ppc_intervals <- bind_rows(empirical_exact, empirical_initial)
write_csv(empirical_ppc_intervals, "data/production_v10_ppc_empirical_intervals.csv")
empirical_ppc_intervals <- read_csv(
  "data/production_v10_ppc_empirical_intervals.csv",
  show_col_types = FALSE
)

ppc_join_keys <- c(
  "combination", "relevant_property", "sharpness", "summary_type", "category"
)
production_ppc <- production_ppc %>%
  left_join(empirical_ppc_intervals, by = ppc_join_keys)

ppc_zoom <- production_ppc %>%
  filter(
    summary_type == "Initial adjective",
    combination == "dimension_color",
    relevant_property == "both",
    sharpness == "blurred"
  ) %>%
  transmute(
    source = model,
    category = recode(category, D = "Size", C = "Colour", F = "Form"),
    mean = predicted_mean,
    lower = predicted_q025,
    upper = predicted_q975
  ) %>%
  bind_rows(
    production_ppc %>%
      filter(
        model == "Plan-guided",
        summary_type == "Initial adjective",
        combination == "dimension_color",
        relevant_property == "both",
        sharpness == "blurred"
      ) %>%
      transmute(
        source = "Observed",
        category = recode(category, D = "Size", C = "Colour", F = "Form"),
        mean = observed_mean,
        lower = observed_lower,
        upper = observed_upper
      )
  ) %>%
  mutate(
    source = factor(
      source,
      levels = c("Observed", "Global", "Plan-guided", "Fully incremental")
    ),
    category = factor(category, levels = c("Size", "Colour", "Form"))
  )
write_csv(ppc_zoom, "data/production_v10_ppc_zoom.csv")
ppc_zoom <- read_csv("data/production_v10_ppc_zoom.csv", show_col_types = FALSE) %>%
  mutate(
    source = factor(
      source,
      levels = c("Observed", "Global", "Plan-guided", "Fully incremental")
    ),
    category = factor(category, levels = c("Size", "Colour", "Form"))
  )

source_palette <- c(
  "Observed" = CSP_COLORS[["text"]],
  "Global" = CSP_COLORS[["main"]],
  "Plan-guided" = CSP_COLORS[["green"]],
  "Fully incremental" = CSP_COLORS[["emphasis"]]
)
source_shapes <- c("Observed" = 21, "Global" = 16, "Plan-guided" = 17, "Fully incremental" = 15)

p_ppc_zoom <- ggplot(
  ppc_zoom,
  aes(x = category, y = mean, colour = source, shape = source, group = source)
) +
  geom_errorbar(
    aes(ymin = lower, ymax = upper),
    position = position_dodge(width = .62),
    width = .13,
    linewidth = .55
  ) +
  geom_point(position = position_dodge(width = .62), size = 3.1, stroke = 1) +
  scale_colour_manual(values = source_palette, name = NULL) +
  scale_shape_manual(values = source_shapes, name = NULL) +
  scale_y_continuous(
    labels = percent_format(accuracy = 1),
    limits = c(0, .58),
    breaks = seq(0, .5, .1),
    expand = expansion(mult = c(0, .02))
  ) +
  labs(x = "Initial adjective", y = "Proportion") +
  theme_csp() +
  theme(
    legend.position = "bottom",
    panel.grid.major.x = element_blank(),
    plot.margin = margin(18, 6, 6, 12)
  ) +
  guides(colour = guide_legend(nrow = 2, byrow = TRUE))

response_order <- c(
  "D", "DC", "DCF", "DF", "DFC",
  "C", "CD", "CDF", "CF", "CFD",
  "F", "FD", "FDC", "FC", "FCD"
)
ppc_size_colour <- production_ppc %>%
  filter(
    model == "Plan-guided",
    summary_type == "Exact response",
    combination == "dimension_color"
  ) %>%
  group_by(category) %>%
  mutate(display_response = max(c(observed_mean, predicted_mean)) >= .05) %>%
  ungroup() %>%
  filter(display_response) %>%
  mutate(
    category = recode(category, D = "S", DC = "SC", DCF = "SCF", DF = "SF", DFC = "SFC"),
    category = factor(
      category,
      levels = rev(recode(response_order, D = "S", DC = "SC", DCF = "SCF", DF = "SF", DFC = "SFC"))
    ),
    context = recode(
      relevant_property,
      first = "Size sufficient",
      both = "Both necessary",
      second = "Colour sufficient"
    ),
    context = factor(context, levels = c("Size sufficient", "Both necessary", "Colour sufficient")),
    discriminability = recode(sharpness, blurred = "Low", sharp = "High")
  )
write_csv(ppc_size_colour, "data/production_v10_ppc_size_colour.csv")
ppc_size_colour <- read_csv(
  "data/production_v10_ppc_size_colour.csv",
  show_col_types = FALSE
) %>%
  mutate(
    category = factor(category, levels = levels(ppc_size_colour$category)),
    context = factor(context, levels = c("Size sufficient", "Both necessary", "Colour sufficient"))
  )

p_ppc_distribution <- ggplot(ppc_size_colour, aes(y = category)) +
  geom_segment(
    aes(x = observed_mean, xend = predicted_mean, yend = category),
    colour = CSP_COLORS[["grey"]],
    linewidth = .8
  ) +
  geom_errorbar(
    aes(xmin = predicted_q025, xmax = predicted_q975, x = predicted_mean),
    orientation = "y",
    width = .18,
    linewidth = .55,
    colour = CSP_COLORS[["green"]]
  ) +
  geom_point(
    aes(x = observed_mean, colour = "Observed", shape = "Observed"),
    size = 2.6,
    stroke = 1
  ) +
  geom_point(
    aes(x = predicted_mean, colour = "Plan-guided", shape = "Plan-guided"),
    size = 2.6
  ) +
  facet_grid(discriminability ~ context) +
  scale_colour_manual(
    values = c("Observed" = CSP_COLORS[["text"]], "Plan-guided" = CSP_COLORS[["green"]]),
    name = NULL
  ) +
  scale_shape_manual(values = c("Observed" = 21, "Plan-guided" = 17), name = NULL) +
  scale_x_continuous(
    labels = percent_format(accuracy = 1),
    limits = c(0, .85),
    breaks = seq(0, .8, .2),
    expand = expansion(mult = c(0, .02))
  ) +
  labs(x = "Response proportion", y = "Utterance") +
  theme_csp() +
  theme(
    legend.position = "top",
    panel.grid.major.y = element_blank(),
    axis.ticks.y = element_blank(),
    panel.spacing = unit(.7, "lines")
  )

ppc_correlation <- production_ppc %>%
  filter(model == "Plan-guided", summary_type == "Exact response") %>%
  mutate(
    family = recode(
      combination,
      dimension_color = "Size-colour",
      dimension_form = "Size-form",
      color_form = "Colour-form"
    )
  )
correlation_stats <- ppc_correlation %>%
  summarise(
    n_cells = n(),
    pearson_r = cor(observed_mean, predicted_mean),
    r_squared = pearson_r^2,
    rmse = sqrt(mean((observed_mean - predicted_mean)^2)),
    mae = mean(abs(observed_mean - predicted_mean))
  )
write_csv(correlation_stats, "data/production_v10_ppc_correlation_stats.csv")
correlation_stats <- read_csv(
  "data/production_v10_ppc_correlation_stats.csv",
  show_col_types = FALSE
)

p_ppc_correlation <- ggplot(
  ppc_correlation,
  aes(x = observed_mean, y = predicted_mean, colour = family)
) +
  geom_abline(
    intercept = 0,
    slope = 1,
    linetype = "dashed",
    colour = CSP_COLORS[["grey"]],
    linewidth = .7
  ) +
  geom_point(size = 2.0, alpha = .68) +
  scale_colour_manual(
    values = c(
      "Size-colour" = CSP_COLORS[["main"]],
      "Size-form" = CSP_COLORS[["gold_dark"]],
      "Colour-form" = CSP_COLORS[["accent_dark"]]
    ),
    name = NULL
  ) +
  scale_x_continuous(labels = percent_format(accuracy = 1), limits = c(0, .9)) +
  scale_y_continuous(labels = percent_format(accuracy = 1), limits = c(0, .9)) +
  annotate(
    "text",
    x = .03,
    y = .87,
    hjust = 0,
    vjust = 1,
    size = 4.3,
    colour = CSP_COLORS[["text"]],
    label = sprintf(
      "r = %.3f\nR² = %.3f\nRMSE = %.3f",
      correlation_stats$pearson_r,
      correlation_stats$r_squared,
      correlation_stats$rmse
    )
  ) +
  labs(x = "Observed proportion", y = "Predicted proportion") +
  coord_fixed() +
  theme_csp() +
  theme(
    legend.position = "bottom",
    plot.margin = margin(18, 6, 6, 12)
  ) +
  guides(colour = guide_legend(nrow = 2, byrow = TRUE))

ppc_layout <- "
AC
BB
"
production_ppc_figure <- p_ppc_zoom + p_ppc_distribution + p_ppc_correlation +
  plot_layout(design = ppc_layout, heights = c(.86, 1.55)) +
  plot_annotation(tag_levels = "A")

save_csp_pdf(
  production_ppc_figure,
  "figures/production_architecture_ppc.pdf",
  8.4,
  9.2
)

cat("Wrote revised CSP figures and their supporting summary CSVs.\n")

# The architecture diagnostics replace the earlier exploratory Figure 7/8
# layouts and add the participant-variation Figure 9.
source("scripts/plot_architecture_diagnostics.R")

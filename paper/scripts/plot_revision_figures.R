# Behavioural production figures from the manuscript summaries.
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
source("scripts/plot_slider_interactions.R")

# Current model-comparison figures read the frozen summaries directly.
source("scripts/plot_architecture_diagnostics.R")

cat("Wrote revised CSP figures.\n")

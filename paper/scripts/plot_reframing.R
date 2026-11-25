# Overview and zoom-in figures of the matched model contrasts (minimal reframing).
# Run from paper/ after analysis/reframing_contrasts.py has written
# data/reframing_contrasts.csv. Every value is model A minus model B.

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
})
source("scripts/csp_figure_style.R")

measures <- c("Production: which adjectives", "Production: order",
              "Ratings: canonical preference", "Ratings: context gradient",
              "Ratings: R2")
measure_labels <- c("Production: which\nadjectives", "Production:\norder",
                    "Ratings: preference\nfor conventional order", "Ratings:\ncontext gradient", "Ratings:\nvariance explained")
direction_levels <- c("A better", "no credible difference", "B better")
direction_fill <- c("A better" = CSP_COLORS[["main_light"]],
                    "no credible difference" = "white",
                    "B better" = CSP_COLORS[["emphasis_light"]])
direction_colour <- c("A better" = CSP_COLORS[["navy"]],
                      "no credible difference" = CSP_COLORS[["text"]],
                      "B better" = CSP_COLORS[["dark_red"]])
architectures <- c("Global", "Intermediate", "Fully incremental")

# Row labels of the overview: the claim, then the two models that are compared.
rows <- tribble(
  ~contrast, ~architecture, ~row,
  "Stable order term", "Intermediate",
  "Conventionalised efficiency\nA: with stable order term\nB: without",
  "Sequential context updating", "Intermediate",
  "Hierarchical composition\nA: sequential context updating\nB: context-fixed",
  "Informativeness", "Intermediate",
  "Discriminatory strength\nA: with informativeness\nB: without",
  "Salience", "Intermediate",
  "Availability\nA: with salience\nB: without"
)

contrasts <- read_csv("data/reframing_contrasts.csv", show_col_types = FALSE) %>%
  mutate(
    rating = grepl("^Ratings", measure),
    label = ifelse(rating,
                   sprintf("%+.2f\n[%.2f, %.2f]", estimate, lower, upper),
                   sprintf("%+.0f\n[%.0f, %.0f]", estimate, lower, upper)),
    measure = factor(measure, levels = measures, labels = measure_labels),
    credible = factor(credible, levels = direction_levels)
  )

# ---- Overview ---------------------------------------------------------------
# The overview and the zoom-ins show the rating prediction under a linear link.
shown <- c(1, 2, 5)
overview <- contrasts %>%
  filter(measure %in% measure_labels[shown]) %>%
  mutate(measure = droplevels(measure)) %>%
  inner_join(rows, by = c("contrast", "architecture")) %>%
  mutate(row = factor(row, levels = rev(rows$row)))

overview_plot <- ggplot(overview, aes(x = measure, y = row)) +
  geom_tile(aes(fill = credible), colour = CSP_COLORS[["grey"]], linewidth = .5,
            width = .96, height = .92) +
  geom_text(aes(label = label), size = 4.2, lineheight = .95, colour = CSP_COLORS[["text"]]) +
  scale_x_discrete(position = "top") +
  scale_fill_manual(values = direction_fill, drop = FALSE, name = "A minus B") +
  labs(x = NULL, y = NULL) +
  theme_csp() +
  theme(
    panel.grid = element_blank(),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    axis.ticks = element_blank(),
    axis.text.x = element_text(size = 13, face = "bold", lineheight = .95),
    axis.text.y = element_text(size = 12, hjust = 0, lineheight = .95),
    legend.position = "bottom",
    legend.text = element_text(size = 12),
    legend.title = element_text(size = 12)
  )
save_csp_pdf(overview_plot, "figures/reframing_overview.pdf", width = 9.5, height = 5.2)

# ---- Zoom-ins ---------------------------------------------------------------
zoom_plot <- function(data, y_levels) {
  data <- mutate(data, architecture = factor(architecture, levels = rev(y_levels)))
  ggplot(data, aes(x = estimate, y = architecture, colour = credible)) +
    geom_vline(xintercept = 0, colour = CSP_COLORS[["text"]], linewidth = .4) +
    geom_errorbar(aes(xmin = lower, xmax = upper), width = 0, linewidth = 1.1,
                  orientation = "y") +
    geom_point(size = 3.2) +
    facet_grid(panel ~ measure, scales = "free_x", switch = "y") +
    scale_colour_manual(values = direction_colour, name = "A minus B") +
    scale_x_continuous(expand = expansion(mult = .12), n.breaks = 3) +
    labs(x = "A minus B (production: ELPD; ratings: difference in R2)", y = NULL) +
    theme_csp() +
    theme(
      strip.placement = "outside",
      strip.text.x = element_text(size = 12, lineheight = .95),
      strip.text.y.left = element_text(size = 12, angle = 0, hjust = 1, lineheight = .95),
      axis.text.x = element_text(size = 11),
      axis.text.y = element_text(size = 12),
      axis.title.x = element_text(size = 13),
      panel.spacing.x = unit(1.6, "lines"),
      legend.position = "bottom",
      legend.text = element_text(size = 12),
      legend.title = element_text(size = 12)
    )
}

zoom <- function(panels, keep_measures, y_levels, filename, width, height) {
  data <- contrasts %>%
    inner_join(panels, by = "contrast") %>%
    filter(measure %in% measure_labels[keep_measures]) %>%
    mutate(panel = factor(panel, levels = panels$panel), measure = droplevels(measure))
  save_csp_pdf(zoom_plot(data, y_levels), filename, width = width, height = height)
}

zoom(tribble(~contrast, ~panel,
             "Stable order term", "A: with stable order term\nB: without"),
     shown, architectures, "figures/reframing_zoom_lexical.pdf", 10.5, 3.3)

zoom(tribble(~contrast, ~panel,
             "Sequential context updating",
             "With stable order term\nA: updating\nB: context-fixed",
             "Sequential context updating, no stable order term",
             "Without stable order term\nA: updating\nB: context-fixed"),
     shown, architectures, "figures/reframing_zoom_composition.pdf", 10.5, 4.8)

zoom(tribble(~contrast, ~panel,
             "Informativeness", "A: with informativeness\nB: without"),
     shown, architectures, "figures/reframing_zoom_incremental.pdf", 10.5, 3.3)

zoom(tribble(~contrast, ~panel,
             "Salience", "A: with salience\nB: without"),
     shown, architectures, "figures/reframing_zoom_availability.pdf", 10.5, 3.3)

zoom(tribble(~contrast, ~panel,
             "Local competition (intermediate vs global)",
             "Evaluating each adjective as produced\nA: intermediate\nB: global",
             "Complete-utterance evaluation (intermediate vs fully incremental)",
             "Looking ahead to the complete utterance\nA: intermediate\nB: fully incremental"),
     shown, c("Context-fixed", "Updating"), "figures/reframing_zoom_architecture.pdf", 10.5, 4.2)

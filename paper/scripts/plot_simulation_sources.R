# Size-first advantage for the listener when the sources of the production model
# are switched on: composition (semantics), convention (stable order term) and
# availability (salience), for the three speakers, with all fifteen utterances as
# alternatives, averaged over the semantic grid (A) and at the production values (B). Informativeness is always on.
# Run from paper/. Data: data/simulation_sources.csv
# (simulation/run_convention_salience.py, context_summary_overall.csv).

suppressPackageStartupMessages({library(ggplot2); library(dplyr); library(readr)})
source("scripts/csp_figure_style.R")

config_levels <- c("Informativeness", "+ order term", "+ salience", "+ both")
arch_levels <- c("Global", "Intermediate", "Fully incremental")
arch_colours <- c("Global" = CSP_COLORS[["emphasis"]], "Intermediate" = CSP_COLORS[["green"]],
                  "Fully incremental" = CSP_COLORS[["gold_dark"]])

d <- read_csv("data/simulation_sources.csv", show_col_types = FALSE) %>%
  filter(metric == "listener_advantage", support == "fifteen") %>%
  mutate(
    config = case_when(order == "neutral" & salience == "none" ~ config_levels[1],
                       order == "stable" & salience == "none" ~ config_levels[2],
                       order == "neutral" & salience == "salient" ~ config_levels[3],
                       TRUE ~ config_levels[4]),
    config = factor(config, levels = config_levels),
    architecture = factor(c(global = "Global", intermediate = "Intermediate",
                            incremental = "Fully incremental")[architecture], levels = arch_levels),
    semantics = factor(semantics, levels = c("fixed", "updating"),
                       labels = c("Context-fixed", "Sequential context updating")),
    panel = factor(parameter_set, levels = c("grid", "production_constants"),
                   labels = c("A  Semantic parameter grid", "B  Values of the production models")),
    estimate = 100 * mean,
    x = as.integer(config) + (as.integer(architecture) - 2) * .22 +
      ifelse(semantics == "Context-fixed", -.05, .05))

p <- ggplot(d, aes(x = x, y = estimate, colour = architecture)) +
  geom_hline(yintercept = 0, colour = CSP_COLORS[["text"]], linewidth = .5) +
  geom_line(aes(group = interaction(config, architecture)), linewidth = .8) +
  geom_point(aes(shape = semantics), size = 3) +
  scale_x_continuous(breaks = seq_along(config_levels), labels = config_levels) +
  scale_colour_manual(values = arch_colours, name = NULL) +
  scale_shape_manual(values = c(16, 15), name = NULL) +
  facet_wrap(~panel, ncol = 2) +
  labs(x = NULL, y = "Size-first advantage\n(percentage points)") +
  theme_csp() +
  theme(legend.position = "bottom", legend.box = "vertical",
        panel.grid.major.x = element_blank(), panel.grid.minor.x = element_blank(),
        strip.text = element_text(hjust = 0, face = "bold"))
save_csp_pdf(p, "figures/simulation_sources.pdf", width = 10, height = 4.6)

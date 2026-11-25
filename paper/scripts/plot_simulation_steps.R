# Simulation, step by step: replication of the earlier simulations, extension to
# more alternatives and the production speakers, extension to the other model
# components, and the semantic values of the production models.
# Panel A: size-first advantage for the listener without composition (context-fixed).
# Panel B: what composition adds (sequential context updating minus context-fixed).
# Run from paper/. Data: data/simulation_controlled_overall.csv (steps 1-3) and
# data/simulation_sources.csv (steps 4-7, simulation/run_convention_salience.py).

suppressPackageStartupMessages({library(ggplot2); library(dplyr); library(readr); library(tidyr)})
source("scripts/csp_figure_style.R")

arch_levels <- c("Global", "Intermediate", "Fully incremental")
arch_colours <- c("Global" = CSP_COLORS[["emphasis"]], "Intermediate" = CSP_COLORS[["green"]],
                  "Fully incremental" = CSP_COLORS[["gold_dark"]])
step_levels <- c("1  Two orders", "2  + single adjectives", "3  All fifteen utterances",
                 "4  + stable order term", "5  + salience", "6  + both",
                 "7  Production values")
rule_arch <- c(prefix_k0 = "Global", prefix_k05 = "Intermediate", prefix_k1 = "Fully incremental",
               terminal = "Global")

old <- read_csv("data/simulation_controlled_overall.csv", show_col_types = FALSE) %>%
  filter(metric == "listener_advantage", order == "neutral", parameter_set == "original_grid",
         (support == "two" & rule %in% c("terminal", "prefix_k05", "prefix_k1")) |
           (support %in% c("four", "fifteen") & rule %in% c("prefix_k0", "prefix_k05", "prefix_k1"))) %>%
  mutate(architecture = rule_arch[rule],
         step = step_levels[c(two = 1, four = 2, fifteen = 3)[support]])

new <- read_csv("data/simulation_sources.csv", show_col_types = FALSE) %>%
  filter(metric == "listener_advantage", support == "fifteen",
         (parameter_set == "grid" & !(order == "neutral" & salience == "none")) |
           (parameter_set == "production_constants" & order == "stable" & salience == "salient")) %>%
  mutate(architecture = c(global = "Global", intermediate = "Intermediate",
                          incremental = "Fully incremental")[architecture],
         step = case_when(parameter_set == "production_constants" ~ step_levels[7],
                          order == "stable" & salience == "none" ~ step_levels[4],
                          order == "neutral" ~ step_levels[5],
                          TRUE ~ step_levels[6]))

d <- bind_rows(old, new) %>%
  select(step, architecture, semantics, mean) %>%
  pivot_wider(names_from = semantics, values_from = mean) %>%
  transmute(step = factor(step, levels = rev(step_levels)),
            architecture = factor(architecture, levels = arch_levels),
            `A  Without composition` = 100 * fixed,
            `B  Added by composition` = 100 * (updating - fixed)) %>%
  pivot_longer(-c(step, architecture), names_to = "panel", values_to = "estimate")
stopifnot(nrow(d) == 7 * 3 * 2, !anyNA(d$estimate))
write_csv(d, "data/simulation_steps.csv")

p <- ggplot(d, aes(x = estimate, y = step, colour = architecture)) +
  annotate("rect", xmin = -Inf, xmax = Inf, ymin = 1.5, ymax = 4.5, alpha = .06) +
  geom_vline(xintercept = 0, colour = CSP_COLORS[["text"]], linewidth = .4) +
  geom_point(size = 2.6, position = position_dodge(width = .6)) +
  facet_wrap(~panel, scales = "free_x") +
  scale_colour_manual(values = arch_colours, name = NULL) +
  labs(x = "Percentage points", y = NULL) +
  theme_csp() +
  theme(legend.position = "bottom", strip.text = element_text(hjust = 0, face = "bold"),
        panel.grid.major.y = element_blank(), panel.spacing.x = unit(1.5, "lines"))
save_csp_pdf(p, "figures/simulation_steps.pdf", width = 8, height = 4.2)

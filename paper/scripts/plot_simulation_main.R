# Main simulation figure: does hierarchical composition give size-first order an
# advantage for the listener? Replication of the earlier simulations (global
# speaker, two orders) and the three speakers of the production models choosing
# among all fifteen utterances, averaged over the semantic parameter grid.
# Run from paper/. Data: data/simulation_controlled_overall.csv.

suppressPackageStartupMessages({library(ggplot2); library(dplyr); library(readr)})
source("scripts/csp_figure_style.R")

group_levels <- c("Earlier simulations\n(global speaker,\ntwo orders)", "Global\nspeaker",
                  "Intermediate\nspeaker", "Fully incremental\nspeaker")
d <- read_csv("data/simulation_controlled_overall.csv", show_col_types = FALSE) %>%
  filter(metric == "listener_advantage", order == "neutral", parameter_set == "original_grid",
         (rule == "terminal" & support == "two") |
           (rule %in% c("prefix_k0", "prefix_k05", "prefix_k1") & support == "fifteen")) %>%
  mutate(group = factor(group_levels[c(terminal = 1, prefix_k0 = 2, prefix_k05 = 3, prefix_k1 = 4)[rule]],
                        levels = group_levels),
         semantics = factor(semantics, levels = c("fixed", "updating"),
                            labels = c("Context-fixed", "Sequential context updating")),
         estimate = 100 * mean)
stopifnot(nrow(d) == 8)
write_csv(select(d, group, semantics, estimate, mcse), "data/simulation_main.csv")

p <- ggplot(d, aes(x = group, y = estimate, fill = semantics)) +
  geom_hline(yintercept = 0, colour = CSP_COLORS[["text"]], linewidth = .4) +
  geom_col(position = position_dodge(width = .7), width = .62) +
  geom_text(aes(label = ifelse(abs(estimate) < .05, "0.0", sprintf("%.1f", estimate))), position = position_dodge(width = .7),
            vjust = -.4, size = 3.6, colour = CSP_COLORS[["text"]]) +
  geom_vline(xintercept = 1.5, colour = CSP_COLORS[["grey"]], linetype = "dashed") +
  annotate("text", x = 1, y = 3.6, label = "Replication", size = 3.8, colour = CSP_COLORS[["text"]]) +
  annotate("text", x = 3, y = 3.6, label = "Production speakers, all fifteen utterances", size = 3.8, colour = CSP_COLORS[["text"]]) +
  scale_fill_manual(values = c(CSP_COLORS[["main_light"]], CSP_COLORS[["main"]]), name = NULL) +
  scale_y_continuous(expand = expansion(mult = c(.02, .15))) +
  labs(x = NULL, y = "Size-first advantage\n(percentage points)") +
  theme_csp() +
  theme(legend.position = "bottom", panel.grid.major.x = element_blank())
save_csp_pdf(p, "figures/simulation_main.pdf", width = 7, height = 4)

# Observed against predicted utterance proportions of the best-fitting model
# (plan-guided, context-fixed, all other components), one point per response
# type and display condition. Data: data/best_model_ppc.csv.

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(scales)
})
source("scripts/csp_figure_style.R")

cells <- read_csv("data/best_model_ppc.csv", show_col_types = FALSE) %>%
  filter(summary_type == "Exact response")
r <- cor(cells$observed_proportion, cells$predicted_mean)

best_plot <- ggplot(cells, aes(x = observed_proportion, y = predicted_mean)) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed",
              colour = CSP_COLORS[["grey"]], linewidth = .7) +
  geom_errorbar(aes(ymin = predicted_q025, ymax = predicted_q975),
                width = 0, linewidth = .32, alpha = .32, colour = CSP_COLORS[["main"]]) +
  geom_point(size = 2.15, alpha = .7, colour = CSP_COLORS[["main"]]) +
  annotate("text", x = .035, y = .865, hjust = 0, vjust = 1, size = 4.2,
           colour = CSP_COLORS[["text"]], label = sprintf("r = %.2f", r)) +
  scale_x_continuous(labels = percent_format(accuracy = 1), limits = c(0, .9)) +
  scale_y_continuous(labels = percent_format(accuracy = 1), limits = c(0, .9)) +
  labs(x = "Observed proportion", y = "Predicted proportion") +
  coord_fixed() +
  theme_csp()
save_csp_pdf(best_plot, "figures/best_model_fit.pdf", width = 4.6, height = 4.6)

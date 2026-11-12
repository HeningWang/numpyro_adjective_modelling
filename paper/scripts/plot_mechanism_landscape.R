# Account landscape: which mechanism predicts which part of production.
# Run from paper/ after analysis/exports/mechanism_components.py has written the data.

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(scales)
})
source("scripts/csp_figure_style.R")

architecture_palette <- c(
  "Global" = CSP_COLORS[["text"]],
  "Plan-guided" = CSP_COLORS[["green"]],
  "Fully incremental" = CSP_COLORS[["emphasis"]]
)
architecture_shapes <- c("Global" = 1, "Plan-guided" = 17, "Fully incremental" = 15)
account_levels <- c(
  "Lexical prediction", "Discriminatory efficiency", "Availability",
  "Incremental planning", "Hierarchical composition"
)
mechanism_labels <- c(
  "Stable order term" = "Stable order term",
  "Informativeness" = "Informativeness (alpha)",
  "Prefix salience" = "Salience",
  "Architecture vs global" = "Architecture vs global",
  "Context updating" = "Context updating"
)
component_levels <- c("Number of adjectives", "Which adjectives", "Order")
elpd_breaks <- c(-300, -30, 0, 30, 300, 3000)

gains <- read_csv("data/mechanism_component_gains.csv", show_col_types = FALSE) %>%
  mutate(
    account = factor(account, levels = rev(account_levels)),
    mechanism = factor(mechanism_labels[mechanism], levels = unname(mechanism_labels)),
    component = factor(component, levels = component_levels),
    architecture = factor(architecture, levels = names(architecture_palette))
  )

landscape <- ggplot(
  gains,
  aes(x = estimate_elpd, y = account, colour = architecture, shape = architecture)
) +
  geom_vline(xintercept = 0, colour = CSP_COLORS[["grey"]], linewidth = .6) +
  geom_pointrange(
    aes(xmin = credible_lower_95, xmax = credible_upper_95),
    position = position_dodge(width = .6), size = .45, linewidth = .7, fill = "white"
  ) +
  facet_wrap(~component, nrow = 1) +
  scale_x_continuous(
    trans = pseudo_log_trans(sigma = 10, base = 10),
    breaks = elpd_breaks, labels = label_comma()
  ) +
  scale_y_discrete(labels = function(x) {
    mech <- gains %>% distinct(account, mechanism)
    paste0(x, "\n", mech$mechanism[match(x, mech$account)])
  }) +
  scale_colour_manual(values = architecture_palette, name = NULL) +
  scale_shape_manual(values = architecture_shapes, name = NULL) +
  labs(x = "ELPD gain (signed log scale)", y = NULL) +
  theme_csp() +
  theme(
    legend.position = "top",
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(size = 11, angle = 45, hjust = 1),
    axis.text.y = element_text(size = 12, lineheight = .9)
  )

save_csp_pdf(landscape, "figures/mechanism_landscape.pdf", width = 12, height = 5.5)
ggsave("figures/mechanism_landscape.png", landscape, width = 12, height = 5.5, dpi = 200)

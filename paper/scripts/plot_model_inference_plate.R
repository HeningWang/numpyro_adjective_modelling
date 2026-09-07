# Formal plate diagram for Section 5.1.
# Run from paper/: Rscript scripts/plot_model_inference_plate.R

suppressPackageStartupMessages({
  library(ggplot2)
})

source("scripts/csp_figure_style.R")

dir.create("figures", showWarnings = FALSE)

arrow_style <- grid::arrow(
  length = grid::unit(.10, "inches"),
  type = "closed"
)

edges <- data.frame(
  x = c(2.1, 7.9, 10.6, 12.2, 14.0, 2.1, 5.0, 7.9, 3.25, 8.65),
  y = c(6.50, 6.50, 6.50, 6.55, 6.55, 4.64, 4.64, 4.64, 2.50, 2.50),
  xend = c(2.1, 7.9, 8.15, 5.45, 8.60, 7.55, 7.75, 8.00, 7.45, 11.65),
  yend = c(5.36, 5.36, 3.04, 5.22, 3.05, 3.00, 3.00, 3.00, 2.50, 2.50),
  linetype = c("solid", "solid", "solid", "solid", "solid", "solid", "solid", "solid", "solid", "solid")
)

plate <- ggplot() +
  # Nested plates.
  annotate(
    "rect", xmin = .45, xmax = 14.65, ymin = .72, ymax = 5.92,
    fill = NA, colour = CSP_COLORS[["text"]], linewidth = .65
  ) +
  annotate(
    "rect", xmin = .82, xmax = 14.28, ymin = 1.08, ymax = 4.05,
    fill = NA, colour = CSP_COLORS[["text"]], linewidth = .5
  ) +
  # Directed dependencies.
  geom_segment(
    data = edges,
    aes(x = x, y = y, xend = xend, yend = yend, linetype = linetype),
    colour = CSP_COLORS[["text"]], linewidth = .48,
    arrow = arrow_style, show.legend = FALSE
  ) +
  annotate(
    "segment", x = 5.0, y = 6.50, xend = 5.0, yend = 5.36,
    colour = CSP_COLORS[["green"]], linewidth = .58, linetype = "dashed",
    arrow = arrow_style
  ) +
  # Fixed model choices and trial inputs.
  annotate(
    "rect", xmin = 11.68, xmax = 12.72, ymin = 6.56, ymax = 7.34,
    fill = "white", colour = CSP_COLORS[["text"]], linewidth = .58
  ) +
  annotate(
    "rect", xmin = 13.48, xmax = 14.52, ymin = 6.56, ymax = 7.34,
    fill = "white", colour = CSP_COLORS[["text"]], linewidth = .58
  ) +
  annotate(
    "rect", xmin = 1.20, xmax = 3.25, ymin = 2.05, ymax = 2.95,
    fill = "white", colour = CSP_COLORS[["text"]], linewidth = .58
  ) +
  # Population and participant-level latent quantities.
  annotate(
    "point", x = 2.1, y = 6.92, shape = 21, size = 18,
    stroke = .65, fill = "white", colour = CSP_COLORS[["text"]]
  ) +
  annotate(
    "point", x = 5.0, y = 6.92, shape = 21, size = 18,
    stroke = .75, fill = CSP_COLORS[["green_light"]], colour = CSP_COLORS[["green"]]
  ) +
  annotate(
    "point", x = 7.9, y = 6.92, shape = 21, size = 18,
    stroke = .65, fill = "white", colour = CSP_COLORS[["text"]]
  ) +
  annotate(
    "point", x = 10.6, y = 6.92, shape = 21, size = 18,
    stroke = .65, fill = "white", colour = CSP_COLORS[["text"]]
  ) +
  annotate(
    "point", x = 2.1, y = 5.0, shape = 21, size = 16,
    stroke = .65, fill = "white", colour = CSP_COLORS[["text"]]
  ) +
  annotate(
    "point", x = 5.0, y = 5.0, shape = 21, size = 16,
    stroke = .75, fill = CSP_COLORS[["green_light"]], colour = CSP_COLORS[["green"]]
  ) +
  annotate(
    "point", x = 7.9, y = 5.0, shape = 21, size = 16,
    stroke = .65, fill = "white", colour = CSP_COLORS[["text"]]
  ) +
  # Deterministic probability node and observed response.
  annotate(
    "point", x = 8.15, y = 2.50, shape = 21, size = 20,
    stroke = 1.25, fill = "white", colour = CSP_COLORS[["text"]]
  ) +
  annotate(
    "point", x = 8.15, y = 2.50, shape = 21, size = 17.5,
    stroke = .45, fill = "white", colour = CSP_COLORS[["text"]]
  ) +
  annotate(
    "point", x = 12.25, y = 2.50, shape = 21, size = 17,
    stroke = .65, fill = CSP_COLORS[["main_light"]], colour = CSP_COLORS[["text"]]
  ) +
  # Mathematical node labels.
  annotate("text", x = 2.1, y = 6.92, label = "atop(alpha[0], tau[log~alpha])", parse = TRUE, size = 3.8) +
  annotate("text", x = 5.0, y = 6.92, label = "atop(mu[kappa], tau[kappa])", parse = TRUE, size = 4.0) +
  annotate("text", x = 7.9, y = 6.92, label = "atop(mu[log~beta], tau[beta])", parse = TRUE, size = 3.8) +
  annotate("text", x = 10.6, y = 6.92, label = "bold(lambda)", parse = TRUE, size = 4.4) +
  annotate("text", x = 12.2, y = 6.95, label = "A", size = 4.6, fontface = "bold") +
  annotate("text", x = 14.0, y = 6.95, label = "R", size = 4.6, fontface = "bold") +
  annotate("text", x = 2.1, y = 5.0, label = "alpha[i]", parse = TRUE, size = 4.5) +
  annotate("text", x = 5.0, y = 5.0, label = "kappa[i]", parse = TRUE, size = 4.5) +
  annotate("text", x = 7.9, y = 5.0, label = "beta[i]", parse = TRUE, size = 4.5) +
  annotate("text", x = 2.22, y = 2.50, label = "X[it] * ',' ~ Omega(u)", parse = TRUE, size = 4.2) +
  annotate("text", x = 8.15, y = 2.50, label = "pi[it](u)", parse = TRUE, size = 4.4) +
  annotate("text", x = 12.25, y = 2.50, label = "y[it]", parse = TRUE, size = 4.6) +
  # Labels explain the fixed model choices and the repeated units.
  annotate(
    "text", x = 12.2, y = 7.62, label = "Production\narchitecture",
    size = 3.35, lineheight = .92, colour = CSP_COLORS[["text"]]
  ) +
  annotate(
    "text", x = 14.0, y = 7.62, label = "Semantic\nregime",
    size = 3.35, lineheight = .92, colour = CSP_COLORS[["text"]]
  ) +
  annotate(
    "text", x = .62, y = 5.76, hjust = 0, vjust = 1,
    label = "participant  i = 1, ..., 113", size = 3.65,
    colour = CSP_COLORS[["text"]]
  ) +
  annotate(
    "text", x = 1.00, y = 3.88, hjust = 0, vjust = 1,
    label = "trial  t = 1, ..., n_i", size = 3.65,
    colour = CSP_COLORS[["text"]]
  ) +
  coord_cartesian(xlim = c(0, 15.1), ylim = c(.55, 8.05), clip = "off") +
  theme_void() +
  theme(plot.margin = margin(10, 10, 8, 10))

save_csp_pdf(plate, "figures/production_inference_plate.pdf", 9.4, 4.75)

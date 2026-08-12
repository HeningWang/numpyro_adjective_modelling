suppressPackageStartupMessages({
  library(aida)
  library(ggplot2)
})

CSP_COLORS <- c(
  main = "#7581B3",
  accent = "#99C2C2",
  emphasis = "#C65353",
  gold = "#E2BA78",
  green = "#1B6B3A",
  text = "#575463",
  main_light = "#B0B7D4",
  accent_dark = "#66A3A3",
  emphasis_light = "#DB9494",
  gold_dark = "#D49735",
  green_light = "#A7CCB2",
  grey = "#D4D3D9",
  navy = "#414C76",
  dark_red = "#993333"
)

# Figure 3 is the manuscript typography anchor: 14 pt tick and legend text,
# 16 pt axis titles, and 14 pt facet labels at the saved figure size.
theme_csp <- function() {
  suppressWarnings(theme_aida()) +
    theme(
      text = element_text(colour = CSP_COLORS[["text"]]),
      axis.text.x = element_text(size = 14, colour = CSP_COLORS[["text"]]),
      axis.text.y = element_text(size = 14, colour = CSP_COLORS[["text"]]),
      axis.title.x = element_text(size = 16, colour = CSP_COLORS[["text"]]),
      axis.title.y = element_text(size = 16, colour = CSP_COLORS[["text"]]),
      legend.text = element_text(size = 14, colour = CSP_COLORS[["text"]]),
      legend.title = element_text(size = 14, colour = CSP_COLORS[["text"]]),
      strip.text = element_text(size = 14, face = "bold", colour = CSP_COLORS[["text"]]),
      plot.tag = element_text(size = 16, face = "bold", colour = CSP_COLORS[["text"]]),
      plot.tag.position = c(0, 1),
      plot.margin = margin(8, 10, 8, 10)
    )
}

save_csp_pdf <- function(plot, filename, width, height) {
  ggsave(
    filename = filename,
    plot = plot,
    width = width,
    height = height,
    units = "in",
    device = "pdf"
  )
}

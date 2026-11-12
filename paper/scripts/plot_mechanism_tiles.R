# Mechanism tiles: ELPD gains on order and content for each model cell.
# Run from paper/ after copying analysis/summaries/component_gains/component_gains.csv
# to data/component_gains.csv.

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(tidyr)
})
source("scripts/csp_figure_style.R")

architectures <- c(G = "Global", K = "Plan-guided", I = "Fully incremental")
# Plotmath labels draw the Greek letters on every graphics device.
block_levels <- c('atop("Architecture", "vs global, fixed")',
                  'atop("Stable order term", "gain from" ~ beta)',
                  'atop("Informativeness", "gain from" ~ alpha)',
                  'atop("Salience", "gain from" ~ lambda)')
account_fill <- c(
  "Discriminatory efficiency, left to right (Fukumura, Rubio-Fernandez)" = CSP_COLORS[["main_light"]],
  "Composition (Scontras, Franke)" = CSP_COLORS[["emphasis_light"]],
  "Lexical prediction (Hahn, Futrell)" = CSP_COLORS[["accent"]],
  "Availability (salience)" = CSP_COLORS[["gold"]],
  "Other" = "white"
)

parse_contrast <- function(contrast) {
  model <- sub("[: ].*", "", contrast)
  tibble(
    architecture = unname(architectures[substr(model, 1, 1)]),
    semantics = ifelse(grepl("UPD", model), "updating", "fixed"),
    block = case_when(
      grepl("stable order term", contrast) ~ block_levels[2],
      grepl("informativeness", contrast) ~ block_levels[3],
      grepl("prefix salience", contrast) ~ block_levels[4],
      TRUE ~ block_levels[1]
    )
  )
}

gains <- bind_rows(
  read_csv("data/component_gains.csv", show_col_types = FALSE),
  read_csv("data/salience_gains.csv", show_col_types = FALSE)
) %>%
  filter(factor != "response") %>%
  mutate(part = ifelse(factor == "order_given_set", "order", "content")) %>%
  group_by(contrast, part) %>%
  summarise(elpd = sum(estimate_elpd), .groups = "drop") %>%
  pivot_wider(names_from = part, values_from = elpd) %>%
  bind_cols(parse_contrast(.$contrast)) %>%
  bind_rows(tibble(contrast = "baseline", architecture = "Global", semantics = "fixed",
                   block = block_levels[1], order = NA, content = NA))

tiles <- gains %>%
  mutate(
    row = ifelse(block %in% block_levels[3:4], "", semantics),
    account = case_when(
      block == block_levels[2] ~ names(account_fill)[3],
      block == block_levels[3] ~ names(account_fill)[1],
      block == block_levels[4] ~ names(account_fill)[4],
      block == block_levels[1] & architecture == "Fully incremental" & semantics == "fixed" ~
        names(account_fill)[1],
      block == block_levels[1] & architecture == "Global" & semantics == "updating" ~
        names(account_fill)[2],
      TRUE ~ "Other"
    ),
    label = ifelse(contrast == "baseline", "baseline",
                   sprintf("order %+.0f\ncontent %+.0f", order, content)),
    best = block == block_levels[1] & architecture == "Plan-guided" & semantics == "fixed",
    baseline = contrast == "baseline",
    architecture = factor(architecture, levels = architectures),
    block = factor(block, levels = block_levels),
    row = factor(row, levels = c("updating", "fixed", ""))
  )
write_csv(select(tiles, block, row, architecture, account, order, content),
          "data/mechanism_tiles.csv")

tile_plot <- ggplot(tiles, aes(x = architecture, y = row)) +
  geom_tile(aes(fill = account), colour = CSP_COLORS[["grey"]], linewidth = .5,
            width = .94, height = .9) +
  geom_tile(data = filter(tiles, best), fill = NA, colour = CSP_COLORS[["text"]],
            linewidth = 1.4, width = .94, height = .9) +
  geom_tile(data = filter(tiles, baseline), fill = "white", colour = CSP_COLORS[["text"]],
            linetype = "dashed", linewidth = .6, width = .94, height = .9) +
  geom_text(aes(label = label), size = 4.6, lineheight = .95, colour = CSP_COLORS[["text"]]) +
  facet_grid(block ~ ., scales = "free_y", space = "free_y", switch = "y",
             labeller = label_parsed) +
  scale_x_discrete(position = "top") +
  scale_fill_manual(values = account_fill, breaks = names(account_fill)[1:4], name = NULL) +
  guides(fill = guide_legend(ncol = 1)) +
  labs(x = NULL, y = NULL) +
  theme_csp() +
  theme(
    panel.grid = element_blank(),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    axis.ticks = element_blank(),
    strip.placement = "outside",
    strip.text.y.left = element_text(angle = 0, hjust = 1, size = 13),
    axis.text.y = element_text(size = 12),
    legend.position = "bottom",
    legend.direction = "horizontal",
    legend.text = element_text(size = 12)
  )

save_csp_pdf(tile_plot, "figures/mechanism_tiles.pdf", width = 9, height = 7.6)
ggsave("figures/mechanism_tiles.png", tile_plot, width = 9, height = 6.5, dpi = 200)

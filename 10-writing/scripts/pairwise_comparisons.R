# Pairwise comparisons for slider experiment:
# context x discriminability interaction
#
# Run from 10-writing/scripts/:
#   Rscript pairwise_comparisons.R

library(lme4)
library(emmeans)

# ── Load data ────────────────────────────────────────────────────────────────
df <- read.csv("../../01-dataset/01-slider-data-preprocessed.csv")

# Check column names
cat("Columns:", paste(names(df), collapse=", "), "\n")
cat("N observations:", nrow(df), "\n")

# ── Identify key columns ────────────────────────────────────────────────────
# Expecting: prefer_first_1st (response), conditions (context), sharpness (discriminability)
# id (participant), item

# Centre response around 0
df$rating_centered <- df$prefer_first_1st - 0.5

# Context factor
df$context <- factor(df$conditions)
cat("Context levels:", levels(df$context), "\n")

# Discriminability
df$discrim <- factor(df$sharpness)
cat("Discriminability levels:", levels(df$discrim), "\n")

# ── Fit LMM ──────────────────────────────────────────────────────────────────
cat("\nFitting linear mixed model...\n")
m <- lmer(rating_centered ~ context * discrim + (1 | id) + (1 | item), data = df)
cat("\nModel summary:\n")
print(summary(m))

# ── Pairwise comparisons ────────────────────────────────────────────────────
cat("\n\n=== Pairwise comparisons of context within each discriminability level ===\n")
emm_ctx_by_disc <- emmeans(m, pairwise ~ context | discrim, adjust = "bonferroni")
print(summary(emm_ctx_by_disc))

cat("\n\n=== Pairwise comparisons of discriminability within each context ===\n")
emm_disc_by_ctx <- emmeans(m, pairwise ~ discrim | context, adjust = "bonferroni")
print(summary(emm_disc_by_ctx))

cat("\n\n=== All pairwise comparisons (context x discrim cells) ===\n")
emm_all <- emmeans(m, pairwise ~ context * discrim, adjust = "bonferroni")
print(summary(emm_all$contrasts))

# ── Save results ─────────────────────────────────────────────────────────────
sink("pairwise_results.txt")
cat("=== Pairwise comparisons of context within each discriminability level ===\n")
print(summary(emm_ctx_by_disc))
cat("\n\n=== Pairwise comparisons of discriminability within each context ===\n")
print(summary(emm_disc_by_ctx))
cat("\n\n=== All pairwise comparisons (context x discrim cells) ===\n")
print(summary(emm_all$contrasts))
sink()
cat("\nResults saved to pairwise_results.txt\n")

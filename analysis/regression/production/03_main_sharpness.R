library(tidyr)
library(ggplot2)
library(gridExtra)
library(trimr)
library(stringr)
library(lme4)
library(lmerTest)
library(dplyr)
library(ordinal)
library(effects)
library(optimx)
library(emmeans)
library(pbkrtest)
library(multcomp)
library(sjPlot)
library(sjmisc)
library(ggplot2)
library(texreg)
library(memisc)
library(xtable)
library(RColorBrewer)

rm(list=ls())
#setwd("~/GitHub/taishan/analysis")
plots_dir<- file.path(getwd(),"plots")
if(!dir.exists(plots_dir)) dir.create(plots_dir, recursive = TRUE, showWarnings = FALSE)


data <- readRDS("data_preprocessed.Rdata")

# Recode combination to dimension_X
data %>% 
  mutate(combination = ifelse(combination == "dimension_color" | 
                                combination == "dimension_form", 
                              "dimension_X", 
                              "color_form")) %>% filter(combination == "dimension_X")-> df_dx

# Recode annotation with D-first and others
data %>% mutate(encoding = ifelse(substr(annotation, 1, 1) == "D", 1, 0)) -> data

# Perform GlMER analysis

# m0_dfirst <- glmer(encoding~relevant_property*dist*combination + (1 | id) + (1 | item), data = data, family = binomial)
# Using above random structure, model fails to converge, try another:
# The model converged if there are no random strucutres, using GLM instead
m0_dfirst <- glm(encoding~relevant_property*dist*combination, data = data, family = binomial)
summary(m0_dfirst) # sharp strength the effect of relevance in its direction, in line with the effect in slider data
m1_dfirst <- update(m0_dfirst, .~. -relevant_property:dist:combination)                      
anova(m0_dfirst, m1_dfirst, test="Chisq") # sig. three way interaction

# Break down three-way interactions:
# relevant_propertysecond:distsharp:combinationdimension_color -1.57257    0.59107  -2.661  0.00780 ** (sharp strengthen color first in color relevnat)
# relevant_propertyfirst:distsharp:combinationdimension_form    2.15802    0.34009   6.345 2.22e-10 *** (sharp strengthen size first in size relevant)
# relevant_propertyfirst:distsharp:combinationdimension_color  -0.06688    0.34140  -0.196  0.84468 (interessted effect but no sig.)
# relevant_propertysecond:distsharp:combinationdimension_form   0.95103    0.52193   1.822  0.06843 . (opposite effect, sharp strengthen size first in form relevant)
# We cannot resolve the sig. three-way interaction.
# So we cannot interpret this: distsharp                                                     0.42168    0.15953   2.643  0.00821 ** 

# We also interested in main effect of dist in dimension_X.
# But see 02 interaction: 1. there is no sig. main effect of sharpness; 2. Cannot resolve two-way interaction

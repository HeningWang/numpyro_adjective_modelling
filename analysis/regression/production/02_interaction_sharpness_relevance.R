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
                              "color_form")) %>% 
  filter(combination == "dimension_X") -> df_dx

# Recode annotation with D-first and others
df_dx %>% mutate(encoding = ifelse(substr(df_dx$annotation, 1, 1) == "D", 1, 0)) -> df_dx

# Perform GlMER analysis

m0_dfirst <- glmer(encoding~relevant_property*dist + (1 | id) + (1 | item), data = df_dx, family = binomial)
summary(m0_dfirst) # sharp strength the effect of relevance in its direction, in line with the effect in slider data
m1_dfirst <- update(m0_dfirst, .~. -relevant_property:dist)                      
anova(m0_dfirst, m1_dfirst, test="Chisq") # sig. interaction



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

# Recode annotation with CDF and others
data %>% mutate(encoding = ifelse(substr(annotation, 1, 1) == "D", 1, 0)) -> data

# Recode annotation with D-only and others
df_dx %>% mutate(encoding = ifelse(annotation == "D", 1, 0)) -> df_dx

# Perform GlMER analysis

m0_donly <- glmer(encoding~relevant_property*dist + (1 | id) + (1 | item), data = df_dx, family = binomial)
summary(m0_donly) # relevant_propertyfirst:distsharp    2.3810     1.2702   1.875   0.0609 .
# sharp lead to less overinformativness than blurred (marginal sig.). This interaction is also attributed to adjective selection rather than just adjective ordering!
m1_donly <- update(m0_donly, .~. -relevant_property:dist)                   
anova(m0_donly, m1_donly, test="Chisq") # no sig. interaction

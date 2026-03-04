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

rm(list = ls())
data = readRDS(file = "data_preprocessed.Rdata")

#set up ordinal coding for clmm analysis
data$coding <- ifelse(data$annotation == "D"|
                        data$annotation == "DC"|
                        data$annotation == "DF"|
                        data$annotation == "DCF"|
                        data$annotation == "DFC", 1, 
                      ifelse(data$annotation == "CD"|
                               data$annotation == "FD"|
                               data$annotation == "CDF"|
                               data$annotation == "FDC", 2, 
                             ifelse(data$annotation == "CFD"|
                                    data$annotation == "FCD", 3, 4)))
data$coding <- as.factor(data$coding)
factor(data$coding)
levels(data$coding)
str(data)

#check proportional odd assumption
m_check_1 <- clm(coding~combination*dist*relevant_property, data = data) # not passed
m_check_2 <- clm(coding~combination+dist*relevant_property, data = data) # also not passed
data_dimension <- subset(data, combination != "color_form" )
m_check_3 <- clm(coding~combination+dist*relevant_property, data = data_dimension) # passed, using this model for analysis of distribution
# fitting the model for analysis of relevance and combination
m_relevant <- clmm(coding~combination*dist*relevant_property + (1|item) + (1|id), data = data)
summary(m_relevant)
m_relevant_1 <- update(m_relevant, .~.-combination:dist:relevant_property)
anova(m_relevant,m_relevant_1) #104.59  4  < 2.2e-16 *** significant three way interaction

#subset data by dist sharp
data_sharp <- subset(data, dist == 'sharp')
m_check_4  <- clm(coding~combination*relevant_property, data = data_sharp) #passed, using this model for analysis of relevance and combination 
m_sharp <- clmm(coding~combination*relevant_property + (1|item) + (1|id), data = data_sharp)
m_sharp_1 <- update(m_sharp, .~.-combination:relevant_property)
anova(m_sharp,m_sharp_1) #309.76  4  < 2.2e-16 *** significant two way interaction

#subset data by combination color form for effect of relevance
data_cf <- subset(data_sharp, combination == 'color_form')
m_cf <- clmm(coding~relevant_property + (1|item) + (1|id), data = data_cf)
m_cf_1 <- clmm(coding~1 + (1|item) + (1|id), data = data_cf)
anova(m_cf,m_cf_1) # 77.45  2  < 2.2e-16 *** effect of relevance

#subset data by relevance both for effect of combination
data_both <- subset(data_sharp, relevant_property == 'both')
m_both <- clmm(coding~combination + (1|item) + (1|id), data = data_both)
m_both_1 <- clmm(coding~1 + (1|item) + (1|id), data = data_both)
anova(m_both,m_both_1) # 359.93  2  < 2.2e-16 *** effect of combination


# fitting model for analysis of distribution
data_sub <- subset(data, combination != "color_form" )
m_relevant_sub <- clmm(coding~combination+dist*relevant_property + (1|item) + (1|id), data = data_sub)
summary(m_relevant_sub)
m_relevant_sub1 <-  update(m_relevant_sub, .~.-dist:relevant_property)
anova(m_relevant_sub1,m_relevant_sub) # 10.818  2   0.004476 ** significant two way interaction between dist and relevance


# create interaction plot
# why two-way interaction? The difference between "both" and "first" (effect of discriminative strength) is greater under sharp than under blurred.
w <- plot_model(m_relevant_sub, type = "int", title = "Plot of interaction effect between relevance property and distribution", axis.title  = "P(outcome)") # plot of interactions shows the similar pattern of Exp I (more D-first under sharp)
w  + theme_sjplot() + theme(axis.text.x = element_text(angle = 45, hjust = 1))
w$data$response.level <- factor(c("D first", "D second", "D third", "D null"), levels =c("D first", "D second", "D third", "D null"))
levels(w$data$response.level)
w$dist$data$response.level <- as.factor(c("C first", "D first", "F first"))
w

emmeans(m_relevant_sub, specs = ~relevant_property * dist, mode = "prob")

# further subset for interaction effect

data_sub_sub <- subset(data_sub, relevant_property == "first")
m_relevant_sub_sub <- clmm(coding~combination+dist + (1|item) + (1|id), data = data_sub_sub)
m_relevant_sub_sub1 <-  update(m_relevant_sub_sub, .~.-dist)
anova(m_relevant_sub_sub1, m_relevant_sub_sub) #1.4332  1     0.2312
plot_model(m_relevant_sub, type = "pred")

data_sub_blurred <- subset(data_sub, dist == "blurred")
m_relevant_sub_blurred <- clmm(coding~combination+relevant_property + (1|item) + (1|id), data = data_sub_blurred)
summary(m_relevant_sub_blurred)
m_relevant_sub_sub1 <-  update(m_relevant_sub_sub, .~.-relevant_property)
anova(m_relevant_sub_sub1, m_relevant_sub_sub) # 0.0442  1     0.8335

data_sub_sharp <- subset(data_sub, dist == "sharp")
m_relevant_sub_sharp <- clmm(coding~combination+relevant_property + (1|item) + (1|id), data = data_sub_sharp)
summary(m_relevant_sub_sharp)
m_relevant_sub_sub1 <-  update(m_relevant_sub_sub, .~.-relevant_property)
anova(m_relevant_sub_sub1, m_relevant_sub_sub) # 0.0442  1     0.8335


# hypothesis driven glmer analysis
data_glm <- data_sub
data_glm$coding <- ifelse(data_glm$coding == "1", 1, 0)
m_glm <- glmer(coding ~ combination + dist * relevant_property + (1|item) + (1|id),
               family = "binomial", data = data_glm)
summary(m_glm)
plot_model(m_glm, type = "int")
plot_model(m_glm, type = "eff")



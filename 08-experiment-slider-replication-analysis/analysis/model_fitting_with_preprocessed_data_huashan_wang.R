rm(list = ls())
load("data_preprocessed_huashan.RData")
library(tidyr)
library(ggplot2)
library(gridExtra)
library(trimr)
library(stringr)
library(lme4)
library(lmerTest)
library(dplyr)
library(effects)
library(optimx)
library(emmeans)
library(pbkrtest)
# set up contrast coding
mean(data$prefer_first_1st)
contrasts(data$combination)
contrasts(data$relevant_property)
contrasts(data$dist) <- c(-0.5, 0.5)
colnames(contrasts(data$dist)) <- c("sharp")

#maximaze the random effect structure





# log transformation 
data$sli<-(data$prefer_first_1st+50.5)/101
m0 <- lmer(data$sli~relevant_property*combination*dist
           +(1|id)
           +(1|item), data=data)
m0.1 <- lmer(log(sli)~relevant_property*combination*dist
           +(1|id)
           +(1|item), data=data)
data$transli<-qlogis(jitter(data$sli))
m0.2 <- lmer(transli~relevant_property*combination*dist
             +(1|id)
             +(1|item), data=data)
par(mfrow=c(1,3))
qqnorm(residuals(m0), main = "default value")
qqnorm(residuals(m0.1), main = "value log transformed")
qqnorm(residuals(m0.2), main = "value logit transformed")

# use m0.2 for inference statistic
summary(m0.2)
anova(m0.2)
step(m0.2)


#subset data for post hoc analysis of interaction 

data_cf <- subset(data, combination == "color_form")
data_d <- subset(data, combination == "dimension_X")

# analyse cf
data_cf$combination <- as.factor(droplevels(data_cf$combination))
m_cf <- lmer(transli~relevant_property*dist
                   +(1|id)
                   +(1|item), data=data_cf)

summary(m_cf)
anova(m_cf)
plot(allEffects(m_cf))
interaction.plot(x.factor = data_cf$relevant_property,
                 trace.factor = data_cf$dist, 
                 response = data_cf$prefer_first_1st)

# analyse dimension_X
data_d$combination <- as.factor(droplevels(data_d$combination))
m_d <- lmer(transli~relevant_property*dist
             +(1|id)
             +(1|item), data=data_d)

summary(m_d)
anova(m_d)

data_dfirst <- subset(data_d, relevant_property == "first")
data_dfirst$relevant_property <- as.factor(droplevels(data_dfirst$relevant_property))
m_dfirst <- lmer(transli~dist
             +(1|id)
             +(1|item), data=data_dfirst)

summary(m_dfirst)
anova(m_dfirst)

data_dsecond <- subset(data_d, relevant_property == "second")
data_dsecond$relevant_property <- as.factor(droplevels(data_dsecond$relevant_property))
m_dsecond <- lmer(transli~dist
                  +(1|id)
                  +(1|item), data=data_dsecond)

summary(m_dsecond)
anova(m_dsecond)

data_dsharp <- subset(data_d, dist == "sharp")
data_dsharp$dist <- droplevels(as.factor(data_dsharp$dist))
m_dsharp <- lmer(transli~relevant_property
                   +(1|id)
                   +(1|item), data=data_dsharp)

summary(m_dsharp)
anova(m_dsharp)

data_dblurred <- subset(data_d, dist == "blurred")
data_dblurred$dist <- droplevels(as.factor(data_dblurred$dist))
m_dblurred <- lmer(transli~relevant_property
                  +(1|id)
                  +(1|item), data=data_dblurred)

summary(m_dblurred)
anova(m_dblurred)

data_dboth <- subset(data_d, relevant_property == "both")
data_dboth$relevant_property <- as.factor(droplevels(data_dboth$relevant_property))
m_dboth <- lmer(transli~dist
                  +(1|id)
                  +(1|item), data=data_dboth)

summary(m_dboth)
anova(m_dboth)







m1 <-  update(m0, .~.-combination:dist:relevant_property)
anova(m1,m0)

m2 <-  update(m1, .~.-combination:dist)
anova(m2,m1)

m3 <-  update(m2, .~.-dist:relevant_property)
anova(m3,m2)

m4 <-  update(m3, .~.-combination:relevant_property)
anova(m4,m3)

m5 <-  update(m4, .~.-combination)
anova(m5,m4)

m6 <-  update(m5, .~.-dist)
anova(m6,m5)

m7 <-  update(m6, .~.-relevant_property)
anova(m7,m6)


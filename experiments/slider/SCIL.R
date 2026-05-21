rm(list = ls())
#setwd("~/GitHub/huashan/analysis")
load ("data_preprocessed_huashan.RData")
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
data_d <- subset(data, combination == "dimension_form" | combination == "dimension_color")
data_df <- subset(data_d, combination == "dimension_form")
data_dc <- subset(data_d, combination == "dimension_color")


data$combination1 <- ifelse(data$combination == "dimension_form" | data$combination == "dimension_color", "dimension_x", "color_form")
m0 <- lmer(transli~relevant_property*combination1*dist
           +(1|id)
           +(1|item), data=data)
summary(m0)
step(m0)
anova(m0)

m0.1 <- update(m0, .~. -relevant_property:combination1:dist)
summary(m0.1)
anova(m0.1, m0)
m0.2<- update(m0.1, .~. -combination1:dist)
anova(m0.2,m0.1)
summary(m0.2)
m0.3<- update(m0.2, .~. -combination1:relevant_property)
anova(m0.2,m0.3)
summary(m0.3)
# haupteffect for combination
m0.4 <- update(m0.3, .~. -combination1)
anova(m0.4,m0.3)
# interaktion effect between relevance and dist
m0.5 <- update(m0.3, .~. -relevant_property:dist)
anova(m0.5,m0.3)

# analyse dx
data_dx <- subset(data, combination1 == "dimension_x")
m1 <- lmer(transli~relevant_property*dist
           +(1|id)
           +(1|item), data=data_dx)
summary(m1)
m1.1 <- update(m1, .~. -dist:relevant_property)
anova(m1,m1.1)


# analyse cf
data_cf$combination <- as.factor(droplevels(data_cf$combination))
m_cf <- lmer(transli~relevant_property*dist
                   +(1|id)
                   +(1|item), data=data_cf)

summary(m_cf)
anova(m_cf)
m_cf.1 <- update(m_cf, .~. -dist:relevant_property)
anova(m_cf, m_cf.1)

# subset dx for relevance
data_dx_blur <- subset(data_dx, dist == "blurred")
data_dx_sharp <- subset(data_dx, dist == "sharp")

m_dx_blur <- lmer(transli~relevant_property
                  +(1|id), data=data_dx_blur)

m_dx_sharp <- lmer(transli~relevant_property
                  +(1|id), data=data_dx_sharp)

m_rel_blur <- lmer(transli~1
                +(1|id), data=data_dx_blur)

m_rel_sharp <- lmer(transli~1
                   +(1|id), data=data_dx_sharp)

anova(m_dx_blur,m_rel_blur)

anova(m_dx_sharp,m_rel_sharp)




plot(allEffects(m_cf))
interaction.plot(x.factor = data_cf$relevanty_property, 
                 trace.factor = data_cf$dist, 
                 response = data_cf$prefer_first_1st)

# analyse d
data_d$combination <- as.factor(droplevels(data_d$combination))
m_d <- lmer(transli~combination*relevant_property*dist
             +(1|id)
             +(1|item), data=data_d)

summary(m_d)
anova(m_d)




# analyse dc
data_dc$combination <- as.factor(droplevels(data_dc$combination))
m_dc <- lmer(transli~relevant_property*dist
            +(1|id)
            +(1|item), data=data_dc)
summary(m_dc)
anova(m_dc)

data_dcfirst <- subset(data_dc, relevant_property == "first")
data_dcfirst$relevant_property <- as.factor(droplevels(data_dcfirst$relevant_property))
m_dcfirst <- lmer(transli~dist
             +(1|id)
             +(1|item), data=data_dcfirst)

summary(m_dcfirst)
anova(m_dcfirst)

data_dcsecond <- subset(data_dc, relevant_property == "second")
data_dcsecond$relevant_property <- as.factor(droplevels(data_dcsecond$relevant_property))
m_dcsecond <- lmer(transli~dist
                  +(1|id)
                  +(1|item), data=data_dcsecond)

summary(m_dcsecond)
anova(m_dcsecond)


# analyse df post hoc
data_df$combination <- as.factor(droplevels(data_df$combination))
m_df <- lmer(transli~relevant_property*dist
            +(1|id)
            +(1|item), data=data_df)

summary(m_df)
anova(m_df)

data_dffirst <- subset(data_df, relevant_property == "first")
data_dffirst$relevant_property <- as.factor(droplevels(data_dffirst$relevant_property))
m_dffirst <- lmer(transli~dist
                  +(1|id)
                  +(1|item), data=data_dffirst)

summary(m_dffirst)
anova(m_dffirst)

data_dfsecond <- subset(data_df, relevant_property == "second")
data_dfsecond$relevant_property <- as.factor(droplevels(data_dfsecond$relevant_property))
m_dfsecond <- lmer(transli~dist
                   +(1|id)
                   +(1|item), data=data_dfsecond)

summary(m_dfsecond)
anova(m_dfsecond)

data_dfsharp <- subset(data_df, dist == "sharp")
data_dfsharp$dist <- as.factor(droplevels(data_dfsharp$sharp))
m_dfsharp <- lmer(transli~relevant_property
                   +(1|id)
                   +(1|item), data=data_dfsharp)

summary(m_dfsharp)
anova(m_dfsharp)

data_dfblurred <- subset(data_df, dist == "sharp")
data_dfsharp$dist <- as.factor(droplevels(data_dfsharp$sharp))
m_dfsharp <- lmer(transli~relevant_property
                  +(1|id)
                  +(1|item), data=data_dfsharp)

summary(m_dfsharp)
anova(m_dfsharp)



data_dfboth <- subset(data_df, relevant_property == "both")
data_dfboth$relevant_property <- as.factor(droplevels(data_dfboth$relevant_property))
m_dfboth <- lmer(transli~dist
                  +(1|id)
                  +(1|item), data=data_dfboth)

summary(m_dfboth)
anova(m_dfboth)







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


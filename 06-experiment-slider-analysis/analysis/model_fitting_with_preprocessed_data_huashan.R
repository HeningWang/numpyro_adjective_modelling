rm(list = ls())
load ("data_preprocessed_for_analysis_of_none_fit_huashan.RData")
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

#FS
data_filler <- subset(data_preprocessed_for_analysis_of_none_fit_huashan, str_sub(data_preprocessed_for_analysis_of_none_fit_huashan$conditions, 1, 1) == "f")
data_preprocessed_for_analysis_of_none_fit_huashan <- droplevels(subset(data_preprocessed_for_analysis_of_none_fit_huashan, str_sub(data_preprocessed_for_analysis_of_none_fit_huashan$conditions, 1, 1) != "f"))
#SF

# analyse none fits using glmer
nonefit_mh1 <- glmer(none_fits ~ 1 + conditions*dist
                     #+ (1|id)#FS
                     + (1|item)
                     , data=data_preprocessed_for_analysis_of_none_fit_huashan, 
family=poisson, nAGQ=1, control = glmerControl(optimizer ='bobyqa', optCtrl=list(maxfun= 10^6)))#FS: increase maxfun (iterations)

#FS
nonefit_mh1.1 <- update(nonefit_mh1, .~.-conditions:dist)
anova(nonefit_mh1.1,nonefit_mh1)
summary(nonefit_mh1)

xtabs(none_fits ~ conditions+dist, data=data_preprocessed_for_analysis_of_none_fit_huashan)/xtabs(~ conditions+dist, data=data_preprocessed_for_analysis_of_none_fit_huashan)
#SF

summary(nonefit_mh1)
nonefit_mh0 <- glmer(none_fits ~ 1 + (1|id) + (1|item), data=data_preprocessed_for_analysis_of_none_fit_huashan, 
                    family=binomial, nAGQ=0, control = glmerControl(optimizer ='bobyqa'))
summary(nonefit_mh0)
anova(nonefit_mh0,nonefit_mh1)
#mm <- allFit(nonefit_m0)
#nonefit_m1 <- update(nonefit_m0, .~.-(1|item)) 
nonefit_m2 <- glm(none_fits ~ conditions*dist, data = data_preprocessed_for_analysis_of_none_fit_huashan, family = binomial) 
summary(nonefit_m2)
anova(nonefit_mh1,nonefit_m2)
missing_dist <- which(is.na(data_preprocessed_for_analysis_of_none_fit_huashan$dist)==TRUE)
data_preprocessed_for_analysis_of_none_fit_huashan[missing_dist,]
#plot(allEffects(nonefit_m1))

#FS#play around with transformations
dat <- data_preprocessed_for_analysis_of_none_fit_huashan
hist(dat$slider_value)
qqnorm(dat$slider_value)
dat$sli<-(dat$slider_value+0.5)/101
dat$transli<-qlogis(jitter(dat$sli))
hist(dat$transli)
qqnorm(dat$transli)
m0 <- lmer(dat$sli ~ conditions*dist+ (1|id) + (1|item), data = dat) 
m0_log <- lmer(log(dat$sli) ~ conditions*dist+ (1|id) + (1|item), data = dat) 
m0_trans <- lmer(dat$transli ~ conditions*dist+ (1|id) + (1|item), data = dat) 
par(mfrow=c(1,3))
qqnorm(residuals(m0))
qqnorm(residuals(m0_trans))
qqnorm(residuals(m0_log))
par(mfrow=c(1,1))
plot(m0_log)
summary(m0)
#SF

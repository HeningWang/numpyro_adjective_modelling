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


data <- read.csv(file = "../../../dataset/taishan_full_annotiert.csv")
subj_info <-read.csv(file = "../../../dataset/taishan_subj_info.csv")

#show how many pps
sum(xtabs(~id,data=subj_info))


#show conditions
xtabs(~id+conditions, data=data)




#exclude pps with more than three s.e. away from mean experiment time
hist(subj_info$time_in_minutes)
mean(subj_info$time_in_minutes)
sd(subj_info$time_in_minutes)
x <- mean(subj_info$time_in_minutes) + 3 * sd(subj_info$time_in_minutes)
y <- mean(subj_info$time_in_minutes) - 3 * sd(subj_info$time_in_minutes)
filter(subj_info, time_in_minutes < y | time_in_minutes > x)
exclude <- filter(subj_info, time_in_minutes < y | time_in_minutes > x)$id


#exclude pps with more than 20% identical responses
data_ex <- as.data.frame(xtabs(~id+annotation, data = data))
data_ex$Freq <- as.numeric(data_ex$Freq)
exclude <- append(exclude, levels(droplevels(filter(data_ex, Freq > 108)$id)))


#exclude pps with more than 10% NA responses
data_na <- subset(data, is.na(data$annotation))
exclude <- append(exclude, names(which(xtabs(~id,data_na)>10)))

#subset excluded pps
data <- subset(data, !data$id %in% exclude)

# show lists for nachhebung
xtabs(~id+list, data=data)
20 - round(xtabs(~list, data=data)/135)
sum(xtabs(~id,data=data)/81)



#subset filler items
data_filler <- subset(data, str_sub(data$conditions, 1, 1) == "f")
data <- droplevels(subset(data, str_sub(data$conditions, 1, 1) != "f"))


as.character(data$conditions)
data$conditions[data$conditions=="erdf"] <- "frdf"
data$conditions[data$conditions=="erdc"] <- "frdc"
data$conditions[data$conditions=="ercf"] <- "frcf"

data$conditions[data$conditions=="zrdf"] <- "srdf"
data$conditions[data$conditions=="zrdc"] <- "srdc"
data$conditions[data$conditions=="zrcf"] <- "srcf"



# show how many NAs
data$none_fits <- ifelse(is.na(data$annotation), 1, 0)
xtabs(~none_fits, data = data)

#coercion NA value
data <- subset(data, !is.na(data$annotation))


# set up factors
data$combination <- as.factor(ifelse(str_sub(data$conditions, 3, 4)=="cf", "color_form",
                                     ifelse(str_sub(data$conditions, 3, 4)=="dc", "dimension_color",
                                            "dimension_form")))


data$relevant_property <- as.factor(ifelse(str_sub(data$conditions, 1, 1)== "f", "first",
                                           ifelse(str_sub(data$conditions, 1, 1)== "s", "second",
                                                  "both")))

data$annotation <- as.factor(data$annotation)

data$dist <- ifelse(data$list<4, "sharp", "blurred")
data$dist <- as.factor(data$dist)


xtabs(~annotation, data)
#subset freq
data_sharp <- subset(data, dist == "sharp")

dat <- as.data.frame(xtabs(~conditions+annotation, data = data_sharp))
data_blurred <- subset(data, dist == "blurred")
dat_blurred <- as.data.frame(xtabs(~conditions+annotation, data = data_blurred))

saveRDS(data, file = "data_preprocessed.Rdata")


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
library(lattice)
library(MASS)
rm(list=ls())

plots_dir<- file.path(getwd(),"plots")
if(!dir.exists(plots_dir)) dir.create(plots_dir, recursive = TRUE, showWarnings = FALSE)
setwd("~/GitHub/huashan/analysis")

data <- read.csv(file = "../data/huashan.csv")
subj_info <-read.csv(file = "../data/huashan_subj_info.csv")


#do stuff with subj_info here:
#...

#show how many pps
sum(xtabs(~id,data=data)/180)

# show lists
xtabs(~id+list, data=data)
round(xtabs(~list, data=data)/180)

#show conditions
xtabs(~id+conditions, data=data)

# show distribution for randomized left and right per trials
xtabs(~id+leftright_trial, data=data) #TODO: check the following extrem values: 6,5,4,3,2,1, because of missing slider values
sum_left <- sum(xtabs(~leftright_trial=="1left", data=data)[2])
sum_right <- sum(xtabs(~leftright_trial=="1right", data=data)[2])
z <- c(sum_left,sum_right)
labels <- c("left","right")
piepercent<- round(100*z/sum(z), 1)
pie(z, labels=piepercent, main="Distribution of left and right", col = rainbow(length(z)))
legend("topright", c("left","right"), cex = 0.8, fill = rainbow(length(z)))


# set up levels and factors 
data$conditions <- droplevels(as.factor(data$conditions))
data$id <- droplevels(as.factor(data$id))
data$slider_value <- as.numeric(data$slider_value)
data$dist <- ifelse(data$list<4, "sharp", "blurred")
data$dist <- droplevels(as.factor(data$dist))

#coercion slider value when missing list to NA
missing_list <- which(is.na(data$list)==TRUE)
data[missing_list,]$slider_value <- NA

#check none fit and add it to df
data$none_fits <- ifelse(data$slider_value==-1 | is.na(data$list), 1, 0)
xtabs(~item+conditions+none_fits, data)

#coercion slider value when none fit to NA
data$slider_value <- ifelse(data$slider_value==-1, NA, data$slider_value)

#exclude pps with missing list more than than 5:
xtabs(~id, data=data[is.na(data$list),])>5 
exclude <- c("6242d7b615bf4", "6242d82c12542", "6242e450b3b5c", "624d80a5bdbf2"
             ,"62557f3d79780","626167b76e314")

#exclude pps with more than three s.e. away from mean experiment time
hist(subj_info$time_in_minutes)
mean(subj_info$time_in_minutes)
sd(subj_info$time_in_minutes)
x <- mean(subj_info$time_in_minutes) + 3 * sd(subj_info$time_in_minutes)
y <- mean(subj_info$time_in_minutes) - 3 * sd(subj_info$time_in_minutes)
filter(subj_info, time_in_minutes < y | time_in_minutes > x)
exclude <- append(exclude,c("6242bd7514401",
                            "624da2cd1ef0b",
                            "624b22243fd06"))

# exclude pps with more than 3 false responses to control items
# item 34-38 condition ferdc, fercf, ferdf, fzrcf, fzrdc, fzrdf
# b.t.w item 38 condition fzrcf should be none fit
filter(data, item %in% c(34,35,36,37,38) & conditions %in% c("ferdc","fercf","ferdf","fzrdc","fzrdf") 
       & none_fits == 1)
filter(data, item %in% c(38) & conditions %in% c("fzrcf") 
       & none_fits==0)
exclude <- append(exclude,c("624b303d6bbb7","6242d016c0e11","624b0affa1dad",
                            "624b0b35b7990","62420e37579a8","626167b76e314",
                            "624d80a5bdbf2","62420e37579a8","6242d7b615bf4"
                            ))

# exclude pps with extrem lang reading times
##split rt list
rt_as_lists <- with(data, strsplit(gsub("\\[|\\]", "", read_time), ","))
unlist(rt_as_lists)
len <- nrow(data)
rt <- rep(NA, len)
for (i in 1:len){
  rt[i] <- as.numeric(unlist(rt_as_lists[i]))
   }
rt <- as.data.frame(rt)
data <- cbind(data[,1:3], rt, data[,5:10])
mean(data$rt)
plot(data$rt)
max(data$rt)
filter(data,rt>100000)
exclude <- append(exclude,c("6241fc4a4e8ee","6242bd7514401","6242197550443",
                            "624b22243fd06","6242cc346032c","6242173eeab01",
                            "6242d1a5bc481","624b10a84ecda","624d710db494a",
                            "625596dfdfa50","626167b76e314","626179bd7e471 "))
# ...after subsetting filler items
# exclude more than 5 none fits in experimental items
xtabs(~data$slider_value==-1, data = data)
xtabs(~none_fits + id, data = data)
#"624da2cd1ef0b","624d94a33c1bb","624d6af77ae58""624da2cd1ef0b","624b303d6bbb7","6242d2a1afd92","6242d016c0e11",
exclude <- append(exclude,
                  c("6242d0cc141b3","6242cc346032c","6242c9a5256b1",
                    "6242c6ebdcebd","6242197550443","6242140018a74","6241eec9bf12f","62408e90eaa0f",
                    "6241fe6a604b4","6241f580649f8"))

# exclude pps und subset data
data_exclude <- subset(data, id %in% exclude)
round(xtabs(~id+list, data = data_exclude)/81) # list for second fill up
round(xtabs(~list, data = data_exclude)/81) # list for second fill up
data <- subset(data, !(id %in% exclude))

# subset NA from data
data <- subset(data, !is.na(data$slider_value))

# set up fillers and subset filler from data
data_filler <- subset(data, str_sub(data$conditions, 1, 1) == "f")
data <- droplevels(subset(data, str_sub(data$conditions, 1, 1) != "f"))

#TODO: adjust later, rename variable
data$prefer_first_1st <- ifelse(data$leftright_trial=="1left", 100-data$slider_value, data$slider_value)
data$prefer_first_1st <- data$prefer_first_1st - 50
#aggregate data
aggregate(data$prefer_first_1st, list(data$conditions), FUN = function(x){c(mean(x), sd(x)/sqrt(length(x)))})
aggregate(data$prefer_first_1st, list(data$conditions, data$dist), FUN = function(x){c(mean(x), sd(x)/sqrt(length(x)))})




data$combination <- as.factor(ifelse(str_sub(data$conditions, 3, 4)=="cf", "color_form",
                           ifelse(str_sub(data$conditions, 3, 4)=="dc", "dimension_color",
                                  "dimension_form")))


data$relevant_property <- as.factor(ifelse(str_sub(data$conditions, 1, 1)== "e", "first",
                                           ifelse(str_sub(data$conditions, 1, 1)== "z", "second",
                                           "both")))
                                           

ratings_aggregated<-aggregate(prefer_first_1st~combination+relevant_property+dist, data=data, mean)
#TODO: find another way to compute ses (SEwithin?)
ses_ratings<-aggregate(prefer_first_1st~combination+relevant_property+dist, data=data, function(x) {sd(x)/sqrt(length(x))})
ratings_aggregated<-cbind(ratings_aggregated,ses_ratings[,4])

names(ratings_aggregated)[5]<-"se"

# make bar plot for descriptive analysis
p <- ggplot(ratings_aggregated, aes(x=combination, y=prefer_first_1st, fill=relevant_property)) +
  ylab("preference for first adjective in position I")+
  ggtitle("Mittelwerte der Akzeptabilitätswerte für Präferenzen der ersten Adjektiven aus Kombinationen \n an der Position I in der scharfen oder unscharfen Größenverteilung")+
  geom_bar(stat="identity", position=position_dodge()) +
  geom_errorbar(aes(ymin=prefer_first_1st-se, ymax=prefer_first_1st+se), width=.2,
                position=position_dodge(.9))+
  facet_wrap(~dist)+
  scale_fill_brewer(palette="Paired") + theme_minimal()+
  theme(axis.text.x = element_text(angle = 45, hjust=1))
p
pdf(file = file.path(plots_dir,"mean_preference_facet.pdf"))
p
dev.off()


#export current data for none fit analysis
save(data, file="data_preprocessed_huashan.Rdata")


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


data <- read.csv(file = "../data/taishan_full_annotiert.csv")
subj_info <-read.csv(file = "../data/taishan_subj_info.csv")

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

# set up colors according to factors
colors_c <- brewer.pal(5, "Set1")
names(colors_c) <- c("C","CD","CDF","CF","CFD")
colors_d <- brewer.pal(5, "Set3")
names(colors_d) <- c("DC","D","DCF","DF","DFC")
colors_f <- brewer.pal(5, "Set2")
names(colors_f) <- c("F","FC","FD","FDC","FCD")
my_colors <- append(colors_c, colors_d)
my_colors <- append(my_colors, colors_f)

custom_colors <- scale_fill_manual(name = "annotation", 
                                   #levels(dat$annotation), 
                                   values = my_colors)


xtabs(~conditions+annotation,data = data_sharp)
xtabs(~conditions+annotation,data = data_blurred)



o <- ggplot(data = dat, aes(x = conditions, y = Freq ,fill = annotation))+
  geom_bar(position = "fill", stat = "identity")+
  ylab("frequency")+
  custom_colors+ 
  guides(fill=guide_legend(ncol=2))+
  labs(annotation = "annotation")+ theme_minimal()+
  theme(text = element_text(size = 20),
    axis.text.x = element_text(angle = 45, hjust=1, size = 30),
    #axis.title = element_text(size = 30),
        legend.key.size = unit(1.2, 'cm'))
o
pdf(file = file.path(plots_dir,"frequency_annotation_sharp.pdf"))
o
dev.off()



  
  
# set up colors according to factors

custom_colors <- scale_fill_manual(name = levels(dat_blurred$annotation), values = my_colors)

p <- ggplot(data = dat_blurred, aes(x = conditions, y = Freq ,fill = annotation))+
  geom_bar(position = "fill", stat = "identity")+
  ylab("frequency")+
  guides(fill=guide_legend(ncol=2))+
  custom_colors + theme_minimal() +
  theme(text = element_text(size = 20),
        axis.text.x = element_text(angle = 45, hjust=1, size = 30),
        #axis.title = element_text(size = 30),
        legend.key.size = unit(1.2, 'cm'))
p
pdf(file = file.path(plots_dir,"frequency_annotation_blurred.pdf"))
p
dev.off()

# barplot of frequency of annotations
q <- barchartGC(~annotation, data = data, main = "frequency of different adjective orders from annotation")
q







#data <- subset(data, !is.na(data$coding))
#m1 <- glmer(coding~relevant_property*dist+(1|id)+(1|item), data = data, family = binomial)
#anova(m1)
#summary(m1)


#ifelse(substr(data$annotation, 1, 1) == "D", 1 , ifelse(substr(data$annotation, 2, 2) == "D", 2 ))
#m1.1 <- update(m1, .~.-relevant_property:dist)
#anova(m1.1,m1)
#m1.2 <- update(m1.1, .~.-relevant_property)
#anova(m1.2,m1.1)
#C F relevant_property chi(2) = 83.713, p < 0.001
#CF FC chi0.98650  0.9865

# testing hypothesis relating to combination
#data_combination <- subset(data, relevant_property == "both")
# dependent variable coding: (in multiple adjective string) D first as 1, D not first as 0, D not in it as NA
#data_combination$coding <- ifelse(data_combination$annotation == "DC"|
#                        data_combination$annotation == "DCF"|
#                        data_combination$annotation == "DF"|
#                        data_combination$annotation == "DFC", 1, ifelse(data_combination$annotation == "CD" |
#                                                              data_combination$annotation == "FD"|
#                                                              data_combination$annotation == "CDF"|
#                                                              data_combination$annotation == "CFD"|
#                                                              data_combination$annotation == "FCD"|
#                                                              data_combination$annotation == "FDC", 0, NA))



#m_combination <- glmer(coding~combination*dist+(1|id)+(1|item), 
#                       data = data_combination, 
#                       family = binomial,
#                       glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 100000)))

#summary(m_combination)

#m_combination_1 <- update(m_combination, .~.-combination:dist)
#anova(m_combination_1, m_combination) #9.1463  2    0.01033 *

#m_combination_2 <- update(m_combination_1, .~.-combination)
#anova(m_combination_2, m_combination_1) #17.135  2  0.0001902 ***

#m_combination_3 <- update(m_combination_2, .~.-dist)
#anova(m_combination_3, m_combination_2) #0.05691 .


# testing hypothesis relating to dist
# ?why this coding

# subset data in combination: dimension_color, dimension_form
#data_d <- subset(data, combination == "dimension_form" | combination == "dimension_color")
# subset data in relevant_property: first
#data_d_first <- subset(data_d, relevant_property == "first")


#data_d_first$coding <- ifelse(data_d_first$annotation == "D", 1, 
#                                ifelse(data_d_first$annotation == "DC", 2,
#                                ifelse(data_d_first$annotation == "DF", 3,
#                                ifelse(data_d_first$annotation == "DCF", 4,
#                                ifelse(data_d_first$annotation == "DFC", 5, 0)))))
#data_d_first$coding <- as.factor(data_d_first$coding)
#m_dist <- clmm(coding~combination*dist+(1|id)+(1|item), data = data_d_first)
#summary(m_dist)

#m_dist_1 <- update(m_dist, .~.-combination:dist)
#anova(m_dist_1, m_dist) #23.031  1  1.594e-06 ***

#m_dist_2 <- update(m_dist_1, .~.-combination)
#anova(m_dist_2, m_dist_1) #14.748  1  0.0001229 ***

#m_dist_3 <- clmm(coding~1+(1|id)+(1|item), data = data_d_first)
#anova(m_dist_3, m_dist_2) #0.2822  1     0.5953

# testing hypothesis relevant_property and combination with clm
# subset data in combination
# full model is of all there relevant_properties; 
# abhängige variabl coding: C and F, C and D, D and F, CF and FC, DC and CD, DF and FD

data$coding <- ifelse(data$annotation == "D"|
                      data$annotation == "DC"|
                      data$annotation == "DF"|
                      data$annotation == "DCF"|
                      data$annotation == "DFC", "D_first", 
                              ifelse(data$annotation == "C"|
                                     data$annotation == "CF"|
                                     data$annotation == "CD"|
                                     data$annotation == "CDF"|
                                     data$annotation == "CFD", "C_first", "F_first"))

data$coding2 <- ifelse(
                        data$annotation == "DC"|
                        data$annotation == "DF"
                            , "D_first", 
                      ifelse(
                               data$annotation == "CF"|
                               data$annotation == "CD", "C_first", 
                               ifelse(data$annotation == "FC"|
                                        data$annotation == "FD"
                                        , "F_first", NA)))

data$coding <- as.factor(data$coding)
data$coding2 <- as.factor(data$coding2)

m_relevant <- clmm(coding~combination*dist*relevant_property + (1|item) + (1|id), data = data)
saveRDS(m_relevant, "model.rds")
m_relevant <- readRDS("model.rds")
plot_model(m_relevant, type = "pred")
summary(m_relevant)
step(m_relevant)
saveRDS(m_relevant, "m0.rds")
m_relevant <- readRDS("m0.rds")
plot_model(m_relevant, type = "int")


m_relevant_1 <-  update(m_relevant, .~.-combination:dist:relevant_property)
anova(m_relevant_1,m_relevant) #30.96  4  3.119e-06 ***

m_relevant_2 <-  update(m_relevant_1, .~.-dist:relevant_property)
anova(m_relevant_2,m_relevant_1) # 4.4704  2      0.107


data_two_way <- subset(data, combination == "dimension_color" )

data_sharp <- subset(data, dist == "sharp")
data_blurred <- subset(data, dist == "blurred")



m_blurred <- clmm(coding2~combination*relevant_property + (1|item) + (1|id), data = data_blurred)
summary(m_blurred)
m_blurred_1 <-  update(m_blurred, .~.-combination:relevant_property)
anova(m_blurred_1,m_blurred) #182.37  4  < 2.2e-16 ***

data_blurred_cf <- subset(data_blurred, combination == "color_form")
data_blurred_dc <- subset(data_blurred, combination == "dimension_color")
data_blurred_df <- subset(data_blurred, combination == "dimension_form")
data_blurred_cf$combination <- as.factor(droplevels(data_blurred_cf$combination))

m_blurred_cf <- clmm(coding2~relevant_property + (1|item) + (1|id), data = data_blurred_cf)
summary(m_blurred_cf)
m_blurred_cf_1 <-  clmm(coding2~1 + (1|item) + (1|id), data = data_blurred_cf)
anova(m_blurred_cf_1,m_blurred_cf)#13.665  2   0.001078 **


data_blurred_br <- subset(data_blurred, relevant_property == "both")
m_combination <- clmm(coding2~combination + (1|item) + (1|id), data = data_blurred_br)
summary(m_combination)
m_combination_1 <-  clmm(coding2~1 + (1|item) + (1|id), data = data_blurred_br)
anova(m_combination_1,m_combination) #320.73  2 
plot_model(m_combination, type = "pred")



m_sharp <- clmm(coding2~combination*relevant_property + (1|item) + (1|id), data = data_sharp)
summary(m_sharp)
m_sharp_1 <-  update(m_sharp, .~.-combination:relevant_property)
anova(m_sharp_1,m_sharp) #70.681  4

data_sharp_cf <- subset(data_sharp, combination == "color_form")
data_sharp_dc <- subset(data_sharp, combination == "dimension_color")
data_sharp_df <- subset(data_sharp, combination == "dimension_form")
data_sharp_cf$combination <- as.factor(droplevels(data_sharp_cf$combination))

m_sharp_cf <- clmm(coding2~relevant_property + (1|item) + (1|id), data = data_sharp_cf)
summary(m_sharp_cf)
m_sharp_cf_1 <-  clmm(coding2~1 + (1|item) + (1|id), data = data_sharp_cf)
anova(m_sharp_cf_1,m_sharp_cf)

w <- plot_model(m_relevant, type = "pred",
                title = "Wahrscheinlichkeit der produzierten Adjektivenreihenfolgen \n unter dem Haupteffekt Größenverteilung", 
                axis.title  = "P(outcome)")
w  + theme_sjplot(label_angle(angle.x = 45), font_size(labels.x = 1))

w$combination$data$response.level <- as.factor(c("C first", "D first", "F first"))
w$relevant_property$data$response.level <- as.factor(c("C first", "D first", "F first"))
w$dist$data$response.level <- as.factor(c("C first", "D first", "F first"))
w



m_sharp_df <- clmm(coding2~relevant_property + (1|item) + (1|id), data = data_sharp_df)
summary(m_sharp_df)
m_sharp_df_1 <-  clmm(coding2~1 + (1|item) + (1|id), data = data_sharp_df)
anova(m_sharp_df_1,m_sharp_df)

m_sharp_dc <- clmm(coding2~relevant_property + (1|item) + (1|id), data = data_sharp_dc)
summary(m_sharp_dc)
m_sharp_dc_1 <-  clmm(coding2~1 + (1|item) + (1|id), data = data_sharp_dc)
anova(m_sharp_dc_1,m_sharp_dc)


data_er <- subset(data, relevant_property == "first")
data_er_d <- subset(data, combination == "dimension_color" | combination == "dimension_form")
data_df <- subset(data, combination == "df")
data_er_d$combination <- as.factor(droplevels(data_er_d$combination))

m_er <- clmm(coding2~combination*dist + (1|item) + (1|id), data = data_er)
summary(m_er)
m_er_1 <-  clmm(coding2~combination+dist + (1|item) + (1|id), data = data_sharp_dc)
anova(m_sharp_dc_1,m_sharp_dc)

m_er_d <- clmm(coding2~combination*dist + (1|item) + (1|id), data = data_er_d)
summary(m_er_d)
m_er_d_1 <-  clmm(coding2~combination+dist + (1|item) + (1|id), data = data_er_d)
anova(m_er_d_1,m_er_d) #0.3892  1     0.5327 
m_er_d_2 <-  clmm(coding2~combination + (1|item) + (1|id), data = data_er_d)
anova(m_er_d_2,m_er_d_1) #0.92  1     0.3375
plot_model(m_er_d, type = "int")


data_sharp_br <- subset(data_sharp, relevant_property == "both")

m_combination <- clmm(coding2~combination + (1|item) + (1|id), data = data_sharp_br)
summary(m_combination)
m_combination_1 <-  clmm(coding2~1 + (1|item) + (1|id), data = data_sharp_br)
anova(m_combination_1,m_combination) #171.03  2  < 2.2e-16 ***

m <- plot_model(m_combination, type = "pred",
           title = "Wahrscheinlichkeit der produzierten Adjektivenreihenfolgen unter dem Haupteffekt Kombination", 
           axis.title  = "P(outcome)")
m  + theme_sjplot(label_angle(angle.x = 45), font_size(labels.x = 1))

m$combination$data$response.level <- as.factor(c("C first", "D first", "F first"))
m




# why three way interaction? 1. not predicted by hypothesis 
#2. perhaps each factors contribute to variance in its own way
#3. we do a posthoc analysis starting by setset data in dist

# refitting model in data_sharp for two way interaction between combination and relevant property
data_sharp <- subset(data, dist == "sharp")
m_sharp_0 <- clmm(coding~relevant_property*combination + (1|item) + (1|id), data = data_sharp)

summary(m_sharp_0)
m_sharp_1 <-  update(m_sharp_0, .~.-combination:relevant_property)
anova(m_sharp_1,m_sharp_0)#2434.5  4  < 2.2e-16 ***
# there is significant two way interaction between combination and relevant property unter sharp
m <- plot_model(m_sharp_0, type = "int", 
                title = "Interaction between relevant property and combination \n after subsetting data by distribution ''sharp''", 
                axis.title  = "Frequncy",
                grid = TRUE,
                grid.breaks = 100,
                set_theme(axis.angle.x = 45)) + theme(axis.text.x = element_text(angle = 45, hjust=1))
m$data$response.level <- as.factor(c("C first", "D first", "F first"))
m

# this interaction is due to relevant property hypothesis 


# refitting model in both relevant for two way interaction between combination and dist
data_both <- subset(data, relevant_property == "both")
m_both_0 <- clmm(coding~ combination*dist + (1|item) + (1|id), data = data_both)
summary(m_both_0)
m_both_1 <-  update(m_both_0, .~.-combination:dist)
anova(m_both_1,m_both_0)#23.871  2  6.554e-06 ***
# there is significant two way interaction between combination and dist unter both
m <- plot_model(m_both_0, type = "int", 
                title = "Interaction between dist(ribution) and combination \n after subsetting data by relevant property ''both''", 
                axis.title  = "Frequncy",
                grid = TRUE,
                grid.breaks = 100,
                set_theme(axis.angle.x = 45)) + theme(axis.text.x = element_text(angle = 45, hjust=1))
m$data$response.level <- as.factor(c("C first", "D first", "F first"))
m
# this interaction is due to combination hypothesis 


# refitting model in both relevant for two way interaction between relevant property and dist
data_df <- subset(data, combination == "dimension_form")
m_df_0 <- clmm(coding~ relevant_property*dist + (1|item) + (1|id), data = data_df)
summary(m_df_0)
m_df_1 <-  update(m_df_0, .~.-relevant_property:dist)
anova(m_df_1,m_df_0)#27.35  2  1.151e-06 ***
# there is significant two way interaction between combination and dist unter both
m <- plot_model(m_df_0, type = "int", 
                title = "Interaction between dist(ribution) and relevant property \n after subsetting data by combination ''dimension and form''", 
                axis.title  = "Frequncy",
                grid = TRUE,
                grid.breaks = 100,
                set_theme(axis.angle.x = 45)) + theme(axis.text.x = element_text(angle = 45, hjust=1))
m$data$response.level <- as.factor(c("C first", "D first", "F first"))
m
# this interaction is due to dist hypothesis 

data_dc <- subset(data, combination == "dimension_color")
m_dc_0 <- clmm(coding~ relevant_property*dist + (1|item) + (1|id), data = data_dc)
summary(m_dc_0)
m_dc_1 <-  update(m_dc_0, .~.-relevant_property:dist)
anova(m_dc_1,m_dc_0)#27.35  2  1.151e-06 ***
# there is significant two way interaction between combination and dist unter both
plot_model(m_dc_0, type = "int", title = "interaction between relevant_property and dist(ribution)", axis.title  = "preference for 1st adj at position I")
m_dc_2 <-  update(m_dc_1, .~.-dist)
anova(m_dc_2,m_dc_1) #0.1869  1     0.6655 with dc there is no sign for dist effect





# subset data furthermore and refitting the model unter two way interaction dist and combination, both
data_both_sharp <- subset(data_both, dist == "sharp")
m_both_sharp_0 <- clmm(coding~combination + (1|item) + (1|id), data = data_both_sharp)
summary(m_both_sharp_0)
m_both_sharp_1 <- clmm(coding~1 + (1|item) + (1|id), data = data_both_sharp)
anova(m_both_sharp_1, m_both_sharp_0) #272.31  2  < 2.2e-16 *** this confirm the combination hypothesis

data_both_df <- subset(data_both, combination == "dimension_form")
m_both_df_0 <- clmm(coding~dist + (1|item) + (1|id), data = data_both_df)
summary(m_both_df_0)
m_both_df_1 <- clmm(coding~1 + (1|item) + (1|id), data = data_both_df)
anova(m_both_df_1, m_both_df_0) #13.131  1  0.0002905 *** this confirm the dist hypothesis
plot_model(m_both_df_0, type = "pred", title = "interaction between relevant_property and dist(ribution)", axis.title  = "preference for 1st adj at position I")


# subset data furthermore and refitting the model unter two way interaction relevant property and combination, sharp
data_sharp_cf <- subset(data_sharp, combination == "color_form")
m_sharp_cf_0 <- clmm(coding~relevant_property + (1|item) + (1|id), data = data_sharp_cf)
summary(m_sharp_cf_0)
m_sharp_cf_1 <- clmm(coding~1 + (1|item) + (1|id), data = data_sharp_cf)
anova(m_sharp_cf_1, m_sharp_cf_0) #1005  2  < 2.2e-16 *** this can confirm the effect of relevantproperty

















m_relevant_2 <-  update(m_relevant_1, .~.-combination:relevant_property)
anova(m_relevant_2,m_relevant_1) #4519.8  4  < 2.2e-16 ***

m_relevant_3 <-  update(m_relevant_2, .~.-combination:dist)
anova(m_relevant_3,m_relevant_2) #5.364  2    0.06843 .

m_relevant_4 <-  update(m_relevant_3, .~.-dist:relevant_property)
anova(m_relevant_4,m_relevant_3) # 6.3838  2    0.04109 *

m_relevant_5 <-  update(m_relevant_4, .~.-combination)
anova(m_relevant_5,m_relevant_4) # 442.21  2  < 2.2e-16 ***

m_relevant_6 <-  update(m_relevant_5, .~.-dist)
anova(m_relevant_6,m_relevant_5) # 0.06234 .

m_relevant_7 <-  clmm(coding~1 + (1|item) + (1|id), data = data)

anova(m_relevant_7,m_relevant_6) # 568.76  2  < 2.2e-16 ***


## testing hypothesis relating to dist (alternative 2)
data_d_first$coding <- ifelse(data_d_first$annotation == "DC"|
                                    data_d_first$annotation == "DCF"|
                                    data_d_first$annotation == "DF"|
                                    data_d_first$annotation == "DFC", 1, ifelse(data_d_first$annotation == "CD" |
                                                                                      data_d_first$annotation == "FD"|
                                                                                      data_d_first$annotation == "CDF"|
                                                                                      data_d_first$annotation == "CFD"|
                                                                                      data_d_first$annotation == "FCD"|
                                                                                      data_d_first$annotation == "FDC", 0, NA))
data_d_first$coding <- as.factor(data_d_$coding)
m_dist_alt <- glmer(coding~combination*dist+(1|id)+(1|item), 
                       data = data_d_first, 
                       family = binomial,
                       glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 100000)))
summary(m_dist_alt)

m_dist_alt_1 <- update(m_dist_alt, .~.-combination:dist)
anova(m_dist_alt_1, m_dist_alt) #1.8231  1     0.1769

m_dist_alt_2 <- update(m_dist_alt_1, .~.-combination)
anova(m_dist_alt_2, m_dist_alt_1) #2.076  1     0.1496

m_dist_alt_3 <- update(m_dist_alt_2, .~.-dist)
anova(m_dist_alt_3, m_dist_alt_2) #0.3004  1     0.5837




## testing hypothesis relating to dist (alternative 3)
data_d_second <- subset(data_d, relevant_property == "second")

data_d_second$coding <- ifelse(data_d_second$annotation == "D", 1, 
                              ifelse(data_d_second$annotation == "DC", 2,
                                     ifelse(data_d_second$annotation == "DF", 3,
                                            ifelse(data_d_second$annotation == "DCF", 4,
                                                   ifelse(data_d_second$annotation == "DFC", 5, 0)))))
data_d_second$coding <- as.factor(data_d_second$coding)
m_dist_2 <- clmm(coding~combination*dist+(1|id)+(1|item), data = data_d_second)
summary(m_dist_2)

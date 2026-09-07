# Reproduce the existing descriptive slider summary with common CSP labels.
# Run from paper/: Rscript scripts/plot_slider_empirical.R
suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(readr)
})
source('scripts/csp_figure_style.R')
slider <- read_csv('data/slider_empirical.csv', show_col_types=FALSE)
# Same trial-rating mean and t interval as plotting_main_paper.R, Figure 1.
summary <- slider %>%
  mutate(slider_centered=human_slider-.5) %>%
  group_by(relevant_property, sharpness) %>%
  summarise(estimate=mean(slider_centered),
            lower=estimate-qt(.975,n()-1)*sd(slider_centered)/sqrt(n()),
            upper=estimate+qt(.975,n()-1)*sd(slider_centered)/sqrt(n()),
            n_ratings=n(), .groups='drop')
write_csv(summary,'data/slider_empirical_figure_summary.csv')
plot_data <- read_csv('data/slider_empirical_figure_summary.csv',show_col_types=FALSE) %>%
  mutate(context=factor(relevant_property,levels=c('first','both','second'),
                        labels=c('Size sufficient','Both necessary','Colour sufficient')),
         discriminability=factor(sharpness,levels=c('blurred','sharp'),labels=c('Low','High')))
p <- ggplot(plot_data,aes(x=context,y=estimate,fill=discriminability)) +
  geom_col(position=position_dodge(.8),width=.7,alpha=.85) +
  geom_errorbar(aes(ymin=lower,ymax=upper),position=position_dodge(.8),width=.2,linewidth=.6) +
  geom_hline(yintercept=0,linetype='dashed',colour=CSP_COLORS[['text']]) +
  scale_fill_manual(values=c(Low=CSP_COLORS[['accent_dark']],High=CSP_COLORS[['main']]),name='Size discriminability') +
  labs(x='Referential context',y='Mean centred size-first rating') +
  theme_csp() + theme(legend.position='top')
save_csp_pdf(p,'figures/slider_empirical.pdf',7,4.5)

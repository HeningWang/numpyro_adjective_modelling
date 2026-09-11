# Plot frozen behavioural posterior summaries; no model fitting or aggregation.
# Run from paper/: Rscript scripts/plot_slider_interactions.R
suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(readr)
})
source('scripts/csp_figure_style.R')
plot_data <- read_csv('data/slider_interaction_figure_summary.csv', show_col_types=FALSE) %>%
  mutate(study=factor(study,levels=c('study1a_slider','study1b_slider'),
                      labels=c('Study 1a','Study 1b')),
         context=factor(context,levels=c('size_sufficient','both_necessary','colour_sufficient'),
                        labels=c('Size\nsufficient','Both\nnecessary','Colour\nsufficient')),
         discriminability=factor(discriminability,levels=c('low','high'),labels=c('Low','High')))
p <- ggplot(plot_data,aes(context,median,colour=discriminability,group=discriminability)) +
  geom_hline(yintercept=0,linetype='dashed',colour=CSP_COLORS[['text']],linewidth=.4) +
  geom_line(linewidth=.65,position=position_dodge(.12)) +
  geom_errorbar(aes(ymin=lower,ymax=upper),width=.10,linewidth=.6,position=position_dodge(.12)) +
  geom_point(size=2.7,position=position_dodge(.12)) +
  facet_wrap(~study,nrow=1) +
  scale_colour_manual(values=c(Low=CSP_COLORS[['accent_dark']],High=CSP_COLORS[['main']]),
                      name='Size discriminability') +
  scale_y_continuous(breaks=seq(-.1,.4,.1)) +
  labs(x='Referential context',y='Centred size-first rating') +
  theme_csp() + theme(legend.position='top',panel.grid.minor=element_blank())
save_csp_pdf(p,'figures/study1_slider_replication.pdf',8.5,4.5)

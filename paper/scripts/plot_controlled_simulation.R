# Run from paper/. Frozen summaries only; no model execution or new contrasts.
suppressPackageStartupMessages({library(ggplot2); library(dplyr); library(readr)})
source('scripts/csp_figure_style.R')
summaries <- read_csv('data/simulation_controlled_overall.csv',show_col_types=FALSE) %>% filter(metric=='listener_advantage')
semantics_levels <- c('Context-fixed','Context-updating')
palette <- setNames(unname(CSP_COLORS[c('main','emphasis')]),semantics_levels)
shapes <- setNames(c(16,17),semantics_levels)
prepare <- function(d) d %>% mutate(semantics=factor(recode(semantics,fixed='Context-fixed',updating='Context-updating'),levels=semantics_levels),estimate=100*mean,lower=100*lower_mc95,upper=100*upper_mc95)
settings <- tibble(
 rule=c('terminal',rep('prefix_k0',4)),
 support=c('two','two','four','fifteen','fifteen'),
 parameter_set=c(rep('original_grid',4),'fitted_constants'),
 order='neutral',
 setting=c('Completed-utterance baseline\n2 orders','Global model\n2 orders',
           'Global model\n4 utterances','Global model\n15 utterances',
           'Global model, 15 utterances\nProduction constants'))
main <- settings %>% inner_join(summaries,by=c('rule','support','parameter_set','order')) %>% prepare()
stopifnot(nrow(main)==10L, all(complete.cases(main)))
write_csv(main,'data/simulation_controlled_main_figure.csv')
main <- read_csv('data/simulation_controlled_main_figure.csv',show_col_types=FALSE) %>%
 mutate(setting=factor(setting,levels=rev(settings$setting)),
        semantics=factor(semantics,levels=semantics_levels))
p <- ggplot(main,aes(x=estimate,y=setting,colour=semantics,shape=semantics)) +
 geom_vline(xintercept=0,colour=CSP_COLORS[['text']],linetype='dashed',linewidth=.45) +
 geom_errorbar(aes(xmin=lower,xmax=upper),orientation='y',width=.13,position=position_dodge(.42),linewidth=.65) +
 geom_point(position=position_dodge(.42),size=2.8) +
 scale_colour_manual(values=palette,name=NULL) + scale_shape_manual(values=shapes,name=NULL) +
 scale_x_continuous(breaks=c(-2,0,2)) +
 labs(x='Target-probability difference\n(size-first - colour-first; percentage points)',y=NULL) +
 theme_csp() + theme(legend.position='top',panel.grid.major.y=element_blank(),panel.grid.minor=element_blank())
save_csp_pdf(p,'figures/simulation_controlled_comparison.pdf',9,4.7)
if (!('--main-only' %in% commandArgs(trailingOnly=TRUE))) {
arch <- summaries %>% filter(support=='fifteen',(parameter_set=='original_grid' & order=='neutral') | parameter_set=='fitted_constants') %>% prepare() %>% mutate(setting=case_when(parameter_set=='original_grid'~'Original parameter grid',order=='neutral'~'Production constants',TRUE~'Constants + order score'),setting=factor(setting,levels=c('Original parameter grid','Production constants','Constants + order score')),rule=factor(rule,levels=c('terminal','prefix_k0','prefix_k05','prefix_k1'),labels=c('Terminal','Prefix\n0','Prefix\n0.5','Prefix\n1')))
stopifnot(nrow(arch)==24L,all(complete.cases(arch)))
p <- ggplot(arch,aes(x=rule,y=estimate,colour=semantics,shape=semantics)) +
 geom_hline(yintercept=0,colour=CSP_COLORS[['text']],linetype='dashed',linewidth=.45) +
 geom_errorbar(aes(ymin=lower,ymax=upper),width=.13,position=position_dodge(.28),linewidth=.65) + geom_point(size=2.8,position=position_dodge(.28)) +
 facet_wrap(~setting,nrow=1) + scale_colour_manual(values=palette,name=NULL) + scale_shape_manual(values=shapes,name=NULL) +
 labs(x='Evaluation rule and successive-choice weight',y='Target-probability difference\n(percentage points)') +
 theme_csp() + theme(legend.position='top',panel.grid.major.x=element_blank(),panel.grid.minor=element_blank())
save_csp_pdf(p,'figures/simulation_controlled_architectures.pdf',9.8,4.7)

}

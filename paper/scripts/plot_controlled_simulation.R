# Run from paper/. Frozen summaries only; no model execution or new contrasts.
suppressPackageStartupMessages({library(ggplot2); library(dplyr); library(readr)})
source('scripts/csp_figure_style.R')
summaries <- read_csv('data/simulation_controlled_overall.csv',show_col_types=FALSE) %>% filter(metric=='listener_advantage')
semantics_levels <- c('Context-fixed','Context-updating')
palette <- setNames(unname(CSP_COLORS[c('main','emphasis')]),semantics_levels)
shapes <- setNames(c(16,17),semantics_levels)
prepare <- function(d) d %>% mutate(semantics=factor(recode(semantics,fixed='Context-fixed',updating='Context-updating'),levels=semantics_levels),estimate=100*mean,lower=100*lower_mc95,upper=100*upper_mc95)
if (!('--architectures-only' %in% commandArgs(trailingOnly=TRUE))) {
# Panel A changes evaluation and alternatives while global choice stays fixed.
settings <- tibble(
 rule = c('terminal', rep('prefix_k0', 3)),
 support = c('two', 'two', 'four', 'fifteen'),
 parameter_set = 'original_grid',
 order = 'neutral',
 setting = c('Complete-utterance evaluation\n2 orders',
             'Incremental evaluation\n2 orders',
             'Incremental evaluation\n4 utterances',
             'Incremental evaluation\n15 utterances'))
main_a <- settings %>%
 inner_join(summaries, by = c('rule', 'support', 'parameter_set', 'order')) %>%
 prepare() %>% mutate(panel = 'A', setting = factor(setting, levels = rev(settings$setting)))
# Panel B changes architecture at common production semantic constants.
main_b <- summaries %>%
 filter(support == 'fifteen', parameter_set == 'fitted_constants', order == 'neutral',
        rule %in% c('prefix_k0', 'prefix_k05', 'prefix_k1')) %>%
 prepare() %>%
 mutate(panel = 'B', setting = factor(rule,
   levels = c('prefix_k1', 'prefix_k05', 'prefix_k0'),
   labels = c('Fully incremental choice', 'Plan-guided choice', 'Global choice')))
stopifnot(nrow(main_a) == 8L, nrow(main_b) == 6L,
          all(complete.cases(main_a)), all(complete.cases(main_b)))
write_csv(bind_rows(main_a, main_b), 'data/simulation_controlled_main_figure.csv')
comparison_panel <- function(d, title) {
 ggplot(d, aes(x = estimate, y = setting, colour = semantics, shape = semantics)) +
  geom_vline(xintercept = 0, colour = CSP_COLORS[['text']], linetype = 'dashed', linewidth = .45) +
  geom_errorbar(aes(xmin = lower, xmax = upper), orientation = 'y', width = .12,
                position = position_dodge(.38), linewidth = .65) +
  geom_point(position = position_dodge(.38), size = 2.8) +
  scale_colour_manual(values = palette, name = NULL,
    labels = c('Context-fixed', 'Sequential context updating')) +
  scale_shape_manual(values = setNames(c(17, 15), semantics_levels), name = NULL,
    labels = c('Context-fixed', 'Sequential context updating')) +
  scale_x_continuous(limits = c(-3.2, 3.5), breaks = c(-2, 0, 2)) +
  labs(title = title, x = NULL, y = NULL) +
  theme_csp() +
  theme(legend.position = 'top', panel.grid.major.y = element_blank(),
        panel.grid.minor = element_blank(),
        plot.title = element_text(size = 15, face = 'bold'),
        axis.text.y = element_text(size = 12))
}
p_a <- comparison_panel(main_a, 'A  Utterance evaluation and alternatives')
p_b <- comparison_panel(main_b, 'B  Production architecture') +
 labs(x = 'Target-probability difference\n(size-first - colour-first; percentage points)')
p <- patchwork::wrap_plots(p_a, p_b, ncol = 1, heights = c(4, 3.4), guides = 'collect') &
 theme(legend.position = 'top')
save_csp_pdf(p, 'figures/simulation_controlled_comparison.pdf', 9, 7.1)
}
if (!('--main-only' %in% commandArgs(trailingOnly=TRUE))) {
arch <- summaries %>% filter(support=='fifteen',(parameter_set=='original_grid' & order=='neutral') | parameter_set=='fitted_constants') %>% prepare() %>% mutate(setting=case_when(parameter_set=='original_grid'~'Original parameter grid',order=='neutral'~'Production constants',TRUE~'Constants + order score'),setting=factor(setting,levels=c('Original parameter grid','Production constants','Constants + order score')),rule=factor(rule,levels=c('terminal','prefix_k0','prefix_k05','prefix_k1'),labels=c('Completed-\nutterance\nbaseline','Global','Plan-guided','Fully\nincremental')))
stopifnot(nrow(arch)==24L,all(complete.cases(arch)))
p <- ggplot(arch,aes(x=rule,y=estimate,colour=semantics,shape=semantics)) +
 geom_hline(yintercept=0,colour=CSP_COLORS[['text']],linetype='dashed',linewidth=.45) +
 geom_errorbar(aes(ymin=lower,ymax=upper),width=.13,position=position_dodge(.28),linewidth=.65) + geom_point(size=2.8,position=position_dodge(.28)) +
 facet_wrap(~setting,nrow=1) + scale_colour_manual(values=palette,name=NULL) + scale_shape_manual(values=shapes,name=NULL) +
 labs(x='Utterance evaluation and production architecture',y='Target-probability difference\n(percentage points)') +
 theme_csp() + theme(legend.position='top',axis.text.x=element_text(size=10),panel.grid.major.x=element_blank(),panel.grid.minor=element_blank())
save_csp_pdf(p,'figures/simulation_controlled_architectures.pdf',9.8,4.7)

}

# Run from paper/. Frozen summaries only; no model execution or new contrasts.
suppressPackageStartupMessages({library(ggplot2); library(dplyr); library(readr)})
source('scripts/csp_figure_style.R')
summaries <- read_csv('data/simulation_controlled_overall.csv',show_col_types=FALSE) %>% filter(metric=='listener_advantage')
semantics_levels <- c('Context-fixed','Context-updating')
palette <- setNames(unname(CSP_COLORS[c('main','emphasis')]),semantics_levels)
shapes <- setNames(c(16,17),semantics_levels)
prepare <- function(d) d %>% mutate(semantics=factor(recode(semantics,fixed='Context-fixed',updating='Context-updating'),levels=semantics_levels),estimate=100*mean,lower=100*lower_mc95,upper=100*upper_mc95)
if (!('--architectures-only' %in% commandArgs(trailingOnly=TRUE))) {
# Main figure: two orders (global speaker = reproduction of the earlier simulations;
# plan-guided and fully incremental = production models), then the six speaker models
# under two further sets of alternatives.
# Lines connect context-fixed and context-updating semantics within an architecture.
group_labels <- c('Two orders', 'Two orders +\nsingle adjectives', 'Fifteen\nutterances')
arch_levels <- c('Global', 'Plan-guided', 'Fully incremental')
arch_colours <- c('Global' = CSP_COLORS[['emphasis']], 'Plan-guided' = CSP_COLORS[['green']],
                  'Fully incremental' = CSP_COLORS[['gold_dark']])
overview_data <- function(parameter_set_name) {
 models <- summaries %>%
  filter(order == 'neutral', parameter_set == parameter_set_name,
         (rule %in% c('prefix_k0', 'prefix_k05', 'prefix_k1') & support %in% c('four', 'fifteen')) |
           (rule %in% c('prefix_k05', 'prefix_k1') & support == 'two')) %>%
  mutate(group = c(two = 1, four = 2, fifteen = 3)[support],
         architecture = c(prefix_k0 = 'Global', prefix_k05 = 'Plan-guided', prefix_k1 = 'Fully incremental')[rule])
 reproduction <- summaries %>%
  filter(order == 'neutral', parameter_set == parameter_set_name, rule == 'terminal', support == 'two') %>%
  mutate(group = 1, architecture = 'Global')
 bind_rows(reproduction, models) %>%
  mutate(estimate = 100 * mean,
         architecture = factor(architecture, levels = arch_levels),
         semantics = factor(semantics, levels = c('fixed', 'updating'),
                            labels = c('Context-fixed', 'Context updating')),
         x = group + (as.integer(architecture) - 2) * .24 +
           ifelse(semantics == 'Context-fixed', -.06, .06))
}
overview_plot <- function(d) {
 ggplot(d, aes(x = x, y = estimate, colour = architecture)) +
  geom_hline(yintercept = 0, colour = CSP_COLORS[['text']], linewidth = .5) +
  geom_line(aes(group = interaction(group, architecture)), linewidth = .8) +
  geom_point(aes(shape = semantics), size = 3) +
  annotate('text', x = .55, y = 3.45, label = 'size-first preference', hjust = 0, size = 3.5,
           colour = CSP_COLORS[['text']]) +
  annotate('text', x = 3.48, y = -3.05, label = 'colour-first preference', hjust = 1, size = 3.5,
           colour = CSP_COLORS[['text']]) +
  scale_colour_manual(values = arch_colours, name = NULL) +
  scale_shape_manual(values = c('Context-fixed' = 16, 'Context updating' = 15), name = NULL) +
  scale_x_continuous(breaks = 1:3, labels = group_labels, limits = c(.5, 3.5)) +
  scale_y_continuous(limits = c(-3.2, 3.5), breaks = c(-2, 0, 2)) +
  labs(x = NULL, y = 'Size-first preference\n(percentage points)') +
  theme_csp() +
  theme(legend.position = 'top', legend.box = 'vertical', legend.spacing.y = unit(0, 'pt'),
        panel.grid.minor = element_blank(), panel.grid.major.x = element_blank(),
        axis.text.x = element_text(size = 11))
}
main <- overview_data('original_grid')
production <- overview_data('fitted_constants')
stopifnot(nrow(main) == 18L, nrow(production) == 18L, all(complete.cases(main)))
write_csv(bind_rows(mutate(main, setting = 'grid'), mutate(production, setting = 'production')),
          'data/simulation_controlled_main_figure.csv')
save_csp_pdf(overview_plot(main), 'figures/simulation_controlled_comparison.pdf', 7.5, 4.6)
}
if (!('--main-only' %in% commandArgs(trailingOnly=TRUE))) {
arch <- summaries %>% filter(support=='fifteen',(parameter_set=='original_grid' & order=='neutral') | parameter_set=='fitted_constants') %>% prepare() %>% mutate(setting=case_when(parameter_set=='original_grid'~'Original parameter grid',order=='neutral'~'Production constants',TRUE~'Constants + order score'),setting=factor(setting,levels=c('Original parameter grid','Production constants','Constants + order score')),rule=factor(rule,levels=c('terminal','prefix_k0','prefix_k05','prefix_k1'),labels=c('Terminal\nutility','Global','Plan-guided','Fully\nincremental')))
stopifnot(nrow(arch)==24L,all(complete.cases(arch)))
p <- ggplot(arch,aes(x=rule,y=estimate,colour=semantics,shape=semantics)) +
 geom_hline(yintercept=0,colour=CSP_COLORS[['text']],linetype='dashed',linewidth=.45) +
 geom_errorbar(aes(ymin=lower,ymax=upper),width=.13,position=position_dodge(.28),linewidth=.65) + geom_point(size=2.8,position=position_dodge(.28)) +
 facet_wrap(~setting,nrow=1) + scale_colour_manual(values=palette,name=NULL) + scale_shape_manual(values=shapes,name=NULL) +
 labs(x='Utterance utility and production architecture',y='Target-probability difference\n(percentage points)') +
 theme_csp() + theme(legend.position='top',axis.text.x=element_text(size=10),panel.grid.major.x=element_blank(),panel.grid.minor=element_blank())
save_csp_pdf(p,'figures/simulation_controlled_architectures.pdf',9.8,4.7)

}

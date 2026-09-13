# Run from paper/. All displayed statistics are read from frozen summary CSVs.
suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(readr)
  library(patchwork)
})
source("scripts/csp_figure_style.R")

if (!("--sweeps-only" %in% commandArgs(trailingOnly=TRUE))) {
model_labels <- c(`G-HO`="Global", `I-HO`="Fully incremental",
                  `K-HO`="Plan-guided: shared weight", `K-HKO`="Plan-guided: participant weights")
factor_labels <- c(response="Complete response", length="Length",
                   set_given_length="Adjective set | length", order_given_set="Order | adjective set")
data <- read_csv("data/semantic_factor_contrasts.csv", show_col_types=FALSE) %>%
  mutate(model=factor(unname(model_labels[fixed]), levels=unname(model_labels)),
         factor=factor(unname(factor_labels[factor]), levels=rev(unname(factor_labels))))
p <- ggplot(data, aes(x=delta_elpd, y=factor)) +
  geom_vline(xintercept=0, colour=CSP_COLORS[["grey"]]) +
  geom_errorbar(aes(xmin=lower_95, xmax=upper_95), orientation="y", width=.12,
                colour=CSP_COLORS[["green"]]) +
  geom_point(colour=CSP_COLORS[["green"]], size=2.6) +
  facet_wrap(~model, ncol=2) +
  labs(x="ELPD difference (updating - fixed)", y=NULL) + theme_csp() +
  theme(panel.grid.major.y=element_blank(), strip.text=element_text(size=12))
save_csp_pdf(p, "figures/production_semantic_decomposition.pdf", 10, 5.7)

bridge <- read_csv("data/simulation_fitted_bridge_summary.csv", show_col_types=FALSE) %>%
  filter(relevant_property=="all", metric %in% c("speaker_size_first_given_set", "listener_advantage")) %>%
  mutate(fixed=sub("-UPD", "", model),
         architecture=factor(unname(model_labels[fixed]), levels=rev(unname(model_labels))),
         semantics=if_else(grepl("UPD",model), "Updating", "Fixed"))
panel <- function(metric_name, label, reference) {
  ggplot(filter(bridge, metric==metric_name), aes(x=mean,y=architecture,colour=semantics)) +
    geom_vline(xintercept=reference, colour=CSP_COLORS[["grey"]], linetype="dashed") +
    geom_errorbar(aes(xmin=lower_95,xmax=upper_95),orientation="y",width=.12,
                  position=position_dodge(.5)) +
    geom_point(position=position_dodge(.5),size=2.7) +
    scale_colour_manual(values=c(Fixed=CSP_COLORS[["main"]],Updating=CSP_COLORS[["emphasis"]]),name=NULL) +
    labs(x=label,y=NULL) + theme_csp() +
    theme(legend.position="top",panel.grid.major.y=element_blank())
}
bridge_plot <- panel("speaker_size_first_given_set", "P(size-first | size-colour set)", .5) /
  panel("listener_advantage", "Pragmatic target-probability difference\n(size-first - colour-first)", 0) +
  plot_layout(guides="collect") + plot_annotation(tag_levels="A")
save_csp_pdf(bridge_plot, "figures/simulation_fitted_bridge.pdf", 8.8, 7.2)

}

sweep <- read_csv("data/simulation_size_first_advantage_summary.csv",show_col_types=FALSE) %>%
  mutate(semantics=recode(semantics,static="Context-fixed",recursive="Sequential context updating"),
         speaker=factor(recode(speaker,global_speaker="Original global",incremental_speaker="Original incremental"),
                        levels=c("Original incremental","Original global")),
         spread=factor(sd_spread,levels=c(2,7.75,15)))
p <- ggplot(sweep,aes(x=nobj,y=mean_advantage,colour=spread,fill=spread)) +
  geom_hline(yintercept=0,colour=CSP_COLORS[["grey"]],linetype="dashed") +
  geom_ribbon(aes(ymin=mean_advantage-1.96*mcse_advantage,ymax=mean_advantage+1.96*mcse_advantage),
              alpha=.15,colour=NA) + geom_line(linewidth=.7) + geom_point(size=2) +
  facet_grid(semantics~speaker) +
  scale_colour_manual(values=unname(CSP_COLORS[c("emphasis","accent_dark","main")]),name="Size spread") +
  scale_fill_manual(values=unname(CSP_COLORS[c("emphasis","accent_dark","main")]),guide="none") +
  scale_x_continuous(breaks=seq(2,30,4)) +
  labs(x="Number of objects",y="Target-probability difference\n(size-first - colour-first)") + theme_csp() +
  theme(legend.position="top",strip.text=element_text(size=12))
save_csp_pdf(p,"figures/sim_advantage_nobj.pdf",9,6.8)

marginal <- read_csv("data/simulation_parameter_summary.csv",show_col_types=FALSE) %>%
  mutate(semantics=recode(semantics,static="Context-fixed",recursive="Sequential context updating"),
         speaker=factor(recode(speaker,global_speaker="Original global",incremental_speaker="Original incremental"),
                        levels=c("Original incremental","Original global")))
for (parameter_name in c("k","wf","color_semvalue")) {
  x_label <- c(k="Threshold parameter k",wf="Perceptual blur",color_semvalue="Colour reliability")[[parameter_name]]
  p <- ggplot(filter(marginal,parameter==parameter_name),aes(x=value,y=mean,colour=order,fill=order)) +
    geom_ribbon(aes(ymin=mean-1.96*mcse,ymax=mean+1.96*mcse),alpha=.15,colour=NA) +
    geom_line(linewidth=.7) + geom_point(size=2) + facet_grid(semantics~speaker) +
    scale_colour_manual(values=c("big blue"=CSP_COLORS[["main"]],"blue big"=CSP_COLORS[["emphasis"]]),name=NULL) +
    scale_fill_manual(values=c("big blue"=CSP_COLORS[["main"]],"blue big"=CSP_COLORS[["emphasis"]]),guide="none") +
    labs(x=x_label,y="P(target | utterance)") + theme_csp() +
    theme(legend.position="top",strip.text=element_text(size=12))
  suffix <- if (parameter_name=="color_semvalue") "colorsemval" else parameter_name
  save_csp_pdf(p,paste0("figures/sim_",suffix,".pdf"),9,6.2)
}

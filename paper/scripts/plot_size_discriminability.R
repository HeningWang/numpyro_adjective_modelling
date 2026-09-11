# Schematic size comparison using recorded stimulus levels, not a trial screenshot.
# Run from paper/: Rscript scripts/plot_size_discriminability.R
library(grid)
d <- read.csv('data/size_discriminability_illustration.csv')
pdf('figures/size_discriminability.pdf',width=4.0,height=3.0,family='Helvetica',useDingbats=FALSE)
grid.newpage()
for(i in seq_len(nrow(d))) {
  y <- c(.66,.23)[i]
  grid.text(d$discriminability[i],x=.02,y=y,just='left',gp=gpar(fontsize=13,col='#575463'))
  for(j in 1:2) {
    x <- c(.47,.82)[j]
    size <- c(d$target_size[i],d$competitor_size[i])[j]
    grid.circle(x,y,r=unit(size*.035,'inches'),gp=gpar(fill='#7581B3',col='#575463',lwd=.6))
  }
}
grid.text('Target',x=.47,y=.95,gp=gpar(fontsize=12,col='#575463'))
grid.text('Competitor',x=.82,y=.95,gp=gpar(fontsize=12,col='#575463'))
dev.off()

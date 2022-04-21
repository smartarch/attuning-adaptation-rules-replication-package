grDevices:::initPSandPDFfonts()

legends = c('combined', 'random')
labels = c('128', '256', '512', '1024', '128-128', '256-256', '512-512', '1024-1024')
data = array(dim=c(length(legends), length(labels)))
data[1,] = c(97.15833306312561, 97.8125, 98.38708400726318, 98.39125037193298, 98.8658332824707, 99.39291715621948, 98.70749950408936, 95.22249937057495)
data[2,] = c(97.91333317756653, 97.89708375930786, 97.99708485603333, 97.99749970436096, 99.28333282470703, 99.15916562080383, 99.1587507724762, 99.28541660308838)

imgWidth=8
imgHeight=5
marks = 1:length(labels)
line_width = 2.8
colors = c(rgb(0.85, 0.85, 1.0), rgb(0.75, 1.0, 0.75))
lab_cex=1.3
axis_cex=1.15
main_cex=1.5
name_cex=1.1

X11(width=imgWidth, height=imgHeight)

bplt = barplot(data, beside=TRUE, ylim=c(92,101), xpd=FALSE, col=colors, names.arg=labels, cex.axis=axis_cex, cex.names=name_cex, las=3, cex.main=main_cex)
title(ylab="accuracy [%]", cex.lab=lab_cex)
legend("topleft", legends, bty="n", cex=1.2, fill=colors, horiz=TRUE)
box()

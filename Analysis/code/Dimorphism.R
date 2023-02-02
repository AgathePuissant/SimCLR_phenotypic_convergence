#################

source("./code/basis_functions/match_tree.R")
source("./code/basis_functions/get_phenotype.R")
library(ggtree)

list_get_phenotype = get_phenotype("M","D", level="form", mode = 'mean')
meanphen_M <- list_get_phenotype[[1]]
data_M <- list_get_phenotype[[2]]
rm(list=c("list_get_phenotype"))

list_get_phenotype = get_phenotype("F","D", level="form",mode = 'mean')
meanphen_F <- list_get_phenotype[[1]]
data_F <- list_get_phenotype[[2]]
rm(list=c("list_get_phenotype"))

meanphen_M = meanphen_M[rownames(meanphen_F),]
meanphen_F = meanphen_F[rownames(meanphen_M),]

list_match <- match_tree(meanphen_match = meanphen_M, data_match = data_M, add_poly=T)
subtree <- list_match[[1]]
meanphen_M <- list_match[[2]]

list_match <- match_tree(meanphen_match = meanphen_F, data_match=data_F, add_poly=T)
meanphen_F <- list_match[[2]]

rm(list=c("list_match"))


######################

meanphen_dimorph = diag(as.matrix(dist(meanphen_M,meanphen_F)))
meanphen_dimorph = as.data.frame(meanphen_dimorph)

rng = range(meanphen_dimorph)

p <- ggtree(subtree, layout="circular")

colnames(meanphen_dimorph) <- c("Dimorphism")
p2 <- gheatmap(p, meanphen_dimorph,offset=0, width=.05, colnames = F,legend_title = "Dimorphism")

p2

#################
dM<-as.matrix(proxy::dist(meanphen_M))
dF<-as.matrix(proxy::dist(meanphen_F))


dratio <- dM/dF

library(diverge)
sis=extract_sisters(subtree)
sis$ratio <- 0
sis$dm <- 0
sis$df <- 0

for (sp in 1:dim(sis)[1]){
  sis[sp,]$ratio<-dratio[sis[sp,1],sis[sp,2]]
  sis[sp,]$dm <-dM[sis[sp,1],sis[sp,2]]
  sis[sp,]$df <-dF[sis[sp,1],sis[sp,2]]
}

sis$dimorphism <- apply(cbind(meanphen_dimorph[sis$sp1,],meanphen_dimorph[sis$sp2,]),1,max)


val= sis$ratio
names(val) <- sis$sp1

p<-ggtree(subtree, layout="circular")#plot_genres(subtree)
p2 <- gheatmap(p,data.frame(val),offset=1, width=.1, colnames = F, legend_title = "Ratio contrast male / contrast female")+
  scale_fill_gradient2(midpoint = 1, na.value="white", mid="lightyellow")+
  labs(fill="Ratio contrast male / contrast female")

library(ggnewscale)

p2=p2+new_scale_fill()
p3 <- gheatmap(p2, meanphen_dimorph,offset=0, width=.03, low="white",high="black",colnames = F,legend_title = "Dimorphism")+
  labs(fill="Dimorphism")
p3

summary(sis$dimorphism)


################

library(scales)
library(RColorBrewer)
library(ggridges)

sis$bin<-sis$dimorphism>0.3
sis$bin <- factor(sis$bin)
levels(sis$bin) <- c("Monomorphic\nLess than 0.3","Dimorphic\nAbove 0.3")

densdimorph <- ggplot(sis, aes(x=ratio,y=factor(bin),fill=after_stat(x), height = stat(density)))+
  geom_density_ridges_gradient(jittered_points=T, point_alpha = .3,quantile_lines = TRUE, quantiles = 2)+
  scale_fill_gradient2(midpoint=1, mid="lightyellow") +
  theme_bw()+
  xlab("Ratio contrast male / contrast female")+
  ylab("Density")+ 
  theme_ridges()+
  theme(legend.position = "none")
densdimorph



##############

wilcox.test(sis[sis$dimorphism>0.3,]$ratio, mu=1, alternative = "less")

##############

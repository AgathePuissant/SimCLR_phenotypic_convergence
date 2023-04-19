#---------------------- Loading and preparing data
source("./code/basis_functions/match_tree.R")
source("./code/basis_functions/get_phenotype.R")
source('./code/basis_functions/plot_phylogenus.R')

library(phyr)
library(ggplot2)
library(viridis)
library(reshape2)

# Here choose the sex and wing side that you want, F for female and V for ventral
# level can be sp to be at the species level or form to be at the form level


list_get_phenotype = get_phenotype(c("M"),c("D"), mode = 'mean', level = "sp") 
meanphen <- list_get_phenotype[[1]]
data <- list_get_phenotype[[2]]
sp_data <- list_get_phenotype[[4]]
rm(list=c("list_get_phenotype"))


list_match <- match_tree(meanphen_match = meanphen, data_match = data, add_poly=F)
subtree <- list_match[[1]]
meanphen <- list_match[[2]]

#---------------------- Prepare pairwise data

dreal <- as.matrix(dist(meanphen))
df <- melt(as.matrix(dreal), varnames = c("row", "col"))
df = df[as.numeric(df$row) > as.numeric(df$col), ]

distphylo<-cophenetic.phylo(subtree)
df_phy <- melt(as.matrix(distphylo), varnames = c("row", "col"))
df_phy = df_phy[as.numeric(df_phy$row) > as.numeric(df_phy$col), ]

df_tot = merge(df, df_phy, by=c("row","col"))
colnames(df_tot) <- c("tip1","tip2","distpheno","distphylo")

#---------------------- Match geographical overlap data

overlap<-read.csv2("./data/jaccard.csv")
colnames(overlap)<- c("index","tip1form","tip2form","overlap")

df_tot2 <- df_tot
df_tot2$tip1form <- gsub("_[^_]*_[^_]*$","",df_tot2$tip1)
df_tot2$tip2form <- gsub("_[^_]*_[^_]*$","",df_tot2$tip2)

df_tot <- df_tot2


df_tot <- merge(df_tot, overlap, by=c("tip1form","tip2form"))
df_tot <- df_tot[!(is.na(df_tot$overlap)),]

df_tot$tip1form <- NULL
df_tot$tip2form <- NULL

meanphen = meanphen[as.vector(unique(c(df_tot$tip1,df_tot$tip2))),]
subtree=match.phylo.data(subtree,meanphen)$phy

#---------------------- PLMM

df_plmm <- df_tot
df_plmm$tip1 <- as.factor(df_plmm$tip1)
df_plmm$tip2 <- as.factor(df_plmm$tip2)
df_plmm <- df_plmm %>% mutate_at(c("overlap","distphylo"), ~(scale(.) %>% as.vector))
df_plmm$distphylosquared=df_plmm$distphylo**2

#Warning: very slow to run (a few hours)
print("Model with quadratic terms")
z_quadra <- pglmm(distpheno ~ overlap + distphylo + distphylosquared + overlap:distphylo + overlap:distphylosquared + (1|tip1__)+(1|tip2__),data = df_plmm,
                  cov_ranef = list(tip1 = subtree, tip2=subtree), REML = TRUE, verbose = TRUE, s2.init = .1)


z_quadra

#------------------------ To get predicted values

#Uncomment male or female depending on wanted values
plmm_pred_withquadra <- function(ov,dp,dp2){
  return(0.89324427+ov*(-0.01156030)+dp*0.00478737+dp2*(-0.04647549)+ov*dp*(0.00920400)+ov*dp2*(0.00481016)) #males
  # return(0.87972025+ov*(-0.01541798)+dp*0.00574119+dp2*(-0.03948247)+ov*dp*(0.00899770)+ov*dp2*(0.00372303)) #females
}

seqov = seq(-0.2,11,0.1)
facdp=seq(-3.7,0.8,0.1)

df_pred = data.frame(ov=rep(seqov,length(facdp)))
df_pred$dp = unlist(lapply(facdp,function(x){rep(x,length(seqov))}))
df_pred$dp2=df_pred$dp**2
df_pred$distpheno = apply(df_pred,1,function(x){plmm_pred_withquadra(x[1],x[2],x[3])})

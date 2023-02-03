#---------------------- Load and prepare data
source("./code/basis_functions/match_tree.R")
source("./code/basis_functions/get_phenotype.R")
source('./code/basis_functions/permutations_functions')

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


#Phylogenetic correction
fit <- lm(distpheno~distphylo,data=df_tot)
summary(fit)
plot(df_tot$distphylo,df_tot$distpheno)
abline(fit)
res<-fit$residuals
df_tot$res <- res


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


#---------------------- Permutations

#Function to perform Lapointe-Garland permutations
LG.permute <- function(tre,vec,k, loc){
  
  pmat <- scale((k-cophenetic.phylo(tre)/max(cophenetic.phylo(tre))), center = FALSE, scale = colSums((k-cophenetic.phylo(tre)/max(cophenetic.phylo(tre)))))
  out <- matrix(nrow=dim(vec)[1],ncol=dim(vec)[2])
  rownames(out)<-rownames(vec)
  colnames(out)<-colnames(vec)
  shuf <- rownames(vec)
  for(i in seq(nrow(vec))){
    index <- sample(seq(nrow(vec)),1,prob=pmat[,i])
    out[index,] <- as.numeric(vec[i,])
    shuf[index] <- rownames(vec)[i]
    pmat[index,] <- 0}
  return(list("new" = shuf, "old" = rownames(vec)))}


nsim=100000 #Number of permutations

listconv = list()
listdiv = list()

df_loc <- df_tot
  
simconv <- matrix(0, nrow = nrow(df_loc), ncol=1)
simdiv <- matrix(0, nrow = nrow(df_loc), ncol=1)

meansim <- matrix(0, nrow = nrow(df_loc), ncol=1)

#Permute and compute median residuals for sympatric and allopatric pairs
for (i in c(1:nsim)){
  
  #Print progress
  if (i%%10==0){
    print(i)
  }
  
  shuf <- as.data.frame(LG.permute(subtree, meanphen, 1, 0))
  
  
  res_sim <- df_loc
  
  #This block of code is necessary to match the permuted residuals to the right permuted pairs
  res_sim$tip1 <- shuf[match(as.character(res_sim$tip1), as.character(shuf$old)),1]
  res_sim$tip2 <- shuf[match(as.character(res_sim$tip2), as.character(shuf$old)),1]
  newpairs<-res_sim %>%
    mutate(var = paste(pmin(tip1, tip2), pmax(tip1, tip2))) %>%
    group_by(var)
  df_loc$tip1<-as.character(df_loc$tip1)
  df_loc$tip2<-as.character(df_loc$tip2)
  oldpairs<-df_loc %>%
    mutate(var = paste(pmin(tip1, tip2), pmax(tip1, tip2))) %>%
    group_by(var)
  res_sim<-res_sim[match(oldpairs$var,newpairs$var),]

  simconv <- simconv + as.numeric(df_loc$res<res_sim$res)
  simdiv <- simdiv + as.numeric(df_loc$res>res_sim$res)
  
  meansim <- meansim + res_sim$res
}


#Set the level of significance, compute p-values and store results
conv <- simconv/nsim > (1-(0.01))
div <- simdiv/nsim > (1-(0.01))

euc.dist <- function(x1, x2) sqrt(sum((x1 - x2) ^ 2))

forceconv <- NULL
for(i in 1:nrow(df_loc)) forceconv[i] <- euc.dist(df_loc[i,5],(meansim/nsim)[i])
forceconv <- forceconv[conv]


forcediv <- NULL
for(i in 1:nrow(df_loc)) forcediv[i] <- euc.dist(df_loc[i,5],(meansim/nsim)[i])
forcediv <- forcediv[div]

nameconv = df_loc[conv==T,c(1,2)]
namediv = df_loc[div==T,c(1,2)]

df_conv = data.frame(tip1 = nameconv$tip1, tip2 = nameconv$tip2, forceconv)
df_div = data.frame(tip1 = namediv$tip1, tip2 = namediv$tip2, forcediv)


df_conv=merge(df_conv,df_tot[,c(1,2,4)],by=c("tip1","tip2"))
df_div=merge(df_div,df_tot[,c(1,2,4)],by=c("tip1","tip2"))

listconv <- append(listconv, list(df_conv))
listdiv <- append(listdiv, list(df_div))

#----------------------
source("./code/basis_functions/match_tree.R")
source("./code/basis_functions/get_phenotype.R")
source('./code/basis_functions/permutations_functions')

library(reshape2)

list_get_phenotype = get_phenotype(c("M"),c("D"), mode = 'mean', level = "sp")
meanphen <- list_get_phenotype[[1]]
data <- list_get_phenotype[[2]]
sp_data <- list_get_phenotype[[4]]
rm(list=c("list_get_phenotype"))


list_match <- match_tree(meanphen_match = meanphen, data_match = data, add_poly=F)
subtree <- list_match[[1]]
meanphen <- list_match[[2]]

########

dreal <- as.matrix(dist(meanphen))
df <- melt(as.matrix(dreal), varnames = c("row", "col"))
df = df[as.numeric(df$row) > as.numeric(df$col), ]

distphylo<-cophenetic.phylo(subtree)
df_phy <- melt(as.matrix(distphylo), varnames = c("row", "col"))
df_phy = df_phy[as.numeric(df_phy$row) > as.numeric(df_phy$col), ]

df_tot = merge(df, df_phy, by=c("row","col"))
colnames(df_tot) <- c("tip1","tip2","distpheno","distphylo")
fit <- lm(distpheno~distphylo,data=df_tot)
summary(fit)
plot(df_tot$distphylo,df_tot$distpheno)
abline(fit)
res<-fit$residuals
df_tot$res <- res


#########

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


#########




nsim=100000

listconv = list()
listdiv = list()

df_loc <- df_tot
  
simconv <- matrix(0, nrow = nrow(df_loc), ncol=1)
simdiv <- matrix(0, nrow = nrow(df_loc), ncol=1)

meansim <- matrix(0, nrow = nrow(df_loc), ncol=1)

for (i in c(1:nsim)){
  
  if (i%%10==0){
    print(i)
  }
  
  shuf <- as.data.frame(LG.permute(subtree, meanphen, 1, 0))
  
  
  res_sim <- df_loc
  # res_sim$res <- df_loc$res[sample(c(1:length(df_loc$res)),length(df_loc$res))]
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





plot(df_conv$distphylo,df_conv$forceconv)
res <- lm(df_conv$forceconv~df_conv$distphylo)
summary(res)
abline(res)

plot(df_div$distphylo,df_div$forcediv)
res <- lm(df_div$forcediv~df_div$distphylo)
summary(res)
abline(res)


save.image(file="ws_resdist_FD_p99.RData")


df_tot$id <- df_tot$overlap>0.2
df_tot$id[df_tot$id==T]='Sympatry'
df_tot$id[df_tot$id==F]='Allopatry'
res=aov(data=df_tot, res~id*distphylo)
summary(res)
ggplot(data=df_tot, aes(y=res, x=distphylo, color=id))+geom_point()
wilcox.test(df_tot$res~df_tot$id)

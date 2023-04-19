#---------------------- Loading and preparing data
source("./code/basis_functions/match_tree.R")
source("./code/basis_functions/get_phenotype.R")
source('./code/basis_functions/plot_phylogenus.R')

library(mvMORPH)
library(motmot)
library(reshape2)
library(data.table)

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

#------------------------ fit models of evolution

naxes = 20
Y=as.matrix(meanphen[,c(1:naxes)])
phendata<-list(Y = Y)
fit1 <- mvgls(Y~1, data=phendata, subtree, model="BM", penalty="RidgeArch", error=T)
fit2 <- mvgls(Y~1, data=phendata, subtree, model="OU", penalty="RidgeArch", error=T)
fit3 <- mvgls(Y~1, data=phendata, subtree, model="EB", penalty="RidgeArch", error=T)
fit4 <- mvgls(Y~1, data=phendata, subtree, model="lambda", penalty="RidgeArch")
GIC(fit1); GIC(fit2); GIC(fit3); GIC(fit4)

l=as.numeric(fit4$param[[1]])
ltrns=transformPhylo(subtree,model="lambda",lambda=l)

fitBM <- mvBM(subtree, meanphen[,c(1:naxes)],method="pic")
summary(fitBM)


#------------------------ Run simulations and register mean standardized distance

#Warning: slow to run

nsimu=10000#Number of simulations to run (lower the number to run quickly)
threshsymp=0.2 #Sympatry cutoff threshold

simul<-mvSIM(ltrns,nsim=nsimu,param=fitBM)

#Prepare the dataframe to store the simulations results
dsimul <- lapply(simul, dist)
dsimul = lapply(dsimul, function(x){melt(as.matrix(x)/mean(as.matrix(x)), value.name = "distpheno")})
dsimul = lapply(dsimul, function(x){x[as.numeric(x[,1])>as.numeric(x[,2]),]})


#Prepare the data into to compute observed standardized distance values
dreal=as.matrix(dist(meanphen[,c(1:naxes)]))
drealst = melt(dreal/mean(dreal), value.name = "distpheno")
drealst = drealst[as.numeric(drealst$Var1)>as.numeric(drealst$Var2),]
overlap<-read.csv2(paste(getwd(),"./data/jaccard.csv",sep="/"))
colnames(overlap)<- c("index","Var1form","Var2form","value")
drealst2 <- drealst
drealst2$Var1form <- gsub("_[^_]*_[^_]*$","",drealst2$Var1)
drealst2$Var2form <- gsub("_[^_]*_[^_]*$","",drealst2$Var2)
drealst <- drealst2
drealst <- merge(drealst, overlap, by=c("Var1form","Var2form"))
drealst <- drealst[!(is.na(drealst$value)),]
drealst$Var1form <- NULL
drealst$Var2form <- NULL
drealst$symp=drealst$value>threshsymp
dphyl = melt(cophenetic.phylo(subtree),value.name = "distphylo")
drealst <- merge(drealst, dphyl, by=c("Var1","Var2"))


meansymp = mean(drealst$distpheno[drealst$symp==T],na.rm=T)
meanallo = mean(drealst$distpheno[drealst$symp==F],na.rm=T)
meansymp
meanallo

#Select sympatric pairs using data.table (slightly faster)

# convert the data frames in dsimul and drealst to data.tables
dsimul_dt <- lapply(dsimul, as.data.table)
drealst_dt <- as.data.table(drealst)

# create a new column in drealst with the concatenated values of Var1 and Var2
drealst_dt[, comb := paste0(Var1, Var2)]

# create a vector with the concatenated values from drealst where symp == T
symp_vec <- drealst_dt[symp == T, comb]
allo_vec <- drealst_dt[symp == F, comb]

allo_vec = allo_vec[sample(c(1:length(allo_vec)),length(symp_vec))] #To get same sample size for sympatric and allopatric pairs

id_symp = paste0(dsimul_dt[[1]]$Var1, dsimul_dt[[1]]$Var2) %in% symp_vec | paste0(dsimul_dt[[1]]$Var2, dsimul_dt[[1]]$Var1) %in% symp_vec
id_allo = paste0(dsimul_dt[[1]]$Var1, dsimul_dt[[1]]$Var2) %in% allo_vec | paste0(dsimul_dt[[1]]$Var2, dsimul_dt[[1]]$Var1) %in% allo_vec

# apply the function to each data.table in the dsimul_dt list using lapply
dsimulsymp <- lapply(dsimul_dt, function(dt) {dt[id_symp,]$distpheno
})

dsimulallo <- lapply(dsimul_dt, function(dt) {dt[id_allo,]$distpheno
})

#Get mean standardized distances
dsimulsymp <- lapply(dsimulsymp, function(dt) {mean(dt,na.rm=T)
})
dsimulallo <- lapply(dsimulallo, function(dt) {mean(dt,na.rm=T)
})



dsimulsymp = unlist(dsimulsymp)
dsimulallo = unlist(dsimulallo)


df = data.frame(value=c(dsimulallo,dsimulsymp),group=c(rep("Allopatry",nsimu),rep("Sympatry",nsimu)))

p1 <- ggplot(data=df)+
  geom_density(aes(x=value, after_stat(scaled), fill=group),alpha=0.3)+
  geom_vline(aes(xintercept=median(meanallo)),color="coral3", linetype="dashed", size=1)+
  geom_vline(aes(xintercept=median(meansymp)),color="aquamarine4", linetype="dashed", size=1)+
  theme_classic()+
  xlab("Null distribution of mean standardized distances")+
  ylab("Density")+
  theme(legend.title = element_blank())
p1

#pvalues
sum(dsimulsymp<meansymp)/nsimu
sum(dsimulallo<meanallo)/nsimu



#---------------------------------------------------------------------------------------

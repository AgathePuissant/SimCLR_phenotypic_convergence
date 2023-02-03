#------------------- Libraries and parameters ----------------

thresh_overlap=0.2 #Here you can choose the thershold to consider pairs sympatric or allopatric
sex = "M" #Choose which sex


library(ggtree)
library(ggplot2)
library(reshape2)
library(ggpubr)
library(dplyr)
library(RColorBrewer)

load(paste0("./data/saved_pairs_",sex,".RData"))

#------------------- Prepare sympatric and allopatric pair data ----------------

#Loading overlap data
overlap <- read.csv2("./data/overlap.csv")
colnames(overlap)<- c("index","tip1","tip2","percent")
overlap$index <- NULL

  
#Preparing convergent pairs data
dfconv <- df_conv #Getting the list of convergent pairs from the saved workspace
dfconv$tip1ori <- dfconv$tip1
dfconv$tip2ori <- dfconv$tip2
dfconv$tip1 <- gsub("_[^_]*_[^_]*$","",dfconv$tip1) #To match with other data, we need the name at the species level
dfconv$tip2 <- gsub("_[^_]*_[^_]*$","",dfconv$tip2)
dfconv <- dfconv[!(dfconv$distphylo==0),]

#Getting the range overlaps
dfconv$overlap1 <- overlap[match(paste0(dfconv$tip1,dfconv$tip2),paste0(overlap$tip1,overlap$tip2)),]$percent
dfconv$overlap2 <- overlap[match(paste0(dfconv$tip2,dfconv$tip1),paste0(overlap$tip1,overlap$tip2)),]$percent

#Creating a mask to split sympatric pairs from allopatric pairs
mask = dfconv$overlap1<thresh_overlap | dfconv$overlap2<thresh_overlap
dfconv_allo <- dfconv[mask,]
dfconv <- dfconv[!mask,]

dfconv <- dfconv[!duplicated(dfconv),] #To clean data

#Same with divergent pairs
dfdiv <- df_div
dfdiv$tip1ori <- dfdiv$tip1
dfdiv$tip2ori <- dfdiv$tip2
dfdiv$tip1 <- gsub("_[^_]*_[^_]*$","",dfdiv$tip1)
dfdiv$tip2 <- gsub("_[^_]*_[^_]*$","",dfdiv$tip2)
dfdiv <- dfdiv[!(dfdiv$distphylo==0),]

dfdiv$overlap1 <- overlap[match(paste0(dfdiv$tip1,dfdiv$tip2),paste0(overlap$tip1,overlap$tip2)),]$percent
dfdiv$overlap2 <- overlap[match(paste0(dfdiv$tip2,dfdiv$tip1),paste0(overlap$tip1,overlap$tip2)),]$percent

mask = dfdiv$overlap1<thresh_overlap | dfdiv$overlap2<thresh_overlap
dfdiv_allo <- dfdiv[mask,]
dfdiv <- dfdiv[!mask,]

dfdiv <- dfdiv[!duplicated(dfdiv),]


#Preparing some dataframes to make it easier to plot afterwards
dfconv$id <- factor('Convergence')
dfdiv$id <- factor('Divergence')
dfconv_allo$id <- factor('Convergence')
dfdiv_allo$id <- factor('Divergence')
names(dfconv)[names(dfconv) == 'forceconv'] <- 'strength'
names(dfdiv)[names(dfdiv) == 'forcediv'] <- 'strength'
names(dfconv_allo)[names(dfconv_allo) == 'forceconv'] <- 'strength'
names(dfdiv_allo)[names(dfdiv_allo) == 'forcediv'] <- 'strength'
df <- rbind(dfconv,dfdiv)
df_allo <- rbind(dfconv_allo,dfdiv_allo)
df$context <- factor('Sympatry')
df_allo$context <- factor("Allopatry")
df_context<-rbind(df,df_allo)

mypal <- brewer.pal(n=6,name="Set2")[c(1,2)]
mypal2 <- brewer.pal(n=6,name="Dark2")[c(1,2)]
mypalall <- c(mypal2[1],mypal[1],mypal2[2],mypal[2])

#----- Testing the effect of sympatry vs allopatry on convergence and divergence ----

#Chi 2 to test the effect of sympatry on the number of convergence and divergence
matcount = matrix(c(nrow(df[df$id=="Convergence",]), nrow(df_conv), nrow(df[df$id=="Divergence",]),nrow(df_div)),nrow=2,ncol=2)
matcount
res = chisq.test(matcount)
res

#Testing the effect of sympatry on convergence and divergence strength
pairwise.wilcox.test(df_context$strength,interaction(df_context$id,df_context$context))

#------------------- Plots and tests on sympatric pairs ------------------------

#Testing strength difference between convergent and divergent pairs
ggplot(df, aes(x=id, y=strength,fill=id,color=id,alpha=0.5))+
  stat_compare_means()+
  geom_boxplot()+
  labs(fill = "Direction", y = "Strength", x = "Direction")+
  theme_bw()+
  theme(legend.position="none")+
  scale_colour_manual(values = mypal)+
  scale_fill_manual(values = mypal)

#Testing differences in phylogenetic relatedness between convergent and divergent pairs

ggplot(df, aes(x=id, y=distphylo,fill=id,color=id,alpha=0.5))+
  stat_compare_means()+
  geom_boxplot()+
  geom_jitter(position=position_jitter(0.2))+
  labs(fill = "Direction", y = "Phylogenetic distance", x = "Direction")+
  theme_bw()+
  theme(legend.position="none")+
  scale_colour_manual(values = mypal)+
  scale_fill_manual(values = mypal)

#------------------- Plot the pairs on the phylogeny ---------------------------
subtree <- read.tree(paste0("./data/subtree_",sex,".tre"))

p4 <- ggtree(subtree, layout="inward_circular", xlim=c(150, 0)) +
  geom_taxalink(data=dfconv,
                mapping=aes(taxa1=tip1ori,
                            taxa2=tip2ori,
                            color=strength,
                            size=strength
                ),
                ncp=10,
                offset=0.15)+
  scale_color_continuous(low='lightgreen', high='darkgreen')+
  scale_size_continuous(range=c(0,1))
p4



p5 <- ggtree(subtree, layout="inward_circular", xlim=c(150, 0)) +
  geom_taxalink(data=dfdiv,
                mapping=aes(taxa1=tip1ori,
                            taxa2=tip2ori,
                            size=strength,
                            color=strength
                ),
                ncp=10,
                offset=0.15)+
  scale_color_gradient(name="Divergence",low='burlywood1', high='darkorange3')+
  scale_size_continuous(name="Divergence",range=c(0,1))
p5


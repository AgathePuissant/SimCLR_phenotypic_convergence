#######
include_polytomy = F
valseuil=0.2


library(ggplot2)
library(reshape2)
library(ggpubr)
library(dplyr)
library(RColorBrewer)
overlap<-read.csv2("./data/jaccard.csv")
seuillage <- read.csv2("C:/Users/Agathe/Mon Drive/Codes/gbif/overlap2.csv")

a <- apply(biomes, MARGIN =  1, FUN = which.max)
a <- a==16
biomes <- biomes[!a,]

colnames(overlap)<- c("index","tip1","tip2","overlap")
colnames(p1_p2)<- c("index","tip1","tip2","p1_p2")
colnames(seuillage)<- c("index","tip1","tip2","seuil")

overlap$index <- NULL
p1_p2$index <- NULL
seuillage$index <- NULL


sp_biomes_names<-biomes$X
biomes <- biomes[,c(-1, -16)]
biomes <-sapply(biomes, as.numeric)
rownames(biomes)<-sp_biomes_names
biomes <- biomes[apply(biomes[,-1], 1, function(x) !all(x==0)),]
a <- apply(biomes, MARGIN =  1, FUN = which.max)
maxbiomes <- colnames(biomes)[a]
names(maxbiomes) <- rownames(biomes)
maxbiomes<-as.data.frame(maxbiomes)
  

df_conv2 <- df_conv
df_conv2 <- merge(df_conv2, df_tot[,c(1,2,5)], by=c("tip1","tip2"))
df_conv2$tip1ori <- df_conv2$tip1
df_conv2$tip2ori <- df_conv2$tip2
df_conv2$tip1 <- gsub("_[^_]*_[^_]*$","",df_conv2$tip1)
df_conv2$tip2 <- gsub("_[^_]*_[^_]*$","",df_conv2$tip2)

dfinv <- cbind(df_conv2$tip2,df_conv2$tip1,df_conv2[,3:7])
colnames(dfinv)<-colnames(df_conv2)
dfinv <- merge(dfinv, overlap, by=c('tip1','tip2'))

dfconv <- merge(df_conv2, overlap, by=c('tip1','tip2'))


dfconv<- merge(dfconv, seuillage, by=c('tip1','tip2'))
dfinv<- merge(dfinv, seuillage, by=c('tip1','tip2'))

dfconv <- dfconv[order(dfconv$forceconv),]
dfinv <- dfinv[order(dfinv$forceconv),]

dfconv$biome <- apply(dfconv,1,function(x) maxbiomes[x[1],]==maxbiomes[x[2],])
dfinv$biome <- apply(dfinv,1,function(x) maxbiomes[x[1],]==maxbiomes[x[2],])

dfconv_allo <- dfconv
dfconv_allo <- dfconv_allo[!is.na(dfconv_allo$overlap),]
dfconv_allo <- dfconv_allo[(dfconv_allo$overlap<=valseuil),]

if (include_polytomy==F){
dfconv_allo <- dfconv_allo[!(dfconv_allo$distphylo==0),]}

maskseuil = dfconv$seuil<valseuil | dfinv$seuil<valseuil


maskseuil[is.na(maskseuil)] = T


dfconv <- dfconv[!maskseuil,]
dfinv <- dfinv[!maskseuil,]

dfconv <- dfconv[!is.na(dfconv$overlap),]

dfconv <- dfconv[!(dfconv$overlap==0),]
if (include_polytomy==F){
dfconv <- dfconv[!(dfconv$distphylo==0),]}

dfconv <- dfconv[!duplicated(dfconv),]
dfconv_allo <- dfconv_allo[!duplicated(dfconv_allo),]





df_div2 <- df_div
df_div2 <- merge(df_div2, df_tot[,c(1,2,4,5)], by=c("tip1","tip2")) #changer ici
df_div2$tip1ori <- df_div2$tip1
df_div2$tip2ori <- df_div2$tip2
df_div2$tip1 <- gsub("_[^_]*_[^_]*$","",df_div2$tip1)
df_div2$tip2 <- gsub("_[^_]*_[^_]*$","",df_div2$tip2)

dfinv <- cbind(df_div2$tip2,df_div2$tip1,df_div2[,3:7])
colnames(dfinv)<-colnames(df_div2)
dfinv <- merge(dfinv, overlap, by=c('tip1','tip2'))


dfdiv <- merge(df_div2, overlap, by=c('tip1','tip2'))


dfdiv<- merge(dfdiv, seuillage, by=c('tip1','tip2'))
dfinv<- merge(dfinv, seuillage, by=c('tip1','tip2'))

dfdiv <- dfdiv[order(dfdiv$forcediv),]
dfinv <- dfinv[order(dfinv$forcediv),]

dfdiv$biome <- apply(dfdiv,1,function(x) maxbiomes[x[1],]==maxbiomes[x[2],])
dfinv$biome <- apply(dfinv,1,function(x) maxbiomes[x[1],]==maxbiomes[x[2],])

dfdiv_allo <- dfdiv
dfdiv_allo <- dfdiv_allo[!is.na(dfdiv_allo$overlap),]
dfdiv_allo <- dfdiv_allo[(dfdiv_allo$overlap<=valseuil),]
if (include_polytomy==F){
dfdiv_allo <- dfdiv_allo[!(dfdiv_allo$distphylo==0),]}


maskseuil = dfdiv$seuil<valseuil | dfinv$seuil<valseuil


maskseuil[is.na(maskseuil)] = T


dfdiv <- dfdiv[!maskseuil,]
dfinv <- dfinv[!maskseuil,]

dfdiv <- dfdiv[!is.na(dfdiv$overlap),]
dfdiv <- dfdiv[!(dfdiv$overlap==0),]
if (include_polytomy==F){
dfdiv <- dfdiv[!(dfdiv$distphylo==0),]}


dfdiv <- dfdiv[!duplicated(dfdiv),]
dfdiv_allo <- dfdiv_allo[!duplicated(dfdiv_allo),]



dfconv$id <- factor('Convergence')
dfdiv$id <- factor('Divergence')
dfconv_allo$id <- factor('Convergence')
dfdiv_allo$id <- factor('Divergence')
names(dfconv)[names(dfconv) == 'forceconv'] <- 'force'
names(dfdiv)[names(dfdiv) == 'forcediv'] <- 'force'
names(dfconv_allo)[names(dfconv_allo) == 'forceconv'] <- 'force'
names(dfdiv_allo)[names(dfdiv_allo) == 'forcediv'] <- 'force'
df <- rbind(dfconv,dfdiv)
df_allo <- rbind(dfconv_allo,dfdiv_allo)
df$context <- factor('Sympatry')
df_allo$context <- factor("Allopatry")

# df_allo$seuil <- NULL
df_context<-rbind(df,df_allo)

mypal <- brewer.pal(n=6,name="Set2")[c(1,2)]
mypal2 <- brewer.pal(n=6,name="Dark2")[c(1,2)]
mypalall <- c(mypal2[1],mypal[1],mypal2[2],mypal[2])

########

pdf("bp_force.pdf",7,5)
ggplot(df, aes(x=id, y=force,fill=id,color=id,alpha=0.5))+
  stat_compare_means()+
  geom_boxplot()+
  labs(fill = "Direction", y = "Strength", x = "Direction")+
  theme_bw()+
  theme(legend.position="none")+
  scale_colour_manual(values = mypal)+
  scale_fill_manual(values = mypal)
dev.off()

####Overlap####
pdf("bp_overlap.pdf",7,5)
ggplot(df,aes(x=id, y=overlap,fill=id,color=id,alpha=0.5))+
  stat_compare_means()+
  geom_boxplot(notch=F)+
  geom_jitter(position=position_jitter(0.2))+
  labs(fill = "Direction", y = "% of range overlap")+
  theme_bw()+
  theme(legend.position="none")+
  scale_colour_manual(values = mypal)+
  scale_fill_manual(values = mypal)
summary(dfconv$overlap)
summary(dfdiv$overlap)
dev.off()

####Distphylo####
pdf("bp_distphylo.pdf",7,5)
ggplot(df, aes(x=id, y=distphylo,fill=id,color=id,alpha=0.5))+
  stat_compare_means()+
  geom_boxplot()+
  geom_jitter(position=position_jitter(0.2))+
  labs(fill = "Direction", y = "Phylogenetic distance", x = "Direction")+
  theme_bw()+
  theme(legend.position="none")+
  scale_colour_manual(values = mypal)+
  scale_fill_manual(values = mypal)
summary(dfconv$distphylo)
summary(dfdiv$distphylo)
dev.off()

####Allo vs Symp####
matcount = matrix(c(nrow(df[df$id=="Convergence",]), nrow(df_conv), nrow(df[df$id=="Divergence",]),nrow(df_div)),nrow=2,ncol=2)


matcount
res = chisq.test(matcount)
res
res$expected
res$residuals

pdf("bp_force_allosymp.pdf",7,5)
ggplot(df_context, aes(x=interaction(id,context), y=force,color=interaction(id,context),fill=interaction(id,context),alpha=0.5))+
  stat_compare_means(comparisons = list(c("Convergence.Sympatry", "Convergence.Allopatry"), c("Divergence.Sympatry","Divergence.Allopatry")))+
  geom_boxplot(notch=F)+
  labs(fill = "Direction", y = "Strength", x = "Context")+
  theme_bw()+
  theme(legend.position="none")+
  scale_colour_manual(values = c(mypal,mypal2))+
  scale_fill_manual(values = mypalall)
dev.off()

pdf("bp_distphylo_allosymp.pdf",7,5)
ggplot(df_context, aes(x=interaction(id,context), y=distphylo,color=interaction(id,context),fill=interaction(id,context),alpha=0.5))+
  stat_compare_means(comparisons = list(c("Convergence.Sympatry", "Convergence.Allopatry"), c("Divergence.Sympatry","Divergence.Allopatry")))+
  geom_boxplot()+
  geom_jitter(position=position_jitter(0.2))+
  labs(fill = "Direction", y = "Phylogenetic distance", x = "Context")+
  theme_bw()+
  theme(legend.position="none")+
  scale_colour_manual(values = c(mypal,mypal2))+
  scale_fill_manual(values = mypalall)
dev.off()

pairwise.wilcox.test(df_context$force,interaction(df_context$id,df_context$context))
pairwise.wilcox.test(df_context$distphylo,interaction(df_context$id,df_context$context))


####Plot assiociations####

dfconv$g1 <- gsub("_.*?$","",dfconv$tip1)
dfconv$g2 <- gsub("_.*?$","",dfconv$tip2)

p4 <- ggtree(subtree, layout="inward_circular", xlim=c(150, 0)) +
  geom_taxalink(data=dfconv,
                mapping=aes(taxa1=tip1ori,
                            taxa2=tip2ori,
                            color=force,
                            size=force
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
                            size=force,
                            color=force
                ),
                ncp=10,
                offset=0.15)+
  scale_color_gradient(name="Divergence",low='burlywood1', high='darkorange3')+
  scale_size_continuous(name="Divergence",range=c(0,1))
p5


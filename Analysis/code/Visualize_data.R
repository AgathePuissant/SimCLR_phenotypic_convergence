#---------------------- Loading and preparing data
source("./code/basis_functions/match_tree.R")
source("./code/basis_functions/get_phenotype.R")
source('./code/basis_functions/plot_phylogenus.R')

library(ggimage)
library(ggrepel)
library(geomorph)
library(cowplot)

# Here choose the sex and wing side that you want, F for female and V for ventral
# level can be sp to be at the species level or form to be at the form level


list_get_phenotype = get_phenotype(c("F"),c("D"), mode = 'mean', level = "form") 
meanphen <- list_get_phenotype[[1]]
data <- list_get_phenotype[[2]]
sp_data <- list_get_phenotype[[4]]
rm(list=c("list_get_phenotype"))


list_match <- match_tree(meanphen_match = meanphen, data_match = data, add_poly=T)
subtree <- list_match[[1]]
meanphen <- list_match[[2]]

# PCA on species means ----------------------------------------------------

#Do not execute this first part if you do not have the images to scatter

randomsample = data %>%
  group_by(genresp,sex) %>%
  sample_n(1)

namesrandomsample <- paste0(randomsample$genresp)
randomsample <- randomsample[,1]
randomsample$id <- apply(randomsample,1,function(x) paste('./data/dataset_bg/',x,'D.png',sep=''))
rownames(randomsample) <- namesrandomsample

randomsample <- cbind(meanphen,randomsample[rownames(meanphen),])

p<-ggplot(randomsample, mapping=aes(coord0, coord1, image=id)) + geom_image(by="height", asp=1)+
  theme_void(base_size = 22)+
  xlim(c(-0.5,0.5))+
  ylim(c(-0.5,0.5))+
  labs(x="PC1 (11.9%)",y="PC2 (8%)")+
  theme(axis.title.x = element_blank(),
        axis.title.y = element_blank(),
        title=element_blank(),
        aspect.ratio = 1,
        legend.position="none")
p


########

listgenretree <- c("Baronia", "Archon", "Luehdorfia", "Sericinus", "Bhutanitis", "Zerynthia", "Allancastria", "Hypermnestra", "Parnassius",
                   "Iphiclides", "Lamproptera", "Protographium", "Mimoides", "Eurytides", "Protesilaus", "Graphium", "Teinopalpus", "Battus",
                   "Pharmacophagus", "Losaria", "Pachliopta", "Atrophaneura", "Byasa", "Cressida", "Euryades", "Parides", "Trogonoptera",
                   "Ornithoptera", "Troides", "Meandrusa", "Papilio")


pal1 <- palette_genus(subtree)
tip.cols<-pal1[as.factor(listgenretree)]
names(tip.cols)<-listgenretree
cols<-c(tip.cols[listgenretree],rep("#000000",subtree$Nnode))
names(cols)<-1:(length(listgenretree)+subtree$Nnode)

meanphen$group = as.data.frame(str_split(rownames(meanphen),"_",simplify=T))$V1
meanphen$taxon = as.data.frame(str_split(rownames(meanphen),"_",simplify=T))$V2
meanphen$group <- factor(meanphen$group, levels = listgenretree)

pms1 <- ggphylomorpho_custom(subtree,meanphen,xvar=coord0,yvar=coord1, edge.width = 0.1)
pms1 <- pms1 + 
  scale_colour_manual(values = pal1)+
  labs(colour="Genus")+
  theme(axis.title.x = element_blank(),
        axis.title.y = element_blank(),
        title=element_blank(),
        aspect.ratio = 1)+
  xlim(c(-0.5,0.5))+
  ylim(c(-0.5,0.5))

legend <- cowplot::get_legend(pms1)

grid.newpage()
grid.draw(legend)
  
pms1 <- pms1 + theme(legend.position="none")
pms1

meanphen$group<-NULL
meanphen$taxon<-NULL
meanphen$id <- NULL


# Phylogenetic signal ---------------------------------------------------


#Should be computed on species mean and not form.

subtree <- multi2di(subtree)
geomorph::physignal(as.matrix(meanphen), subtree, iter=999)





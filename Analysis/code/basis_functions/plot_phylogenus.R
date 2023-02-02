library(nodiv)
library(ggtree)
library(RColorBrewer)
library(randomcoloR)

palette_genus <- function(subtree){
  
  listsptree <- subtree$tip.label
  listgenretree <- gsub("_.*?$","",listsptree)
  listsptree <- gsub("^.*?_","",listsptree)
  species <- listsptree
  new_genre_F <- as.factor(listgenretree)
  colors <- brewer.pal(11, "Spectral")
  pal1 <- colorRampPalette(colors)
  pal1 <- pal1(dim(table(new_genre_F)))
  
  return(pal1)
  
}


plot_genres <- function(subtree){
  
  pal1 <- palette_genus(subtree)
  
  p <- ggtree(subtree, layout = 'circular')
  
  genus <- gsub("_.*?$","",subtree$tip.label)
  
  dfgenus <- data.frame(id = c(1:length(genus)), genre = genus)
  ss <- split.data.frame(dfgenre,dfgenus$genus)
  listnodes <- lapply(ss,"[[","id")
  
  
  for (i in c(1:length(listnodes))){
    n = MostRecentAncestor(tree = subtree, tips = listnodes[[i]])
    # print(n)
    if (length(n)<1){
      n=listnodes[[i]]
    }
    nnames = names(listnodes[i])
    
    p <- p + 
      geom_hilight(node=n, 
                   fill = pal1[i]) + 
      geom_cladelab(node=n, 
                    label=nnames, 
                    align=TRUE, offset = 3.5,  
                    barcolor=pal1[i],angle="auto", 
                    fontsize=2.5)
  }
  return (p)
}


ggphylomorpho_custom <- function(tree,
                          tipinfo,
                          xvar=PC1,
                          yvar=PC2,
                          factorvar=group,
                          labelvar=taxon,
                          title="Phylomorphospace",
                          xlab="PC1",
                          ylab="PC2",
                          repel=TRUE,
                          edge.width=1,
                          fontface="italic",
                          tree.alpha = 0.7)
{
  
  require(ggplot2)
  require(phytools)
  require(ggrepel)
  
  ## create matrix for use in phytools::fastAnc()
  # mat <- cbind(eval(substitute(xvar), tipinfo),eval(substitute(yvar), tipinfo))
  # rownames(mat) <- eval(substitute(labelvar), tipinfo)
  mat <- cbind(tipinfo$coord0,tipinfo$coord1) #CHANGER ICI
  rownames(mat) <- rownames(tipinfo)
  stopifnot(length(setdiff(tree$tip.label, rownames(mat))) == 0)
  
  xAnc <- fastAnc(tree, mat[,1])
  yAnc <- fastAnc(tree, mat[,2])
  
  all_node_coords <-
    data.frame(
      #put PC values for all nodes and tips in a dataframe
      #tips go first in order of tip labels, then numerical order for nodes
      x=c(mat[tree$tip.label,1], xAnc),
      y=c(mat[tree$tip.label,2], yAnc),
      nodeid=1:(tree$Nnode + length(tree$tip.label))
    )
  
  #get edge list from tree object
  edges <- data.frame(tree$edge)
  names(edges) <- c("node1", "node2")
  #translate tip/node numbers into PC coordinates in all_node_coords dataframe
  edgecoords <- merge(
    merge(edges, all_node_coords, by.x="node1", by.y="nodeid"),
    all_node_coords, by.x="node2", by.y="nodeid")
  
  pointsForPlot <-
    data.frame(x=tipinfo$coord0,
               y=tipinfo$coord1,
               color=tipinfo$group,
               label=tipinfo$taxon)
  
  theplot <-
    ggplot() +
    geom_segment(data=edgecoords,aes(x=x.x,xend=x.y, y=y.x, yend=y.y), size=edge.width, alpha=tree.alpha) +
    geom_point(data=pointsForPlot, aes(x=x, y=y, color=color), size=5) +
    # geom_image(data=pointsForPlot, aes(x=x, y=y, image=im))+
    labs(title=title, x=xlab, y=ylab) +
    theme_bw(20) +
    theme(legend.position='right')
  # if(repel){
  #   theplot <- theplot + geom_text_repel(data=pointsForPlot, aes(x=x, y=y, label=label), segment.alpha=0.5, fontface=fontface)
  # } else{
  #   theplot <- theplot + geom_text(data=pointsForPlot, aes(x=x, y=y, label=label), fontface=fontface)
  # }
  return(theplot)
}
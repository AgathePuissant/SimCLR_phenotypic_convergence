
library(picante)
library(ape)
library(phangorn)
library(phytools)
library(geiger)


hush=function(code){ #Pour pas print
  sink("NUL") # use /dev/null in UNIX
  tmp = code
  sink()
  return(tmp)
}

match_tree <- function(add_poly = F, tree_to_match = list(), tree_path = "./data/Papilionidae_MCC_clean.tre" , meanphen_match, data_match, plot_match = F, genre=0){
  
  if (length(tree_to_match)==0){
    tree <-read.tree(file=tree_path)
  }else {
    tree <- tree_to_match
  }
  
  
  subtree <-tree
  subtree <- drop.tip(subtree, grep('[0-9]',subtree$tip.label))
  if (typeof(genre)!="double"){
    subtree<-keep.tip(subtree, grep(paste(genre,collapse = "|"),subtree$tip.label))
  }
  
  
  if (add_poly==T){
    gps <- data_match %>% group_by(tipsgenre)%>%
      group_rows()
    
    for (i in c(1:length(gps))){
      if ((as.character(unique(data_match$tipsgenre[gps[[i]]])) %in% subtree$tip.label)){
        subtree<-add.cherry(subtree,as.character(unique(data_match$tipsgenre[gps[[i]]])),unique(as.character(data_match$genresp[gps[[i]]])))
      }
    }
  }
  
  matching <-hush(match.phylo.data(subtree, meanphen_match))
  meanphen_match<-matching$data
  subtree<-matching$phy
  
  name.check(subtree, meanphen_match)
  
  if (plot_match == T){
    plotTree(subtree,type="fan",fsize=0.2,lwd=1)
  }
  
  return(list(subtree, meanphen_match))
}



## Function for adding a cherry to a tree where a single tip was before
add.cherry <- function(tree, tip, new.tips) {
  
  ## Find the edge leading to the tip
  tip_id <- match(tip, tree$tip.label)
  
  ## Create the new cherry
  tree_to_add <- ape::stree(length(c(tip, new.tips)))
  
  ## Naming the tips
  tree_to_add$tip.label <- c(tip, new.tips)
  
  ## Add 0 branch length
  tree_to_add$edge.length <- rep(0, Nedge(tree_to_add))
  
  ## Binding both trees
  return(bind.tree(tree, tree_to_add, where = tip_id))
}


permut <- function(subtree, oldmeanphen, loc =0){
  
  tipnames <- rownames(oldmeanphen)
  if (loc != 0) {
    oldmeanphen <- oldmeanphen[tipnames %in% Distrib_Papilionidae[Distrib_Papilionidae[,1+loc] == 1,1],]
    oldrownames <- rownames(oldmeanphen)
    olddata <- data[data$tipsgenre %in% Distrib_Papilionidae[Distrib_Papilionidae[,1+loc] == 1,1],]
  }
  else {
    oldrownames <- rownames(oldmeanphen)
    olddata <- data
  }
  list_match <- match_tree(meanphen_match = oldmeanphen, data_match = olddata, add_poly=F)#, genre=as.character(levels(data$Genre)[-21]))
  subtreeold <- list_match[[1]]
  meanphenold <- list_match[[2]]
  listpermute<-phylo.permute(subtreeold,meanphenold,margin=0.1)
  oldmeanphen<-listpermute$vec
  shuf <- listpermute$shuf
  
  return(list("new" = shuf, "old" = rownames(oldmeanphen)))
}

LG.permute <- function(tre,vec,k, loc){
  
  tipnames <- gsub("_[^_]*_[^_]*$","",rownames(vec))
  if (loc != 0) {
    vec <- vec[tipnames %in% Distrib_Papilionidae[Distrib_Papilionidae[,1+loc] == 1,1],]
    oldrownames <- rownames(vec)
    olddata <- data[data$tipsgenre %in% Distrib_Papilionidae[Distrib_Papilionidae[,1+loc] == 1,1],]
  }else{
    oldrownames <- rownames(vec)
    olddata <- data
  }
  list_match <- match_tree(meanphen_match = vec, data_match = olddata, add_poly=T)#, genre=as.character(levels(data$Genre)[-21]))
  tre <- list_match[[1]]
  vec <- list_match[[2]]
  
  pmat <- (k-cophenetic.phylo(tre)/max(cophenetic.phylo(tre)))/sum(cophenetic.phylo(tre)[,1])
  out <- matrix(nrow=dim(vec)[1],ncol=dim(vec)[2])
  rownames(out)<-rownames(vec)
  colnames(out)<-colnames(vec)
  shuf <- rownames(vec)
  for(i in seq(nrow(vec))){
    index <- sample(seq(nrow(vec)),1,prob=pmat[i,])
    # print(index)
    out[index,] <- as.numeric(vec[i,])
    shuf[index] <- rownames(vec)[i]
    pmat[,index] <- 0}
  return(list("new" = shuf, "old" = rownames(vec)))}
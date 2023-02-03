library(dplyr)
library(StatMatch)
# library(HDMD)
library(ade4)
library(stringr)
library(readr)
library(sampling)

harmonize <- function(path_data_photos= "./data/data_photos.csv", path_coords = "./data/pca_embeddings.csv", level="ssp"){
  data<-read.csv(file=path_coords, sep =';', header = T)
  data_photos<-read.csv(file=path_data_photos, sep =';', header = T, row.names = 1, fileEncoding="latin1")

  data<-merge(x = data, y = data_photos[,c("id","genus","sex","sp","ssp","form")], by = "id") 
  
  
  data$ssp[data$ssp=='-'] <- data$sp[data$ssp=='-']
  data$form[is.na(data$form)] <- data$ssp[is.na(data$form)]
  data$form[data$form==''] <- data$ssp[data$form=='']
  
  data$tipsgenre <- paste(data$genus,data$sp,sep="_")
  
  if (level=="ssp"){
    data$genresp <- paste(data$genus,data$sp,data$ssp,sep="_")
  }
  else if (level=="sp"){
    data$genresp <- data$tipsgenre
  }
  else if (level=="form"){
    data$genresp <- paste(data$genus,data$sp,data$ssp,data$form,sep="_")
  }
  
  data<-data %>% 
    mutate(across(where(is.character), str_remove_all, pattern = fixed(" ")))
  
  data$sp<-as.factor(data$sp)
  data$genus<-as.factor(data$genus)
  data$tipsgenre <- as.factor(data$tipsgenre)
  data$genresp <- as.factor(data$genresp)
  data$ssp <- as.factor(data$ssp)
  
  genresp_keep <- table(data[,c("tipsgenre","sex","view")])
  
  matD <- as.matrix(genresp_keep[ , , 1])
  matV <- as.matrix(genresp_keep[ , , 2])
  maskmat <- matD[,1]>0 & matD[,3]>0 & matV[,1]>0 & matV[,3]>0
  
  data = data[maskmat[data$tipsgenre],]
}

get_phenotype <- function(sex, view, wing=0, side=0, level = "ssp", mode = 'mean', path_data_photos = "./data/data_photos.csv", path_coords = "./data/pca_embeddings.csv", indices = 0, mode_indices="id"){
  
  data<-harmonize(path_data_photos = path_data_photos, path_coords = path_coords, level=level)
  
  if(mode_indices=="ssp"){
    data <- data %>%                                        # Create ID by group
      group_by(genus, sp, ssp) %>%
      dplyr::mutate(ID = cur_group_id())
  }
  else {
    data <- data %>%                                        # Create ID by group
      group_by(id) %>%
      dplyr::mutate(ID = cur_group_id())
  }
  
  data_sex <- data[ which(data$sex %in% sex & data$view %in% view), ]
  
  if (wing != 0){
    data_sex <- data_sex[ which(data_sex$wing %in% wing), ]
    }
  if (side != 0){
    data_sex <- data_sex[ which(data_sex$side %in% side), ]
    }
  
  data_sex = data_sex[order(data_sex$genresp),]
  
  if (mode == 'mean'){
    meanphen <- aggregate(dplyr::select( data.frame(data_sex), contains('coord')), list(data_sex$genresp), FUN=mean)
    meanphen2 <- meanphen[,-1]
    rownames(meanphen2) <- meanphen[,1]
    meanphen <- meanphen2
    ids = NaN
  }
  else if (mode == 'random'){
    x <- sampling::strata(data = data_sex, stratanames = "genresp", size = rep(1,nlevels(data_sex$genresp)),method="srswor")
    ids = data_sex[x$ID_unit,]$ID
    meanphen = data.frame(data_sex[x$ID_unit,])
    rownames(meanphen) <- meanphen$genresp
    meanphen = dplyr::select(meanphen, contains('coord'))
   }
  else if (mode == 'indices'){
    data_sex2 = data.frame(data_sex[data_sex$ID %in% indices,])
    x <- sampling::strata(data = data_sex2, stratanames = "ID", size = rep(1,nlevels(data_sex2$genresp)),method="srswor")
    meanphen = data.frame(data_sex2[x$ID_unit,])
    rownames(meanphen) <- meanphen$genresp
    meanphen = dplyr::select(meanphen, contains('coord'))
    ids = NaN
  }
  
  genre_data<-unique(data_sex$genus)
  sp_data <- unique(data_sex$sp)
  tipsgenre_data <- unique(data_sex$tipsgenre)
  genresp_data <- unique(data_sex$genresp)
  
  return (list(meanphen, data_sex, genre_data, sp_data, genresp_data, tipsgenre_data, ids))
  
}
#%%

from pixellib.instance import custom_segmentation
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import shutil
from tqdm import tqdm

from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("--path", type=str, default = r'J:\Segmentation\test_code\\') #path to the segmentation folder
parser.add_argument("--path_jpg", type=str, default =r'J:\Data\test_images\\') #path to the jsp image files
parser.add_argument("--wing_name", type=list, default =['_Ant_g','_Post_g','_Ant_d','_Post_d']) #List of names for the 4 wings
parser.add_argument("--save_imwing_raw", type=bool, default =True) #True if you want to save raw segmentation files
parser.add_argument("--save_as_tif", type=bool, default =False) #True if you want to save refined segmentation as tiff files
parser.add_argument("--dim_images", type=int, default =3) #number of channels of the images

args = parser.parse_args()
dict_args = vars(args)

path = dict_args['path']
path_jpg = dict_args['path_jpg']
wing_name = dict_args['wing_name']
save_imwing_raw = dict_args['save_imwing_raw']
save_as_tif = dict_args['save_as_tif']
dim_images = dict_args['dim_images']

model = "mask_rcnn_model.085-0.066738.h5" # segmentation model

to_correct = np.load(path+"to_correct.npy")
to_correct = [x[:-7]+".jpg" for x in to_correct]

#%%


os.chdir(path+"segmented") #Where to save the segmentation results


def SegmentWings_correct(path,save_as_tif,path_jpg,wing_name,model,
                 save_imwing_raw, to_correct) :
    '''Function to segment butterfly images, with a first part of automatic segmentation
        and a second part of refinement by image processing with opencv
    '''
    
    
    def flip(x, y):
        """Flips the x and y coordinate values"""
        return y, x
    
    if not os.path.isdir(path+"/rembg_out") : 
        os.mkdir(path+"/rembg_out") #Create a directory to store images without background
    os.system("rembg p "+path_jpg+" "+path+"/rembg_out")
    
    #Loading the automatic segmentation model with PixelLib
    segment_image = custom_segmentation()
    segment_image.inferConfig(num_classes= 1, class_names= ["BG","aile"])
    segment_image.load_model(path+model)
    
    listim = to_correct
    
    #Loop over the list of images to be processed
    for numim in tqdm(range(len(listim))) :
        
        #Loading the original image and creating the results storage folder for this image
        os.chdir(path_jpg)
        
        indexim = listim[numim]
        img_ori = plt.imread(indexim)
        shutil.rmtree(path+'segmented/'+indexim[:-4]+'_output')
        os.mkdir(path+'segmented/'+indexim[:-4]+'_output')
        
        if len(img_ori.shape)>2:
            dim_images = img_ori.shape[-1]
        else :
            dim_images = 1
        
        #Automatic segmentation with PixelLib
        if save_imwing_raw == True :
            segmask, output = segment_image.segmentImage(indexim, show_bboxes=True, output_image_name=path+'segmented/'+indexim[:-4]+'_output/'+indexim[:-4]+"_out.jpg")
        else :
            segmask, output = segment_image.segmentImage(indexim)
        
        #If 4 wings were detected with the automatic segmentation : 
        if len(segmask['class_ids'])==4 :
            
            stop = False #Variable to stop refining if there is a problem
            
            os.chdir(path+"segmented")
            
            #Numbering of the 4 wings counterclockwise
            minx=[]
            miny=[]
            for i in range (4) :
                minx.append(np.min(np.where(segmask['masks'][:,:,i]==True)[1]))
                miny.append(np.min(np.where(segmask['masks'][:,:,i]==True)[0]))
            minx=np.argsort(np.argsort(minx))
            miny=np.argsort(np.argsort(miny))
            aile=np.zeros(4)
            aile[minx<=1] = np.argsort(miny[minx<=1])
            aile[minx>1] = np.argsort(miny[minx>1]) + 2
            
            #Creation of empty images on which we will " paste " the semgented wings at the end
            imaile = np.zeros(shape = (segmask["masks"][:,:,i].shape[0],segmask["masks"][:,:,i].shape[1],dim_images)).astype('uint8').squeeze()
            imailebrut = np.zeros(shape = (segmask["masks"][:,:,i].shape[0],segmask["masks"][:,:,i].shape[1],dim_images)).squeeze()
            
            #Boucle sur les 4 ailes
            for k in range(4) :
                
                os.chdir(path+"segmented")
                i=np.where(aile==k)[0][0] #To recover the desired wing in the segmentation masks
                    
                #Recovery of raw segmentation masks and copies for future use
                img = img_ori.copy()
                a_mask = segmask["masks"][:,:,i].copy()
                mask = 1*np.stack([a_mask]*dim_images,axis=2).squeeze() #we stack the binary mask so that it is on the 3 channels (RGB)
                b_mask = segmask["masks"][:,:,i].copy()
                b_mask = 1*b_mask.astype('uint8')
                
                #Image segmentation with the raw segmentation mask
                img = img*mask
                
                imailebrut = imailebrut + img #" Pasting " of the raw segmented wing on a blank image
                
            
                #Preparation of the final masks and pasting of the wings on empty image
                
                #Individual wing masks
                im = Image.fromarray(255*mask.astype('uint8'))
                
                im.save(path+'segmented/'+indexim[:-4]+'_output/'+indexim[:-4]+wing_name[k]+'.jpg')
               
            #Saving images with the 4 segmented wings " pasted "
            os.chdir(path+'segmented/'+indexim[:-4]+'_output')
            im = Image.fromarray(imaile)
            if save_imwing_raw :
                imailebrut = imailebrut.astype('uint8')
                imailebrut[imailebrut==0] = 255
                im = Image.fromarray(imailebrut)
                im.save("imailebrut_"+indexim)
            
        else :
            print('segmentation error')
            
            with open(path+"segmented/errors.txt", 'a') as file:
                file.write("\n "+"Segmentation error - "+indexim+" \n")
            file.close()
            file.close()
            shutil.rmtree(path+'segmented/'+indexim[:-4]+'_output')
            
#%%

#Run to segment all the images in a row
if __name__=="__main__" :
    SegmentWings_correct(path,save_as_tif,path_jpg,wing_name,model,
                     save_imwing_raw, to_correct)




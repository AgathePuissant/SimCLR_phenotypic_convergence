#%%

from pixellib.instance import custom_segmentation
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from shapely.geometry import Polygon,Point,LineString
from shapely.ops import split
import scipy
import shutil
from os.path import isfile, join
from tqdm import tqdm
from shapely.ops import transform
from skimage.draw import polygon


from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("--path", type=str, default = r'J:\Segmentation\test_code\\') #path to the segmentation folder
parser.add_argument("--path_jpg", type=str, default =r'J:\Data\test_images\\') #path to the jsp image files
parser.add_argument("--wing_name", type=list, default =['_Ant_g','_Post_g','_Ant_d','_Post_d']) #List of names for the 4 wings
parser.add_argument("--save_imwing_raw", type=bool, default =False) #True if you want to save raw segmentation files
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

'''
Before starting the segmentation, the following folder structure must be created in the segmentation folder
(path) :

path/
    |_segmented/
    model.h5
'''

#%%

os.mkdir(path+"segmented")
os.chdir(path+"segmented") #Where to save the segmentation results


def SegmentWings(path,save_as_tif,path_jpg,wing_name,model,
                 save_imwing_raw) :
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
    
    #Creation of the list of images to be processed
    listim = os.listdir(path_jpg)
    listim = [f for f in listim if isfile(join(path_jpg, f))]
    listim = [x for x in listim if x[-6:]!="output" and x[-4:]!=".png" and x[-4:]!=".tif"]
    listim = [x for x in listim if x[:-4]+'_output' not in os.listdir(path+"/segmented")]
    
    #Loop over the list of images to be processed
    for numim in tqdm(range(len(listim))) :
        
        #Loading the original image and creating the results storage folder for this image
        os.chdir(path_jpg)
        indexim = listim[numim]
        img_ori = plt.imread(indexim)
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
                
                #Find wing overlap using bbox from segmentation
                if k==0 :
                    j=np.where(aile==1)[0][0]
                elif k==1 :
                    j=np.where(aile==0)[0][0]
                elif k==2 :
                    j=np.where(aile==3)[0][0]
                elif k==3 :
                    j=np.where(aile==2)[0][0]
                box_1 = [Point(list(segmask["rois"][i][2:4])),Point([segmask["rois"][i][0],segmask["rois"][i][3]]),Point(list(segmask["rois"][i][0:2])),Point([segmask["rois"][i][2],segmask["rois"][i][1]])]
                box_2 = [Point(list(segmask["rois"][j][2:4])),Point([segmask["rois"][j][0],segmask["rois"][j][3]]),Point(list(segmask["rois"][j][0:2])),Point([segmask["rois"][j][2],segmask["rois"][j][1]])]
                poly_1 = Polygon(box_1)
                poly_2 = Polygon(box_2)
                difference = poly_1.difference(poly_2)
                poly_1 = transform(flip,poly_1)
                difference = transform(flip, difference)
                x = np.array(difference.exterior.coords)[:,0] #x
                y = np.array(difference.exterior.coords)[:,1] #y
                if len(np.unique(x))>2 and len(np.unique(y))>2 :
                    if aile[i]==0 and aile[j]==1 :
                        line = LineString([Point(min(x),np.unique(np.sort(y))[-2]),Point(max(x),np.unique(np.sort(y))[-2])])
                    elif aile [i] == 3 and aile[j]==2 :
                        line = LineString([Point(min(x),np.unique(np.sort(y))[-2]),Point(max(x),np.unique(np.sort(y))[-2])])
                    elif aile[i]==1 and aile[j]==0 :
                        line = LineString([Point(min(x),np.unique(np.sort(y))[1]),Point(max(x),np.unique(np.sort(y))[1])])
                    elif aile[i]==2 and aile[j]==3 :
                        line = LineString([Point(min(x),np.unique(np.sort(y))[1]),Point(max(x),np.unique(np.sort(y))[1])])
                    result = split(difference, line)
                    finalbox = max(result, key=lambda a: a.area)
                else :
                    finalbox = difference
                poly_coordinates = np.array(list(finalbox.exterior.coords))
                x=poly_coordinates[:,0]
                y=poly_coordinates[:,1]
                if k==0 :
                    a=np.array([[0,np.max(y)],[np.max(x),np.max(y)],[np.max(x),0],[0,0]])
                elif k==1 :
                    a=np.array([[0,img.shape[0]],[np.max(x),img.shape[0]],[np.max(x),np.min(y)],[0,np.min(y)]])
                elif k==2 :
                    a=np.array([[img.shape[1],0],[img.shape[1],np.max(y)],[np.min(x),np.max(y)],[np.min(x),0]])
                elif k==3 :
                    a=np.array([[np.min(x),img.shape[0]],[img.shape[1],img.shape[0]],[img.shape[1],np.min(y)],[np.min(x),np.min(y)]])
                rr, cc = polygon(a[:,0],a[:,1],(segmask["masks"][:,:,i].shape[1],segmask["masks"][:,:,i].shape[0]))
                a_mask = segmask["masks"][:,:,i]
                a_mask[cc,rr] = True        
                
                #Segmentation of the wing according to the raw mask, taking only the part of the image corresponding to the desired wing
                img = img_ori.copy()
                mask = 1*np.stack([a_mask]*dim_images,axis=2).squeeze()
                mask = mask.astype('uint8')
                masked_arr = img*mask
                
                
                
                
                #Remove the background of the whole image
                img_without_background = plt.imread(path+'/rembg_out/'+indexim[:-4]+'.png')
                mask_remove_background = img_without_background[...,3]*255
                mask_remove_background = cv2.threshold(mask_remove_background,200,255,cv2.THRESH_BINARY)[1]
                mask_remove_background = mask_remove_background.astype('uint8')
                mask_remove_background = np.stack([mask_remove_background]*dim_images,axis=2).squeeze()
                comb_mask_2 = mask_remove_background
                masked_arr = masked_arr*comb_mask_2
                  
            
                
                masked_arr[masked_arr==0] = 255 #black (0) is changed to white (255)
            
                #Remove parts that touch wings and are in the mask (antennae, parts of the body)
                mask_final = np.zeros(masked_arr.shape)
                mask_final[masked_arr!=255] = 1
                kernel = np.ones((50,50),np.uint8)
                
                if dim_images ==1 :
                    temp_mask = mask_final
                else :
                    temp_mask = mask_final[:,:,0]
                    
                tophat = cv2.morphologyEx(temp_mask.astype('uint8').copy(), cv2.MORPH_TOPHAT, kernel)
                tophat = tophat - cv2.bitwise_and(tophat, b_mask)
                if np.sum(tophat)!=0 : #Check if there are things left in the mask to remove
                    #Remove objects on the outer right or left
                    if k==0 or k==1 : #Right
                        mask_touch = np.zeros(tophat.shape)
                        cnts,hierarchy = cv2.findContours(tophat.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                        cnts_array = [cnts[x] for x in range(len(cnts))]
                        cnts_array = np.vstack(cnts)
                        extr_val=np.max(cnts_array[:,0,0])
                        cnts = [ cnts[x] for x in range(len(cnts)) if extr_val in cnts[x][:,0,0] ]
                        mask_touch = cv2.fillPoly(mask_touch, pts =cnts, color=(255,255,255))
                    else : #Left
                        mask_touch = np.zeros(tophat.shape)
                        cnts,hierarchy = cv2.findContours(tophat.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                        cnts_array = [cnts[x] for x in range(len(cnts))]
                        cnts_array = np.vstack(cnts)
                        extr_val=np.min(cnts_array[:,0,0])
                        cnts = [ cnts[x] for x in range(len(cnts)) if extr_val in cnts[x][:,0,0] ]
                        mask_touch = cv2.fillPoly(mask_touch, pts =cnts, color=(255,255,255))    
                    mask_final = temp_mask*cv2.bitwise_not(mask_touch.astype('uint8'))
        
                    #Remove objects smaller than the wing (eg antennae and parts of body)
                    binary_flat = mask_final.astype('uint8')
                    mask_final = np.zeros(binary_flat.shape)
                    cnts,hierarchy = cv2.findContours(binary_flat.copy(), cv2.RETR_EXTERNAL,
                     	cv2.CHAIN_APPROX_SIMPLE)
                    area = [cv2.contourArea(x) for x in cnts]   
                    cnts = [ cnts[x] for x in range(len(area)) if x==np.argmax(area) ]
                    mask_final = cv2.fillPoly(mask_final, pts =cnts, color=(255,255,255))
                    #fill holes in the mask
                    mask_final = scipy.ndimage.morphology.binary_fill_holes(mask_final)
                    mask_final = np.stack([mask_final]*dim_images,axis=2).astype('float').squeeze()
                
                #Preparation of the final refined masks and pasting of the wings on empty image
                
                #Individual wing masks
                im_cpm = (masked_arr*mask_final)
                
                if dim_images!=1 :
                    im_cpm = im_cpm[:,:,0]
                
                mask_cpm = np.zeros(im_cpm.shape)
                mask_cpm[im_cpm!=0] = 1
                
                #Collage des ailes segmentées raffinées à partir des tif
                img = plt.imread(path_jpg+indexim[:-4]+'.jpg')
                
                
                img = img*np.stack([mask_cpm]*dim_images,axis=2).astype('uint8').squeeze()
                img[img==0] = 255
                imaile = imaile + img
                
                #Saving the wing mask
                mask_cpm = mask_cpm*255
                mask_cpm=mask_cpm.astype('uint8')
                im = Image.fromarray(mask_cpm)
                
                im.save(path+'segmented/'+indexim[:-4]+'_output/'+indexim[:-4]+wing_name[k]+'.jpg')
               
            #Saving images with the 4 segmented wings " pasted "
            os.chdir(path+'segmented/'+indexim[:-4]+'_output')
            im = Image.fromarray(imaile)
            if save_as_tif==True :
                im.save("imaile_"+indexim[:-4]+'.tif')
            else :
                im.save("imaile_"+indexim[:-4]+'.jpg')
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
    SegmentWings(path,save_as_tif,path_jpg,wing_name,model,
                     save_imwing_raw)




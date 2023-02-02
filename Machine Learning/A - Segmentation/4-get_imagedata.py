import os
import cv2
import tqdm

from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("--path_segment", type=str, default = "J:/Segmentation/test_code/segmented") #path to the segmentation folder
parser.add_argument("--path_save", type=str, default =r"C:\Users\Agathe\Desktop\dataset_test") #path to the jsp image files

args = parser.parse_args()
dict_args = vars(args)

path_segment = dict_args['path_segment']
path_save = dict_args['path_save']
os.chdir(path_segment)

listdirs = os.listdir(path_segment)

for d in tqdm.tqdm(listdirs) :
    im = cv2.imread(d+'/imaile_'+d.replace('_output','')+'.jpg')
    
    if im is None :
        im = cv2.imread(d+'/imailebrut_'+d.replace('_output','')+'.jpg')
    
    cv2.imwrite(path_save+'/'+d.replace('_output','')+'.jpg', im)
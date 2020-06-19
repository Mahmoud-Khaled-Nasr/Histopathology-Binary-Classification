import tqdm
import time
import numpy as np
import cv2
from sklearn.metrics import (roc_curve, auc, accuracy_score, f1_score, precision_score, 
                             recall_score, classification_report, confusion_matrix)
import seaborn as sn
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, StratifiedKFold
from dataio import *
from preprocess import *
from colorfeatures import *
from classify import *





### Feature Extraction
def extract_features(image,mask=None):    
    # Color Spaces: I/O -------------------------------------------------------------------------------------------------------------------------------------
    img_RGB               = image
    img_GL                = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2GRAY)
    img_HSV               = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2HSV)
    img_LAB               = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2Lab)
    img_YCrCb             = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2YCrCb)
    img_luv               = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2Luv)
    #---------------------------------------------------------------------------------------------------------------------------------------------------------

    
    # Color Moments ------------------------------------------------------------------------------------------------------------------------------------------
    mean_R, std_R, skew_R, kurt_R, mean_G,  std_G,  skew_G,  kurt_G,  mean_B,  std_B,  skew_B,  kurt_B   = color_moments(img_RGB,     channel=3)
    mean_H, std_H, skew_H, kurt_H, mean_S,  std_S,  skew_S,  kurt_S,  mean_V,  std_V,  skew_V,  kurt_V   = color_moments(img_HSV,     channel=3)
    mean_L, std_L, skew_L, kurt_L, mean_A,  std_A,  skew_A,  kurt_A,  mean_b,  std_b,  skew_b,  kurt_b   = color_moments(img_LAB,     channel=3)
    mean_Y, std_Y, skew_Y, kurt_Y, mean_Cr, std_Cr, skew_Cr, kurt_Cr, mean_Cb, std_Cb, skew_Cb, kurt_Cb  = color_moments(img_YCrCb,   channel=3)
    mean_l, std_l, skew_l, kurt_l, mean_u,  std_u,  skew_u,  kurt_u,  mean_v,  std_v,  skew_v,  kurt_v   = color_moments(img_luv,     channel=3)
    #---------------------------------------------------------------------------------------------------------------------------------------------------------
    
    # Graylevel Co-Occurrence Matrix ----------------------
    GLCM_RGB   = GLCM(img_RGB,   channel=3)
    GLCM_HSV   = GLCM(img_HSV,   channel=3)    
    GLCM_LAB   = GLCM(img_LAB,   channel=3)
    #------------------------------------------------------
    
    # Local Binary Patterns --------------------------------------------------------------------
    lbp_R, lbp_G, lbp_B    = LBP(img_RGB,   channel=3)
    lbp_H, lbp_S, lbp_V    = LBP(img_HSV,   channel=3)
    lbp_Y, lbp_Cr, lbp_Cb  = LBP(img_YCrCb, channel=3) 
    lbp_GL                 = LBP(img_GL,    channel=1) 
    LBP_CGLF  = np.concatenate((lbp_R,lbp_G,lbp_B,lbp_H,lbp_S,lbp_V,lbp_Y,lbp_Cr,lbp_Cb,lbp_GL),axis=0)
    #-------------------------------------------------------------------------------------------
    
    # Smoothness, Uniformity, Entropy -----------------------------
    smoothness_GL, uniformity_GL, entropy_GL = entropyplus(img_GL)
    #--------------------------------------------------------------   
    
    # Graph Features ---------------------------------------------------------------------------------------------
    voronoi     = voronoi_features(img_GL)
    delaunay    = delaunay_features(img_GL)
    #--------------------------------------------------------------------------------------------------------------
    
    # Gabor Filter Features ---------------------------------------------------------------------------------------
    gabor_filters    = []
    gabor_energy     = []
    kernel_size      = 8
    thetas           = [0, np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2, 5*np.pi/8, 3*np.pi/4, 7*np.pi/8, np.pi]
    for theta in thetas:
        kern = cv2.getGaborKernel((kernel_size,kernel_size), 3.25, theta, 9.0, 1.0, 1.0, ktype=cv2.CV_32F)
        kern /= 1.5*kern.sum()
        gabor_filters.append(kern) 
    for kern in gabor_filters:
        fimg         = cv2.filter2D(img_GL, cv2.CV_8UC3, kern)
        GLCM_gabor   = greycomatrix(fimg,  [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=2**8)
        gabor_energy.append(greycoprops(GLCM_gabor,'energy').mean())
    #--------------------------------------------------------------------------------------------------------------
    
    
    features = [ mean_R, std_R, skew_R, mean_G,  std_G,  skew_G,  mean_B,  std_B,  skew_B,   
                 mean_H, std_H, skew_H, mean_S,  std_S,  skew_S,  mean_V,  std_V,  skew_V,   
                 mean_L, std_L, skew_L, mean_A,  std_A,  skew_A,  mean_b,  std_b,  skew_b,   
                 mean_Y, std_Y, skew_Y, mean_Cr, std_Cr, skew_Cr, mean_Cb, std_Cb, skew_Cb, 
                 mean_l, std_l, skew_l, mean_u,  std_u,  skew_u,  mean_v,  std_v,  skew_v,
                 smoothness_GL, uniformity_GL, entropy_GL ]
 
    features = np.concatenate((features, GLCM_RGB, GLCM_HSV, GLCM_LAB, LBP_CGLF, voronoi, delaunay, gabor_energy),axis=0)

    return features
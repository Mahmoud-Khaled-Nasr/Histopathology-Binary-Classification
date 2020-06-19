### Libraries
import numpy as np
import math
from scipy.stats import skew
from scipy.stats import kurtosis
from skimage.feature import greycomatrix, greycoprops
from skimage.feature import local_binary_pattern
from scipy.spatial import ConvexHull, Voronoi, voronoi_plot_2d, distance, Delaunay
from scipy.stats import entropy as entropy_
from sklearn import preprocessing
from skimage.morphology.convex_hull import convex_hull_image
from skimage.color import rgb2hed, rgb2gray
import cv2





### Color Features
def color_moments(image, channel=3):
    if (channel==3):        
        mean_0 = np.mean(image[:,:,0])
        mean_1 = np.mean(image[:,:,1])
        mean_2 = np.mean(image[:,:,2])
        std_0  = np.std(image[:,:,0])
        std_1  = np.std(image[:,:,1])
        std_2  = np.std(image[:,:,2])
        skew_0 = skew(image[:,:,0].reshape(-1))
        skew_1 = skew(image[:,:,1].reshape(-1))
        skew_2 = skew(image[:,:,2].reshape(-1))
        kurt_0 = kurtosis(image[:,:,0].reshape(-1))
        kurt_1 = kurtosis(image[:,:,1].reshape(-1))
        kurt_2 = kurtosis(image[:,:,2].reshape(-1))
        return mean_0, std_0, skew_0, kurt_0, mean_1, std_1, skew_1, kurt_1, mean_2, std_2, skew_2, kurt_2
    
    elif (channel==1):        
        mean_0 = np.mean(image)
        std_0  = np.std(image)
        skew_0 = skew(image.reshape(-1))
        return mean_0, std_0, skew_0
    
    else:
        assert False, "ERROR: The function supports only 1 or 3-channel image formats."
    


    
def GLCM(image, channel=3, bit_depth=8):
    if (channel==3):  
        GLCM_0  = greycomatrix(image[:,:,0],  [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=2**bit_depth)
        GLCM_1  = greycomatrix(image[:,:,1],  [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=2**bit_depth)
        GLCM_2  = greycomatrix(image[:,:,2],  [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=2**bit_depth)
        contrast_0  = greycoprops(GLCM_0,  'contrast').mean()
        contrast_1  = greycoprops(GLCM_1,  'contrast').mean()
        contrast_2  = greycoprops(GLCM_2,  'contrast').mean()
        dissim_0    = greycoprops(GLCM_0,  'dissimilarity').mean()
        dissim_1    = greycoprops(GLCM_1,  'dissimilarity').mean()
        dissim_2    = greycoprops(GLCM_2,  'dissimilarity').mean()
        correl_0    = greycoprops(GLCM_0,  'correlation').mean()
        correl_1    = greycoprops(GLCM_1,  'correlation').mean()
        correl_2    = greycoprops(GLCM_2,  'correlation').mean()
        homo_0      = greycoprops(GLCM_0,  'homogeneity').mean()
        homo_1      = greycoprops(GLCM_1,  'homogeneity').mean()
        homo_2      = greycoprops(GLCM_2,  'homogeneity').mean()
        return [ contrast_0, dissim_0, correl_0, homo_0, contrast_1, dissim_1,
                 correl_1, homo_1, contrast_2, dissim_2, correl_2, homo_2 ]
    
    elif (channel==1):
        GLCM_0  = greycomatrix(image,  [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=2**bit_depth)
        contrast_0  = greycoprops(GLCM_0,  'contrast').mean()
        dissim_0    = greycoprops(GLCM_0,  'dissimilarity').mean()
        correl_0    = greycoprops(GLCM_0,  'correlation').mean()
        homo_0      = greycoprops(GLCM_0,  'homogeneity').mean()
        return [ contrast_0, dissim_0, correl_0, homo_0 ]
    
    else:
        assert False, "ERROR: The function supports only 1 or 3-channel image formats."




def melanoma_color_markers(image,mask):
    CM_black      = np.count_nonzero(((image[:,:,0].astype(float)/255)<0.20)
                                    &((image[:,:,1].astype(float)/255)<0.20)
                                    &((image[:,:,2].astype(float)/255)<0.20))*(100/np.sum(mask))
    CM_red        = np.count_nonzero(((image[:,:,0].astype(float)/255)>0.80)
                                    &((image[:,:,1].astype(float)/255)<0.20)
                                    &((image[:,:,2].astype(float)/255)<0.20))*(100/np.sum(mask))
    CM_bluegray   = np.count_nonzero(((image[:,:,0].astype(float)/255)<0.20)
                                    &((image[:,:,1].astype(float)/255)<0.72)
                                    &((image[:,:,1].astype(float)/255)>0.30)
                                    &((image[:,:,2].astype(float)/255)<0.74)
                                    &((image[:,:,2].astype(float)/255)>0.34))*(100/np.sum(mask))
    CM_white      = np.count_nonzero(((image[:,:,0].astype(float)/255)>0.80)
                                    &((image[:,:,1].astype(float)/255)>0.80)
                                    &((image[:,:,2].astype(float)/255)>0.80))*(100/np.sum(mask))
    CM_lightbrown = np.count_nonzero(((image[:,:,0].astype(float)/255)<1.00)
                                    &((image[:,:,0].astype(float)/255)>0.60)
                                    &((image[:,:,1].astype(float)/255)<0.72)
                                    &((image[:,:,1].astype(float)/255)>0.32)
                                    &((image[:,:,2].astype(float)/255)<0.45)
                                    &((image[:,:,2].astype(float)/255)>0.05))*(100/np.sum(mask))
    CM_darkbrown  = np.count_nonzero(((image[:,:,0].astype(float)/255)<0.60)
                                    &((image[:,:,0].astype(float)/255)>0.20)
                                    &((image[:,:,1].astype(float)/255)<0.46)
                                    &((image[:,:,1].astype(float)/255)>0.06)
                                    &((image[:,:,2].astype(float)/255)<0.33))*(100/np.sum(mask))
    return CM_black, CM_red, CM_bluegray, CM_white, CM_lightbrown, CM_darkbrown




def LBP(image, channel=3, P=8, R=2, bins=10):
    if (channel==3):
        lbp       = local_binary_pattern(image[:,:,0], P=P, R=R, method="uniform")
        lbp_0, _  = np.histogram(lbp, density=True, bins=bins, range=(0,int(lbp.max()+1)))
        lbp       = local_binary_pattern(image[:,:,1], P=P, R=R, method="uniform")
        lbp_1, _  = np.histogram(lbp, density=True, bins=bins, range=(0,int(lbp.max()+1)))
        lbp       = local_binary_pattern(image[:,:,2], P=P, R=R, method="uniform")
        lbp_2, _  = np.histogram(lbp, density=True, bins=bins, range=(0,int(lbp.max()+1)))
        return lbp_0, lbp_1, lbp_2
    
    elif (channel==1):
        lbp       = local_binary_pattern(image, P=P, R=R, method="uniform")
        lbp_0, _  = np.histogram(lbp, density=True, bins=bins, range=(0,int(lbp.max()+1)))
        return lbp_0
    else:
        assert False, "ERROR: The function supports only 1 or 3-channel image formats."

        

def entropyplus(image):   
    histogram         = np.histogram(image, bins=2**8, range=(0,(2**8)-1), density=True)
    histogram_prob    = histogram[0]/sum(histogram[0])    
    single_entropy    = np.zeros((len(histogram_prob)), dtype = float)
    for i in range(len(histogram_prob)):
        if(histogram_prob[i] == 0):
            single_entropy[i] = 0;
        else:
            single_entropy[i] = histogram_prob[i]*np.log2(histogram_prob[i]);
    smoothness   = 1- 1/(1 + np.var(image/2**8))            
    uniformity   = sum(histogram_prob**2);        
    entropy      = -(histogram_prob*single_entropy).sum()
    return smoothness, uniformity, entropy



def entropyplus_3(image):
    smoothness_0, uniformity_0, entropy_0 = entropyplus(image[:,:,0])
    smoothness_1, uniformity_1, entropy_1 = entropyplus(image[:,:,1])
    smoothness_2, uniformity_2, entropy_2 = entropyplus(image[:,:,2])
    return [ smoothness_0, uniformity_0, entropy_0, smoothness_1, uniformity_1, 
             entropy_1, smoothness_2, uniformity_2, entropy_2 ]



def voronoi_volumes(points):
    v = Voronoi(points)
    vol = np.zeros(v.npoints)
    for i, reg_num in enumerate(v.point_region):
        indices = v.regions[reg_num]
        if -1 in indices: # some regions can be opened
            vol[i] = np.inf
        else:
            try:
                vol[i] = ConvexHull(v.vertices[indices]).volume
            except:
                vol[i] = 0
    return vol



def remove_values_from_list(the_list, val):
    return [value for value in the_list if value != val]



def find_max_dist(vor, point, array):
    
    all_dist = []
    
    for x in array:
        all_dist.append(distance.euclidean(vor.ridge_points[point], vor.ridge_points[x]))
        
    return np.max(all_dist)




def get_tri_info(a, b, c):
    def distance_(p1, p2):
        return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

    side_a = distance_(a, b)
    side_b = distance_(b, c)
    side_c = distance_(c, a)
    s = 0.5 * ( side_a + side_b + side_c)
    
    return math.sqrt(s * (s - side_a) * (s - side_b) * (s - side_c)), side_a, side_b, side_c





def delaunay_features(image, perc=0.30):
    # Blur and Generate Difference Image
    blur          = cv2.GaussianBlur(image,(3,3),0)
    min_intensity = np.min(blur)
    max_intensity = np.max(blur)
    diff          = (max_intensity - min_intensity) * perc
    
    # Apply Threshold to Extract Nuclei
    thresh        = blur.copy()
    thresh[thresh > (min_intensity + diff)] = 0
    thresh[thresh > 0]                      = 1
    
    # Extract Connected Component Statistics
    connectivity  = 4
    output        = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_8S)
    centroids     = output[3]
    

    if len(centroids) > 3:    
        # Extract Triangles and Vertices
        tri              = Delaunay(centroids)
        faces            = tri.simplices
        all_edge_lengths = []
        all_areas        = []

        # For Each Face, Extract Features
        for face in faces:
            try:
                area, d1, d2, d3 = get_tri_info([tri.points[face[0]][0], tri.points[face[0]][1]], 
                                                [tri.points[face[1]][0], tri.points[face[1]][1]], 
                                                [tri.points[face[2]][0], tri.points[face[2]][1]])
                all_edge_lengths.append(d1)
                all_edge_lengths.append(d2)
                all_edge_lengths.append(d3)
                if (area==0):
                    all_areas.append(1e-50)
                else:
                    all_areas.append(area)
            except:
                pass

        # Aggregate Delaunay Features
        d_area_avg              = np.mean(all_areas)
        d_area_std              = np.std(all_areas)
        d_area_min_max_ratio    = np.min(all_areas)/np.max(all_areas)
        d_area_disorder         = entropy_(all_areas)
        d_lengths_avg           = np.mean(all_edge_lengths)
        d_lengths_std           = np.std(all_edge_lengths)
        d_lengths_min_max_ratio = np.min(all_edge_lengths)/np.max(all_edge_lengths)
        d_lengths_disorder      = entropy_(all_edge_lengths)  

    else:
        d_area_avg    = d_area_std    = d_area_min_max_ratio    = d_area_disorder    = 0
        d_lengths_avg = d_lengths_std = d_lengths_min_max_ratio = d_lengths_disorder = 0

    return [d_area_avg, d_area_std, d_area_min_max_ratio, d_area_disorder, 
            d_lengths_avg, d_lengths_std, d_lengths_min_max_ratio, d_lengths_disorder]




def voronoi_features(image, perc=0.30):
    # Blur and Generate Difference Image
    blur          = cv2.GaussianBlur(image,(3,3),0)
    min_intensity = np.min(blur)
    max_intensity = np.max(blur)
    diff          = (max_intensity - min_intensity) * perc
    
    # Apply Threshold to Extract Nuclei
    thresh        = blur.copy()
    thresh[thresh > (min_intensity + diff)] = 0
    thresh[thresh > 0]                      = 1
    
    # Extract Connected Component Statistics
    connectivity  = 4
    output        = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_8S)
    centroids     = output[3]
    
    if len(centroids) > 3:       

        # Extract Polygon Object Areas
        vor      = Voronoi(centroids)
        all_area = voronoi_volumes(centroids)

        # Redundancy Check (remove any occurences of 'inf')
        new_all_area = []
        for area in all_area:
            if str(area) != 'inf':
                new_all_area.append(area)
        if (len(new_all_area)==0):
            area_avg  = area_std_dev  = area_min_max_ratio  = area_entropy  = 0
            chord_avg = chord_std_dev = chord_min_max_ratio = chord_entropy = 0
        else:
            all_regions       = vor.regions
            all_chord_lengths = []

            for region in all_regions:
                if (len(region)) == 0 or -1 in region:
                    continue

                # Loop Through All Points and Store Max Chord Length
                for point in region:
                    all_max_dist = []
                    region_copy  = region.copy()
                    region_copy.remove(point)
                    max_dist     = find_max_dist(vor, point, region_copy)
                    all_max_dist.append(max_dist)
                all_chord_lengths.append(np.max(all_max_dist))

            # Compute Area Statistics
            area_avg     = np.mean(new_all_area)
            area_std_dev = np.std(new_all_area)

            # Redundancy Check
            if len(new_all_area) == 0:
                area_min_max_ratio = 0
            else:
                area_min_max_ratio = (np.min(new_all_area)/np.max(new_all_area))
            area_entropy = entropy_(new_all_area)

            # Compute Chord Statistics
            chord_avg     = np.mean(all_chord_lengths)
            chord_std_dev = np.std(all_chord_lengths)

            if len(all_chord_lengths) == 0:
                chord_min_max_ratio = 0
            else:
                chord_min_max_ratio = (np.min(new_all_area)/np.max(new_all_area))
            chord_entropy = entropy_(all_chord_lengths)
    else:        
        area_avg  = area_std_dev  = area_min_max_ratio  = area_entropy  = 0
        chord_avg = chord_std_dev = chord_min_max_ratio = chord_entropy = 0

    return [ area_avg, area_std_dev, area_min_max_ratio, area_entropy,
             chord_avg, chord_std_dev, chord_min_max_ratio, chord_entropy ]
     
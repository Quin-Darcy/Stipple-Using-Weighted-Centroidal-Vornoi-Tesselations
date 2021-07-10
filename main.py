#=================================================================================#
#   AUTHOR: Quin Darcy
#   PROJECT: Stipple Generator
#   DATE: 07-02-2021
#   Description: This program takes a given image, applies a greyscale filter
#       to it, follows by a halftoner. After this, we use the (?) method along with
#       Lloyd-Relaxation to obtain a weighted centroidal Vornoi Diagram, whose
#       generators are the stipple points.
#=================================================================================#
import numpy as np
from numpy import linalg as la
import math
import time
import cmath
import random
from PIL import Image
import itertools
from itertools import groupby
from itertools import product
import scipy
import gc
import my_class as mc
from scipy.spatial.distance import cdist

#=================================================================================#
#                                  CONSTANTS                                      #
#=================================================================================#                                 
EPSILON = 0.05
#=================================================================================#
#                                  GREYSCALER                                     #
#=================================================================================#
def get_grey(pixel):
    try:
        theta = abs(math.pi / 4 - math.atan(pixel[1] / pixel[0]))
    except ZeroDivisionError:
        theta = math.pi / 2
    C = math.sqrt(math.pow(pixel[0], 2)+math.pow(pixel[1], 2)) * math.cos(theta) / math.sqrt(2)
    C = math.floor(C)
    return tuple([C, C, C])

def greyscale():
    img = Image.open('head.jpg')
    px = img.load()
    sz = img.size

    for i, j in product(range(sz[0]), range(sz[1])):
        px[i, j] = get_grey(px[i, j])
    img.save('greyscale_result.jpg')
    img.close()
    return sz
#=================================================================================#
#                                   HALFTONER                                     #
#=================================================================================#                               
def halftone(num):
    img = Image.open('greyscale_result.jpg')
    px = img.load()
    sz = img.size
    K = 0
    C0 = 0
    C1 = 0
    C2 = 0
    C3 = 0
    C4 = 0
    err = 0

    for i, j in product(range(sz[0]), range(sz[1])):
        if (px[i, j][0] > random.randint(44, 191)):
            px[i, j] = (255,255,255)
        else:
            px[i, j] = (0,0,0)
    
    img.save('stipple/stipple_'+str(num)+'.jpg')
    img.close()
#=================================================================================#
#                                GENERATORS                                       #
#=================================================================================#
def get_sites():
    img = Image.open('stipple/stipple_0.jpg')
    px = img.load()
    sz = img.size
    sites = []

    for i, j in product(range(sz[0]), range(sz[1])):
        if (px[i, j][0] == 0):
            sites.append([i, j])

    img.close()
    return np.array(sites)
#=================================================================================#
#                                 REGIONS                                         #
#=================================================================================#
def gen(m, n, sites):
    arr = [[0 for i in range(m)] for j in range(n)]
    for i in range(len(sites)):
        arr[sites[i][1]][sites[i][0]] = i + 1
    return np.array(arr)

def get_regions(num, sites, sz):
    m = sz[0]
    n = sz[1]
    regions = [[] for i in range(len(sites))]
    arr = gen(m, n, sites)
    root = mc.Node(arr, sites, [[0,0], [m-1,0], [0,n-1], [m-1,n-1]], 0, 1)
    root.div_conq()
    for i, j in product(range(m), range(n)):
        regions[root.arr[j][i] - 1].append([i, j]) 
    #del arr
    root.delete()
    gc.collect()
    return regions, arr
#=================================================================================#
#                                 CENTROIDS                                       #
#=================================================================================#
def get_center(reg, num):
    img = Image.open('stipple/stipple_'+str(num)+'.jpg')
    px = img.load()
    L = len(reg)
    b_sum = 0
    num_x = 0
    den_x = 0
    num_y = 0
    den_y = 0

    for i in range(L):
        if (px[reg[i][0], reg[i][1]][0] == 0):
            b_sum += 1
    
    pb = b_sum / L
    pw = (L-b_sum) / L

    for i in range(L):
        if (px[reg[i][0], reg[i][1]][0] == 0):
            num_x += reg[i][0] * pb
            den_x += pb
            num_y += reg[i][1] * pb
            den_y += pb
        else:
            num_x += reg[i][0] * pw
            den_x += pw
            num_y += reg[i][1] * pw
            den_y += pw

    img.close()
    return [int(num_x / den_x), int(num_y / den_y)]

def get_centroids(regions, num):
    L = len(regions)
    C = []

    for i in range(L):
        C.append(get_center(regions[i], num))

    return C
#=================================================================================#
#                                     IMAGE                                       #
#=================================================================================#
def make_img(cent, num, sz):
    L = len(cent)
    img = Image.new('RGB', sz, color=(255, 255, 255))
    px = img.load()

    for i in range(L):
        x = cent[i][0]
        y = cent[i][1]
        px[x, y] = (0,0,0)

    img.save('stipple/stipple_'+str(num+1)+'.jpg')
    img.close()
#=================================================================================#
#                                    THRESHOLD                                    #
#=================================================================================#
def get_avg_dist(cent, sites):
    L = len(cent)
    sum = 0
    for i in range(L):
        a = np.array([cent[i][0], cent[i][1]])
        b = np.array([sites[i][0], sites[i][1]])
        sum += la.norm(a-b)
    return sum / L
#=================================================================================#
#                                    VORONOI                                      #
#=================================================================================#
def start(sites, sz, num):
    # INITIAL VORONOI TESSELATION
    print('GENERATING VORONOI TESSELATION #'+str(num+1)+' ... ')
    regions, arr = get_regions(num, sites, sz)

    # CALCULATE WEIGHTED CENTROIDS
    print('CALCULATING WEIGHTED CENTROIDS ... ')
    centroids = get_centroids(regions, num)

    # SET SITES TO CENTROIDS 
    print('GENERATING IMAGE BASED ON CENTROIDS ... ')
    make_img(centroids, num, sz)

    # CALCULATE SITE-TO-CENTROID DISTANCES
    print('CALCULATING SITE-TO-CENTROID DISTANCES ... ')
    avg = get_avg_dist(centroids, sites)
    print('AVERAGE DISTANCE: ', '{:.2f}'.format(avg))
    sites = centroids
    del centroids
    gc.collect()
    sites.sort()
    sites = list(sites for sites,_ in itertools.groupby(sites))

    return sites
#=================================================================================#
#                                     MAIN                                        #
#=================================================================================#
def main():
    num = 0
    avg = EPSILON + 1
    # GRAYSCALE SECTION
    print('RENDERING GREYSCALE IMAGE ... ')
    sz = greyscale()

    # HALFTONE SECTION
    print('RENDERING HALFTONE IMAGE ... ')
    halftone(num)
    sites = get_sites()

    for i in range(15):
        sites = start(sites, sz, num)
        num += 1



if __name__=='__main__':
    main()



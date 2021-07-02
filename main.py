import numpy as np
from numpy import linalg as la
import math
import time
import cmath
import random
from PIL import Image
import itertools
from itertools import groupby

EPSILON = 0.05

def get_grey(pixel):
    theta = abs(math.pi / 4 - math.atan(pixel[1] / (pixel[0] + 0.0000000001)))
    C = math.sqrt(math.pow(pixel[0], 2)+math.pow(pixel[1], 2)) * math.cos(theta) / math.sqrt(2)
    C = math.floor(C)
    return tuple([C, C, C])

def greyscale():
    img = Image.open('face.jpg')
    px = img.load()
    sz = img.size

    for i in range(sz[0]):
        for j in range(sz[1]):
            px[i, j] = get_grey(px[i, j])
    img.save('greyscale_result.jpg')
    img.close()
    return sz

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

    '''
    for i in range(sz[0]):
        for j in range(sz[1]):
            if (px[i, j][0] > random.randint(74, 201)):
                px[i, j] = (255,255,255)
            else:
                px[i, j] = (0,0,0)
    '''

    for i in range(1, sz[0] - 1):
        for j in range(1, sz[1]):
            if (px[i, j][0] > random.randint(64, 191)):
                K = 255
            else:
                K = 0
            # Floyd-Steinberg error diffusion
            C0 = px[i, j][0]
            err = C0 - K
            C1 = px[i+1, j][0]
            temp = math.floor(C1+7*err/16)
            px[i+1, j] = tuple([temp, temp, temp])
            C2 = px[i-1, j-1][0]
            temp = math.floor(C2+3*err/16)
            px[i-1, j-1] = tuple([temp, temp, temp])
            C3 = px[i, j-1][0]
            temp = math.floor(C3+5*err/16)
            px[i, j-1] = tuple([temp, temp, temp])
            C4 = px[i+1, j-1][0]
            temp = math.floor(C4+err/16)
            px[i+1, j-1] = tuple([temp, temp, temp])
            px[i, j] = tuple([K, K, K])     


    img.save('stipple/stipple_'+str(num)+'.jpg')
    img.close()

def get_sites():
    img = Image.open('stipple/stipple_0.jpg')
    px = img.load()
    sz = img.size
    sites = []

    for i in range(sz[0]):
        for j in range(sz[1]):
            if (px[i, j][0] == 0):
                sites.append([i, j])
    img.close()
    return sites

def get_dist(a, b, c, d):
    return math.sqrt(math.pow(a-c, 2) + math.pow(b-d, 2))

def get_index(a, b, sites):
    dists = []
    for i in range(len(sites)):
        dists.append(get_dist(a, b, sites[i][0], sites[i][1]))
    MIN = min(dists)
    for i in range(len(dists)):
        if (MIN == dists[i]):
            return i

def get_regions(num, sites):
    L = len(sites)
    img = Image.open('stipple/stipple_'+str(num)+'.jpg')
    px = img.load()
    sz = img.size
    regions = [[] for i in range(L)]

    for i in range(sz[0]):
        for j in range(sz[1]):
            k = get_index(i, j, sites)
            regions[k].append([i, j])

    img.close()
    return regions

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

def make_img(cent, num, sz):
    L = len(cent)
    img = Image.new('RGB', sz, color=(255, 255, 255))
    px = img.load()

    for i in range(sz[0]):
        for j in range(sz[1]):
            px[i, j] = (255,255,255)

    for i in range(L):
        x = cent[i][0]
        y = cent[i][1]
        px[x, y] = (0,0,0)

    img.save('stipple/stipple_'+str(num+1)+'.jpg')
    img.close()

def get_avg_dist(cent, sites):
    L = len(cent)
    dist = []
    sum = 0
    for i in range(L):
        sum += get_dist(cent[i][0], cent[i][1], sites[i][0], sites[i][1])
    return sum / L

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

    while (avg > EPSILON):
        # INITIAL VORNOI TESSELATION
        print('GENERATING VORNOI TESSELATION #'+str(num)+' ... ')
        regions = get_regions(num, sites)

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
        sites.sort()
        sites = list(sites for sites,_ in itertools.groupby(sites))
        num += 1



if __name__=='__main__':
    main()

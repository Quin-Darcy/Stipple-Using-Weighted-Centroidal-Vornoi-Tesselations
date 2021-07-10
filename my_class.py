import itertools
from itertools import product
import numpy as np
import gc
import scipy
import math
from scipy.spatial.distance import cdist

class Node:
    def __init__(self, arr, sites, corners, depth, node):
        self.depth = depth
        self.node = node
        self.arr = arr
        self.sites = sites
        self.corners = corners
        self.c_sites = self.closest_sites()
        self.ul = None
        self.ur = None
        self.ll = None
        self.lr = None

    def delete(self):
        del self.arr
        del self.sites
        del self.c_sites
        del self.corners
        del self.ul
        del self.ur
        del self.ll
        del self.lr
        gc.collect()

    def closest_sites(self):
        S = np.array(self.sites)
        temp = []
        for i in range(4):
            P = np.array([[self.corners[i][0], self.corners[i][1]]])
            d = cdist(P, S)
            index = np.argmin(d[0])
            temp.append(self.sites[index])
        return np.array(temp)
    
    def fill(self):
        point = self.c_sites
        p = self.arr[point[0][1]][point[0][0]]
        for i, j in product(range(self.corners[0][0], self.corners[1][0]+1),
                    range(self.corners[0][1], self.corners[2][1]+1)):
            self.arr[j][i] = p
        return

    def same_sites(self):
        t = [list(a) for a in self.c_sites]
        t.sort()
        t = list(t for t,_ in itertools.groupby(t))
        if (len(t) == 1):
            return True
        return False

    def subdivide(self):
        c1x = self.corners[0][0]
        c1y = self.corners[0][1]
        c2x = self.corners[1][0]
        c2y = self.corners[1][1]
        c3x = self.corners[2][0]
        c3y = self.corners[2][1]
        c4x = self.corners[3][0]
        c4y = self.corners[3][1]
        
        d = self.depth + 1

        self.ul = Node(self.arr, 
                        self.sites, 
                        [[c1x, c1y], 
                        [(c2x - c1x) // 2 + c1x, c1y], 
                        [c1x, (c3y - c1y) // 2 + c1y], 
                        [(c2x - c1x) // 2 + c1x, (c3y - c1y) // 2 + c1y]], d,
                        self.node * 4 - 3)

        self.ur = Node(self.arr, 
                        self.sites, 
                        [[int(math.ceil((c2x - c1x) / 2) + c1x), c2y], 
                        [c2x, c2y], 
                        [int(math.ceil((c2x - c1x) / 2) + c1x), (c4y - c2y) // 2 + c2y], 
                        [c2x, (c4y - c2y) // 2 + c2y]], d, self.node * 4 - 2)

        self.ll = Node(self.arr, 
                        self.sites, 
                        [[c3x, int(math.ceil((c3y - c1y) / 2 + c1y))], 
                        [(c4x - c3x) // 2 + c3x, int(math.ceil((c3y - c1y) / 2 + c1y))], 
                        [c3x, c3y],
                        [(c4x - c3x) // 2 + c3x, c3y]], d, self.node * 4 - 1)

        self.lr = Node(self.arr, 
                        self.sites, 
                        [[int(math.ceil((c4x - c3x) / 2 + c3x)), int(math.ceil((c4y - c2y) / 2 + c2y))], 
                        [c4x, int(math.ceil((c4y - c2y) / 2 + c2y))], 
                        [int(math.ceil((c4x - c3x) / 2 + c3x)), c4y], 
                        [c4x, c4y]], d, self.node * 4)   
        return

    def div_conq(self):
        if (self.same_sites()):
            self.fill()
            return
        self.subdivide()
        self.ul.div_conq()
        self.ur.div_conq()
        self.ll.div_conq()
        self.lr.div_conq()



'''
    adjust normals according to which SH we want to use
'''
import numpy as np
import sys
from utils_shtools import *
from pyshtools.rotate import djpi2, SHRotateRealCoef

class sh_cvt():
    '''
        the normal direction we get from projection is:

                    > z
                |  /
                | /
                |/            
        --------------------------> x
                |
                |
                v y

         the x, y, z direction of SH from SHtools is
                ^ z  > y
                |  /
                | /
                |/            
        --------------------------> x
                |
                |

        the bip lighting coordinate is
                    > z
                |  /
                | /
                |/            
        <--------------------------
        x       |
                |
                v y

        the sfs lighting coordinate is
                | 
                |            
        --------------------------> y
              / |
             /  |
          z /   v x
    '''
    def __init__(self):
        self.SH_DEGREE = 2
        self.dj = djpi2(self.SH_DEGREE)


    def cvt2shtools(self, normalImages):
        '''
            align coordinates of normal with shtools
        '''
        newNormals = normalImages.copy()
        # new y is the old z 
        newNormals[:,:,1] =  normalImages[:,:,2]
        # new z is the negative old y
        newNormals[:,:,2] = -1*normalImages[:,:,1]
        return newNormals

    def bip2shtools(self, lighting):
        '''
            lighting is n x 9 matrix of bip lighting, we want to convert it 
            to the coordinate of shtools so we can use the same coordinate
            --we use shtools to rotate the coordinate:
            we use shtools to rotate the object:
            we need to use x convention, 
            alpha_x = -pi (contour clock-wise rotate along z by pi)
            beta_x = -pi/2 (contour clock-wise rotate along new x by pi/2)
            gamma_x = 0
            then y convention is:
            alpha_y = alpha_x - pi/2 = 0
            beta_y = beta_x = -pi/2
            gamma_y = gamma_x + pi/2 = pi/2
            reference: https://shtools.oca.eu/shtools/pyshrotaterealcoef.html
        '''
        new_lighting = np.zeros(lighting.shape)
        n = lighting.shape[0]
        for i in range(n):
            shMatrix = shtools_sh2matrix(lighting[i,:], self.SH_DEGREE)
            # rotate coordinate
            shMatrix = SHRotateRealCoef(shMatrix, np.array([0, -np.pi/2, np.pi/2]), self.dj)
            # rotate object
            #shMatrix = SHRotateRealCoef(shMatrix, np.array([-np.pi/2, np.pi/2, -np.pi/2]), self.dj)
            new_lighting[i,:] = shtools_matrix2vec(shMatrix)
        return new_lighting

    def sfs2shtools(self, lighting):
        '''
            convert sfs SH to shtools
            --we use shtools to rotate the coordinate:
            we use shtools to rotate the object:

            we need to use x convention, 
            we use shtools to rotate the coordinate:
            we need to use x convention, 
            alpha_x = pi/2 (clock-wise rotate along z axis by pi/2)
            beta_x = -pi/2 (contour clock-wise rotate along new x by pi/2)
            gamma_x = 0
            then y convention is:
            alpha_y = alpha_x - pi/2 = 0
            beta_y = beta_x = -pi/2
            gamma_y = gamma_x + pi/2 = pi/2
            reference: https://shtools.oca.eu/shtools/pyshrotaterealcoef.html
        '''
        new_lighting = np.zeros(lighting.shape)
        n = lighting.shape[0]
        for i in range(n):
            shMatrix = shtools_sh2matrix(lighting[i,:], self.SH_DEGREE)
            # rotate coordinate
            shMatrix = SHRotateRealCoef(shMatrix, np.array([0, -np.pi/2, np.pi/2]), self.dj)
            # rotate object
            #shMatrix = SHRotateRealCoef(shMatrix, np.array([np.pi/2, -np.pi/2, 0]), self.dj)
            new_lighting[i,:] = shtools_matrix2vec(shMatrix)
        return new_lighting

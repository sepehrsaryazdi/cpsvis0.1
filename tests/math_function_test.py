import unittest
import numpy as np
from helper_functions.add_new_triangle_functions import *

class TestAddNewTriangle(unittest.TestCase):
    def setUp(self):
        self.t = 1
        self.e01 = 1
        self.e10 = 1
        self.e02 = 1
        self.e20 = 1
        self.e12 = 1
        self.e21 = 1
        self.e03 = 6.1
        self.e30 = 3.0
        self.e23 = 7.8
        self.e32 = 1.7
        self.A023 = 1.3
        self.c0 = [1,0,0]
        self.c1 = [0,self.t,0]
        self.c2 = [0,0, 1]
        self.r0 = [0, self.e01/self.t, self.e02]
        self.r1 = [self.e10, 0, self.e12]
        self.r2 = [self.e20, self.e21/self.t, 0]


    def test_compute_m_inverse(self):

        #print(self.r0, self.r2, self.c0, self.c2,self.e03, self.e23)

        r2_cross_r0 = [[-1],[1],[-1]]
        #print(np.cross(self.r0,self.r2))
        A = [[10/61,0,0],[0,5/39,0],[0,0,50/2379]]
        c0 = [[1],[0],[0]]
        c2 = [[0],[0],[1]]
        B = [[0,1,-1],[0,0,1],[1,0,-1]]
        AB = [[0,10/61, -10/61], [0,0,5/39],[50/2379,0,-50/2379]]


        m_inverse = compute_m_inverse(self.r0, self.r2, self.c0, self.c2,self.e03, self.e23)

        self.assertEqual(np.allclose(m_inverse, np.array(AB)),True, 'Incorrect M inverse')


    def test_compute_c3(self):
        #print(self.e03, self.e23, self.A023)

        B = [[6.1], [7.8], [1.3]]
        m_inverse = [[0,10/61, -10/61], [0,0,5/39],[50/2379,0,-50/2379]]
        m_inverseB = [1.06557,0.166667,0.10088]

        m_inverse = compute_m_inverse(self.r0, self.r2, self.c0, self.c2, self.e03, self.e23)
        c3 = compute_c3(m_inverse, self.e03, self.e23, self.A023)
        self.assertEqual(np.allclose(m_inverseB,c3,rtol=1.e-4),True,'Incorrect C3')

    def test_compute_r3(self):
        m_inverse = compute_m_inverse(self.r0, self.r2, self.c0, self.c2, self.e03, self.e23)
        self.c3 = compute_c3(m_inverse, self.e03, self.e23, self.A023)
        r3_actual = [3,-20.20926344,1.7]
        r3 = compute_r3(self.c0, self.c2, self.c3, self.e30, self.e32)
        self.assertEqual(np.allclose(r3_actual,r3),True,'Incorrect r3')
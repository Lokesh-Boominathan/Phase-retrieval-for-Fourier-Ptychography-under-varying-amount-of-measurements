from __future__ import division

import imageio
import skimage.measure as ms

import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
import cv2
from ops import *
from utils import *
import scipy.misc as io
import matplotlib.pyplot as plt

class pix2pix(object):
    def __init__(self, sess, x_max, x_min, image_size=64,
                 batch_size=1, sample_size=1, output_size=256,
                 gf_dim=64, df_dim=64, L1_lambda=100,
                 input_c_dim=81, output_c_dim=2, dataset_name='25',
                 checkpoint_dir=None, sample_dir=None,test_image=1,CGAN_FP=1):
 
        self.sess = sess
        self.is_grayscale = (input_c_dim == 1)
        self.batch_size = batch_size
        self.image_size = image_size
        self.sample_size = sample_size
        self.output_size = output_size
        self.x_max = x_max
        self.x_min = x_min
        self.gf_dim = gf_dim
        self.df_dim = df_dim
	self.test_image=test_image
        self.input_c_dim = input_c_dim
        self.output_c_dim = output_c_dim
	self.CGAN_FP=CGAN_FP
        self.L1_lambda = L1_lambda
        self.fake_B_size = 256
        self.xmax = x_max[40]

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn_e2 = batch_norm(name='g_bn_e2')
        self.g_bn_e3 = batch_norm(name='g_bn_e3')
        self.g_bn_e4 = batch_norm(name='g_bn_e4')
        self.g_bn_e5 = batch_norm(name='g_bn_e5')
        self.g_bn_e6 = batch_norm(name='g_bn_e6')
        self.g_bn_e7 = batch_norm(name='g_bn_e7')
        self.g_bn_e8 = batch_norm(name='g_bn_e8')

        self.g_bn_d1 = batch_norm(name='g_bn_d1')
        self.g_bn_d2 = batch_norm(name='g_bn_d2')
        self.g_bn_d3 = batch_norm(name='g_bn_d3')
        self.g_bn_d4 = batch_norm(name='g_bn_d4')
        self.g_bn_d5 = batch_norm(name='g_bn_d5')
        self.g_bn_d6 = batch_norm(name='g_bn_d6')
        self.g_bn_d7 = batch_norm(name='g_bn_d7')

        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.build_model()

    def build_model(self):
        self.real_A = tf.placeholder(tf.float32,
                                        [self.batch_size, self.image_size, self.image_size,
                                         self.input_c_dim],
                                        name='real_A_images')
        # self.real_A = tf.image.resize_images(self.real_A, [256, 256])
        
        self.real_B = tf.placeholder(tf.float32,
                                        [self.batch_size, self.output_size, self.output_size,
                                         self.output_c_dim],
                                        name='real_B_images')
        #pass input through generator
        self.fake_B = self.generator(self.real_A)
	#max and min of input, channelwise, This is used in unscaled forward model loss
        self.x_max = tf.constant(self.x_max,dtype = tf.float32)
        self.x_min = tf.constant(self.x_min,dtype = tf.float32)
	#scaling generated output intensity and phase for forward model loss
        self.scaled_fake_B1 = tf.scalar_mul(255,self.fake_B[:,:,:,0]);
        self.scaled_fake_B2 = tf.scalar_mul(2*np.pi,self.fake_B[:,:,:,1]);
	#Making complex object input
        self.real = tf.multiply(tf.cos(self.scaled_fake_B2) , tf.sqrt(self.scaled_fake_B1))
        self.imag = tf.multiply(tf.sin(self.scaled_fake_B2) , tf.sqrt(self.scaled_fake_B1))

	#Fourier transform and shift applied on input complex object
        self.z = tf.complex(self.real,self.imag)
        self.z = tf.cast(self.z,tf.complex64)
        self.y = tf.fft2d(self.z)
        self.real1 = tf.real(self.y)
        self.imag1 = tf.imag(self.y)

        self.tempA = tf.concat([self.real1,self.real1],1)
        self.tempB = tf.concat([self.tempA, self.tempA],2)
        self.real2 = tf.slice(self.tempB,[0,128,128],[-1,256,256])

        self.tempA1 = tf.concat([self.imag1,self.imag1],1)
        self.tempB1 = tf.concat([self.tempA1, self.tempA1],2)
        self.imag2 = tf.slice(self.tempB1, [0,128,128],[-1,256,256])

        self.objectFT = tf.complex(self.real2,self.imag2)
	#sampling from fourier transform of complex object using pupil function and led locations
        self.imlowres1 = self.cap_int(self.objectFT, self.input_c_dim)
        self.imlowres = tf.cast(self.imlowres1,tf.float32)
	#unscaling the low resolution measurements using x_max and x_min
	
	self.realA0 = tf.add(tf.multiply(self.real_A[:,:,:,0],tf.subtract(self.x_max[0],self.x_min[0])),self.x_min[0])
        self.realA1 = tf.add(tf.multiply(self.real_A[:,:,:,1],tf.subtract(self.x_max[1],self.x_min[1])),self.x_min[1])
        self.realA2 = tf.add(tf.multiply(self.real_A[:,:,:,2],tf.subtract(self.x_max[2],self.x_min[2])),self.x_min[2])
	self.realA3 = tf.add(tf.multiply(self.real_A[:,:,:,3],tf.subtract(self.x_max[3],self.x_min[3])),self.x_min[3])
	self.realA4 = tf.add(tf.multiply(self.real_A[:,:,:,4],tf.subtract(self.x_max[4],self.x_min[4])),self.x_min[4])
	self.realA5 = tf.add(tf.multiply(self.real_A[:,:,:,5],tf.subtract(self.x_max[5],self.x_min[5])),self.x_min[5])

	self.realA6 = tf.add(tf.multiply(self.real_A[:,:,:,6],tf.subtract(self.x_max[6],self.x_min[6])),self.x_min[6])
        self.realA7 = tf.add(tf.multiply(self.real_A[:,:,:,7],tf.subtract(self.x_max[7],self.x_min[7])),self.x_min[7])
        self.realA8 = tf.add(tf.multiply(self.real_A[:,:,:,8],tf.subtract(self.x_max[8],self.x_min[8])),self.x_min[8])
	self.realA9 = tf.add(tf.multiply(self.real_A[:,:,:,9],tf.subtract(self.x_max[9],self.x_min[9])),self.x_min[9])
	self.realA10 = tf.add(tf.multiply(self.real_A[:,:,:,10],tf.subtract(self.x_max[10],self.x_min[10])),self.x_min[10])
	

        self.realA11 = tf.add(tf.multiply(self.real_A[:,:,:,11],tf.subtract(self.x_max[11],self.x_min[11])),self.x_min[11])
        self.realA12 = tf.add(tf.multiply(self.real_A[:,:,:,12],tf.subtract(self.x_max[12],self.x_min[12])),self.x_min[12])
	self.realA13 = tf.add(tf.multiply(self.real_A[:,:,:,13],tf.subtract(self.x_max[13],self.x_min[13])),self.x_min[13])
	self.realA14 = tf.add(tf.multiply(self.real_A[:,:,:,14],tf.subtract(self.x_max[14],self.x_min[14])),self.x_min[14])
	self.realA15 = tf.add(tf.multiply(self.real_A[:,:,:,15],tf.subtract(self.x_max[15],self.x_min[15])),self.x_min[15])

	self.realA16 = tf.add(tf.multiply(self.real_A[:,:,:,16],tf.subtract(self.x_max[16],self.x_min[16])),self.x_min[16])
        self.realA17 = tf.add(tf.multiply(self.real_A[:,:,:,17],tf.subtract(self.x_max[17],self.x_min[17])),self.x_min[17])
        self.realA18 = tf.add(tf.multiply(self.real_A[:,:,:,18],tf.subtract(self.x_max[18],self.x_min[18])),self.x_min[18])
	self.realA19 = tf.add(tf.multiply(self.real_A[:,:,:,19],tf.subtract(self.x_max[19],self.x_min[19])),self.x_min[19])
	self.realA20 = tf.add(tf.multiply(self.real_A[:,:,:,20],tf.subtract(self.x_max[20],self.x_min[20])),self.x_min[20])


        self.realA21 = tf.add(tf.multiply(self.real_A[:,:,:,21],tf.subtract(self.x_max[21],self.x_min[21])),self.x_min[21])
        self.realA22 = tf.add(tf.multiply(self.real_A[:,:,:,22],tf.subtract(self.x_max[22],self.x_min[22])),self.x_min[22])
	self.realA23 = tf.add(tf.multiply(self.real_A[:,:,:,23],tf.subtract(self.x_max[23],self.x_min[23])),self.x_min[23])
	self.realA24 = tf.add(tf.multiply(self.real_A[:,:,:,24],tf.subtract(self.x_max[24],self.x_min[24])),self.x_min[24])
	self.realA25 = tf.add(tf.multiply(self.real_A[:,:,:,25],tf.subtract(self.x_max[25],self.x_min[25])),self.x_min[25])

	self.realA26 = tf.add(tf.multiply(self.real_A[:,:,:,26],tf.subtract(self.x_max[26],self.x_min[26])),self.x_min[26])
        self.realA27 = tf.add(tf.multiply(self.real_A[:,:,:,27],tf.subtract(self.x_max[27],self.x_min[27])),self.x_min[27])
        self.realA28 = tf.add(tf.multiply(self.real_A[:,:,:,28],tf.subtract(self.x_max[28],self.x_min[28])),self.x_min[28])
	self.realA29 = tf.add(tf.multiply(self.real_A[:,:,:,29],tf.subtract(self.x_max[29],self.x_min[29])),self.x_min[29])
	self.realA30 = tf.add(tf.multiply(self.real_A[:,:,:,30],tf.subtract(self.x_max[30],self.x_min[30])),self.x_min[30])


        self.realA31 = tf.add(tf.multiply(self.real_A[:,:,:,31],tf.subtract(self.x_max[31],self.x_min[31])),self.x_min[31])
        self.realA32 = tf.add(tf.multiply(self.real_A[:,:,:,32],tf.subtract(self.x_max[32],self.x_min[32])),self.x_min[32])
	self.realA33 = tf.add(tf.multiply(self.real_A[:,:,:,33],tf.subtract(self.x_max[33],self.x_min[33])),self.x_min[33])
	self.realA34 = tf.add(tf.multiply(self.real_A[:,:,:,34],tf.subtract(self.x_max[34],self.x_min[34])),self.x_min[34])
	self.realA35 = tf.add(tf.multiply(self.real_A[:,:,:,35],tf.subtract(self.x_max[35],self.x_min[35])),self.x_min[35])

	self.realA36 = tf.add(tf.multiply(self.real_A[:,:,:,36],tf.subtract(self.x_max[36],self.x_min[36])),self.x_min[36])
        self.realA37 = tf.add(tf.multiply(self.real_A[:,:,:,37],tf.subtract(self.x_max[37],self.x_min[37])),self.x_min[37])
        self.realA38 = tf.add(tf.multiply(self.real_A[:,:,:,38],tf.subtract(self.x_max[38],self.x_min[38])),self.x_min[38])
	self.realA39 = tf.add(tf.multiply(self.real_A[:,:,:,39],tf.subtract(self.x_max[39],self.x_min[39])),self.x_min[39])
	self.realA40 = tf.add(tf.multiply(self.real_A[:,:,:,40],tf.subtract(self.x_max[40],self.x_min[40])),self.x_min[40])


        self.realA41 = tf.add(tf.multiply(self.real_A[:,:,:,41],tf.subtract(self.x_max[41],self.x_min[41])),self.x_min[41])
        self.realA42 = tf.add(tf.multiply(self.real_A[:,:,:,42],tf.subtract(self.x_max[42],self.x_min[42])),self.x_min[42])
	self.realA43 = tf.add(tf.multiply(self.real_A[:,:,:,43],tf.subtract(self.x_max[43],self.x_min[43])),self.x_min[43])
	self.realA44 = tf.add(tf.multiply(self.real_A[:,:,:,44],tf.subtract(self.x_max[44],self.x_min[44])),self.x_min[44])
	self.realA45 = tf.add(tf.multiply(self.real_A[:,:,:,45],tf.subtract(self.x_max[45],self.x_min[45])),self.x_min[45])

	self.realA46 = tf.add(tf.multiply(self.real_A[:,:,:,46],tf.subtract(self.x_max[46],self.x_min[46])),self.x_min[46])
        self.realA47 = tf.add(tf.multiply(self.real_A[:,:,:,47],tf.subtract(self.x_max[47],self.x_min[47])),self.x_min[47])
        self.realA48 = tf.add(tf.multiply(self.real_A[:,:,:,48],tf.subtract(self.x_max[48],self.x_min[48])),self.x_min[48])
	self.realA49 = tf.add(tf.multiply(self.real_A[:,:,:,49],tf.subtract(self.x_max[49],self.x_min[49])),self.x_min[49])
	


	self.realA50 = tf.add(tf.multiply(self.real_A[:,:,:,50],tf.subtract(self.x_max[50],self.x_min[50])),self.x_min[50])
	self.realA51 = tf.add(tf.multiply(self.real_A[:,:,:,51],tf.subtract(self.x_max[51],self.x_min[51])),self.x_min[51])
        self.realA52 = tf.add(tf.multiply(self.real_A[:,:,:,52],tf.subtract(self.x_max[52],self.x_min[52])),self.x_min[52])
	self.realA53 = tf.add(tf.multiply(self.real_A[:,:,:,53],tf.subtract(self.x_max[53],self.x_min[53])),self.x_min[53])
	self.realA54 = tf.add(tf.multiply(self.real_A[:,:,:,54],tf.subtract(self.x_max[54],self.x_min[54])),self.x_min[54])
	self.realA55 = tf.add(tf.multiply(self.real_A[:,:,:,55],tf.subtract(self.x_max[55],self.x_min[55])),self.x_min[55])

	self.realA56 = tf.add(tf.multiply(self.real_A[:,:,:,56],tf.subtract(self.x_max[56],self.x_min[56])),self.x_min[56])
        self.realA57 = tf.add(tf.multiply(self.real_A[:,:,:,57],tf.subtract(self.x_max[57],self.x_min[57])),self.x_min[57])
        self.realA58 = tf.add(tf.multiply(self.real_A[:,:,:,58],tf.subtract(self.x_max[58],self.x_min[58])),self.x_min[58])
	self.realA59 = tf.add(tf.multiply(self.real_A[:,:,:,59],tf.subtract(self.x_max[59],self.x_min[59])),self.x_min[59])
	
	self.realA60 = tf.add(tf.multiply(self.real_A[:,:,:,60],tf.subtract(self.x_max[60],self.x_min[60])),self.x_min[60])
	self.realA61 = tf.add(tf.multiply(self.real_A[:,:,:,61],tf.subtract(self.x_max[61],self.x_min[61])),self.x_min[61])
        self.realA62 = tf.add(tf.multiply(self.real_A[:,:,:,62],tf.subtract(self.x_max[62],self.x_min[62])),self.x_min[62])
	self.realA63 = tf.add(tf.multiply(self.real_A[:,:,:,63],tf.subtract(self.x_max[63],self.x_min[63])),self.x_min[63])
	self.realA64 = tf.add(tf.multiply(self.real_A[:,:,:,64],tf.subtract(self.x_max[64],self.x_min[64])),self.x_min[64])
	self.realA65 = tf.add(tf.multiply(self.real_A[:,:,:,65],tf.subtract(self.x_max[65],self.x_min[65])),self.x_min[65])

	self.realA66 = tf.add(tf.multiply(self.real_A[:,:,:,66],tf.subtract(self.x_max[66],self.x_min[66])),self.x_min[66])
        self.realA67 = tf.add(tf.multiply(self.real_A[:,:,:,67],tf.subtract(self.x_max[67],self.x_min[67])),self.x_min[67])
        self.realA68 = tf.add(tf.multiply(self.real_A[:,:,:,68],tf.subtract(self.x_max[68],self.x_min[68])),self.x_min[68])
	self.realA69 = tf.add(tf.multiply(self.real_A[:,:,:,69],tf.subtract(self.x_max[69],self.x_min[69])),self.x_min[69])
	

	self.realA70 = tf.add(tf.multiply(self.real_A[:,:,:,70],tf.subtract(self.x_max[70],self.x_min[70])),self.x_min[70])
	self.realA71 = tf.add(tf.multiply(self.real_A[:,:,:,71],tf.subtract(self.x_max[71],self.x_min[71])),self.x_min[71])
        self.realA72 = tf.add(tf.multiply(self.real_A[:,:,:,72],tf.subtract(self.x_max[72],self.x_min[72])),self.x_min[72])
	self.realA73 = tf.add(tf.multiply(self.real_A[:,:,:,73],tf.subtract(self.x_max[73],self.x_min[73])),self.x_min[73])
	self.realA74 = tf.add(tf.multiply(self.real_A[:,:,:,74],tf.subtract(self.x_max[74],self.x_min[74])),self.x_min[74])
	self.realA75 = tf.add(tf.multiply(self.real_A[:,:,:,75],tf.subtract(self.x_max[75],self.x_min[75])),self.x_min[75])

	self.realA76 = tf.add(tf.multiply(self.real_A[:,:,:,76],tf.subtract(self.x_max[76],self.x_min[76])),self.x_min[76])
        self.realA77 = tf.add(tf.multiply(self.real_A[:,:,:,77],tf.subtract(self.x_max[77],self.x_min[77])),self.x_min[77])
        self.realA78 = tf.add(tf.multiply(self.real_A[:,:,:,78],tf.subtract(self.x_max[78],self.x_min[78])),self.x_min[78])
	self.realA79 = tf.add(tf.multiply(self.real_A[:,:,:,79],tf.subtract(self.x_max[79],self.x_min[79])),self.x_min[79])
	
	self.realA80 = tf.add(tf.multiply(self.real_A[:,:,:,80],tf.subtract(self.x_max[80],self.x_min[80])),self.x_min[80])	

	#channelwise l2 loss between generated low res measurements and actual low res measurements 	
	self.fft_loss0 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,0], self.realA0))
        self.fft_loss1 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,1], self.realA1))
        self.fft_loss2 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,2], self.realA2))
        self.fft_loss3 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,3], self.realA3))
        self.fft_loss4 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,4], self.realA4))
        self.fft_loss5 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,5], self.realA5))
        self.fft_loss6 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,6], self.realA6))
        self.fft_loss7 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,7], self.realA7))
        self.fft_loss8 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,8], self.realA8))
        self.fft_loss9 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,9], self.realA9))
        self.fft_loss10 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,10], self.realA10))
        self.fft_loss11 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,11], self.realA11))
        self.fft_loss12 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,12], self.realA12))
        self.fft_loss13 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,13], self.realA13))
        self.fft_loss14 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,14], self.realA14))
        self.fft_loss15 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,15], self.realA15))
        self.fft_loss16 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,16], self.realA16))
        self.fft_loss17 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,17], self.realA17))
        self.fft_loss18 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,18], self.realA18))
        self.fft_loss19 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,19], self.realA19))

        self.fft_loss20 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,20], self.realA20))
        self.fft_loss21 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,21], self.realA21))
        self.fft_loss22 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,22], self.realA22))
        self.fft_loss23 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,23], self.realA23))
        self.fft_loss24 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,24], self.realA24))
        self.fft_loss25 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,25], self.realA25))
        self.fft_loss26 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,26], self.realA26))
        self.fft_loss27 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,27], self.realA27))
        self.fft_loss28 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,28], self.realA28))
        self.fft_loss29 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,29], self.realA29))

        self.fft_loss30 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,30], self.realA30))
        self.fft_loss31 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,31], self.realA31))
        self.fft_loss32 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,32], self.realA32))
        self.fft_loss33 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,33], self.realA33))
        self.fft_loss34 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,34], self.realA34))
        self.fft_loss35 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,35], self.realA35))
        self.fft_loss36 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,36], self.realA36))
        self.fft_loss37 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,37], self.realA37))
        self.fft_loss38 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,38], self.realA38))
        self.fft_loss39 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,39], self.realA39))

        self.fft_loss40 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,40], self.realA40))
        self.fft_loss41 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,41], self.realA41))
        self.fft_loss42 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,42], self.realA42))
        self.fft_loss43 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,43], self.realA43))
        self.fft_loss44 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,44], self.realA44))
        self.fft_loss45 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,45], self.realA45))
        self.fft_loss46 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,46], self.realA46))
        self.fft_loss47 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,47], self.realA47))
        self.fft_loss48 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,48], self.realA48))
	self.fft_loss49 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,49], self.realA49))
        
        self.fft_loss50 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,50], self.realA50))
        self.fft_loss51 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,51], self.realA51))
        self.fft_loss52 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,52], self.realA52))
        self.fft_loss53 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,53], self.realA53))
        self.fft_loss54 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,54], self.realA54))
        self.fft_loss55 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,55], self.realA55))
        self.fft_loss56 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,56], self.realA56))
        self.fft_loss57 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,57], self.realA57))
        self.fft_loss58 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,58], self.realA58))
        self.fft_loss59 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,59], self.realA59))

        self.fft_loss60 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,60], self.realA60))
        self.fft_loss61 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,61], self.realA61))
        self.fft_loss62 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,62], self.realA62))
        self.fft_loss63 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,63], self.realA63))
        self.fft_loss64 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,64], self.realA64))
        self.fft_loss65 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,65], self.realA65))
        self.fft_loss66 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,66], self.realA66))
        self.fft_loss67 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,67], self.realA67))
        self.fft_loss68 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,68], self.realA68))
        self.fft_loss69 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,69], self.realA69))

        self.fft_loss70 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,70], self.realA70))
        self.fft_loss71 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,71], self.realA71))
        self.fft_loss72 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,72], self.realA72))
        self.fft_loss73 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,73], self.realA73))
        self.fft_loss74 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,74], self.realA74))
        self.fft_loss75 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,75], self.realA75))
        self.fft_loss76 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,76], self.realA76))
        self.fft_loss77 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,77], self.realA77))
        self.fft_loss78 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,78], self.realA78))
        self.fft_loss79 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,79], self.realA79))

        self.fft_loss80 = tf.reduce_mean(tf.losses.mean_squared_error(self.imlowres[:,:,:,80], self.realA80))

 



	


        self.fft_loss0_sum = tf.summary.scalar("f_loss0", tf.reduce_mean(self.fft_loss0))
        self.fft_loss1_sum = tf.summary.scalar("f_loss1", tf.reduce_mean(self.fft_loss1))
        self.fft_loss2_sum = tf.summary.scalar("f_loss2", tf.reduce_mean(self.fft_loss2))
        self.fft_loss3_sum = tf.summary.scalar("f_loss3", tf.reduce_mean(self.fft_loss3))
        self.fft_loss4_sum = tf.summary.scalar("f_loss4", tf.reduce_mean(self.fft_loss4))
        self.fft_loss5_sum = tf.summary.scalar("f_loss5", tf.reduce_mean(self.fft_loss5))
        self.fft_loss6_sum = tf.summary.scalar("f_loss6", tf.reduce_mean(self.fft_loss6))
        self.fft_loss7_sum = tf.summary.scalar("f_loss7", tf.reduce_mean(self.fft_loss7))
        self.fft_loss8_sum = tf.summary.scalar("f_loss8", tf.reduce_mean(self.fft_loss8))
        self.fft_loss9_sum = tf.summary.scalar("f_loss9", tf.reduce_mean(self.fft_loss9))
        self.fft_loss10_sum = tf.summary.scalar("f_loss10", tf.reduce_mean(self.fft_loss10))
        self.fft_loss11_sum = tf.summary.scalar("f_loss11", tf.reduce_mean(self.fft_loss11))
        self.fft_loss12_sum = tf.summary.scalar("f_loss12", tf.reduce_mean(self.fft_loss12))
        self.fft_loss13_sum = tf.summary.scalar("f_loss13", tf.reduce_mean(self.fft_loss13))
        self.fft_loss14_sum = tf.summary.scalar("f_loss14", tf.reduce_mean(self.fft_loss14))
        self.fft_loss15_sum = tf.summary.scalar("f_loss15", tf.reduce_mean(self.fft_loss15))
        self.fft_loss16_sum = tf.summary.scalar("f_loss16", tf.reduce_mean(self.fft_loss16))
        self.fft_loss17_sum = tf.summary.scalar("f_loss17", tf.reduce_mean(self.fft_loss17))
        self.fft_loss18_sum = tf.summary.scalar("f_loss18", tf.reduce_mean(self.fft_loss18))
        self.fft_loss19_sum = tf.summary.scalar("f_loss19", tf.reduce_mean(self.fft_loss19))
        self.fft_loss20_sum = tf.summary.scalar("f_loss20", tf.reduce_mean(self.fft_loss20))
        self.fft_loss21_sum = tf.summary.scalar("f_loss21", tf.reduce_mean(self.fft_loss21))
        self.fft_loss22_sum = tf.summary.scalar("f_loss22", tf.reduce_mean(self.fft_loss22))
        self.fft_loss23_sum = tf.summary.scalar("f_loss23", tf.reduce_mean(self.fft_loss23))
        self.fft_loss24_sum = tf.summary.scalar("f_loss24", tf.reduce_mean(self.fft_loss24))

        self.fft_loss30_sum = tf.summary.scalar("f_loss30", tf.reduce_mean(self.fft_loss30))
        self.fft_loss31_sum = tf.summary.scalar("f_loss31", tf.reduce_mean(self.fft_loss31))
        self.fft_loss32_sum = tf.summary.scalar("f_loss32", tf.reduce_mean(self.fft_loss32))
        self.fft_loss33_sum = tf.summary.scalar("f_loss33", tf.reduce_mean(self.fft_loss33))
        self.fft_loss34_sum = tf.summary.scalar("f_loss34", tf.reduce_mean(self.fft_loss34))
        self.fft_loss35_sum = tf.summary.scalar("f_loss35", tf.reduce_mean(self.fft_loss35))
        self.fft_loss36_sum = tf.summary.scalar("f_loss36", tf.reduce_mean(self.fft_loss36))
        self.fft_loss37_sum = tf.summary.scalar("f_loss37", tf.reduce_mean(self.fft_loss37))
        self.fft_loss38_sum = tf.summary.scalar("f_loss38", tf.reduce_mean(self.fft_loss38))
        self.fft_loss39_sum = tf.summary.scalar("f_loss39", tf.reduce_mean(self.fft_loss39))
        
        self.fft_loss40_sum = tf.summary.scalar("f_loss40", tf.reduce_mean(self.fft_loss40))
        self.fft_loss41_sum = tf.summary.scalar("f_loss41", tf.reduce_mean(self.fft_loss41))
        self.fft_loss42_sum = tf.summary.scalar("f_loss42", tf.reduce_mean(self.fft_loss42))
        self.fft_loss43_sum = tf.summary.scalar("f_loss43", tf.reduce_mean(self.fft_loss43))
        self.fft_loss44_sum = tf.summary.scalar("f_loss44", tf.reduce_mean(self.fft_loss44))
        self.fft_loss45_sum = tf.summary.scalar("f_loss45", tf.reduce_mean(self.fft_loss45))
        self.fft_loss46_sum = tf.summary.scalar("f_loss46", tf.reduce_mean(self.fft_loss46))
        self.fft_loss47_sum = tf.summary.scalar("f_loss47", tf.reduce_mean(self.fft_loss47))
        self.fft_loss48_sum = tf.summary.scalar("f_loss48", tf.reduce_mean(self.fft_loss48))

        


	#summing losses of all channels


        self.fft_loss = self.fft_loss0
        self.fft_loss = tf.add(self.fft_loss,self.fft_loss1)
        self.fft_loss = tf.add(self.fft_loss,self.fft_loss2)
        self.fft_loss = tf.add(self.fft_loss,self.fft_loss3)
        self.fft_loss = tf.add(self.fft_loss,self.fft_loss4)
        self.fft_loss = tf.add(self.fft_loss,self.fft_loss5)
        self.fft_loss = tf.add(self.fft_loss,self.fft_loss6)
        self.fft_loss = tf.add(self.fft_loss,self.fft_loss7)
        self.fft_loss = tf.add(self.fft_loss,self.fft_loss8)
        self.fft_loss = tf.add(self.fft_loss,self.fft_loss9)

        self.fft_loss = tf.add(self.fft_loss,self.fft_loss10)
        self.fft_loss = tf.add(self.fft_loss,self.fft_loss11)
        self.fft_loss = tf.add(self.fft_loss,self.fft_loss12)
        self.fft_loss = tf.add(self.fft_loss,self.fft_loss13)
        self.fft_loss = tf.add(self.fft_loss,self.fft_loss14)
        self.fft_loss = tf.add(self.fft_loss,self.fft_loss15)
        self.fft_loss = tf.add(self.fft_loss,self.fft_loss16)
        self.fft_loss = tf.add(self.fft_loss,self.fft_loss17)
        self.fft_loss = tf.add(self.fft_loss,self.fft_loss18)
        self.fft_loss = tf.add(self.fft_loss,self.fft_loss19)

        self.fft_loss = tf.add(self.fft_loss,self.fft_loss20)
        self.fft_loss = tf.add(self.fft_loss,self.fft_loss21)
        self.fft_loss = tf.add(self.fft_loss,self.fft_loss22)
        self.fft_loss = tf.add(self.fft_loss,self.fft_loss23)
        self.fft_loss = tf.add(self.fft_loss,self.fft_loss24)
        self.fft_loss = tf.add(self.fft_loss,self.fft_loss25)
        self.fft_loss = tf.add(self.fft_loss,self.fft_loss26)
        self.fft_loss = tf.add(self.fft_loss,self.fft_loss27)
        self.fft_loss = tf.add(self.fft_loss,self.fft_loss28)
        self.fft_loss = tf.add(self.fft_loss,self.fft_loss29)

        self.fft_loss = tf.add(self.fft_loss,self.fft_loss30)
        self.fft_loss = tf.add(self.fft_loss,self.fft_loss31)
        self.fft_loss = tf.add(self.fft_loss,self.fft_loss32)
        self.fft_loss = tf.add(self.fft_loss,self.fft_loss33)
        self.fft_loss = tf.add(self.fft_loss,self.fft_loss34)
        self.fft_loss = tf.add(self.fft_loss,self.fft_loss35)
        self.fft_loss = tf.add(self.fft_loss,self.fft_loss36)
        self.fft_loss = tf.add(self.fft_loss,self.fft_loss37)
        self.fft_loss = tf.add(self.fft_loss,self.fft_loss38)
        self.fft_loss = tf.add(self.fft_loss,self.fft_loss39)

        self.fft_loss = tf.add(self.fft_loss,self.fft_loss40)
        self.fft_loss = tf.add(self.fft_loss,self.fft_loss41)
        self.fft_loss = tf.add(self.fft_loss,self.fft_loss42)
        self.fft_loss = tf.add(self.fft_loss,self.fft_loss43)
        self.fft_loss = tf.add(self.fft_loss,self.fft_loss44)
        self.fft_loss = tf.add(self.fft_loss,self.fft_loss45)
        self.fft_loss = tf.add(self.fft_loss,self.fft_loss46)
        self.fft_loss = tf.add(self.fft_loss,self.fft_loss47)
        self.fft_loss = tf.add(self.fft_loss,self.fft_loss48)
	self.fft_loss = tf.add(self.fft_loss,self.fft_loss49)

        self.fft_loss = tf.add(self.fft_loss,self.fft_loss50)
        self.fft_loss = tf.add(self.fft_loss,self.fft_loss51)
        self.fft_loss = tf.add(self.fft_loss,self.fft_loss52)
        self.fft_loss = tf.add(self.fft_loss,self.fft_loss53)
        self.fft_loss = tf.add(self.fft_loss,self.fft_loss54)
        self.fft_loss = tf.add(self.fft_loss,self.fft_loss55)
        self.fft_loss = tf.add(self.fft_loss,self.fft_loss56)
        self.fft_loss = tf.add(self.fft_loss,self.fft_loss57)
        self.fft_loss = tf.add(self.fft_loss,self.fft_loss58)
        self.fft_loss = tf.add(self.fft_loss,self.fft_loss59)

        self.fft_loss = tf.add(self.fft_loss,self.fft_loss60)
        self.fft_loss = tf.add(self.fft_loss,self.fft_loss61)
        self.fft_loss = tf.add(self.fft_loss,self.fft_loss62)
        self.fft_loss = tf.add(self.fft_loss,self.fft_loss63)
        self.fft_loss = tf.add(self.fft_loss,self.fft_loss64)
        self.fft_loss = tf.add(self.fft_loss,self.fft_loss65)
        self.fft_loss = tf.add(self.fft_loss,self.fft_loss66)
        self.fft_loss = tf.add(self.fft_loss,self.fft_loss67)
        self.fft_loss = tf.add(self.fft_loss,self.fft_loss68)
        self.fft_loss = tf.add(self.fft_loss,self.fft_loss69)

        self.fft_loss = tf.add(self.fft_loss,self.fft_loss70)
        self.fft_loss = tf.add(self.fft_loss,self.fft_loss71)
        self.fft_loss = tf.add(self.fft_loss,self.fft_loss72)
        self.fft_loss = tf.add(self.fft_loss,self.fft_loss73)
        self.fft_loss = tf.add(self.fft_loss,self.fft_loss74)
        self.fft_loss = tf.add(self.fft_loss,self.fft_loss75)
        self.fft_loss = tf.add(self.fft_loss,self.fft_loss76)
        self.fft_loss = tf.add(self.fft_loss,self.fft_loss77)
        self.fft_loss = tf.add(self.fft_loss,self.fft_loss78)
        self.fft_loss = tf.add(self.fft_loss,self.fft_loss79)
        
        self.fft_loss = tf.add(self.fft_loss,self.fft_loss80)




        self.fft_loss = tf.div(self.fft_loss,81)











	#loss with ground truth for plotting

        self.loss_with_gt = tf.losses.mean_squared_error(self.real_B ,self.fake_B)


        self.lossy = 0.1*self.fft_loss







        self.f_loss_sum = tf.summary.scalar("forward model loss", self.fft_loss)
        self.loss_with_gt_sum = tf.summary.scalar("loss with ground truth", tf.reduce_mean(self.loss_with_gt))




        t_vars = tf.trainable_variables()


        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver(max_to_keep = 0)
    #function for sampling from fourier domain
    def cap_int(self,objectFT,input_c_dim):
        self.kxl = tf.constant(np.load('Needed_params/kxl_.npy'),dtype = tf.int32)
        self.kxh = tf.constant(np.load('Needed_params/kxh_.npy'),dtype = tf.int32)
        self.kyl = tf.constant(np.load('Needed_params/kyl_.npy'),dtype = tf.int32)
        self.kyh = tf.constant(np.load('Needed_params/kyh_.npy'),dtype = tf.int32)
        self.CTF_tf = tf.constant(np.load('Needed_params/CTF.npy'),dtype = tf.complex64)
        
        counter = 0;
        # CTF_tf = tf.constant(CTF1,dtype = tf.complex64)

        for tt in range(0,self.input_c_dim):
            self.tt = tf.constant(tt,dtype = tf.int32)
            self.xl = self.kxl[0,tt]
            self.xh = self.kxh[0,tt]
            self.yl = self.kyl[0,tt]
            self.yh = self.kyh[0,tt]
     
            self.beginarr = [0,self.xl-1,self.yl-1]
            self.sizearr = [-1,self.xh-self.xl+1,self.yh-self.yl+1]

            self.imSeqLowFT1 = tf.slice(self.objectFT,self.beginarr,self.sizearr)
            self.imSeqLowFT = tf.multiply(self.imSeqLowFT1,self.CTF_tf)
            
            self.real2 = tf.real(self.imSeqLowFT)
            self.imag2 = tf.imag(self.imSeqLowFT)
            
            self.tempA2 = tf.concat([self.real2,self.real2],1)
            self.tempB2 = tf.concat([self.tempA2, self.tempA2],2)
            self.real12 = tf.slice(self.tempB2, [0,32,32],[-1,64,64])
            
            self.tempA3 = tf.concat([self.imag2,self.imag2],1)
            self.tempB3 = tf.concat([self.tempA3, self.tempA3],2)
            self.imag12 = tf.slice(self.tempB3, [0,32,32],[-1,64,64])
            
            self.z1 = tf.complex(self.real12,self.imag12)
            self.z1 = tf.cast(self.z1,tf.complex64)
            self.res= tf.ifft2d(self.z1)
            self.res = tf.reshape(self.res,[tf.shape(self.res)[0],64,64,1])
            self.res = tf.pow(tf.abs(self.res),2)
            if counter == 0:
                self.imlowres = self.res
            else:
                self.imlowres = tf.concat([self.imlowres,self.res],axis = 3)
                
                
            counter = counter +1;

        return self.imlowres


    def train(self, args):
        """optimisation function"""

        g_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                          .minimize(self.lossy, var_list=self.g_vars)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        self.summary_op = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        counter = 1
        start_time = time.time()
	#restoring CGAN-FP weights 
	#restoring CGAN-FP weights 
	if self.CGAN_FP==1:
		print("Loading CGAN_FP_weights")
	    	checkpoint_dir='./CGAN_checkpoint/pix2pix.model-55502'
		saver=tf.train.Saver(var_list=self.g_vars)
		saver.restore(self.sess,checkpoint_dir)
	else:	
		print(" NOT Loading CGAN_FP_weights")

        # #if self.load(self.checkpoint_dir):
        #  #   print(" [*] Load SUCCESS")
        # else:
        #     print(" [!] Load failed...")

        for epoch in xrange(args.epoch+1):
            data_X = sorted(glob('./Needed_params/X/*.npy'))
            data_y = sorted(glob('./Needed_params/y/*.npy'))

            for idx in range(10):
                batchX = data_X[self.test_image]
                batch_X = np.load(batchX)
                batch_X=np.expand_dims(batch_X,axis=0)
                batchy = data_y[self.test_image]
		print('batchy ',batchy)
                batch_y = np.load(batchy)
                batch_y=np.expand_dims(batch_y,axis=0)


                optim_imag,fft_l=self.sess.run([self.fake_B,self.fft_loss],feed_dict={ self.real_A: batch_X,self.real_B: batch_y })
                optim_imag=np.squeeze(optim_imag)

                print("idx ",idx)
                print("epoch ",epoch)

		print("fft_loss ",(fft_l))
                print(np.shape(optim_imag))
                print(np.shape(batch_y))
                int_psnr=ms.compare_psnr(np.float32(np.squeeze(optim_imag[:,:,0])),np.float32(np.squeeze(batch_y[0,:,:,0])))
                phase_psnr=ms.compare_psnr(np.float32(np.squeeze(optim_imag[:,:,1])),np.float32(np.squeeze(batch_y[0,:,:,1])))
		#saving checkpoints and sample images 
		if np.mod(epoch,100) == 0:
			self.save(args.checkpoint_dir, epoch)
                if np.mod(epoch, 10) == 0:
                    #self.save(args.checkpoint_dir, epoch)
                    np.save('./sample/'+str(epoch)+' '+str(idx)+'.npy',optim_imag)
                    imageio.imwrite('./sample/intensity'+str(epoch)+str(idx)+'psnr='+str(int_psnr)+'.png',optim_imag[:,:,0]*255)
                    imageio.imwrite('./sample/phase'+str(epoch)+str(idx)+'psnr='+str(phase_psnr)+'.png',optim_imag[:,:,1]*255)




		#running the optimizer
                _,summary_str = self.sess.run([g_optim,self.summary_op],feed_dict={ self.real_A: batch_X,self.real_B: batch_y})
                

                

                self.writer.add_summary(summary_str, counter)
                

                errG = self.loss_with_gt.eval({self.real_A: batch_X,self.real_B: batch_y})
		

		
                counter += 1

                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, g_loss: %.8f" \
                    % (epoch, idx, 10.0,
                        time.time() - start_time, errG))


    def generator(self, image, y=None):
        with tf.variable_scope("generator") as scope:

            s = self.output_size
            s2, s4, s8, s16, s32, s64, s128 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128)
            resized = tf.image.resize_images(image, [256, 256])
            #print(resized.get_shape())
            # image is (256 x 256 x input_c_dim)
            e1 = conv2d(resized, self.gf_dim, name='g_e1_conv')
            # e1 is (128 x 128 x self.gf_dim)
            e2 = self.g_bn_e2(conv2d(lrelu(e1), self.gf_dim*2, name='g_e2_conv'))
            # e2 is (64 x 64 x self.gf_dim*2)
            e3 = self.g_bn_e3(conv2d(lrelu(e2), self.gf_dim*4, name='g_e3_conv'))
            # e3 is (32 x 32 x self.gf_dim*4)
            e4 = self.g_bn_e4(conv2d(lrelu(e3), self.gf_dim*8, name='g_e4_conv'))
            # e4 is (16 x 16 x self.gf_dim*8)
            e5 = self.g_bn_e5(conv2d(lrelu(e4), self.gf_dim*8, name='g_e5_conv'))
            # e5 is (8 x 8 x self.gf_dim*8)
            e6 = self.g_bn_e6(conv2d(lrelu(e5), self.gf_dim*8, name='g_e6_conv'))
            # e6 is (4 x 4 x self.gf_dim*8)
            e7 = self.g_bn_e7(conv2d(lrelu(e6), self.gf_dim*8, name='g_e7_conv'))
            # e7 is (2 x 2 x self.gf_dim*8)
            e8 = self.g_bn_e8(conv2d(lrelu(e7), self.gf_dim*8, name='g_e8_conv'))
            # e8 is (1 x 1 x self.gf_dim*8)

            self.d1, self.d1_w, self.d1_b = deconv2d(tf.nn.relu(e8),
                [self.batch_size, s128, s128, self.gf_dim*8], name='g_d1', with_w=True)
            d1 = tf.nn.dropout(self.g_bn_d1(self.d1), 0.5)
            d1 = tf.concat([d1, e7], 3)
            # d1 is (2 x 2 x self.gf_dim*8*2)

            self.d2, self.d2_w, self.d2_b = deconv2d(tf.nn.relu(d1),
                [self.batch_size, s64, s64, self.gf_dim*8], name='g_d2', with_w=True)
            d2 = tf.nn.dropout(self.g_bn_d2(self.d2), 0.5)
            d2 = tf.concat([d2, e6], 3)
            # d2 is (4 x 4 x self.gf_dim*8*2)

            self.d3, self.d3_w, self.d3_b = deconv2d(tf.nn.relu(d2),
                [self.batch_size, s32, s32, self.gf_dim*8], name='g_d3', with_w=True)
            d3 = tf.nn.dropout(self.g_bn_d3(self.d3), 0.5)
            d3 = tf.concat([d3, e5], 3)
            # d3 is (8 x 8 x self.gf_dim*8*2)

            self.d4, self.d4_w, self.d4_b = deconv2d(tf.nn.relu(d3),
                [self.batch_size, s16, s16, self.gf_dim*8], name='g_d4', with_w=True)
            d4 = self.g_bn_d4(self.d4)
            d4 = tf.concat([d4, e4], 3)
            # d4 is (16 x 16 x self.gf_dim*8*2)

            self.d5, self.d5_w, self.d5_b = deconv2d(tf.nn.relu(d4),
                [self.batch_size, s8, s8, self.gf_dim*4], name='g_d5', with_w=True)
            d5 = self.g_bn_d5(self.d5)
            d5 = tf.concat([d5, e3], 3)
            # d5 is (32 x 32 x self.gf_dim*4*2)

            self.d6, self.d6_w, self.d6_b = deconv2d(tf.nn.relu(d5),
                [self.batch_size, s4, s4, self.gf_dim*2], name='g_d6', with_w=True)
            d6 = self.g_bn_d6(self.d6)
            d6 = tf.concat([d6, e2], 3)
            # d6 is (64 x 64 x self.gf_dim*2*2)

            self.d7, self.d7_w, self.d7_b = deconv2d(tf.nn.relu(d6),
                [self.batch_size, s2, s2, self.gf_dim], name='g_d7', with_w=True)
            d7 = self.g_bn_d7(self.d7)
            d7 = tf.concat([d7, e1], 3)
            # d7 is (128 x 128 x self.gf_dim*1*2)

            self.d8, self.d8_w, self.d8_b = deconv2d(tf.nn.relu(d7),
                [self.batch_size, s, s, self.output_c_dim], name='g_d8', with_w=True)
            # d8 is (256 x 256 x output_c_dim)

            return tf.nn.sigmoid(self.d8)
#function to save checkpoint
    def save(self, checkpoint_dir, step):
        model_name = "pix2pix.model"
        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)
















    #def load(self, checkpoint_dir):
     #   print(" [*] Reading checkpoint...")
      #  print("checkpoint_dir ",checkpoint_dir)

       # checkpoint_dir="/home/honey/Honey/FPM/Honey/Mayug/CGAN_adaptive/checkpoint_old"
        #model_dir = "%s_%s_%s" % (25, 32, 256)
        #model_dir = "separate_discriminator"
        #checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        #print("checkpoint_dir ",checkpoint_dir)
        #ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        #if ckpt and ckpt.model_checkpoint_path:
        #    ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        #    print("ckpt_name ",ckpt_name)
        #    self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
        #    return True
        #else:
        #    return False

#    def test(self, args):
 #      init_op = tf.global_variables_initializer()
  #      self.sess.run(init_op)

   #     sample_files_X = sorted(glob('/media/data/FPM/datasets/25/test/X/*.npy'.format(self.dataset_name)))
    #    sample_files_y = sorted(glob('/media/data/FPM/datasets/25/test/y_/*.npy'.format(self.dataset_name)))

        # sort testing input
        #n = [int(i) for i in map(lambda x: x.split('/')[-1].split('.jpg')[0], sample_files)]
        #sample_files = [x for (y, x) in sorted(zip(n, sample_files))]

        # load testing input
     #   print("Loading testing images ...")
      #  sample = [load_data(sample_file, is_test=True) for sample_file in sample_files_X]
       # sample_images = np.array(sample).astype(np.float32)
        #sample_y = [load_data(sample_file, is_test=True) for sample_file in sample_files_y]
        #sample_images_y = np.array(sample_y).astype(np.float32)
    #    sample_images = [sample_images[i:i+self.batch_size]
      #                   for i in xrange(0, len(sample_images), self.batch_size)]
     #   sample_images = np.array(sample_images)
      #  sample_images_y = [sample_images_y[i:i+self.batch_size]
      #                   for i in xrange(0, len(sample_images_y), self.batch_size)]
      #  sample_images_y = np.array(sample_images_y)
      #  print(sample_images.shape)
      #  print(sample_images_y.shape)
      #  #sample_images_y = sample_images_y[:,:,:,0]
      #  
      #  start_time = time.time()
      #  self.saver.restore(self.sess, './checkpoint_old/separate_discriminator/unet_fwd.model-40')


#        for i, (sample_image,sample_image_y) in enumerate(list(zip(sample_images,sample_images_y))):
#            idx = i+1
#            print("sampling image ", idx)
#            samples = self.sess.run(
#                self.fake_B_sample,
#                feed_dict={self.real_A: sample_image,self.real_B: sample_image_y})
#            print("max ",np.amax(samples))
#            print("min ",np.amin(samples))
#            np.save('./test/results/test_{:02d}.npy'.format(idx),samples)
#            np.save('./test/gt/test_{:02d}.npy'.format(idx),sample_image_y)

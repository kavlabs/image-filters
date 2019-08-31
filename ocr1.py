import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('download.jpg',1)

class filters:
	def avg_blur(a,k_size):
		kernel = np.ones([k_size,k_size],dtype=np.int16)
		c,d = np.shape(a)
		e = np.zeros(((int(c)-2,int(d)-2)),dtype=np.int16)
		kk1 = 0
		kk2 = 0
		while kk1<=(c-k_size):
			if int(kk2)<=(d-k_size):
				k = np.zeros([k_size,k_size],dtype=np.int16)
				for i in range(3):
					for j in range(3):
						k[i][j]=a[i+kk1][j+kk2]
				kkk = np.multiply(k,kernel)
				kkkk = int(np.sum(kkk)/(k_size*k_size))
				e[kk1][kk2] = int(kkkk)
				kk2+=1
			if int(kk2)>(d-k_size):
				kk2 = 0
				kk1 += 1
		return e

	def avg_blur_rgb(a,k_size):
		c,d,g = np.shape(a)
		kernel = np.ones([k_size,k_size,int(g)],dtype=np.int16)
		e = np.zeros(((int(c)-2,int(d)-2,int(g))),dtype=np.int16)
		kk1 = 0
		kk2 = 0
		while kk1<=(c-k_size):
			if int(kk2)<=(d-k_size):
				k = np.zeros([k_size,k_size,int(g)],dtype=np.int16)
				for i in range(3):
					for j in range(3):
						k[i][j][0]=a[i+kk1][j+kk2][0]
						k[i][j][1]=a[i+kk1][j+kk2][1]
						k[i][j][2]=a[i+kk1][j+kk2][2]
				kkk = np.multiply(k,kernel)
				kkkk = np.sum(kkk,axis=1)//(k_size*k_size)
				e[kk1][kk2] = kkkk[0]
				kk2+=1
			if int(kk2)>(d-k_size):
				kk2 = 0
				kk1 += 1
		return e


	def gaussian_blur(a,k_size):
		kernel = np.array([[1,2,1],[2,4,2],[1,2,1]])
		c,d = np.shape(a)
		e = np.zeros(((int(c)-2,int(d)-2)),dtype=np.int16)
		kk1 = 0
		kk2 = 0
		while kk1<=(c-k_size):
			if int(kk2)<=(d-k_size):
				k = np.zeros([k_size,k_size],dtype=np.int16)
				for i in range(3):
					for j in range(3):
						k[i][j]=a[i+kk1][j+kk2]
				kkk = np.multiply(k,kernel)
				kkkk = int(np.sum(kkk)/(k_size*k_size))
				e[kk1][kk2] = int(kkkk)
				kk2+=1
			if int(kk2)>(d-k_size):
				kk2 = 0
				kk1 += 1
		return e

	def gaussian_blur_rgb(a,k_size):
		# kernel = np.array([np.array([[1,2,1],[2,4,2],[1,2,1]]),np.array([[1,2,1],[2,4,2],[1,2,1]]),np.array([[1,2,1],[2,4,2],[1,2,1]])])
		kernel = np.array([np.array([[1,2,1],[2,4,2],[1,2,1]])])
		c,d,g = np.shape(a)
		e = np.zeros(((int(c)-2,int(d)-2,int(g))),dtype=np.int16)
		kk1 = 0
		kk2 = 0
		while kk1<=(c-k_size):
			if int(kk2)<=(d-k_size):
				k = np.zeros([k_size,k_size,int(g)],dtype=np.int16)
				for i in range(3):
					for j in range(3):
						k[i][j][0]=a[i+kk1][j+kk2][0]
						k[i][j][1]=a[i+kk1][j+kk2][1]
						k[i][j][2]=a[i+kk1][j+kk2][2]
				kkk = np.multiply(k,kernel)
				kkkk = np.sum(kkk,axis=1)//(k_size*k_size)
				e[kk1][kk2] = kkkk[0]
				kk2+=1
			if int(kk2)>(d-k_size):
				kk2 = 0
				kk1 += 1
		return e

	def rgb_to_gray(a,k_size):
		c,d,g = np.shape(a)
		e = np.zeros(((int(c),int(d))),dtype=np.int16)
		kk1 = 0
		kk2 = 0
		while kk1<(c):
			if int(kk2)<(d):
				k1=a[kk1][kk2][0]
				k2=a[kk1][kk2][1]
				k3=a[kk1][kk2][2]
				e[kk1][kk2] = (int(k1)+int(k2)+int(k3))//3
				kk2+=1
			if int(kk2)==(d):
				kk2 = 0
				kk1 += 1
		return e

# k = filters.avg_blur_rgb(img,3)
# np.savetxt("kk.txt",k)
# print(k)
# c,d = np.shape(img)
# t = np.reshape(np.array([k]),[1022,789])
# cv2.imshow('image',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
# plt.show()
# plt.imshow(k, cmap = 'gray', interpolation = 'bicubic')
# plt.show()
k = filters.rgb_to_gray(img,3)
cv2.imwrite('kk_gray.png',k)
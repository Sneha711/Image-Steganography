import numpy as np
import cv2 
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
from collections import deque





im= cv2.imread("pic.png",0) 

#Cropping the image to 6*6 pixels
# im = im[0:6,0:6]
# cv2.imshow("grayscale",img)
# cv2.waitKey(0)
cv2.imwrite("grayscale.png",im)
#print img[0,0]

img = cv2.imread("grayscale.png",0) 

print "The image size is:", img.size
print "Image shape is:",img.shape

rows=img.shape[0]
cols=img.shape[1]

print "Rows are ",rows
print "Columns are ", cols

pixel=np.zeros(shape=(rows,cols))
pixel=pixel.astype(int)

for i in range(rows):
	for j in range(cols):
		pixel[i,j]=img[i,j]

print pixel


# blue = img[:,:,0]
# green = img[:,:,1]
# red= img[:,:,2]

#print blue

# Keys

kr = np.random.uniform(low=0, high=math.pow(2,8)-1, size=(rows,))
kr=kr.astype(int)
print kr
print type(kr)

rotkr=list(reversed(kr))
rotkr=np.asarray(rotkr)
print rotkr
print type(rotkr)

kc = np.random.uniform(low=0, high=math.pow(2,8)-1, size=(cols,))
kc=kc.astype(int)
print kc

rotkc=list(reversed(kc))
rotkc=np.asarray(rotkc)
print rotkc
print type(rotkc)

#rows sum
for i in range(rows):
	sum=0
	for j in range(cols):
		sum=sum+pixel[i,j]
	if(sum%2==0):
		d=deque(pixel[i])
		d.rotate(kr[i]) #right circular shift by kr[i] positions
		d=list(d)
		pixel[i]=d
	else:
		d=deque(pixel[i])
		d.rotate(-kr[i]) #left circular shift by kr[i] positions
		d=list(d)
		pixel[i]=d
#cols sum
for j in range(cols):
	sum=0
	for i in range(rows):
		sum=sum+pixel[i,j]
	if(sum%2==0):
		d=deque(pixel[:,j])
		d.rotate(-kc[j]) #up(left) circular shift by kc[j] positions
		d=list(d)
		pixel[:,j]=d
	else:
		d=deque(pixel[:,j])
		d.rotate(kc[j]) #down(right) circular shift by kc[j] positions
		d=list(d)
		pixel[:,j]=d


print pixel
print type(pixel)

pixel2=np.zeros(shape=(rows,cols))
pixel2=pixel2.astype(int)

pixel3=np.zeros(shape=(rows,cols))
pixel3=pixel3.astype(int)

#row exor
for i in range(rows):
	if(i%2==1):
		pixel2[i]=pixel[i]^kc
	else:
		pixel2[i]=pixel[i]^rotkc
#col exor
print "After row exor, pixel2 is \n",pixel2

for j in range(cols):
	if(j%2==1):
		pixel3[:,j]=pixel2[:,j]^kr
	else:
		pixel3[:,j]=pixel2[:,j]^rotkr

print "After col exor, pixel3 is \n", pixel3
print type(pixel3)

# imgplot=plt.imshow(pixel3)
# plt.show()
# cv2.imshow("encrypted",pixel3)
# cv2.waitKey(0)
cv2.imwrite("encrypted.png",pixel3)



#DECRYPTION

print ("\n\n\nDECRYPTION\n\n\n")

#img2=pixel3
img2 = cv2.imread("encrypted.png",0) 
print img2

#Cropping the image to 6*6 pixels
#img = img[0:6,0:6]
# cv2.imshow("grayscale",img)
# cv2.waitKey(0)
# cv2.imwrite("15IT245_M1_cropped_panda.png",img)
#print img[0,0]

# print img[0,0,0],img[0,0,1],img[0,0,2]
# blue = img2[:,:,0]
# green = img2[:,:,1]
# red= img2[:,:,2]
# print "bgr values are\n", blue,green,red
# print blue[0,0],green[0,0],red[0,0]


print "The image size is:", img2.size
print "Image shape is:",img2.shape

rows2=img2.shape[0]
cols2=img2.shape[1]

print "Rows are ",rows2
print "Columns are ", cols2

pixel3_dec=np.zeros(shape=(rows2,cols2))
pixel3_dec=pixel3_dec.astype(int)

for j in range(cols2):
	if(j%2==1):
		pixel3_dec[:,j]=img2[:,j]^kr
	else:
		pixel3_dec[:,j]=img2[:,j]^rotkr


print "pixel3_dec is \n",pixel3_dec
print "pixel2(encryption) is \n",pixel2


pixel2_dec=np.zeros(shape=(rows2,cols2))
pixel2_dec=pixel2_dec.astype(int)

for i in range(rows2):
	if(i%2==1):
		pixel2_dec[i]=pixel3_dec[i]^kc
	else:
		pixel2_dec[i]=pixel3_dec[i]^rotkc


print "pixel2_dec \n",pixel2_dec
print "pixel is \n",pixel


#unscramble

for j in range(cols2):
	sum=0
	for i in range(rows2):
		sum=sum+pixel2_dec[i,j]
	if(sum%2==0):
		d=deque(pixel2_dec[:,j])
		d.rotate(kc[j]) #down circular shift by kc[j] positions
		d=list(d)
		pixel2_dec[:,j]=d
	else:
		d=deque(pixel2_dec[:,j])
		d.rotate(-kc[j]) #up circular shift by kc[j] positions
		d=list(d)
		pixel2_dec[:,j]=d



for i in range(rows2):
	sum=0
	for j in range(cols2):
		sum=sum+pixel2_dec[i,j]
	if(sum%2==0):
		d=deque(pixel2_dec[i])
		d.rotate(-kr[i]) #right circular shift by kr[i] positions
		d=list(d)
		pixel2_dec[i]=d
	else:
		d=deque(pixel2_dec[i])
		d.rotate(kr[i]) #left circular shift by kr[i] positions
		d=list(d)
		pixel2_dec[i]=d


print "pixel2_dec is \n",pixel2_dec
print "original image is \n",img
cv2.imwrite("decrypted.png",pixel2_dec)

img_gray = cv2.imread("grayscale.png",0)

img_dec = cv2.imread("decrypted.png",0) 


print img_gray
print img_dec

if((pixel2_dec==img_dec).all()):
	print "woohoo"

if((img_dec==img).all()):
	print "yes3"
if((img_dec==im).all()):
	print "yes4"	
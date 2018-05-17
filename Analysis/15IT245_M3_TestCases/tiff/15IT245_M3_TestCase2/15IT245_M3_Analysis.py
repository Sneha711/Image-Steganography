import numpy as np
import cv2 
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
from collections import deque
import pylab
import math





print "Reading image to be encrypted"
im= cv2.imread("15IT245_M3_image.tiff",0) 

#Cropping the image to 6*6 pixels
# im = im[0:3,0:3]
# cv2.imshow("grayscale",img)
# cv2.waitKey(0)

print "Converting image to grayscale and writing"
cv2.imwrite("15IT245_M3_grayscale.tiff",im)
#print img[0,0]

img = cv2.imread("15IT245_M3_grayscale.tiff",0) 

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

# print pixel


# blue = img[:,:,0]
# green = img[:,:,1]
# red= img[:,:,2]

#print blue

# Keys
print "Generating keys kr and kc for encryption"
kr = np.random.uniform(low=0, high=math.pow(2,8)-1, size=(rows,))
kr=kr.astype(int)
print "kr is ",kr
print type(kr)

rotkr=list(reversed(kr))
rotkr=np.asarray(rotkr)
print "rotkr is ",rotkr
print type(rotkr)

kc = np.random.uniform(low=0, high=math.pow(2,8)-1, size=(cols,))
kc=kc.astype(int)
print "kc is ",kc

rotkc=list(reversed(kc))
rotkc=np.asarray(rotkc)
print "rotkc is ",rotkc
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

print "After scrambling rows:\n",pixel
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


# print type(pixel)
print "After scrambling rows and columns:\n",pixel

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
cv2.imwrite("15IT245_M3_encrypted.tiff",pixel3)

print "\n\n\nENCRYPTED IMAGE WRITTEN\n\n\n"

print "STARTING STEGANOGRAPHY\n\n\n"


msg= cv2.imread("15IT245_M3_encrypted.tiff",0) 

print "Pixel values of encrypted image read are:\n",msg



binary=[]

for i in range(rows):
	b=[]
	for j in range(cols):
		val=msg[i,j]
		val1 = '{0:08b}'.format(val)
		b.append(val1)
	binary.append(b)

print "The binary form of the encrypted image\n",binary

print("Reading cover image\n")

img_cover = cv2.imread("15IT245_M3_cover.png") 

#Cropping the image to 6*6 pixels
# img_cover = img_cover[0:6,0:6]
#cv2.imshow("cropped",img)
#cv2.waitKey(0)
#cv2.imwrite("15IT245_M1_cropped_panda.png",img)

print "Cover image is \n",img_cover
print "The image size is:", img_cover.size
print "Image shape is:",img_cover.shape

#For spliting the image into red green and blue channels
blue = img_cover[:,:,0]
green = img_cover[:,:,1]
red= img_cover[:,:,2]

red_stego = red.copy()
blue_stego = blue.copy()
green_stego = green.copy()

#getting the dimensions of the image
shape=img_cover.shape
rows_cover = shape[0]
col_cover = shape[1]

print "Red starting"
#for red
for i in range(rows):
	for j in range(cols):
		red_val = red[i][j]
		# green_val = green[i][j]
		# blue_val = blue[i][j]
		#print "count is",count
		# if count==len(binary[i]):
		# 	break
		bin_2bits = binary[i][j][6:]
		#print "the last two bits extracted from the pixels binary(which is to be embedded in red pixel) is: ",bin_2bits, "for pixel ", count
		
		# count=count+1

		

		#Converting integer values 8 bit binary
		red_bin =  '{0:08b}'.format(red_val)
		# green_bin =  '{0:08b}'.format(green_val)
		# blue_bin =  '{0:08b}'.format(blue_val)

		#print "red_bin value", red_bin

		red_2bits = red_bin[6:]
		#print "Last two bits of red pixel for pixel ",count," is ",red_2bits

		#print "bin2bits[0]",bin_2bits[0]
		
		#exor operation
		
		input_a0 = int(bin_2bits[0],2)
		#print "input_a",input_a0,type(input_a0)
		input_b0 = int(red_2bits[0],2)
		#print "input_b",input_b0,type(input_b0)
		xor_res0 = bin(input_a0^input_b0)
		xor_res0= xor_res0[2:] #after removing 0b

		input_a1 = int(bin_2bits[1],2)
		#print "input_a",input_a1,type(input_a1)
		input_b1 = int(red_2bits[1],2)
		#print "input_b",input_b1,type(input_b1)
		xor_res1 = bin(input_a1^input_b1)
		xor_res1 = xor_res1[2:] #after removing 0b


		#print "result of xor is: ",xor_res0,xor_res1
		#print "type",type(xor_res0)

		stego = red_bin[0:6]+xor_res0+xor_res1
		#print "stego 8 bit binary value is ",stego, " for pixel ", count
		int_red=int(stego,2)
		red_stego[i][j] = int_red
	# if count==len(binary):
	# 		break


print "Encrypted image is\n",msg

print "Red is\n",red
print "Red_stego is\n",red_stego

print "Green starting"
	
	#for green
for i in range(rows):
	#print "gree"
	
	for j in range(cols):
																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																				
		green_val = green[i][j]
		
		#print "count is",count
		# if count==len(binary):
		# 	break
		bin_2bits = binary[i][j][4:6]																																																																																																						     
		#print "The 5th and 6th of message is: ",bin_2bits, " for pixel ", count
		 
		# count=count+1
			#Converting integer values 8 bit binary
		
		green_bin =  '{0:08b}'.format(green_val)
		#print "green_bin value", green_bin
		green_2bits = green_bin[6:]
		#print "last 2 bits of green pixel is: ",green_2bits, " for pixel ", count
			#print "bin2bits[0]",bin_2bits[0]
		
		#exor operation
		
		input_a0 = int(bin_2bits[0],2)
		#print "input_a",input_a0,type(input_a0)
		input_b0 = int(green_2bits[0],2)
		#print "input_b",input_b0,type(input_b0)
		xor_res0 = bin(input_a0^input_b0)
		xor_res0= xor_res0[2:] #after removing 0b
		input_a1 = int(bin_2bits[1],2)
		#print "input_a",input_a1,type(input_a1)
		input_b1 = int(green_2bits[1],2)
		#print "input_b",input_b1,type(input_b1)
		xor_res1 = bin(input_a1^input_b1)
		xor_res1 = xor_res1[2:] #after removing 0b
		#print "result of xor for green pixel is: ",xor_res0,xor_res1
		#print "type",type(xor_res0)
		stego = green_bin[0:6]+xor_res0+xor_res1
		#print "stego 8 bit binary value for green pixel is: ",stego
		int_green=int(stego,2)
		green_stego[i][j] = int_green
		
	
print "Message is \n",msg	
print "Green is\n",green
print "Green Stego is\n",green_stego



print "Blue starting"

#for blue
for i in range(rows):
	
	for j in range(cols):
																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																					
	
		blue_val = blue[i][j]
			
			#print "count is",count
			# if count==len(binary):
			# 	break
		bin_4bits = binary[i][j][0:4]																																																																																																						     
		#print "the first 4 bits of the message are: ",bin_4bits," for pixel ",count
			
			# count=count+1

			

			#Converting integer values 8 bit binary
			
		blue_bin =  '{0:08b}'.format(blue_val)
			

			#print "blue_bin value", blue_bin

		blue_4bits = blue_bin[4:]
		#print "last four bits of blue pixel are: ",blue_4bits, " for pixel ",count
			#print "bin4bits[0]",bin_4bits[0]
		
			#exor operation
			
		input_a0 = int(bin_4bits[0],2)
		#print "input_a",input_a0,type(input_a0)
		input_b0 = int(blue_4bits[0],2)
		#print "input_b",input_b0,type(input_b0)
		xor_res0 = bin(input_a0^input_b0)
		xor_res0= xor_res0[2:] #after removing 0b
		input_a1 = int(bin_4bits[1],2)
		#print "input_a",input_a1,type(input_a1)
		input_b1 = int(blue_4bits[1],2)
			#print "input_b",input_b1,type(input_b1)
		xor_res1 = bin(input_a1^input_b1)
		xor_res1 = xor_res1[2:] #after removing 0b
		input_a2 = int(bin_4bits[2],2)
			#print "input_a",input_a2,type(input_a1)
		input_b2 = int(blue_4bits[2],2)
			#print "input_b",input_b2,type(input_b2)
		xor_res2 = bin(input_a2^input_b2)
		xor_res2 = xor_res2[2:] #after removing 0b

		input_a3 = int(bin_4bits[3],2)
			#print "input_a",input_a3,type(input_a3)
		input_b3 = int(blue_4bits[3],2)
			#print "input_b",input_b3,type(input_b3)
		xor_res3 = bin(input_a3^input_b3)
		xor_res3 = xor_res3[2:] #after removing 0b



		#print "result of xor for blue pixel",xor_res0,xor_res1, " for pixel ", count
			#print "type",type(xor_res0)

		stego = blue_bin[0:4]+xor_res0+xor_res1+xor_res2+xor_res3
		#print "stego 8 bit binary value of blue pixel is: ",stego, "for pixel", count
		int_blue=int(stego,2)
		blue_stego[i][j] = int_blue
	
	

print "Message \n",msg
print "Blue\n",blue
print "Blue stego\n",blue_stego


# Merging the rgb components
print "Merging BGR components"

merge_img =  cv2.merge((blue_stego,green_stego,red_stego))
cv2.imwrite('15IT245_M3_stego.png',merge_img)


bit_xor = merge_img^img_cover
print "The merge image is", merge_img

print "The original image is\n", img_cover
# print "The merge image is", merge_img
print "The bit_xor imge is: ",bit_xor
# print "bit_xor[0][0]", bit_xor[0][0]
print bit_xor.shape


lred=[]
lgreen=[]
lblue=[]
lred_final=[]
lgreen_final=[]
lblue_final=[]
l_encr_words=[]
l_encr_words_int=[]

print "Rows and cols of cover are ",rows_cover,col_cover
#For red
for i in range(rows_cover):
	lred=[]
	for j in range(col_cover):
		# if bit_xor[i][j][0] == 0 and bit_xor[i][j][1] == 0 and bit_xor[i][j][2] ==0:
		# 	break
		# b_bin = '{0:04b}'.format(bit_xor[i][j][0])
		# g_bin = '{0:02b}'.format(bit_xor[i][j][1])
		r_bin = '{0:02b}'.format(bit_xor[i][j][2])

		
		#l_int=int(l_encr, 2)   #Converting into integer
		lred.append(r_bin)
	lred_final.append(lred)
print "After extracting red values \n",lred_final

#For green
for i in range(rows_cover):
	lgreen=[]
	
	for j in range(col_cover):
	
			#b_bin = '{0:04b}'.format(bit_xor[i][j][0])
		g_bin = '{0:02b}'.format(bit_xor[i][j][1])
			#r_bin = '{0:02b}'.format(bit_xor[i][j][2])
			
			#l_int=int(l_encr, 2)   #Converting into integer
		lgreen.append(g_bin)
	lgreen_final.append(lgreen)
print "After extracting green values\n",lgreen_final
#For blue
for i in range(rows_cover):
	lblue=[]
	for j in range(col_cover):
	
		#print "b value before ",bit_xor[i][j][0]
		b_bin = '{0:04b}'.format(bit_xor[i][j][0])
		#print "b value 4 bits ",b_bin
		
		lblue.append(b_bin)
	lblue_final.append(lblue)
# print "Length of list for b is ",len(l_encr_words_blue)
print "After extracting blue values\n",lblue_final
lsemifinal=[]
lfinal=[]
lfinalint=[]
lsemifinalint=[]
for i in range(rows_cover):
	lsemifinalint=[]
	lsemifinal=[]
	for j in range(col_cover):
		# print "blue value ", lblue_final[i][j]
		# print "green value ", lgreen_final[i][j]
		# print "red value ", lred_final[i][j]

		k = lblue_final[i][j] + lgreen_final[i][j] + lred_final[i][j]
		kint=int(k,2)
		# print k
		lsemifinal.append(k)
		lsemifinalint.append(kint)
	lfinal.append(lsemifinal)
	lfinalint.append(lsemifinalint)

print "Encrypted extracted\n",lfinalint
print "Actual encrypted\n",msg


extracted=np.array(lfinalint)
print extracted,type(extracted)

rowzero=np.where(~extracted.any(axis=1))[0]
minrow=rowzero[0]

colzero=np.where(~extracted.any(axis=0))[0]
mincol=colzero[0]

print minrow,mincol

img2=extracted[0:minrow,0:mincol]
print "Image extracted after removing rows and columns which are zero\n",img2

print ("\n\n\nDECRYPTION\n\n\n")

#img2=pixel3
# img2 = cv2.imread("encrypted.png",0) 
# print img2

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


pixel2_dec=np.zeros(shape=(rows2,cols2))
pixel2_dec=pixel2_dec.astype(int)

for i in range(rows2):
	if(i%2==1):
		pixel2_dec[i]=pixel3_dec[i]^kc
	else:
		pixel2_dec[i]=pixel3_dec[i]^rotkc


print "pixel2_dec \n",pixel2_dec


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
cv2.imwrite("15IT245_M3_decrypted.tiff",pixel2_dec)

img_gray = cv2.imread("15IT245_M3_grayscale.tiff",0)

img_dec = cv2.imread("15IT245_M3_decrypted.tiff",0) 


print img_gray
print img_dec

if((pixel2_dec==img_dec).all()):
	print "woohoo"


if((img2==img_gray).all()):
	print "yes1"
if((img2==img).all()):
	print "yes2"
if((img_dec==img).all()):
	print "yes3"
if((img_dec==im).all()):
	print "yes4"	


print "-------------------------------------ANALYSIS---------------------------------------------------------------------------"
#msg is the encrypted image and img is the original image
a=0
print msg.shape
for i in range(rows):
	for  j in range(cols):
		if msg[i][j]==pixel3[i][j]:
			a=a+1

print "The value of a is:",a



print "VISUAL TESTING"
d=0
for i in range(rows):
	for j in range(cols):
		if pixel3[i][j] == img[i][j]:
			d=d+0
		else:
			d=d+1

NPCR =  (d / (rows*cols*1.000000))*100


diff_sum=0
for i in range(rows):
	for j in range(cols):
		diff = pixel3[i][j] - img[i][j]
		if diff < 0:
			diff = diff*(-1)
		diff_sum = diff_sum + diff

UACI = (diff_sum/(255*rows*cols*1.0000000))*100

print "The NCPR value is: ", NPCR
print "The UACI value is: ", UACI

print "STATISTICAL ANALYSIS"

print "1] Histogram Analysis"

plt.subplot(211)
plt.hist(img.ravel(),256,[0,256])
plt.title('Histogram of original and encrypted image')
plt.ylabel('Number of pixels')
plt.xlabel('Intensity values')



plt.subplot(212)
plt.hist(msg.ravel(),256,[0,256])
plt.ylabel('Number of pixels')
plt.xlabel('Intensity values')
plt.savefig('histogram.png')



print "2] Correlations computation of the adjacent pixels in encrypted image"
hori_mat = img
hori_mat_enc = msg

for i in range(rows):
	for j in range(cols):
		if j < cols -1:
			hori_mat[i][j] = img[i][j+1] 
			hori_mat_enc[i][j] = msg[i][j+1]



Ex=0
Ey=0
Ex_enc=0
Ey_enc=0
for i in range(rows):
	for j in range(cols-1):
		Ex=Ex+img[i][j]
		Ey=Ey+img[i][j+1]
		Ex_enc=Ex_enc+msg[i][j]
		Ey_enc=Ey_enc+msg[i][j+1]
Ex = Ex/(rows*(cols-1)*1.000)
Ey = Ey/(rows*(cols-1)*1.000)
Ex_enc = Ex_enc/(rows*(cols-1)*1.000)
Ey_enc = Ey_enc/(rows*(cols-1)*1.000)


Dx=0
Dy=0
Dx_enc=0
Dy_enc=0

for i in range(rows):
	for j in range(cols-1):
		Dx=Dx+math.pow(img[i][j]-Ex,2)
		Dy=Dy+math.pow(img[i][j+1]-Ey,2)
		Dx_enc=Dx_enc+math.pow(msg[i][j]-Ex_enc,2)
		Dy_enc=Dy_enc+math.pow(msg[i][j+1]-Ey_enc,2)


Dx = Dx/(rows*(cols-1)*1.000)
Dy = Dy/(rows*(cols-1)*1.000)
Dx_enc = Dx_enc/(rows*(cols-1)*1.000)
Dy_enc = Dy_enc/(rows*(cols-1)*1.000)

cov=0	
cov_enc=0
for i in range(rows):
	for j in range(cols-1):
		cov = cov + (img[i][j] - Ex)*(img[i][j+1] - Ey)
		cov_enc = cov_enc + (msg[i][j] - Ex_enc)*(msg[i][j+1] - Ey_enc)

cov = cov/(rows*(cols-1)*1.0000)
cov_enc = cov_enc/(rows*(cols-1)*1.0000)


coef = cov/((math.pow(Dx,0.5)*math.pow(Dy,0.5))*1.0000)
coef_enc = cov_enc/((math.pow(Dx_enc,0.5)*math.pow(Dy_enc,0.5))*1.0000)


print "The coefficient of original image is:", coef
print "The coefficient of encrypted image is:", coef_enc




Ex_ver=0
Ey_ver=0
Ex_enc_ver=0
Ey_enc_ver=0
for i in range(rows-1):
	for j in range(cols):
		Ex_ver=Ex_ver+img[i][j]
		Ey_ver=Ey_ver+img[i+1][j]
		Ex_enc_ver=Ex_enc_ver+msg[i][j]
		Ey_enc_ver=Ey_enc_ver+msg[i+1][j]
Ex_ver = Ex_ver/((rows-1)*(cols)*1.000)
Ey_ver = Ey_ver/((rows-1)*(cols)*1.000)
Ex_enc_ver = Ex_enc_ver/(rows*(cols-1)*1.000)
Ey_enc_ver = Ey_enc_ver/(rows*(cols-1)*1.000)


Dx_ver=0
Dy_ver=0
Dx_enc_ver=0
Dy_enc_ver=0

for i in range(rows-1):
	for j in range(cols):
		Dx_ver=Dx_ver+math.pow(img[i][j]-Ex_ver,2)
		Dy_ver=Dy_ver+math.pow(img[i+1][j]-Ey_ver,2)
		Dx_enc_ver=Dx_enc_ver+math.pow(msg[i][j]-Ex_enc_ver,2)
		Dy_enc_ver=Dy_enc_ver+math.pow(msg[i+1][j]-Ey_enc_ver,2)


Dx_ver = Dx_ver/((rows-1)*(cols)*1.000)
Dy_ver = Dy_ver/((rows-1)*(cols)*1.000)
Dx_enc_ver = Dx_enc_ver/((rows-1)*(cols)*1.000)
Dy_enc_ver = Dy_enc_ver/((rows-1)*(cols)*1.000)

cov_ver=0	
cov_enc_ver=0
for i in range(rows-1):
	for j in range(cols):
		cov_ver = cov_ver + (img[i][j] - Ex_ver)*(img[i+1][j] - Ey_ver)
		cov_enc_ver = cov_enc_ver + (msg[i][j] - Ex_enc_ver)*(msg[i+1][j] - Ey_enc_ver)

cov_ver = cov_ver/((rows-1)*(cols)*1.0000)
cov_enc_ver = cov_enc_ver/((rows-1)*(cols)*1.0000)


coef_ver = cov_ver/((math.pow(Dx_ver,0.5)*math.pow(Dy_ver,0.5))*1.0000)
coef_enc_ver = cov_enc_ver/((math.pow(Dx_enc_ver,0.5)*math.pow(Dy_enc_ver,0.5))*1.0000)

print "\nii) Vertical"
print "The correlation coefficient of original image is:", coef_ver
print "The correlation coefficient of encrypted image is:", coef_enc_ver



print "ENTROPY"
img_ent = img.flatten()
msg_ent= msg.flatten()

from pyitlib import discrete_random_variable as drv
entropy = drv.entropy(img_ent)
entropy_enc = drv.entropy(msg_ent)
print "The entropy value of original image is:", entropy
print "The entropy value of encrypted image is:", entropy_enc





# print "ANALYSIS AGAINST ATTACKS"

# print "1] Additive noise"
# pad=240
# def to_std_float(img):
# 	img.astype(np.float16, copy = False)
# 	img = np.multiply(img, (1/255))
 
# 	return img

# def to_std_uint8(img):
# 	img = cv2.convertScaleAbs(img, alpha = (255/1))
 
# 	return img

# img_noise = to_std_float(msg)
# noise = np.random.randint(pad, size = (img_noise.shape[0], img_noise.shape[1]))
# img_noise = np.where(noise == 0, 0, img_noise)
# img_noise = np.where(noise == (pad-1), 1, img_noise)
# img_noise = to_std_uint8(img_noise)
# # img_2 = img_noise
# # img_2 = msg
# img_noise = msg
# print ("\n\n\nDECRYPTION\n\n\n")

# #img2=pixel3
# # img2 = cv2.imread("encrypted.png",0) 
# # print img2

# #Cropping the image to 6*6 pixels
# #img = img[0:6,0:6]
# # cv2.imshow("grayscale",img)
# # cv2.waitKey(0)
# # cv2.imwrite("15IT245_M1_cropped_panda.png",img)
# #print img[0,0]

# # print img[0,0,0],img[0,0,1],img[0,0,2]
# # blue = img2[:,:,0]
# # green = img2[:,:,1]
# # red= img2[:,:,2]
# # print "bgr values are\n", blue,green,red
# # print blue[0,0],green[0,0],red[0,0]


# print "The image size is:", img_noise.size
# print "Image shape is:",img_noise.shape

# rows2=img_noise.shape[0]
# cols2=img_noise.shape[1]

# print "Rows are ",rows2
# print "Columns are ", cols2

# pixel3_dec_n=np.zeros(shape=(rows2,cols2))
# pixel3_dec_n=pixel3_dec_n.astype(int)

# for j in range(cols2):
# 	if(j%2==1):
# 		pixel3_dec_n[:,j]=img_noise[:,j]^kr
# 	else:
# 		pixel3_dec_n[:,j]=img_noise[:,j]^rotkr


# print "pixel3_dec is \n",pixel3_dec_n
# print "pixel2(encryption) is \n",pixel2


# pixel2_dec_n=np.zeros(shape=(rows2,cols2))
# pixel2_dec_n=pixel2_dec_n.astype(int)

# for i in range(rows2):
# 	if(i%2==1):
# 		pixel2_dec_n[i]=pixel3_dec_n[i]^kc
# 	else:
# 		pixel2_dec_n[i]=pixel3_dec_n[i]^rotkc


# print "pixel2_dec \n",pixel2_dec_n
# print "pixel is \n",pixel


# #unscramble

# for j in range(cols2):
# 	sum=0
# 	for i in range(rows2):
# 		sum=sum+pixel2_dec_n[i,j]
# 	if(sum%2==0):
# 		d=deque(pixel2_dec_n[:,j])
# 		d.rotate(kc[j]) #down circular shift by kc[j] positions
# 		d=list(d)
# 		pixel2_dec_n[:,j]=d
# 	else:
# 		d=deque(pixel2_dec_n[:,j])
# 		d.rotate(-kc[j]) #up circular shift by kc[j] positions
# 		d=list(d)
# 		pixel2_dec_n[:,j]=d



# for i in range(rows2):
# 	sum=0
# 	for j in range(cols2):
# 		sum=sum+pixel2_dec_n[i,j]
# 	if(sum%2==0):
# 		d=deque(pixel2_dec_n[i])
# 		d.rotate(-kr[i]) #right circular shift by kr[i] positions
# 		d=list(d)
# 		pixel2_dec_n[i]=d
# 	else:
# 		d=deque(pixel2_dec_n[i])
# 		d.rotate(kr[i]) #left circular shift by kr[i] positions
# 		d=list(d)
# 		pixel2_dec_n[i]=d

# cv2.imwrite("decrypted_noise.png",pixel2_dec_n)







		












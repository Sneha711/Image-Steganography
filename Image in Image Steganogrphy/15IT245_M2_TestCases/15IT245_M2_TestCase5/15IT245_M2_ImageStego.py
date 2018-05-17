import numpy as np
import cv2 
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
from collections import deque




print "Reading image to be encrypted"
im= cv2.imread("15IT245_M2_image.png",0) 

#Cropping the image to 6*6 pixels
# im = im[0:3,0:3]
# cv2.imshow("grayscale",img)
# cv2.waitKey(0)

print "Converting image to grayscale and writing"
cv2.imwrite("15IT245_M2_grayscale.png",im)
#print img[0,0]

img = cv2.imread("15IT245_M2_grayscale.png",0) 

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
cv2.imwrite("15IT245_M2_encrypted.png",pixel3)

print "\n\n\nENCRYPTED IMAGE WRITTEN\n\n\n"

print "STARTING STEGANOGRAPHY\n\n\n"


msg= cv2.imread("15IT245_M2_encrypted.png",0) 

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

img_cover = cv2.imread("15IT245_M2_cover.png") 

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
cv2.imwrite('15IT245_M2_stego.png',merge_img)


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
cv2.imwrite("15IT245_M2_decrypted.png",pixel2_dec)

img_gray = cv2.imread("15IT245_M2_grayscale.png",0)

img_dec = cv2.imread("15IT245_M2_decrypted.png",0) 


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

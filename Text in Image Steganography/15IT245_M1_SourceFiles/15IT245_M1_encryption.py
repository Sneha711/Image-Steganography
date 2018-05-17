import numpy as np
import cv2 
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#img = mpimg.imread("cat.jpg")
img = cv2.imread("15IT245_M1_panda.png") 

#Cropping the image to 6*6 pixels
# img = img[0:6,0:6]
#cv2.imshow("cropped",img)
#cv2.waitKey(0)
# cv2.imwrite("15IT245_M1_cropped_panda.png",img)


print "The image size is:", img.size
print "Image shape is:",img.shape

#For spliting the image into red green and blue channels
blue = img[:,:,0]
green = img[:,:,1]
red= img[:,:,2]

#print red

red_stego = red.copy()
blue_stego = blue.copy()
green_stego = green.copy()

#For merging
#merge_img =  cv2.merge((blue,green,red))

#Encryption of the message

msg=raw_input("Enter message ")
ascii=[]
i=0
for c in msg:
	ascii.append(ord(c))
print ascii

key=input("Enter 3 digit key: ")
while key > 10:
	key1=key
	sum=0
	while(key1>0):
		ld=key1%10
		sum=sum+ld
		key1=key1/10
	key=sum
print key

words = msg.split()
print words
for word in words:
	print word
	print len(word)
length=map(len, msg.split())
print length # length of each word
encrypted_msg=""
# exor values
k=0
for i in length:
	#print i
	for j in range(0,i):
		#print j
		word=words[k]
		#print word
		letter = word[j] #extract letter
		print letter
		print "ascii value is ",ord(letter) 
		letter = ord(letter) ^ i #letter exor length
		print "after first exor letter is ",letter
		letter = letter ^ key
		encrypted_msg=encrypted_msg+chr(letter)
	encrypted_msg=encrypted_msg+" "
	print "The encrypted message is:", encrypted_msg
	k=k+1





#message to ascii to binary
asc=[]
binary = [] #contains list of binary values for message

for i in range(len(encrypted_msg)):
	val=ord(encrypted_msg[i])
	asc.append(val)
	val1 = '{0:08b}'.format(val)
	binary.append(val1)

print "The binary form of the message",binary


#getting the dimensions of the image
shape=img.shape
rows = shape[0]
col = shape[1]


#For embedding data into image
count = 0
#for red
for i in range(rows):
	for j in range(col):
		red_val = red[i][j]
		green_val = green[i][j]
		blue_val = blue[i][j]
		#print "count is",count
		if count==len(binary):
			break
		bin_2bits = binary[count][6:]
		print "the last two bits extracted from the message binary(which is to be embedded in red pixel) is: ",bin_2bits, "for pixel ", count
		
		count=count+1

		

		#Converting integer values 8 bit binary
		red_bin =  '{0:08b}'.format(red_val)
		green_bin =  '{0:08b}'.format(green_val)
		blue_bin =  '{0:08b}'.format(blue_val)

		print "red_bin value", red_bin

		red_2bits = red_bin[6:]
		print "Last two bits of red pixel for pixel ",count," is ",red_2bits

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


		print "result of xor is: ",xor_res0,xor_res1
		#print "type",type(xor_res0)

		stego = red_bin[0:6]+xor_res0+xor_res1
		print "stego 8 bit binary value is ",stego, " for pixel ", count
		int_red=int(stego,2)
		red_stego[i][j] = int_red
	if count==len(binary):
			break

print red
print red_stego
																																																																																																																																																																																																																																																																																																																																																																																																																															

print "green"
count = 0
#for green
for i in range(rows):
	#print "gree"
	if i%2 == 0:
		for j in range(col):
																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																					
	
			green_val = green[i][j]
			
			#print "count is",count
			if count==len(binary):
				break
			bin_2bits = binary[count][4:6]																																																																																																						     
			print "The 5th and 6th of message is: ",bin_2bits, " for pixel ", count
			 
			count=count+1

			

			#Converting integer values 8 bit binary
			
			green_bin =  '{0:08b}'.format(green_val)
			

			#print "green_bin value", green_bin

			green_2bits = green_bin[6:]
			print "last 2 bits of green pixel is: ",green_2bits, " for pixel ", count

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


			print "result of xor for green pixel is: ",xor_res0,xor_res1
			#print "type",type(xor_res0)

			stego = green_bin[0:6]+xor_res0+xor_res1
			print "stego 8 bit binary value for green pixel is: ",stego
			int_green=int(stego,2)
			green_stego[i][j] = int_green
	
	else:

		j = col-1
		while j>=0: 
																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																		
	
			green_val = green[i][j]
			
			#print "count is",count
			if count==len(binary):
				break
			bin_2bits = binary[count][4:6]																																																																																																						     
			print "5th and 6th bit of message is: ",bin_2bits
			
			count=count+1

			

			#Converting integer values 8 bit binary
			
			green_bin =  '{0:08b}'.format(green_val)
			

			#print "green_bin value", green_bin

			green_2bits = green_bin[6:]
			print "last two bits of green pixel is: ",green_2bits, " for pixel ", count 

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


			print "result of xor for green pixel is: ",xor_res0,xor_res1, " for pixel ",count
			#print "type",type(xor_res0)

			stego = green_bin[0:6]+xor_res0+xor_res1
			print "stego 8 bit binary value for green pixel is: ",stego, " for pixel ", count
			int_green=int(stego,2)
			green_stego[i][j] = int_green
			j=j-1
	
	if count==len(binary):
			break

print green
print green_stego




print "blue"
count = 0
#for blue
for j in range(col):
	print "blue"
	if j%2 == 0:
		for i in range(rows):
																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																					
	
			blue_val = blue[i][j]
			
			#print "count is",count
			if count==len(binary):
				break
			bin_4bits = binary[count][0:4]																																																																																																						     
			print "the first 4 bits of the message are: ",bin_4bits," for pixel ",count
			
			count=count+1

			

			#Converting integer values 8 bit binary
			
			blue_bin =  '{0:08b}'.format(blue_val)
			

			#print "blue_bin value", blue_bin

			blue_4bits = blue_bin[4:]
			print "last four bits of blue pixel are: ",blue_4bits, " for pixel ",count

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



			print "result of xor for blue pixel",xor_res0,xor_res1, " for pixel ", count
			#print "type",type(xor_res0)

			stego = blue_bin[0:4]+xor_res0+xor_res1+xor_res2+xor_res3
			print "stego 8 bit binary value of blue pixel is: ",stego, "for pixel", count
			int_blue=int(stego,2)
			blue_stego[i][j] = int_blue
	
	else:

		i = rows-1
		while i>=0: 

			blue_val = blue[i][j]
			
			#print "count is",count
			if count==len(binary):
				break
			bin_4bits = binary[count][0:4]																																																																																																						     
			print "first four bits of the message are: ",bin_4bits, " for pixel ",count
			
			count=count+1

			

			#Converting integer values 8 bit binary
			
			blue_bin =  '{0:08b}'.format(blue_val)
			

			#print "blue_bin value", blue_bin

			blue_4bits = blue_bin[4:]
			print "last four bits of blue pixel is: ",blue_4bits, " for pixel ", count

			#print "bin4bits[0]",bin_4bits[0]
			
			#exor operation
			
			input_a0 = int(bin_4bits[0],2)
			#print "input_a0",input_a0,type(input_a0)
			input_b0 = int(blue_4bits[0],2)
			#print "input_b0",input_b0,type(input_b0)
			xor_res0 = bin(input_a0^input_b0)
			xor_res0= xor_res0[2:] #after removing 0b

			input_a1 = int(bin_4bits[1],2)
			print "input_a1",input_a1,type(input_a1)
			input_b1 = int(blue_4bits[1],2)
			print "input_b1",input_b1,type(input_b1)
			xor_res1 = bin(input_a1^input_b1)
			xor_res1 = xor_res1[2:] #after removing 0b

			input_a2 = int(bin_4bits[2],2)
			#print "input_a2",input_a1,type(input_a1)
			input_b2 = int(blue_4bits[2],2)
			#print "input_b2",input_b1,type(input_b1)
			xor_res2 = bin(input_a2^input_b2)
			xor_res2 = xor_res2[2:] #after removing 0b

			input_a3 = int(bin_4bits[3],2)
			#print "input_a3",input_a3,type(input_a3)
			input_b3 = int(blue_4bits[3],2)
			#print "input_b3",input_b3,type(input_b3)
			xor_res3 = bin(input_a3^input_b3)
			xor_res3 = xor_res3[2:] #after removing 0b



			print "result of xor for blue pixel ",xor_res0,xor_res1,xor_res2,xor_res3," for pixel ", count
			#print "type",type(xor_res0)

			stego = blue_bin[0:4]+xor_res0+xor_res1+xor_res2+xor_res3
			print "stego 8 bit binary value for blue pixel is: ",stego, " for pixel ",count
			int_blue=int(stego,2)
			blue_stego[i][j] = int_blue
			i=i-1

	if count==len(binary):
			break

print blue
print blue_stego







# Merging the rgb components
merge_img =  cv2.merge((blue_stego,green_stego,red_stego))
cv2.imwrite('15IT245_M1_stego_panda.jpg',merge_img)
#cv2.imshow('merge',merge_img)
#cv2.waitKey(0)

# print img
# print merge_img




#Decryption part


bit_xor = merge_img^img
print "The original image is", img
print "The merge image is", merge_img
print "The bit_xor imge is: ",bit_xor
print "bit_xor[0][0]", bit_xor[0][0]
print bit_xor.shape

l_encr_words_red=[]
l_encr_words_green=[]
l_encr_words_blue=[]
l_encr_words=[]
l_encr_words_int=[]


#For red
for i in range(rows):
	for j in range(col):
		# if bit_xor[i][j][0] == 0 and bit_xor[i][j][1] == 0 and bit_xor[i][j][2] ==0:
		# 	break
		# b_bin = '{0:04b}'.format(bit_xor[i][j][0])
		# g_bin = '{0:02b}'.format(bit_xor[i][j][1])
		r_bin = '{0:02b}'.format(bit_xor[i][j][2])

		
		#l_int=int(l_encr, 2)   #Converting into integer
		l_encr_words_red.append(r_bin)

#For green
for i in range(rows):
	if i%2==0:
		for j in range(col):
	
			#b_bin = '{0:04b}'.format(bit_xor[i][j][0])
			g_bin = '{0:02b}'.format(bit_xor[i][j][1])
			#r_bin = '{0:02b}'.format(bit_xor[i][j][2])
			
			#l_int=int(l_encr, 2)   #Converting into integer
			l_encr_words_green.append(g_bin)
	else:
		j=col-1
		while j>=0:
			g_bin = '{0:02b}'.format(bit_xor[i][j][1])
			l_encr_words_green.append(g_bin)
			j=j-1

#For blue
for j in range(col):
	if j%2==0:
		for i in range(rows):
	
			print "b value before ",bit_xor[i][j][0]
			b_bin = '{0:04b}'.format(bit_xor[i][j][0])
			print "b value 4 bits ",b_bin
		
			l_encr_words_blue.append(b_bin)
	else:
		i=rows-1
		while i>=0:
			print "b value before ",bit_xor[i][j][0]
			b_bin = '{0:04b}'.format(bit_xor[i][j][0])
			print "b value 4 bits ",b_bin
			l_encr_words_blue.append(b_bin)
			i=i-1
print "Length of list for b is ",len(l_encr_words_blue)
print "List is "
print "Size is ", bit_xor.size

for i in range(bit_xor.size/3):
	print "blue value ", l_encr_words_blue[i]
	print "green value ", l_encr_words_green[i]
	print "red value ", l_encr_words_red[i]

	k = l_encr_words_blue[i] + l_encr_words_green[i] + l_encr_words_red[i]
	print k
	l_encr_words.append(k)
	l_encr_words_int.append(int(k,2))


print l_encr_words_int

l_encr_words_bin=[]

for i in range(len(l_encr_words_int)):
	val2 = '{0:08b}'.format(l_encr_words_int[i])
	l_encr_words_bin.append(val2)

print "Binary form is ",l_encr_words_bin
#print '{0:04b}'.format(bit_xor[1][0][0]),'{0:02b}'.format(bit_xor[0][1][1]),'{0:02b}'.format(bit_xor[0][1][0])
l_encr_words_final=[]

for i in range(len(l_encr_words_int)):
	if(l_encr_words_int[i]<>0):
		l_encr_words_final.append(l_encr_words_int[i])
	else:
		break

print l_encr_words_final

enc_msg=""
for i in range(len(l_encr_words_final)):
	c=(chr)(l_encr_words_final[i])
	enc_msg=enc_msg+c

print enc_msg

wordsenc = enc_msg.split()
print wordsenc
for word in wordsenc:
	print word
	print len(word)
length=map(len, enc_msg.split())
print length # length of each word
decrypted_msg=""
# exor values
k=0
for i in length:
	#print 
	for j in range(0,i):
		#print j
		word=wordsenc[k]
		#print word
		letter = word[j] #extract letter
		print letter
		print "ascii value is ",ord(letter) 
		letter = ord(letter) ^ i #letter exor length
		print "after first exor letter is ",letter
		letter = letter ^ key
		decrypted_msg=decrypted_msg+chr(letter)
	decrypted_msg=decrypted_msg+" "
	print "The decrypted message is:", decrypted_msg
	k=k+1







																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																		
	
			



		

	







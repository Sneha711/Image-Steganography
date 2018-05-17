README


TOPIC: Multiple layer Text security using Variable block size Cryptography and Image Steganography

Two algorithms are used foe encryption of text message: 

1] 
Steps followed for encryting the text(secret message):
a. Read the plain text and their corresponding ASCII values.
b. Calculate the length of the words of the message.
c. Enter the three digits random key and produce the single digit key by folding method.
	eg:
	Let suppose enter the random key is: 969
	1st Round = 9+6+9 = 24
	2nd Round = 2+4 = 6,
	So final key for encryption = 6
d. Cipher Text = (ASCII value XOR key) XOR word length.

2] Hill encryption


Steps followed for hiding the encrypted text message inside the cover image to get stego image:

1) The R, G, B planes of RGB image will be extracted from the cover image.
2) The cipher text will be hidden in R G B planes of cover image –
	• 2 bits will be embedded in 2 LSB of red plane using modified LSB substitution by XORing secret data bits with cover pixels’ bits,
	• 2 bits will be embedded in 2 LSB of green plane using raster scan technique (in first scan hide from left to right and in next scan right to left and soon..) by XORing   secret data bits with cover pixels’ bits,
	• 4 bits will be embedded in 4 LSB of blue plane using raster scan technique (in first scan hide from top to bottom and in next scan bottom to top and so on..) by XORing secret data bits with cover pixels’ bits.
	• This process will continue till the whole cipher text will be hidden in cover image.
3) Stego image will be obtained by combining the three embedded planes.


Source files:

1) 15IT245_M1_encryption.py: Code for encryption of text image using algorithm 1 and hiding it inside an image using the above mentioned steps.

The encryption code is a python code with the name 15IT245_M1_encryption.py. It is run using the following command on the terminal:
$ python 15IT245_M1_encryption.py

The code reads the images which are used to hide the encrypted message. Hence, these images must be in the same folder. Ex: 15IT245_M1_cat.jpg.

The input is read through a file called "15IT245_M1_input.txt" which has 2 lines: the message to be sent and the 3 digit key.
Ex: 15IT245_M1_input.txt:
hello world
345

2) hill_encryption.py: Code for encryption of text image using algorithm 2 i.e Hill Cipher and hiding it inside an image using the above mentioned steps.

It is run using the following command on the terminal:
$ python hill_encryption.py

The code reads the images which are used to hide the encrypted message. Hence, these images must be in the same folder. Ex: 15IT245_M1_cat.jpg.



Output for hill_encryption.py:

The screenshots of the output are included as images. 4 test cases are used, i.e 4 images are used. They are:
1. 15IT245_M1_cat.jpg
	The cropped image is saved as 15IT245_M1_cropped_cat.jpg
 	The image after encryption is saved as 15IT245_M1_stego_cat.jpg
2. 15IT245_M1_lena.png
	The cropped image is saved as 15IT245_M1_cropped_lena.png
 	The image after encryption is saved as 15IT245_M1_stego_lena.png 
3. 15IT245_M1_panda.png
	The cropped image is saved as 15IT245_M1_cropped_panda.png
 	The image after encryption is saved as 15IT245_M1_stego_panda.png
4. 15IT245_M1_dog.jpg
	The cropped image is saved as 15IT245_M1_cropped_dog.jpg
 	The image after encryption is saved as 15IT245_M1_stego_dog.jpg

According to these test cases, the output is stored in 4 log files respectively:
1. 15IT245_M1_log_cat.txt
2. 15IT245_M1_log_lena.txt
3. 15IT245_M1_log_panda.txt
4. 15IT245_M1_log_dog.txt





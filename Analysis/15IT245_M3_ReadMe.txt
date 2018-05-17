README



The code for encrypting the image and hiding the image in cover image is a python code with name 15IT245_M3_Analysis.py. Along with encryption of image and steganography, various analysis is done. It is run using the following command in the terminal:
$python 15IT245_M3_Analysis.py

The tests that were conducted to assess the efficiency and security of the proposed image encryption algorithm are:
1] Visual Testing: NPCR and UACI
2] Statistic Analysis: Histogram analysis and correlation computation
3] Entropy Analysis


The code is applied on various formats of images and the output is included in the 15IT245_M3_TestCases folder.The 15IT245_M3_TestCases folder
containes 4 folders with the name png, bmp, pgm and tiff. Each of these 4 folders cantains 10 test cases each.


The secret images and cover images names as follows:
1. Secret Image: 15IT245_M3_cover.png
2. Cover Image: 15IT245_M3_image.png

After encrypting and hiding the encrypted image in the cover image produces following files as output:
1. 15IT245_M3_decrypted.png
2. 15IT245_M3_encrypted.png: The image generated after encrypting the 15IT245_M2_grayscale.png image
3. 15IT245_M3_stego.png: The stego image generated after hiding the 15IT245_M2_encrypted.png image in 15IT245_M2_cover.png image.



According to these test cases, the output is stored in the corresponding log files named as(for all the 5 testcases):

15IT245_M3_log.txt




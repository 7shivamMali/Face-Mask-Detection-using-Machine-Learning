# Face-Mask-Detection-using-Machine-Learning

by,<br>
Shivam Mali<br>
Abhishek Pandey<br>


This Repository contains source code and dataset of our project "Face Mask Detection using Machine Learning"

### Dataset 

The Face Mask detection using Machine Learning uses the Face Mask Detection Dataset of OMKAR GURAV which is available on kaggle.<br>
dataset link - https://www.kaggle.com/datasets/omkargurav/face-mask-dataset<br>
The Face mask detection Dataset includes 7553 images in total but we have used 4008 to train our model (2004 for each with_mask and without_mask).<br>
dataset structure - <br>
dataset - 
* with_mask
* without_mask<br>

### Source code 

In this project for the face mask detection task we have used a VGG-16 model which reached training accuracy of 96% and testing accuracy of 95%.
The requirements.txt file contain list of all required python libraries for each python file 
The code has three files - 
* train.py - code to train the model
* live_FMD.py - code for live face mask detection
* FMD.py - code for face mask detection on input image

The live_FMD.py and FMD.py uses the trained model (faceMaskDetect.h5) so if anyone want to try the trained model they can simply download the repo install the required libraries and run the live_FMD.py and FMD.py for live face mask detection and face mask detection on image respectively


# Deep learning for localization of two phase high contrast three-dimensional material 
This software is an deep learning application for modeling processing-structure-property (PSP) linkages for two phase high contrast three-dimensional material. It’s a feature-engineering-free framework, which directly takes raw data as input, trains a convolutional neural network (CNN) and outputs output. 

To use this software, what the algorithm requires as input are a numpy array. The shape of this numpy array is (x, 11, 11, 11) where x is the number of focal voxels and the dimension of microstructure centered at focal voxels should be three-dimensional (i.e. 11*11*11). The CNN will establish the PSP linkages in the materials system and predict its local strain.

## Requirements ##
Python 2.7
Numpy 1.12.1 (or higher)
Sklearn 0.19.1 (or higher)
Keras 2.0.0 (or higher)
Pickle
TensorFlow
Scipy

## Files ##
1. contrast10_localization.py: The script to train CNN for contrast 10 dataset and its architecture is presented in the paper in the related publication section.
2. contrast50_localization.py: The script to train CNN for contrast 50 dataset and its architecture is presented in the paper in the related publication section.

## How to run it
1. To run contrast10_localization.py: 
	1. To run this file, use commend ‘python contrast10_localization.py’
	2. The script will train the CNN and save your CNN model.
2. To run contrast50_localization.py: 
	1. To run this file, use commend ‘python contrast50_localization.py’
	2. The script will train the CNN and save your CNN model.

## Acknowledgement
This work is supported in part by the following grants: AFOSR award FA9550-12-1-0458; NIST award 70NANB14H012; NSF award CCF-1409601; DOE awards DESC0007456, DE-SC0014330; and Northwestern Data Science Initiative.

## Related Publications ##
1. Z. Yang, Y. Yabansu, R. Al-Bahrani, W. Liao, A. Choudhary, S. Kalidindi, A. Agrawal, "Establishing structure-property localization linkages for slastic deformation of three-dimensional high contrast composites using deep learning approaches", (Under Review).

## Contact
Zijiang Yang (zijiangyang2016@u.northwestern.edu)
Ankit Agrawal (ankitag@eecs.northwestern.edu)
Alok Choudhary (choudhar@eecs.northwestern.edu)

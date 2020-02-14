# Deep learning for localization of two phase high contrast three-dimensional material 
This software is an deep learning application for modeling processing-structure-property (PSP) linkages for two phase high contrast three-dimensional material. It’s a feature-engineering-free framework, which directly takes raw data as input, trains a convolutional neural network (CNN) and outputs output. 

To use this software, what the algorithm requires as input are a numpy array. The shape of this numpy array is (x, 11, 11, 11) where x is the number of focal voxels and the dimension of microstructure centered at focal voxels should be three-dimensional (i.e. 11x11x11). The CNN will establish the PSP linkages in the materials system and predict its local strain.

## Requirements ##
Python 3.6.3;
Numpy 1.18.1;
Sklearn 0.20.0;
Keras 2.3.1;
Pickle 4.0;
TensorFlow 2.1.0;
h5py 2.9.0;

## Files ##
1. contrast10_localization.py: The script to train CNN for contrast 10 dataset and its architecture is presented in the paper in the related publication section.
2. contrast50_localization.py: The script to train CNN for contrast 50 dataset and its architecture is presented in the paper in the related publication section.
3. sample_data.pkl: Example data for contrast10 dataset, including 10 21x21x21 3D microstructure.
4. sample_data50.pkl: Example data for contrast50 dataset, including 10 21x21x21 3D microstructure.

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
Z. Yang, Y. C. Yabansu, D. Jha, W.-keng Liao, A. N. Choudhary, S. R. Kalidindi, and A. Agrawal, “Establishing structure-property localization linkages for elastic deformation of three-dimensional high contrast composites using deep learning approaches,” Acta Materialia, vol. 166, pp. 335–345, 2019.

## Contact
Zijiang Yang (zyz293@ece.northwestern.edu);
Ankit Agrawal (ankitag@ece.northwestern.edu);
Alok Choudhary (choudhar@ece.northwestern.edu);

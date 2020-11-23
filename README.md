# Aerial_Cactus_Detection

This is my submission to the Aerial Cactus Competition on Kaggle (https://www.kaggle.com/c/aerial-cactus-identification)

Abstract:

To assess the impact of climate change on Earth's flora and fauna, it is vital to quantify how human activities such as logging, mining, and agriculture are impacting our protected natural areas. Researchers in Mexico have created the VIGIA project, which aims to build a system for autonomous surveillance of protected areas. A first step in such an effort is the ability to recognize the vegetation inside the protected areas. In this competition, you are tasked with creation of an algorithm that can identify a specific type of cactus in aerial imagery.


Dataset and Problem Description:

This dataset contains a large number of 32 x 32 thumbnail images containing aerial photos of a columnar cactus (Neobuxbaumia tetetzo). Kaggle has resized the images from the original dataset to make them uniform in size. The file name of an image corresponds to its id.

You must create a classifier capable of predicting whether an images contains a cactus.

Files:

train/ - the training set images
test/ - the test set images (you must predict the labels of these)
train.csv - the training set labels, indicates whether the image has a cactus (has_cactus = 1)
sample_submission.csv - a sample submission file in the correct format

Analysis Summary:

A simple 2DConvNet is created from scratch, achieving 98%+ accuracy.

# Plant Pathology 2020 - FGVC7

## Problematic

The goal of this project was to learn how to use the API of Keras in details and increase my practical knowledge of deep learning. Two year ago, I learned the basics of deep learning with keras in the deep learning specialization of Coursera. Later I have tried and learned how to use Fast.ai.
 
I also wanted to familiarized myself with more advanced notions of deep learning, such as the [one cycle policy](https://arxiv.org/pdf/1803.09820.pdf). To understand them in depth, I reimplemented and modified them using several sources.

The objectives of the ‘Plant Pathology Challenge’ were to train a model using images of leafs to accurately classify a given image into different diseased category or a healthy leaf and accurately distinguish between many diseases, sometimes more than one on a single leaf.

## Problems overcomed

The main difficulty to reimplement the one cycle in R on Keras was the scarcity of the starting ressources in R. While you can find a lot of ready to use code in Python for this purpose, the R community for keras is excessively smaller.

I also encountered a problem when trying to create an adequate model the first times. The option to generate batches of data from images in a directory to train a **multi-output model** using the flow_images_from_directory() function was actually missing from the documentation. This possibility was not present originally in Keras, but was pushed some times ago in the Python version of Keras. This possibility was never documented in the R documentation. This lead to an [accepted PR in the R repository of keras](https://github.com/rstudio/keras/pull/1085/commits/caa11f7d5a80d9edbd120f48176f19381b301854). 

## Material and methods


## Data

The data of this Rmarkdown came from the competition [Plant Pathology 2020 - FGVC7](https://www.kaggle.com/c/plant-pathology-2020-fgvc7). 


## Performance :

Public leaderboard : 0.922
Public leaderboad of the [kernel on kaggle](https://www.kaggle.com/cdk292/simple-resnet50-lr-finder-and-cyclic-lr-with-r) : 0.944

### Performance of the pre-trained model with custom head : 

![Train and Val loss and accuracy](https://github.com/Cdk29/Plant-Pathology/blob/master/resnet50-lr-finder-and-cyclic-lr-with-r_files/figure-gfm/plot_perforance-1.png)


### Performance of the fine_tuned model from epoch 14 : 

![Train and Val loss and accuracy](https://github.com/Cdk29/Plant-Pathology/blob/master/Fine-tuning_files/figure-gfm/history_model_epoch_8-1.png)

### Performance of the model from Epoch 5 after fine tuning of the layer 4 of filters :

![Train and Val loss and accuracy](https://github.com/Cdk29/Plant-Pathology/blob/master/Fine-tune-layer-4_files/figure-gfm/history_model_fine_tuned_res4a-1.png)

### Performance of the model from Epoch 5 after fine tuning of the layer 3 of filters :

![Train and Val loss and accuracy](https://github.com/Cdk29/Plant-Pathology/blob/master/Fine-tune-layer-4_files/figure-gfm/history_model_fine_tuned_res3a-1.png)



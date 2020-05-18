# Plant Pathology 2020 - FGVC7

## My experiments using the Keras API of R. 

Some times ago, I have learned the basics of deep learning with keras. I got more used to Fast.ai since then.
At the moment I relearn Keras with the book *Deep Learning with R*. 

In this github repo I keep track of my progress and update on a classifier for plant disease.

I also reimplement a learning rate finder and a one cycle policy, both for performance and learning goals.

## Data

The data of this Rmarkdown came from the competition [Plant Pathology 2020 - FGVC7](https://www.kaggle.com/c/plant-pathology-2020-fgvc7). 

## Current performance :

### Performance of the pre-trained model with custom head : 

![Train and Val loss and accuracy](https://github.com/Cdk29/Plant-Pathology/blob/master/resnet50-lr-finder-and-cyclic-lr-with-r_files/figure-gfm/plot_perforance-1.png)


### Performance of the fine_tuned model from epoch 14 : 

![Train and Val loss and accuracy](https://github.com/Cdk29/Plant-Pathology/blob/master/Fine-tuning_files/figure-gfm/history_model_epoch_8-1.png)

### Performance of the model from Epoch 5 after fine tuning of the layer 4 of filters :

![Train and Val loss and accuracy](https://github.com/Cdk29/Plant-Pathology/blob/master/Fine-tune-layer-4_files/figure-gfm/history_model_fine_tuned_res4a-1.png)


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

### Fine-tuning of the model generated at epoch 6 after unfreezing of the top layer :

![Train and Val loss and accuracy fine tuned model](https://github.com/Cdk29/Plant-Pathology/blob/master/resnet50-lr-finder-and-cyclic-lr-with-r_files/figure-gfm/plot_perforance_fine_tuned_from_epoch_6-1.png)

### Fine-tuning of the previous model at epoch 1 at a lower learning rate :

![Train and Val loss final](https://github.com/Cdk29/Plant-Pathology/blob/master/Fine-tuning_files/figure-gfm/history_model_epoch_6-1.png)

### Fine-tuning of the pre-trained model with custom head generated at epoch 8 after unfreezing of an other layer of filters :

![Train and Val loss fine tuned epoch 8](https://github.com/Cdk29/Plant-Pathology/blob/master/Fine-tuning_files/figure-gfm/history_model_epoch_8_filter4-1.png)

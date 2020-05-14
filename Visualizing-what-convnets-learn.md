Visualizing-what-convnets-learn
================
Etienne Rolland
04/05/2020

# Visualizing-what-convnets-learn

Most of the code came from this page came from this [chapter of the book
Deep learning with
R](https://jjallaire.github.io/deep-learning-with-r-notebooks/notebooks/5.4-visualizing-what-convnets-learn.nb.html).
The goal is to check what the differents conv\_nets has learned before
submitting. At the moment I can’t run it since the resnet has been built
using the sequential
    API.

## Set up

``` r
library(tidyverse)
```

    ## ── Attaching packages ────────────────────────────────────────────────────────────────────────────────────────────────── tidyverse 1.3.0 ──

    ## ✓ ggplot2 3.3.0     ✓ purrr   0.3.4
    ## ✓ tibble  3.0.1     ✓ dplyr   0.8.5
    ## ✓ tidyr   1.0.2     ✓ stringr 1.4.0
    ## ✓ readr   1.3.1     ✓ forcats 0.5.0

    ## ── Conflicts ───────────────────────────────────────────────────────────────────────────────────────────────────── tidyverse_conflicts() ──
    ## x dplyr::filter() masks stats::filter()
    ## x dplyr::lag()    masks stats::lag()

``` r
library(keras)
library(tensorflow)
library(reticulate)
```

``` r
use_python("/usr/bin/python3.5", required = TRUE)
tf <- tensorflow::tf
sess <- tf$Session()
```

``` r
image_path<-'plant-pathology-2020-fgvc7/images/'
```

``` r
#reticulate::virtualenv_install(packages="pandas") 
```

``` r
keras::use_implementation("keras")
keras::use_backend("tensorflow")
```

``` r
reticulate::py_config()
```

    ## python:         /usr/bin/python3.5
    ## libpython:      /usr/lib/python3.5/config-3.5m-x86_64-linux-gnu/libpython3.5m.so
    ## pythonhome:     //usr://usr
    ## version:        3.5.2 (default, Apr 16 2020, 17:47:17)  [GCC 5.4.0 20160609]
    ## numpy:          /home/proprietaire/.local/lib/python3.5/site-packages/numpy
    ## numpy_version:  1.18.3
    ## tensorflow:     /home/proprietaire/.local/lib/python3.5/site-packages/tensorflow
    ## 
    ## NOTE: Python version was forced by use_python function

## Loading model

``` r
conv_base <- application_resnet50(weights = 'imagenet', include_top = FALSE, input_shape = c(448, 448, 3))
```

``` r
unfreeze_weights(conv_base, from="res5a_branch2a")
```

``` r
model <- keras_model_sequential() %>% 
        conv_base %>% 
        layer_global_max_pooling_2d() %>% 
        layer_batch_normalization() %>%
        layer_dropout(rate=0.5) %>%
        layer_dense(units=4, activation="softmax")
```

``` r
model
```

    ## Model
    ## ________________________________________________________________________________
    ## Layer (type)                        Output Shape                    Param #     
    ## ================================================================================
    ## resnet50 (Model)                    (None, 14, 14, 2048)            23587712    
    ## ________________________________________________________________________________
    ## global_max_pooling2d_1 (GlobalMaxPo (None, 2048)                    0           
    ## ________________________________________________________________________________
    ## batch_normalization_1 (BatchNormali (None, 2048)                    8192        
    ## ________________________________________________________________________________
    ## dropout_1 (Dropout)                 (None, 2048)                    0           
    ## ________________________________________________________________________________
    ## dense_1 (Dense)                     (None, 4)                       8196        
    ## ================================================================================
    ## Total params: 23,604,100
    ## Trainable params: 14,988,292
    ## Non-trainable params: 8,615,808
    ## ________________________________________________________________________________

``` r
checkpoint_dir <- "fine_tuned_models/"

model %>% load_model_weights_hdf5(
  file.path(checkpoint_dir,"Fine_tuned_Resnet50_res5a_05.hdf5")
)
```

# Visualization

``` r
img_path<-"plant-pathology-2020-fgvc7/images/Train_1.jpg"
```

NB : it is really important to use imagenet\_preprocess\_net(),
otherwise the predictions does not reflect what the model has learned.

``` r
img <- image_load(img_path, target_size = c(448, 448)) %>% 
  # Array of shape (224, 224, 3)
  image_to_array() %>% 
  # Adds a dimension to transform the array into a batch of size (1, 224, 224, 3)
  array_reshape(dim = c(1, 448, 448, 3)) %>% 
  # Preprocesses the batch (this does channel-wise color normalization)
  imagenet_preprocess_input()
```

``` r
preds <- model %>% predict(img)
print(c("healthy", "multiple_diseases", "rust", "scab"))
```

    ## [1] "healthy"           "multiple_diseases" "rust"             
    ## [4] "scab"

``` r
print(preds)
```

    ##           [,1]        [,2]       [,3]     [,4]
    ## [1,] 0.4650005 0.001716625 0.01439878 0.518884

``` r
summary(model)
```

    ## ________________________________________________________________________________
    ## Layer (type)                        Output Shape                    Param #     
    ## ================================================================================
    ## resnet50 (Model)                    (None, 14, 14, 2048)            23587712    
    ## ________________________________________________________________________________
    ## global_max_pooling2d_1 (GlobalMaxPo (None, 2048)                    0           
    ## ________________________________________________________________________________
    ## batch_normalization_1 (BatchNormali (None, 2048)                    8192        
    ## ________________________________________________________________________________
    ## dropout_1 (Dropout)                 (None, 2048)                    0           
    ## ________________________________________________________________________________
    ## dense_1 (Dense)                     (None, 4)                       8196        
    ## ================================================================================
    ## Total params: 23,604,100
    ## Trainable params: 14,988,292
    ## Non-trainable params: 8,615,808
    ## ________________________________________________________________________________

Submit\_kaggle
================
Etienne Rolland
04/05/2020

# Submission

Notebook to load a model, made the prediction on the test set and submit
to
    kaggle.

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

Last update of the Rmd doc Fine-tuning save only weights
:

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

## Unfreezing the model

Following line to got the name of the layer we want to unfreeze
(res5a\_branch2a). We load the model from the epoch 14 of from the
notebook resnet50-lr-finder-and-cyclic-lr-with-r.

``` r
checkpoint_dir <- "fine_tuned_models/"

model %>% load_model_weights_hdf5(
  file.path(checkpoint_dir,"Fine_tuned_Resnet50_res5a_05.hdf5")
)
```

## Test set

``` r
test<-read_csv('plant-pathology-2020-fgvc7//test.csv')
```

    ## Parsed with column specification:
    ## cols(
    ##   image_id = col_character()
    ## )

``` r
test$image_id <- paste0(test$image_id, ".jpg")
head(test)
```

    ## # A tibble: 6 x 1
    ##   image_id  
    ##   <chr>     
    ## 1 Test_0.jpg
    ## 2 Test_1.jpg
    ## 3 Test_2.jpg
    ## 4 Test_3.jpg
    ## 5 Test_4.jpg
    ## 6 Test_5.jpg

``` r
image_path<-'plant-pathology-2020-fgvc7/images/'
```

``` r
test_generator <- flow_images_from_dataframe(dataframe = test, 
                                              directory = image_path,
                                              class_mode = NULL,
                                              x_col = "image_id",
                                              y_col = NULL,
                                              target_size = c(448, 448),
                                              batch_size=1)
```

``` r
num_test_images<-1821
```

``` r
pred <- model %>% predict_generator(test_generator, steps=num_test_images)
```

``` r
head(pred)
```

    ##           [,1]         [,2]         [,3]        [,4]
    ## [1,] 0.2197821 3.164808e-03 1.837774e-04 0.776869416
    ## [2,] 0.9355683 7.748817e-03 3.006107e-03 0.053676896
    ## [3,] 0.3568218 1.989747e-02 9.018349e-05 0.623190522
    ## [4,] 0.9894828 7.860666e-05 1.131985e-04 0.010325499
    ## [5,] 0.9992095 1.153802e-04 1.316311e-05 0.000661904
    ## [6,] 0.9687287 5.929744e-05 2.955537e-02 0.001656756

``` r
sample_submission<-read.csv("plant-pathology-2020-fgvc7/sample_submission.csv")
```

``` r
pred<-as.data.frame(cbind("id", pred))
colnames(pred)<-colnames(sample_submission)
head(pred)
```

    ##   image_id           healthy    multiple_diseases                 rust
    ## 1       id 0.219782084226608  0.00316480780020356  0.00018377740343567
    ## 2       id 0.935568273067474  0.00774881709367037  0.00300610740669072
    ## 3       id 0.356821805238724   0.0198974683880806 9.01834937394597e-05
    ## 4       id 0.989482760429382 7.86066593718715e-05 0.000113198548206128
    ## 5       id 0.999209523200989 0.000115380164061207 1.31631068143179e-05
    ## 6       id  0.96872866153717 5.92974356550258e-05   0.0295553747564554
    ##                   scab
    ## 1    0.776869416236877
    ## 2   0.0536768957972527
    ## 3    0.623190522193909
    ## 4   0.0103254988789558
    ## 5 0.000661903992295265
    ## 6  0.00165675557218492

``` r
pred[,1]<-gsub(".jpg","",test$image_id)
head(pred)
```

    ##   image_id           healthy    multiple_diseases                 rust
    ## 1   Test_0 0.219782084226608  0.00316480780020356  0.00018377740343567
    ## 2   Test_1 0.935568273067474  0.00774881709367037  0.00300610740669072
    ## 3   Test_2 0.356821805238724   0.0198974683880806 9.01834937394597e-05
    ## 4   Test_3 0.989482760429382 7.86066593718715e-05 0.000113198548206128
    ## 5   Test_4 0.999209523200989 0.000115380164061207 1.31631068143179e-05
    ## 6   Test_5  0.96872866153717 5.92974356550258e-05   0.0295553747564554
    ##                   scab
    ## 1    0.776869416236877
    ## 2   0.0536768957972527
    ## 3    0.623190522193909
    ## 4   0.0103254988789558
    ## 5 0.000661903992295265
    ## 6  0.00165675557218492

``` r
write.csv(pred, file='submission.csv', row.names=FALSE, quote=FALSE)
```

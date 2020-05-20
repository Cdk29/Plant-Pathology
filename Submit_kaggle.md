Submit\_kaggle
================
Etienne Rolland
04/05/2020

# Submission

Notebook to load a model, made the predictions on the test set and
create the submissions for
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
unfreeze_weights(conv_base, from="res3a_branch2a")
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
    ## Trainable params: 23,320,068
    ## Non-trainable params: 284,032
    ## ________________________________________________________________________________

``` r
checkpoint_dir <- "layer_3_models"

model %>% load_model_weights_hdf5(
  file.path(checkpoint_dir,"Resnet50_res3a_04.hdf5")
)
```

## Test set

``` r
test<-readr::read_csv('plant-pathology-2020-fgvc7//test.csv')
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

With shuffle = FALSE to not mix images and got the right order in the
predictions.

``` r
test_generator <- flow_images_from_dataframe(dataframe = test, 
                                              directory = image_path,
                                              class_mode = NULL,
                                              x_col = "image_id",
                                              y_col = NULL,
                                              target_size = c(448, 448),
                                              shuffle = FALSE,
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

    ##              [,1]         [,2]         [,3]         [,4]
    ## [1,] 3.486477e-02 2.562768e-01 7.015501e-01 7.308291e-03
    ## [2,] 4.600692e-04 4.119723e-03 9.953472e-01 7.301690e-05
    ## [3,] 1.366183e-06 2.282038e-05 4.013441e-07 9.999754e-01
    ## [4,] 9.999837e-01 3.714516e-07 1.490722e-05 1.030114e-06
    ## [5,] 8.004557e-08 3.793443e-05 9.999619e-01 7.255818e-08
    ## [6,] 9.905578e-01 7.911299e-05 9.222937e-03 1.401728e-04

``` r
sample_submission<-read.csv("plant-pathology-2020-fgvc7/sample_submission.csv")
```

``` r
pred<-as.data.frame(cbind("id", pred))
colnames(pred)<-colnames(sample_submission)
head(pred)
```

    ##   image_id              healthy    multiple_diseases                 rust
    ## 1       id   0.0348647721111774    0.256276845932007    0.701550126075745
    ## 2       id 0.000460069160908461  0.00411972310394049    0.995347201824188
    ## 3       id 1.36618257329246e-06 2.28203825827222e-05 4.01344067313403e-07
    ## 4       id    0.999983668327332 3.71451591263394e-07 1.49072220665403e-05
    ## 5       id  8.0045573724874e-08 3.79344310204033e-05    0.999961853027344
    ## 6       id    0.990557789802551 7.91129859862849e-05  0.00922293681651354
    ##                   scab
    ## 1  0.00730829080566764
    ## 2 7.30169049347751e-05
    ## 3    0.999975442886353
    ## 4 1.03011393548513e-06
    ## 5 7.25581799088104e-08
    ## 6 0.000140172778628767

``` r
pred[,1]<-gsub(".jpg","",test$image_id)
head(pred)
```

    ##   image_id              healthy    multiple_diseases                 rust
    ## 1   Test_0   0.0348647721111774    0.256276845932007    0.701550126075745
    ## 2   Test_1 0.000460069160908461  0.00411972310394049    0.995347201824188
    ## 3   Test_2 1.36618257329246e-06 2.28203825827222e-05 4.01344067313403e-07
    ## 4   Test_3    0.999983668327332 3.71451591263394e-07 1.49072220665403e-05
    ## 5   Test_4  8.0045573724874e-08 3.79344310204033e-05    0.999961853027344
    ## 6   Test_5    0.990557789802551 7.91129859862849e-05  0.00922293681651354
    ##                   scab
    ## 1  0.00730829080566764
    ## 2 7.30169049347751e-05
    ## 3    0.999975442886353
    ## 4 1.03011393548513e-06
    ## 5 7.25581799088104e-08
    ## 6 0.000140172778628767

``` r
write.csv(pred, file='submission_resnet_layer_3.csv', row.names=FALSE, quote=FALSE)
```

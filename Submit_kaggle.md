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
unfreeze_weights(conv_base, from="res4a_branch2a")
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
    ## Trainable params: 22,096,900
    ## Non-trainable params: 1,507,200
    ## ________________________________________________________________________________

``` r
checkpoint_dir <- "layer_4_models"

model %>% load_model_weights_hdf5(
  file.path(checkpoint_dir,"Resnet50_res4a_07.hdf5")
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
    ## [1,] 6.891629e-02 8.792665e-01 3.913462e-02 1.268255e-02
    ## [2,] 1.208475e-03 2.334637e-03 9.870152e-01 9.441637e-03
    ## [3,] 2.397761e-06 2.471258e-07 3.480826e-08 9.999974e-01
    ## [4,] 9.999882e-01 2.323823e-07 9.911227e-06 1.621315e-06
    ## [5,] 8.672209e-06 8.631199e-05 9.997025e-01 2.024184e-04
    ## [6,] 9.905082e-01 4.180576e-03 8.257210e-05 5.228636e-03

``` r
sample_submission<-read.csv("plant-pathology-2020-fgvc7/sample_submission.csv")
```

``` r
pred<-as.data.frame(cbind("id", pred))
colnames(pred)<-colnames(sample_submission)
head(pred)
```

    ##   image_id              healthy    multiple_diseases                 rust
    ## 1       id   0.0689162909984589    0.879266500473022   0.0391346178948879
    ## 2       id  0.00120847520884126  0.00233463663607836    0.987015247344971
    ## 3       id 2.39776136368164e-06 2.47125797159242e-07 3.48082558332408e-08
    ## 4       id    0.999988198280334 2.32382348031024e-07  9.9112266980228e-06
    ## 5       id 8.67220933287172e-06 8.63119930727407e-05    0.999702513217926
    ## 6       id    0.990508198738098  0.00418057572096586 8.25720999273472e-05
    ##                   scab
    ## 1   0.0126825487241149
    ## 2  0.00944163650274277
    ## 3     0.99999737739563
    ## 4 1.62131516390218e-06
    ## 5 0.000202418406843208
    ## 6  0.00522863632068038

``` r
pred[,1]<-gsub(".jpg","",test$image_id)
head(pred)
```

    ##   image_id              healthy    multiple_diseases                 rust
    ## 1   Test_0   0.0689162909984589    0.879266500473022   0.0391346178948879
    ## 2   Test_1  0.00120847520884126  0.00233463663607836    0.987015247344971
    ## 3   Test_2 2.39776136368164e-06 2.47125797159242e-07 3.48082558332408e-08
    ## 4   Test_3    0.999988198280334 2.32382348031024e-07  9.9112266980228e-06
    ## 5   Test_4 8.67220933287172e-06 8.63119930727407e-05    0.999702513217926
    ## 6   Test_5    0.990508198738098  0.00418057572096586 8.25720999273472e-05
    ##                   scab
    ## 1   0.0126825487241149
    ## 2  0.00944163650274277
    ## 3     0.99999737739563
    ## 4 1.62131516390218e-06
    ## 5 0.000202418406843208
    ## 6  0.00522863632068038

``` r
write.csv(pred, file='submission_resnet_layer_4.csv', row.names=FALSE, quote=FALSE)
```

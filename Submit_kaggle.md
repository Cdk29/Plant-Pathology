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

``` r
checkpoint_dir <- "fine_tuned_models/"

model %>% load_model_weights_hdf5(
  file.path(checkpoint_dir,"Fine_tuned_Resnet50_res5a_05.hdf5")
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
    ## [1,] 5.742195e-02 1.828957e-01 6.849197e-01 7.476266e-02
    ## [2,] 2.487763e-03 1.961526e-01 7.318514e-01 6.950817e-02
    ## [3,] 1.509694e-06 5.209986e-06 1.279279e-07 9.999932e-01
    ## [4,] 9.999964e-01 3.218626e-07 3.054437e-06 1.363534e-07
    ## [5,] 1.240704e-05 3.909111e-04 9.994234e-01 1.733726e-04
    ## [6,] 9.898796e-01 6.273850e-04 5.344050e-03 4.148873e-03

``` r
sample_submission<-read.csv("plant-pathology-2020-fgvc7/sample_submission.csv")
```

``` r
pred<-as.data.frame(cbind("id", pred))
colnames(pred)<-colnames(sample_submission)
head(pred)
```

    ##   image_id              healthy    multiple_diseases                 rust
    ## 1       id   0.0574219450354576    0.182895749807358    0.684919655323029
    ## 2       id  0.00248776306398213     0.19615264236927    0.731851398944855
    ## 3       id 1.50969390233513e-06 5.20998628417146e-06 1.27927947346507e-07
    ## 4       id    0.999996423721313 3.21862557939312e-07 3.05443722936616e-06
    ## 5       id 1.24070375022711e-05 0.000390911067370325    0.999423384666443
    ## 6       id    0.989879608154297  0.00062738498672843  0.00534405000507832
    ##                   scab
    ## 1   0.0747626572847366
    ## 2   0.0695081725716591
    ## 3    0.999993205070496
    ## 4 1.36353420998603e-07
    ## 5  0.00017337262397632
    ## 6  0.00414887256920338

``` r
pred[,1]<-gsub(".jpg","",test$image_id)
head(pred)
```

    ##   image_id              healthy    multiple_diseases                 rust
    ## 1   Test_0   0.0574219450354576    0.182895749807358    0.684919655323029
    ## 2   Test_1  0.00248776306398213     0.19615264236927    0.731851398944855
    ## 3   Test_2 1.50969390233513e-06 5.20998628417146e-06 1.27927947346507e-07
    ## 4   Test_3    0.999996423721313 3.21862557939312e-07 3.05443722936616e-06
    ## 5   Test_4 1.24070375022711e-05 0.000390911067370325    0.999423384666443
    ## 6   Test_5    0.989879608154297  0.00062738498672843  0.00534405000507832
    ##                   scab
    ## 1   0.0747626572847366
    ## 2   0.0695081725716591
    ## 3    0.999993205070496
    ## 4 1.36353420998603e-07
    ## 5  0.00017337262397632
    ## 6  0.00414887256920338

``` r
write.csv(pred, file='submission.csv', row.names=FALSE, quote=FALSE)
```

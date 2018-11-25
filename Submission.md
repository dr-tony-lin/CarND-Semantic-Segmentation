# Semantic Segmentation Project Submission

## Content of the Submission, Dependency and Usage

This submission includes the following python files:
* main.py: the main program modified from the original main.py
* helper.py: Helper functions enhanced from the original helper.py
* project_tests.py: the test functions modified from the original project_tests to accommodate 2 and 3 classes
* runs: some output images from the runs

### Dependency

. The proram has been tested with Tensorflow 1.10 on Windows, Cuda 9.0 is required
. For Tensorflow 1.11+, it does not work with Cuda 9.2 or Cuda 10.
. Scikit-image is required for image augmentation

### Usage

Run the following command to run the project:
```
python main.py
```

## The Architecture

Two variations of the original work, *Fully Convolutional Networks forSemantic Segmentation *, were experimented.

### Variation 1 

This is implemented in function *layers_regularizer*, it is composed of:

* The encoder: a pre-trained Vgg 16 model from the input layer to layer 7
* The decoder: 4 convolutional layers, with:
    * A one by one fully convolutional layer of shape:
        * 1x1x21: as the original paper does
        * 1x1x1024. 1x1x1024: to experiment if a larger number of channeld could yield a better training result
    * A 2x2x512 convolutional layer with skip from Vgg's layer 4
    * A 4x4x256 convolutional layer with skip from Vgg's layr 3
    * A final 32x32x2 (for 2 classes) or 32x32x3 (for 3 classes) convolutional layer

### Variation 2

This is implemented in function *layers_dropout*, it is the same as variation 1 except that instead of L2 regularizer, a dropout layer is insert between the layers after the one by one fully convolutional layer

### Variation 3

Will a deeper decoder yield better result? the variation 3 attempts to experiment this with 1x1, dropout, 2x2, dropout, 4x4, dropout, 16x16, dropout, and 32x32 convolutional layers. 

### Optimization

The outputs from the last convolutional layer, and the ground truth labels are used to the compute softmax cross entropy.
Then Adam optimizer is used to minimize the cost.

Parameters that were used in the training include:

* Epochs: 250
* Batch size: 16
* Dropout keep proability: 0.5
* L2 regularizer scale: 1e-5
* Learning rate: 1e-4

### Image Augmentation

I also have implemented image augmentation using random affine transformation using scikit-image. The augmentation parameters are: 

* Scale: 95% to 105%
* Rotate: -0.03PI to 0.03PI
* Shear: -0.03PI to 0.03PI
* Translation: -15 to 15 pixels vertically and horizontally

### Project Tests

In order to support semantic segmentation with 3 classes, some project_tests functions have been modified to take the number of classes as an extra parameter.

## Training

The training and testing was performed on a Windows 10 system with Nvidia GTX 1080. The Udacity workspace was unfortunately not useful for the training as the training required several hours, and the workspace's inactive timeout is 30 minutes. 

### Training Time

On my GTX 1080 system, the training time per epoch is aroung 23 seconds without image augmentation, and 29 seconds with image augmentation.
On the Udacity workspace, it tooks around 75 seconds without augmentation, and over 84 seconds with image augmentation.

### Evaluation

Unfortunately, I was not able to evaluate the trained model against the test images since there is no ground-truth provided for the test images. Hyper parameter tuning was performed by visually inspecting the segmentation results. This is of course a highly questionable approach.

### Training Results

#### Segmented Images
The images produced by the segmentation are stored under * runs * folder, there will be a folder of format %Y-%m-%d-%H-%M-%S for each run.

#### Trained Models
For a trained models whose training loss is less than a threshold, it will be saved under * model/%Y-%m-%d-%H-%M-%S * folder. Each saved model has a name of format * model-loss-epoch *, where:

* loss: is the training loss * 10000
* epoch: is the epoch that the model was saved

#### Augmented Images
When image augmentation is applied, I also stored the augmentated images, ground truth images under * data/data_road/augmented/%Y-%m-%d-%H-%M-%S * folder.

Unfortunately, I was not able to train the model with image augmentation as I have to travel, and had difficult to use Udacity's workspace due to inactive timeout issue.

#### Labels of Augmented Images
For 3-class segmentation, the label images will also be stored under * data/data_road/augmented/%Y-%m-%d-%H-%M-%S-labels * folder.

## Results

By visual inspecting the result images, as a proper evaluation could not be done without ground truth data for the test images, it seems to suggest the following:

1. Regularization using dropout may produce slightly better than using L2 regularization, though only marginally.
2. Slightly lower training loss could be obtained by increasing the number of output channels at the one by one convolutional layer. However it may also result in over-fiting, and a larger regularization term (L2 scale or dropout probability) may be required.


The following table presents some of the resulting images:

|         Variation 1                  		|        Variation 2	                    |        Variation 2	                    |
|:-----------------------------------------:|:-----------------------------------------:|:-----------------------------------------:|
| [um_000000.png](runs/2018-11-22-01-37-50-reg-21/um_000000.png) |[um_000000.png](runs/2018-11-21-21-46-46-dropout-1024/um_000000.png)|  [um_000000.png](runs/2018-11-21-08-55-07-deep-1024/um_000000.png) | 
| [um_000010.png](runs/2018-11-22-01-37-50-reg-21/um_000010.png) |[um_000010.png](runs/2018-11-21-21-46-46-dropout-1024/um_000010.png)|  [um_000010.png](runs/2018-11-21-08-55-07-deep-1024/um_000010.png) | 
| [umm_000000.png](runs/2018-11-22-01-37-50-reg-21/umm_000000.png) |[umm_000000.png](runs/2018-11-21-21-46-46-dropout-1024/umm_000000.png)|  [umm_000000.png](runs/2018-11-21-08-55-07-deep-1024/umm_000000.png) | 
| [umm_000010.png](runs/2018-11-22-01-37-50-reg-21/umm_000010.png) |[umm_000010.png](runs/2018-11-21-21-46-46-dropout-1024/umm_000010.png)|  [umm_000010.png](runs/2018-11-21-08-55-07-deep-1024/umm_000010.png) | 
| [uu_000000.png](runs/2018-11-22-01-37-50-reg-21/uu_000000.png) |[uu_000000.png](runs/2018-11-21-21-46-46-dropout-1024/uu_000000.png)|  [uu_000000.png](runs/2018-11-21-08-55-07-deep-1024/uu_000000.png) | 
| [uu_000010.png](runs/2018-11-22-01-37-50-reg-21/uu_000010.png) |[uu_000010.png](runs/2018-11-21-21-46-46-dropout-1024/uu_000010.png)|  [uu_000010.png](runs/2018-11-21-08-55-07-deep-1024/uu_000010.png) | 


The following table shows some augmented images:

|         Augmented Image             		|        Augmented Ground Truth Image       |        Ground Truth Labels	            |
|:-----------------------------------------:|:-----------------------------------------:|:-----------------------------------------:|
| [um_000000.png](runs/augmented/2018-11-25-14-53-50/um_000000.png) |[um_road_000000.png](runs/augmented/2018-11-25-14-53-50/um_road_000000.png)|  [um_road_000000.png](runs/augmented/2018-11-25-14-53-50-labels/um_road_000000.png) | 
| [umm_road_000000.png](runs/augmented/2018-11-25-14-53-50/umm_000000.png) |[umm_road_000000.png](runs/augmented/2018-11-25-14-53-50/umm_road_000000.png)|  [umm_road_000000.png](runs/augmented/2018-11-25-14-53-50-labels/umm_road_000000.png) | 
| [uu_000000.png](runs/augmented/2018-11-25-14-53-50/uu_000000.png) |[uu_road_000000.png](runs/augmented/2018-11-25-14-53-50/uu_road_000000.png)|  [uu_road_000000.png](runs/augmented/2018-11-25-14-53-50-labels/uu_road_000000.png) | 

### TO DO

The following additional works could be carried out:

1. Training the model against the cityscapes database, www.cityscapes-dataset.com requires one to provide information for obtaining the dataset. 
2. Training the model with augmented images
3. Since the test images don't have ground truth data, I could not evaluate the model after each epoch against test images. This may be done at a later time
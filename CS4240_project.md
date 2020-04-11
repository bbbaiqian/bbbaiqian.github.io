# CS4240 Reproducibility Project

## _Between-class Learning for Image Classification_

In this project, we reproduce a learning method for image classification called between-class learning (BC learning). Basically, between-class images are generated through mixing two images from different classes with a random ratio. The aim of BC learning is to train a model which takes the mixed image as input and can output the mixing ratio. This approach is originally designed for digital signals such as sound, and the authors demonstrated that treating input data as waveforms can also work on images and further improve the generalization ability of models. 

The original paper proposed two mixing methods. The first is to simply mix two images using internal divisions, which can improve the classification results compared to using a single image and its label. The second is to treat images as waveforms and take the difference of image energies into consideration to generate ratios, which is called BC+ learning and performs even better.

In our project, we try to reproduce the classification results for CIFAR-10 trained on a 11-layer CNN (Table 1). The reproduction consist of results for standard learning (single image with single label), BC learning (mixed image with mixed label using internal deivsions), as well as BC+ learning (mixed image with mixed label using waveform method). Moreover, instead of training with Chainer, which is used in the publicly available code of the paper, we port the original released code to PyTorch.

Table: (\#tab:tab-comp) Architecture of the 11-layer CNN

| Layer                         |kernel size| stride    | padding   | # filters       | Data type  |
| :---------------------------- |    :----: | :----:    |  :----:   |   :----:        |    :----:  |
|Input                          |           |           |           |                 |  (3,32,32) |
|conv1-1 <br> conv1-2 <br> pool1|3<br>3<br>2|1<br>1<br>2|1<br>1 |64<br>64<br>     | <br> <br> (64,16,16)| 
|conv2-1 <br> conv2-2 <br> pool2|3<br>3<br>2|1<br>1<br>2|1<br>1 |128<br>128<br>   | <br> <br> (128,8,8)| 
|conv3-1 <br> conv3-2 <br> conv3-3 <br> conv3-4 <br> pool3|3<br>3<br>3<br>3<br>2|1<br>1<br>1<br>1<br>2|1<br>1<br>1<br>1 |256<br>256<br>256<br>256<br>   | <br> <br> <br> <br>(256,4,4)|
|fc4 <br> fc5 <br> fc6          |           |           |           |1024<br>1024<br># classes|(1024,)<br>(1024,)<br># classes|

Besides the results mentioned above, we also reproduce all the ablation analysis for CIFAR-10, including data augmentation, mixing method, number of mixed classes, etc. Lastly, a new dataset called Caltech101 that is not mentioned in the paper is used to test this image classification method. 

### Port to PyTorch

When porting the original Chainer code to PyTorch, we looked into the correspondences and differences between the implementation details of such two libraries. Generally, they provide similar methods for getting access to existing image datasets (e.g. CIFAR-10), constructing neural networks, realizing forward and back propagation for a given image. Correspondences of some key functions are easily found, like `optimizer.update()` in Chainer corresponding to `optimizer.step()` in PyTorch, and they will not be documented in detail here. Here we focus on the most important differences when transferring Chainer code to PyTorch, which might have implications on the reproduced classification results.

* Weight initialization for fully connected layers

The authors of this paper 

```
Syntax highlighted code block
```

* Optimization method

```
Syntax highlighted code block
```

* Learning rate schedule

```
Syntax highlighted code block
```

### Results for CIFAR-10 on 11-layer CNN

the average error rate (%) and standard deviation of 5 or 10 trials:

| Learning method| Original paper | Our reproduction |
|      :----:    |     :----:     |      :----:      |
|     Standard   |   6.07 ± 0.04  |    6.59 ± 0.15   | 
|        BC      |   5.40 ± 0.07  |    5.67 ± 0.04   | 
|       BC+      |   5.22 ± 0.04  |    5.51 ± 0.11   | 


### Ablation analysis for CIFAR-10 on 11-layer CNN

No data augmentation:

| Learning method| Original paper | Our reproduction |
|      :----:    |     :----:     |      :----:      |
|     Standard   |   9.68 (6.07)  |    9.64 (6.59)   | 
|        BC      |   8.38 (5.40)  |    8.10 (5.67)   |  

Other comparisons:

|   Comparison of   |     Setting    |  Original paper  |  Our reproduction |
|       :----       |     :----      |      :----:      |       :----:      |
| Mixing method     |None (BC) <br> a <br> a+b <br> a+b+c (BC+) <br> b+c|5.40 <br> 5.45 <br> **5.17** <br> 5.22 <br> 5.26|5.67 <br> 5.66 <br> 5.59 <br> **5.51** <br> 5.61| 
| Label             |Single <br> Multi <br> Ratio (BC+)|6.35 <br> 6.05 <br> **5.22**|6.60 <br> 6.45 <br> **5.51**|
| # mixed classes   |N = 1 <br> N = 1 or 2 <br> N = 2 (BC+) <br> N = 2 or 3 <br> N = 3|5.98 <br> 5.31 <br> 5.22 <br> **5.15** <br> 5.32|6.20 <br> 5.55 <br> 5.51 <br> 5.48 <br> xx |
| Where to mix      |Input (BC) <br> pool1 <br> pool2 <br> pool3 <br> fc4 <br> fc5|**5.40** <br> 5.74 <br> 6.52 <br> 6.05 <br> 6.05 <br> 6.12|5.67 <br> xx <br> xx <br> xx <br> xx <br> xx|


### Results for Caltech101 on 11-layer CNN


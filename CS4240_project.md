# CS4240 Reproducibility Project

## _Between-class Learning for Image Classification_

In this project, we reproduce a learning method for image classification called between-class learning (BC learning). Basically, between-class images are generated through mixing two images from different classes with a random ratio. The aim of BC learning is to train a model which takes the mixed image as input and can output the mixing ratio. This approach is originally designed for digital signals such as sound, and the authors demonstrated that treating input data as waveforms can also work on images and further improve the generalization ability of models. 

The original paper proposed two mixing methods. The first is to simply mix two images using internal divisions, which can improve the classification results compared to using a single image and its label. The second is to treat images as waveforms and take the difference of image energies into consideration to generate ratios, which is called BC+ learning and performs even better.

In our project, we try to reproduce the classification results for CIFAR-10 trained on a 11-layer CNN. The reproduction consist of results for standard learning (single image with single label), BC learning (mixed image with mixed label using internal deivsions), as well as BC+ learning (mixed image with mixed label using waveform method). Moreover, instead of training with Chainer, which is used in the publicly available code of the paper, we port the original released code to PyTorch.

Besides the results mentioned above, we also reproduce all the ablation analysis for CIFAR-10, including data augmentation, mixing method, number of mixed classes, etc. Lastly, a new dataset called Caltech101 that is not mentioned in the paper is used to test this image classification method. 

### Port to PyTorch

Chainer and PyTorch 
find correspondences between them.

```
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
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
| Label             |Single <br> Multi <br> Ratio (proposed)|6.35 <br> 6.05 <br> **5.22**|6.60 <br> 6.45 <br> **5.51**|
| # mixed classes   |N = 1 <br> N = 1 or 2 <br> N = 2 (proposed) <br> N = 2 or 3 <br> N = 3|5.98 <br> 5.31 <br> 5.22 <br> **5.15** <br> 5.32|6.20 <br> 5.55 <br> 5.51 <br> xx <br> xx |
| Where to mix      |Input (proposed) <br> pool1 <br> pool2 <br> pool3 <br> fc4 <br> fc5|**5.40** <br> 5.74 <br> 6.52 <br> 6.05 <br> 6.05 <br> 6.12|5.67 <br> xx <br> xx <br> xx <br> xx <br> xx|


### Results for Caltech101 on 11-layer CNN

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/bbbaiqian/bbbaiqian.github.io/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

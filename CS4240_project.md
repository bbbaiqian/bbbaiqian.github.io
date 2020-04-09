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

the average error of 5 or 10 trials.

|                |          Error rate (%) in        |
| Learning method| Original paper | Our reproduction |
|      :----:    |     :----:     |      :----:      |
|     Standard   |  6.07+_ 0.04   |   6.59 +_ 0.15   | 
|        BC      |  5.40+_ 0.07   |   5.67 +_ 0.04   | 
|       BC+      |  5.22+_ 0.04   |   5.51 +_ 0.11   | 


### Ablation analysis for CIFAR-10 on 11-layer CNN

### Results for Caltech101 on 11-layer CNN

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/bbbaiqian/bbbaiqian.github.io/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

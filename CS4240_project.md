# CS4240 Reproducibility Project

## _Between-class Learning for Image Classification_

In this project, we reproduce a learning method for image classification called between-class learning (BC learning). Basically, between-class images are generated through mixing two images from different classes with a random ratio. The aim of BC learning is to train a model which takes the mixed image as input and can output the mixing ratio. This approach is originally designed for digital signals such as sound, and the authors demonstrated that treating input data as waveforms can also work on images and further improve the generalization ability of models. 

![Figure1](https://github.com/bbbaiqian/bbbaiqian.github.io/blob/master/figs/bc_learning.png)
Figure 1. Illustration of mixing two images

The original paper proposed two mixing methods. The first is to simply mix two images using internal divisions, which can improve the classification results compared to using a single image and its label. The second is to treat images as waveforms and take the difference of image energies into consideration to generate ratios, which is called BC+ learning and performs even better.

In our project, we tried to reproduce the classification results for CIFAR-10 trained on a 11-layer CNN (Table 1). The reproduction consist of results for standard learning (single image with single label), BC learning (mixed image with mixed label using internal deivsions), as well as BC+ learning (mixed image with mixed label using waveform method). Moreover, instead of training with Chainer, which is used in the publicly available code of the paper, we port the original released code to PyTorch.

Table 1: Architecture of the 11-layer CNN

| Layer                         |kernel size| stride    | padding   | # filters       | Data type  |
| :---------------------------- |    :----: | :----:    |  :----:   |   :----:        |    :----:  |
|Input                          |           |           |           |                 |  (3, 32, 32) |
|conv1-1 <br> conv1-2 <br> pool1|3<br>3<br>2|1<br>1<br>2|1<br>1<br>-|64<br>64<br>-    | <br> <br> (64, 16, 16)| 
|conv2-1 <br> conv2-2 <br> pool2|3<br>3<br>2|1<br>1<br>2|1<br>1<br>-|128<br>128<br>-  | <br> <br> (128,8,8)| 
|conv3-1 <br> conv3-2 <br> conv3-3 <br> conv3-4 <br> pool3|3<br>3<br>3<br>3<br>2|1<br>1<br>1<br>1<br>2|1<br>1<br>1<br>1<br>-|256<br>256<br>256<br>256<br>-  | <br> <br> <br> <br>(256, 4, 4)|
|fc4 <br> fc5 <br> fc6          |           |           |           |1024<br>1024<br># classes|(1024,)<br>(1024,)<br># classes|

Besides the results mentioned above, we also reproduced all the ablation analysis for CIFAR-10, including data augmentation, mixing method, number of mixed classes, etc. Lastly, a new dataset called Caltech101 that is not mentioned in the paper is used to test this image classification method. 

### Port to PyTorch

When porting the original Chainer code to PyTorch, we looked into the correspondences and differences between the implementation details of such two libraries. Generally, they provide similar methods for getting access to existing image datasets (e.g. CIFAR-10), constructing neural networks, realizing forward and back propagation for a given image. Correspondences of some key functions are easily found, like `optimizer.update()` in Chainer corresponding to `optimizer.step()` in PyTorch, and they will not be documented in detail here. Here we focus on the most important differences when transferring Chainer code to PyTorch, which might have implications on the reproduced classification results.

- __Weight initialization for fully connected layers__

The authors of this paper proposed to initialize the weights of each fully connected layer using the uniform distribution. Thus, in the original Chainer code, fully connected layers are constructed like:

```
fc4=chainer.links.Linear(256 * 4 * 4, 1024, initialW=Uniform(1. / math.sqrt(256 * 4 * 4)))
```

However, in PyTorch, the weights are initialied as uniform distribution by default when defining a fully connected layer:

```
fc4 = torch.nn.Linear(256 * 4 * 4, 1024)
```

Initializing the weights manually using the same distribution will cause trouble and even make the training not converge as expected.

- __Optimization method__

The optimizer used for this paper is _NesterovAG_ and can be simply invoked using `chainer.optimizers.NesterovAG`. When it comes to PyTorch, such explicit function for _NesterovAG_ is not found. Instead, we can generate such an optimizer through modifying the `nesterov` parameter of Stochastic Gradient Descent (SGD) method:

```
optimizer = torch.optim.SGD(params=model.parameters(), lr=opt.LR, momentum=opt.momentum,
                                weight_decay=opt.weightDecay, nesterov=True)
```

- __Learning rate schedule__

The learning rate for training is expected to be changed as the definition of its schedule. In the original Chainer implementation, a simple function is written to realize the evolution of learning rate:

```
def lr_schedule(self, epoch):
        divide_epoch = np.array([self.opt.nEpochs * i for i in self.opt.schedule])
        decay = sum(epoch > divide_epoch)
        if epoch <= self.opt.warmup:
            decay = 1

        return self.opt.LR * np.power(0.1, decay)
```

However, such function works not well under PyTorch framework. Instead of changing the value of learning rate manually, we can use the learning rate scheduler for optimizers directly provided by PyTorch, as shown below.

```
epoch_milestones = numpy.array([int(self.opt.nEpochs * i) for i in self.opt.schedule]) 
self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, epoch_milestones, gamma=0.1) 
```

### Results for CIFAR-10 on 11-layer CNN

To compare the performance of standard, BC and BC+ learning on CIFAR-10 dataset, we trained the standard 11-layer CNN via Google Colab. The standard shifting/mirroring data augmentation is applied. Each trial includes 250 epochs by default and takes normally around 2.5 hours. In addition, we use the same batch size (128) as that used in the original Chainer code. we also started training with a learning rate of 0.1 and then divided it by 10 at the epoch of {100, 150, 200}.

Table 2 summarizes the results of our experiments on CIFAR-10, and provides a comparison between the original and reproduced results. We tried to train 10 trials for each experiment, however, for BC and BC+ learning we were only able to complete 5 trials due to the GPU access limit of Google Colab. Here we report the average error rate and standard deviation of 5 or 10 trials.

Table 2. Results for CIFAR-10 dataset on the standard 11-layer CNN

| Learning method| Original paper | Our reproduction |
|      :----:    |     :----:     |      :----:      |
|     Standard   |   6.07 ± 0.04  |    6.59 ± 0.15   | 
|        BC      |   5.40 ± 0.07  |    5.67 ± 0.04   | 
|       BC+      |   5.22 ± 0.04  |    5.51 ± 0.11   | 

It can be noticed that reproduced the average classification results are consistently worse than the original ones, no matter which learning method is being used. For standard and BC+ learning, the error rates of 5 or 10 trials are also more dispersive in our reproduction, as illustrated by the standard deviations. Nonetheless, through the reproduced results, we can still conclude that BC learning significantly improved the classification performance, compared to standard learning. Moreover, it was further improved by 0.16% when treating the image data as waveforms (BC+). This conclusion keeps consistent with the original paper.

### Ablation analysis for CIFAR-10 on 11-layer CNN

We also reproduced all the ablation analysis for CIFAR-10 dataset. These experiments are not contained in the released Chainer code, and we implemented them through PyTorch.

Table 3 shows the original and reproduced results for training without data augmentation, with the error rate of experiments using data augmentation indicated between brackets. From our reproduction, we can draw the same conclusion that the classification accuracy drops a lot when discarding data augmentation, and the degree of improvement in the performance through BC learning is roughly at the same level with that when using data augmentation. In addition, our reproduction on PyTorch performs slightly better than the results shown in the original paper, for both standard and BC learning.  

Table 3. Comparison when using no data augmentation. The error rate is averaged over 10 trials.

| Learning method| Original paper | Our reproduction |
|      :----:    |     :----:     |      :----:      |
|     Standard   |   9.68 (6.07)  |    9.64 (6.59)   | 
|        BC      |   8.38 (5.40)  |    8.10 (5.67)   |  

Table 4 reports the classification error rates of other ablation analysis, which offers a better understanding of the important innovation of BC and BC+ learning. Generally, we trained the 11-layer CNN on CIFAR-10 using different settings, including comparisons of mixing methods, types of label, number of mixed classes and the location of mixing.

Table 4. Comparison of training using various settings. The error rate is averaged over 5 trials.

|   Comparison of   |     Setting    |  Original paper  |  Our reproduction |
|       :----       |     :----      |      :----:      |       :----:      |
| Mixing method     |None (BC) <br> a <br> a+b <br> a+b+c (BC+) <br> b+c|5.40 <br> 5.45 <br> **5.17** <br> 5.22 <br> 5.26|5.67 <br> 5.66 <br> 5.59 <br> **5.51** <br> 5.61| 
| Label             |Single <br> Multi <br> Ratio (BC+)|6.35 <br> 6.05 <br> **5.22**|6.60 <br> 6.45 <br> **5.51**|
| # mixed classes   |N = 1 <br> N = 1 or 2 <br> N = 2 (BC+) <br> N = 2 or 3 <br> N = 3|5.98 <br> 5.31 <br> 5.22 <br> **5.15** <br> 5.32|6.20 <br> 5.55 <br> 5.51 <br> 5.48 <br> xx |
| Where to mix      |Input (BC) <br> pool1 <br> pool2 <br> pool3 <br> fc4 <br> fc5|**5.40** <br> 5.74 <br> 6.52 <br> 6.05 <br> 6.05 <br> 6.12|5.67 <br> xx <br> xx <br> xx <br> xx <br> xx|

* Mixing method

The differences between reproduced results for mixing methods are not as obvious as that shown in the original paper. Moreover, we achieved the best accuracy for `a+b+c`, which is just the proposed BC+ learning setting, instead of `a+b`.

* Label

The reproduced results for using different labels are highly consistent with the original paper. Among all these labels, ratio labels that are generated through mixing two labels using a random ratio in _U_(0,1) can help to obtain the best classification performance.

* Number of mixed classes

When experimenting with differen number of mixed classes, we also achieved similar results as the original paper. Although values of the error rate are higher to some extent, the relative order is the same. When using mixtures of three different classes in addition to the mixtures of two different classes (N = 2 or 3) can improve the performance compared to BC+ learning (N = 2), but not significantly.

* Where to mix

### Results for Caltech101 on 11-layer CNN

Table 5.

| Learning method| Error rate (%) |
|      :----:    |     :----:     |
|     Standard   |                | 
|        BC      |                | 
|       BC+      |                |
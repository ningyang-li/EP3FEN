# EP3FEN
## [Hyperspectral remote sensing image classification based on enhanced pseudo 3D features and salient band selection](https://www.researchsquare.com/article/rs-4820019/v1)
Authors: Ningyang Li  
PREPRINT (Version 1) available at Research Square (Ever accepted by Colour and Visual Computing Symposium 2024)  
Environment: Python 3.6., Tensorflow 2.2.2, Keras 2.3.1, Numpy 1.19.  

## Abstract
Hyperspectral classification is a research hotspot in the field of remote sensing. Recently, 3D convolutional neural networks (CNNs) have achieved better classification performances than traditional machine learning algorithms. However, because of the large kernel size and spectral redundancy, the classification accuracy and efficiency of existing CNN-based methods are still restrained. In this paper, a lightweight model based on the enhanced pseudo 3D features and salient band selection is proposed for HSI classification. Specifically, an enhanced pseudo 3D convolution block is constructed to extract spectral-spatial features with less parameters. Then, a salient band selection block without parameters is designed to relieve the spectral redundancy. To obtain the diverse spectral dependency, a local-connected layer is introduced to explore the interactions between adjacent bands. By integrating these blocks, deep spectral-spatial pseudo 3D features can be well prepared for classification. Experiments on three HSI data sets show that the proposed model outperforms the state-of-the-arts.

## Contibutions
1. An end-to-end model which adopts EP3C and SBS were proposed for HSI classification, which is composed of enhanced pseudo 3D convolutional (EP3C) block, salient band selection (SBS) block, and spectral dependency capture (SDC) block.
2. EP3C block aimed to extract the spectral-spatial pseudo 3D features with efficient convolutional kernels.
3. SBS block was designed to collect the important information between neighboring bands, thereby relieving spectral redundancy.
4. SDC block adopted a locally-connected layer to accurately model the spectral dependency of the center pixel.


<img src="https://github.com/ningyang-li/EP3FEN/blob/93d312375a233404392ad6e5fbbc8a97de3751b8/pic/EP3FEN.png" width="700" />  

## Citation
```
N. Li, "Hyperspectral remote sensing image classification based on enhanced pseudo 3D features and salient band selection," PREPRINT (Version 2) available at Research Square [https://www.researchsquare.com/article/rs-4820019/v2].
```


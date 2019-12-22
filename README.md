# Filter Response Normalization Layer
Eliminating Batch Dependence in the Training of Deep Neural Networks
https://arxiv.org/pdf/1911.09737.pdf

> Batch Normalization (BN) is a highly successful and widely used batch dependent training method. Its use of mini-batch statistics to normalize the activations introduces dependence between samples, which can hurt the training if the mini-batch size is too small, or if the samples are correlated. Several alternatives, such as Batch Renormalization and Group Normalization (GN), have been proposed to address these issues. However, they either do not match the performance of BN for large batches, or still exhibit degradation in performance for smaller batches, or introduce artificial constraints on the model architecture. In this paper we propose the Filter Response Normalization (FRN) layer, a novel combination of a normalization and an activation function, that can be used as a drop-in replacement for other normalizations and activations. Our method operates on each activation map of each batch sample independently, eliminating the dependency on other batch samples or channels of the same sample. Our method outperforms BN and all alternatives in a variety of settings for all batch sizes. FRN layer performs ≈0.7−1.0% better on top-1 validation accuracy than BN with large mini-batch sizes on Imagenet classification on InceptionV3 and ResnetV2-50 architectures. Further, it performs >1% better than GN on the same problem in the small mini-batch size regime. For object detection problem on COCO dataset, FRN layer outperforms all other methods by at least 0.3−0.5% in all batch size regimes.

## Usage

It's as simple as using the default Keras BatchNormalization:

```python
from frn import FRN as FilterResponseNormalization

model = Sequential()
# [...]
model.add(FilterResponseNormalization())
```

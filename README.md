# DenseNet
Densely Connected Convolutional Network (DenseNet) is a network architecture where each layer is directly connected to every other layer in a feed-forward fashion. It's quite similar to ResNet but in contrast DenseNet concatenates outputs instead of using summation. If you need a quick introduction about how DenseNet works, please read the [original paper](https://arxiv.org/abs/1608.06993)[1]. It's well written and easy to understand.

I implemented a DenseNet in Python using Keras and TensorFlow as backend. Because of this I can't guarantee that this implementation is working well with Theano or CNTK. I will try to optimize this architecture in my own way with some modifications.
You can find several implementations on [GitHub](https://github.com/liuzhuang13/DenseNet#other-implementations).

## Results
### Fashion-MNIST
I used this [notebook]( https://github.com/cmasch/zalando-fashion-mnist/blob/master/Simple_Convolutional_Neural_Network_Fashion-MNIST.ipynb) to evaluate the model on fashion-MNIST with following parameters:

| Dense Blocks | Depth | Growth Rate | Dropout | Bottlen. | Compress. | BatchSize /<br>Epochs | Training<br>(loss / acc) | Validation<br>(loss / acc) | Test<br>(loss / acc) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 3 | 18 | 25 | 0.2 | False | 0.6 | 250 / 80 | 0.1200 / 0.9641 | 0.1823 / 0.9489 | 0.2459 / 0.93748 |

Feel free to try it on your own with another parameters.

## Requirements
- [Keras 2.2.4](https://keras.io/)
- [TensorFlow 1.10.0](https://www.tensorflow.org/)
- Python 3.6

## Usage
Feel free to use this implementation:<br>
```
import densenet
model = densenet.DenseNet(input_shape=(28,28,1), nb_classes=10, depth=10, growth_rate=25,
                          dropout_rate=0.1, bottleneck=False, compression=0.5)
model.summary()
```
This will build the following model:<br>
<img src="./images/model_3-2.png" height="1024px"></kbd>

## References
[1] [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)<br>
[2] [DenseNet - Lua implementation](https://github.com/liuzhuang13/DenseNet)

## Author
Christopher Masch

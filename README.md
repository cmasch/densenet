# DenseNet
Densely Connected Convolutional Network (DenseNet) is a network architecture where each layer is directly connected to every other layer in a feed-forward fashion. If you need a quick introduction about how DenseNet works, please read the [original paper](https://arxiv.org/abs/1608.06993). It's well written and easy to understand.

I implemented a DenseNet in Python using Keras and TensorFlow as backend. Because of this I can't guarantee that this implementation is working well with Theano or CNTK. I will try to optimize this architecture in my own way with some modifications.
You can find several implementations on [GitHub](https://github.com/liuzhuang13/DenseNet#other-implementations).

Currently I'm evaluating this architecture for different datasets e.g. [fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) and reached with the current implementation / parameter on testset a loss of 0.2310 and accuracy of 93.06%. In the original paper the authors benchmarked their DenseNet on [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html).

I will publish final results in a few days/weeks.

## Requirements
- [Keras 2.x](https://keras.io/)
- [TensorFlow 1.x](https://www.tensorflow.org/)

## Usage
Feel free to use this implementation:<br>
```
import densenet
print('DenseNet-Version: %s' % densenet.__version__)
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

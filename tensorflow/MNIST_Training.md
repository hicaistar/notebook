# Fashion Mnist Training
这个例子用 keras 库训练了一个对衣服分类的模型,Fashion MNIST 数据集包含了 10 类共 70，000 张灰度图片。这些图片都是 28 * 28 像素的。具体见：[训练首个神经网络：基本分类  |  TensorFlow](https://www.tensorflow.org/tutorials/keras/basic_classification)

### 开始

```python
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)
```
### 下载数据

使用 60，000 个图像来训练模型，10，000 个图像来评估模型
```python
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
```
### 数据预处理

将取值缩放到 [0, 1]
```python
train_images = train_images / 255.0
test_images = test_images / 255.0
```
### 创建网络 layers

```python
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(18, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
```
这里，有3层网络，第一层将 2 维数组转为1维 28 * 28 = 784 像素。第二层是一个全连接层，激活函数是 relu。第三层也是一个全连接层，10 路 softmax，返回一个由 10 个概率值（总和为1）组成的数组。每个概率值表示当前图像属于10个类别中某一个的概率。

### 编译网络

在训练模型前，还需要设置模型编译参数：
- Loss function：损失函数，衡量网络在训练数据上的性能。
- Optimizer：基于训练数据和损失函数来更新网络的机制。
- Metric：监控训练和测试的指标。这里用的 accuracy 准确度

```python
model.compile(optimizer=tf.train.AdamOptimizer()， 
              loss=‘sparse_categorical_crossentropy’，
              metrics=[‘accuracy’])
```

### 训练模型

训练模型需要以下步骤：
- 给模型喂数据： train_images 和  train_labels
- 模型根据图像和标签进行学习
- 使用模型在测试集上进行预测

使用 model.fit 来训练数据：
```python
model.fit(train_images, train_labels, epochs=5)
```
上面对所有图像进行了 5 轮训练。

### 评估准确度

```python
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(‘Test accuracy:’, test_acc)
```

### 预测

```python
predictions = model.predict(test_images)
np.argmax(predictions[0])
```
对测试集进行预测，每个图片会有一个对应的数组（长度为10）其中每个值表示模型预测的该图片对应该分类的自信度。

以上即是一个简单的模型训练及评估过程。
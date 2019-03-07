# save and restore models
参见：https://www.tensorflow.org/tutorials/keras/save_and_restore_models
### 简介
模型可以在训练过程中和结束后被保存，这意味着模型可以在中断的地方恢复。同样的，保存意味着可以分享给其他人使用。一般情况下包括：
- 创建模型的代码
- 训练后的权重/参数
可以用不同的方法来保存模型，这取决于用的 API。这里使用了 tf.keras 高级 API。

以 MNIST 数据集为例：
### 创建模型
```python
from __future__ import absolute_import, division, print_function

import os

import tensorflow as tf
from tensorflow import keras

tf.__version__

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

# Returns a short sequential model
def create_model():
  model = tf.keras.models.Sequential([
    keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation=tf.nn.softmax)
  ])
  
  model.compile(optimizer=adam, 
                loss=tf.keras.losses.sparse_categorical_crossentropy,
                metrics=[‘accuracy’])
  
  return model

# Create a basic model instance
model = create_model()
model.summary()
```
上面代码，下载了数据集，并创建了一个模型。

### 保存 checkpoints
主要实现在训练中和训练结束保存 checkpoints。 tf.keras.callbacks.ModelCheckpoint 是一个实现了此功能的回调函数。

```python
checkpoint_path = "checkpoint.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                                 save_weights_only=True,
                                                 verbose=1)

model = create_model()

model.fit(train_images, train_labels,  epochs = 10, 
          validation_data = (test_images,test_labels),
          callbacks = [cp_callback])  # pass callback to training
```
上述代码，首先制定了 checkpoint 文件名及路径，然后定义保存 checkpoint 的回调函数，创建模型，并调用 model.fit 执行训练（将回调函数传入）。下面代码将模型导出并验证：
```python
# load the stored model
model.load_weights(checkpoint_path)
loss,acc = model.evaluate(test_images, test_labels)
```
#### checkpoint callback options
callback 提供了一些选项来配置，如下：定义每5个 epoch 保存一个唯一的 checkpoint 文件：
```python
# include the epoch in the file name. (uses `str.format`)
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True,
    # Save weights, every 5-epochs.
    period=5)

model = create_model()
model.fit(train_images, train_labels,
          epochs = 50, callbacks = [cp_callback],
          validation_data = (test_images,test_labels),
          verbose=0)
```
### What are these files?
```
checkpoint            cp-0030.ckpt.data-00000-of-00001
cp-0005.ckpt.data-00000-of-00001  cp-0030.ckpt.index
cp-0005.ckpt.index        cp-0035.ckpt.data-00000-of-00001
cp-0010.ckpt.data-00000-of-00001  cp-0035.ckpt.index
cp-0010.ckpt.index        cp-0040.ckpt.data-00000-of-00001
cp-0015.ckpt.data-00000-of-00001  cp-0040.ckpt.index
cp-0015.ckpt.index        cp-0045.ckpt.data-00000-of-00001
cp-0020.ckpt.data-00000-of-00001  cp-0045.ckpt.index
cp-0020.ckpt.index        cp-0050.ckpt.data-00000-of-00001
cp-0025.ckpt.data-00000-of-00001  cp-0050.ckpt.index
cp-0025.ckpt.index
```
上面的代码将 weights 保存到了一系列 checkpoint 格式的文件。Checkpoints 包含了：一个或多个包含模型参数的 shard（碎片）。 index 文件标明哪些参数属于哪个 shard。
如果你只在单节点训练了一个模型，应该会得到一个带有 .data-00000-of-00001 的文件。
### 手动保存 weights
```python
# Save the weights
model.save_weights(./checkpoints/my_checkpoint)

# Restore the weights
model = create_model()
model.load_weights(‘./checkpoints/my_checkpoint’)

loss,acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
```
### 保存完整的 model
完整的模型文件包括： weight values, model's configuration 和 optimizer's configuration。
#### HDF5 文件
```python
model = create_model()

# You need to use a keras.optimizer to restore the optimizer state from an HDF5 file.
model.compile(optimizer=‘adam’, 
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=[‘accuracy’])

model.fit(train_images, train_labels, epochs=5)

# Save entire model to a HDF5 file
model.save(‘my_model.h5’)

# Recreate the exact same model, including weights and optimizer.
new_model = keras.models.load_model(‘my_model.h5’)
new_model.summary()

loss, acc = new_model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
```
上面保存了所需的信息：
- weight values
- model's configuration(architecture)
- optimizer configuration

#### saved_model
```python
model = create_model()

model.fit(train_images, train_labels, epochs=5)
saved_model_path = tf.contrib.saved_model.save_keras_model(model, "./saved_models")

new_model = tf.contrib.saved_model.load_keras_model(saved_model_path)

# The optimizer was not restored, re-attach a new one.
new_model.compile(optimizer=‘adam’, 
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=[‘accuracy’])

loss, acc = new_model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

```

### 其他
- [tf.keras guide](https://www.tensorflow.org/guide/keras) 说明了 tf.keras 保存和载入模型的更多信息
- [Saving in eager](https://www.tensorflow.org/guide/eager#object_based_saving) 在 eager execution 过程中保存模型
- [Save and Restore](https://www.tensorflow.org/guide/saved_model) 描述了 TensorFlow 底层实现细节
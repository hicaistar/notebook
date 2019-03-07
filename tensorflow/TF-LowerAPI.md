# TensorFlow Low Level APIs
## Introduction
low-level TensorFlow APIs（Core）编程：
- 管理 TensorFlow 程序（tf.Graph）和 TensorFlow runtime（tf.Session），而不是用 Estimator 来管理
- 使用 tf.Session 运行 TensorFlow operations
- 使用高级别组件(datasets,layers 和 feature_columns)
- 创建 training loop
建议使用高级别 API 来创建模型。出于以下原因来了解 TensorFlow Core：
- 实验和调试更直接
- 直观了解模型训练过程

### Tensor Values
TensorFlow 里核心数据是 tensor。Tensor 是由任意个维度组成的数组。rank 表示 tensor 的维度数量，shape 用一个元组来表示这个数组在每个维度上的长度。例子：
```python
3. # a rank 0 tensor; a scalar with shape [],
[1., 2., 3.] # a rank 1 tensor; a vector with shape [3]
[[1., 2., 3.], [4., 5., 6.]] # a rank 2 tensor; a matrix with shape [2, 3]
[[[1., 2., 3.]], [[7., 8., 9.]]] # a rank 3 tensor with shape [2, 1, 3]
```

### TensorFlow Core Walkthrough
可以将 TensorFlow 看作有两个独立的部分：
- 建立计算图（tf.Graph）
- 运行计算图（tf.Session）

#### Graph
计算图是将一系列 TensorFlow 的操作安排到一个图里。这个图主要有两个类型组成：
- tf.Operation(ops):图中节点。Operations 描述计算（消费和产出 tensors）
- tf.Tensor：图的边。表示流过图的数值。大多数 TensorFlow 函数返回 tf.Tensors。
```python
a = tf.constant(3.0, dtype=tf.float32)
b = tf.constant(4.0) # also tf.float32 implicitly
total = a + b
print(a)
print(b)
print(total)
# output
Tensor("Const:0", shape=(), dtype=float32)
Tensor("Const_1:0", shape=(), dtype=float32)
Tensor("add:0", shape=(), dtype=float32)
```
以上输出结果并没有进行计算，只是构建了计算图。这些 tf.Tensor 只是代表将要运行操作的结果。

#### TensorBoard
可视化计算图

#### Session
为了评估张量，实例化一个 tf.Session。一个 session 封装了 TensorFlow runtime 和运行 TensorFlow operations。如果 tf.Graph 像一个 .py 文件，tf.Session 就像 python 执行器。
```python
sess = tf.Session()
print(sess.run(total))
```
上述代码就对上个 total 张量进行计算并输出结果。

#### Feeding
上面讲述的是常量结果，没有实际意义。一个图可以参数化接收外部输入，以 placeholder 实现。
```python
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
z = x + y
print(sess.run(z, feed_dict={x: 3, y: 4.5}))
print(sess.run(z, feed_dict={x: [1, 3], y: [2, 4]}))
```

#### Datasets
Placeholders 实现了简单的过程，tf.data 处理流数据。用到：tf.data.Iterator 和 tf.data.Iterator.get_next 方法。
```python
r = tf.random_normal([10,3])
dataset = tf.data.Dataset.from_tensor_slices(r)
iterator = dataset.make_initializable_iterator()
next_row = iterator.get_next()

sess.run(iterator.initializer)
while True:
  try:
    print(sess.run(next_row))
  except tf.errors.OutOfRangeError:
    break
```

#### Layers
一个可训练模型必须修改图中的某些值来获取好的结果。tf.layers 实现了添加可训练的参数到图中。
Layers 将参数和操作封装在一起。例如，densely-connected layer 为所有输出对所有的输入进行权重交叉求和，并使用了 activation 函数。连接 weights 和 biases 由 layer 对象管理。

#### Feature columns

### Training
- Define the data
- Define the model
- Loss
  - 定义损失函数来指明优化方向。
- Training
  - 利用 optimizers 实现优化算法。

```python
x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32)
y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32)

linear_model = tf.layers.Dense(units=1)

y_pred = linear_model(x)
loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
for i in range(100):
  _, loss_value = sess.run((train, loss))
  print(loss_value)

print(sess.run(y_pred))
```

## What is a tf.Graph?


























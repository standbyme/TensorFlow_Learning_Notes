# High Level API
## Keras
### 简介
[tf.keras](https://www.tensorflow.org/api_docs/python/tf/keras) 是 TensorFlow 对 [Keras API 规范](https://keras.io/) 的实现<br />支持 TensorFlow 特定功能，如
- [Eager Execution](https://www.tensorflow.org/guide/keras#eager_execution)（所有 [tf.keras](https://www.tensorflow.org/api_docs/python/tf/keras) 模型构建 API 都与 Eager Execution 兼容）
- [tf.data](https://www.tensorflow.org/api_docs/python/tf/data) pipeline
- [Estimator](https://www.tensorflow.org/guide/estimators)
### 模型
#### Sequence
层的简单堆叠，无法表示任意模型
```python
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)
```
#### Functional
可以构建复杂的模型，例如：
- 多输入模型
- 多输出模型
- 具有共享层的模型（同一层被调用多次）
- 具有非序列数据流的模型（例如，residual connections）
```python
inputs = tf.keras.Input(shape=(32,))  # placeholder

x = layers.Dense(64, activation='relu')(inputs)
x = layers.Dense(64, activation='relu')(x)
predictions = layers.Dense(10, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=predictions)

model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(data, labels, batch_size=32, epochs=5)
```
#### Model Subclassing
很像 PyTorch<br />在启用 [Eager Execution](https://www.tensorflow.org/guide/eager) 时特别有用，因为可以命令式地编写前向传播。
```python
class MyModel(tf.keras.Model):

  def __init__(self, num_classes=10):
    super(MyModel, self).__init__(name='my_model')
 
    self.num_classes = num_classes
    self.dense_1 = layers.Dense(32, activation='relu')
    self.dense_2 = layers.Dense(num_classes, activation='sigmoid')

  def call(self, inputs):
    x = self.dense_1(inputs)
    return self.dense_2(x)

  def compute_output_shape(self, input_shape):
    # You need to override this function if you want to use the subclassed model
    # as part of a functional-style model.
    # Otherwise, this method is optional.
    # 指定在给定输入形状的情况下如何计算层的输出形状，供运行时 Graph 编译期 Dimension 推导
    shape = tf.TensorShape(input_shape).as_list()
    shape[-1] = self.num_classes
    return tf.TensorShape(shape)

model = MyModel(num_classes=10)

model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(data, labels, batch_size=32, epochs=5)
```
### 回调
回调是传递给模型的对象，用于在训练期间自定义该模型并扩展其行为。您可以编写自定义回调，也可以使用包含以下方法的内置 [tf.keras.callbacks](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks)
- [tf.keras.callbacks.ModelCheckpoint](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint)：定期保存模型的检查点。
- [tf.keras.callbacks.LearningRateScheduler](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/LearningRateScheduler)：动态更改学习速率。
- [tf.keras.callbacks.EarlyStopping](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping)：在验证效果不再改进时中断训练。
- [tf.keras.callbacks.TensorBoard](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/TensorBoard)：使用 [TensorBoard](https://www.tensorflow.org/guide/summaries_and_tensorboard) 监控模型的行为。
```python
callbacks = [
  # Interrupt training if `val_loss` stops improving for over 2 epochs
  tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
  # Write TensorBoard logs to `./logs` directory
  tf.keras.callbacks.TensorBoard(log_dir='./logs')
]
model.fit(data, labels, batch_size=32, epochs=5, callbacks=callbacks,
          validation_data=(val_data, val_labels))
```
## Estimator
[Estimator](https://www.tensorflow.org/guide/estimators) 用于分布式环境
```python
model = tf.keras.Sequential([layers.Dense(10,activation='softmax'),
                          layers.Dense(10,activation='softmax')])

model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

estimator = tf.keras.estimator.model_to_estimator(model)
```
## Eager
TensorFlow 2.0 中默认开启
```python
x = [[2.]]
m = tf.matmul(x, x)
print("hello, {}".format(m))

------

hello, [[4.]]
```
> Enabling eager execution changes how TensorFlow operations behave—now they immediately evaluate and return their values to Python. [`tf.Tensor`](https://www.tensorflow.org/api_docs/python/tf/Tensor) objects reference concrete values instead of symbolic handles to nodes in a computational graph.
### 一个完整的程序
```python
(mnist_images, mnist_labels), _ = tf.keras.datasets.mnist.load_data()

dataset = tf.data.Dataset.from_tensor_slices(
  (tf.cast(mnist_images[...,tf.newaxis]/255, tf.float32),
   tf.cast(mnist_labels,tf.int64)))
dataset = dataset.shuffle(1000).batch(32)

mnist_model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(16,[3,3], activation='relu'),
  tf.keras.layers.Conv2D(16,[3,3], activation='relu'),
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(10)
])

optimizer = tf.train.AdamOptimizer()

for (batch, (images, labels)) in enumerate(dataset.take(400)):
  if batch % 10 == 0:
    print('.', end='')
  with tf.GradientTape() as tape:
    logits = mnist_model(images, training=True)
    loss_value = tf.losses.sparse_softmax_cross_entropy(labels, logits)

  grads = tape.gradient(loss_value, mnist_model.trainable_variables)
  optimizer.apply_gradients(zip(grads, mnist_model.trainable_variables),
                            global_step=tf.train.get_or_create_global_step())
```
所谓的 Dynamic control flow，不过是[宿主语言的 控制流 + TF 的 数据结构 = Eager 程序]。一定要将这里和 Low Level API - AutoGraph 一节中的 tf.function 加以区分。tf.function 是将[宿主语言的 控制流]转化成[Graph]再进行计算，并非直接利用[宿主语言的 控制流]。

**最主要的用法就是与 Model Subclassing 结合**
### 反向传播
```python
w = tf.Variable([[1.0]])
with tf.GradientTape() as tape:
  loss = w * w

grad = tape.gradient(loss, w)
print(grad)

------

tf.Tensor([[ 2.]], shape=(1, 1), dtype=float32)
```
使用 GradientTape 跟踪正向传播，计算梯度时将 Tape 倒着放。
### Variable Lifetime
**Graph**<br />由 [tf.Session](https://www.tensorflow.org/api_docs/python/tf/Session) 管理<br />**Eager**<br />和它们对应的 Python 对象相同
```python
if tf.test.is_gpu_available():
  with tf.device("gpu:0"):
    v = tf.Variable(tf.random_normal([1000, 1000]))
    v = None  # v 不再占用显存（我存疑，认为GC后才被释放放）
```
### 性能
For compute-heavy models, such as [ResNet50](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/eager/python/examples/resnet50) training on a GPU, eager execution performance is comparable to graph execution. But this gap grows larger for models with less computation and there is work to be done for optimizing hot code paths for models with lots of small operations.
# Low Level API
## 计算图
| 节点 | op |
| --- | --- |
| 边 | Tensor |
## Graph 和 Eager 在体验上的对比
```python
def sum_even(items):
  s = 0
  for c in items:
    if c % 2 > 0:
      continue
    s += c
  return s

print('Eager result: %d' % sum_even(tf.constant([10,12,15,20])))

tf_sum_even = autograph.to_graph(sum_even)

with tf.Graph().as_default(), tf.Session() as sess:
    print('Graph result: %d\n\n' % sess.run(tf_sum_even(tf.constant([10,12,15,20]))))

------

Eager result: 42
Graph result: 42
```
## AutoGraph
**普通**
```python
def square_if_positive(x):
  if x > 0:
    x = x * x
  else:
    x = 0.0
  return x

print(autograph.to_code(square_if_positive))

------

def tf__square_if_positive(x):
  do_return = False
  retval_ = ag__.UndefinedReturnValue()
  cond = x > 0

  def get_state():
    return ()

  def set_state(_):
    pass

  def if_true():
    x_1, = x,
    x_1 = x_1 * x_1
    return x_1

  def if_false():
    x = 0.0
    return x
  x = ag__.if_stmt(cond, if_true, if_false, get_state, set_state)
  do_return = True
  retval_ = x
  cond_1 = ag__.is_undefined_return(retval_)

  def get_state_1():
    return ()

  def set_state_1(_):
    pass

  def if_true_1():
    retval_ = None
    return retval_

  def if_false_1():
    return retval_
  retval_ = ag__.if_stmt(cond_1, if_true_1, if_false_1, get_state_1, set_state_1)
  return retval_
```
**嵌套**
```python
def sum_even(items):
  s = 0
  for c in items:
    if c % 2 > 0:
      continue
    s += c
  return s

print(autograph.to_code(sum_even))

------

def tf__sum_even(items):
  do_return = False
  retval_ = ag__.UndefinedReturnValue()
  s = 0

  def loop_body(loop_vars, s_2):
    c = loop_vars
    continue_ = False
    cond = c % 2 > 0

    def get_state():
      return ()

    def set_state(_):
      pass

    def if_true():
      continue_ = True
      return continue_

    def if_false():
      return continue_
    continue_ = ag__.if_stmt(cond, if_true, if_false, get_state, set_state)
    cond_1 = ag__.not_(continue_)

    def get_state_1():
      return ()

    def set_state_1(_):
      pass

    def if_true_1():
      s_1, = s_2,
      s_1 += c
      return s_1

    def if_false_1():
      return s_2
    s_2 = ag__.if_stmt(cond_1, if_true_1, if_false_1, get_state_1, set_state_1)
    return s_2,
  s, = ag__.for_stmt(items, None, loop_body, (s,))
  do_return = True
  retval_ = s
  cond_2 = ag__.is_undefined_return(retval_)

  def get_state_2():
    return ()

  def set_state_2(_):
    pass

  def if_true_2():
    retval_ = None
    return retval_

  def if_false_2():
    return retval_
  retval_ = ag__.if_stmt(cond_2, if_true_2, if_false_2, get_state_2, set_state_2)
  return retval_
```
### [功能和限制](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/autograph/LIMITATIONS.md)
### tf.function
When you annotate a function with [tf.function](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/function), you can still call it like any other function. But it will be **compiled into a graph**.
```python
@tf.function
def simple_nn_layer(x, y):
  return tf.nn.relu(tf.matmul(x, y))


x = tf.random.uniform((3, 3))
y = tf.random.uniform((3, 3))

simple_nn_layer(x, y)
```
If your code uses multiple functions, you don't need to annotate them all - any functions called from an annotated function will also run in graph mode.
```python
def linear_layer(x):
  return 2 * x + 1


@tf.function
def deep_net(x):
  return tf.nn.relu(linear_layer(x))


deep_net(tf.constant((1, 2, 3)))
```
# 数据输入流水线
_**以下内容基于使用了加速器这个假设**_<br />数据预处理（例如，处理 JPEG 图像的模型将遵循以下流程：从磁盘加载图像，将 JPEG 解码为张量，裁剪并填充），这是一个 ETL 过程，随着 GPU 和其他硬件加速器的速度的提升，数据预处理可能会成为瓶颈。<br />**普通（同步）**：当 CPU 正在准备数据时，加速器处于空闲状态；当加速器正在训练模型时，CPU 处于空闲状态。因此，训练的用时是 CPU 预处理时间和加速器训练时间的总和。<br />![datasets_without_pipelining.png](https://cdn.nlark.com/yuque/0/2019/png/127045/1564057295706-0b087f37-7540-4fd3-9901-8b9c8a984c0b.png#align=left&display=inline&height=160&name=datasets_without_pipelining.png&originHeight=160&originWidth=700&size=20617&status=done&width=700)<br />**流水线（异步）**：当加速器正在执行第 N 个训练步时，CPU 正在准备第 N+1 步的数据。<br />![datasets_with_pipelining.png](https://cdn.nlark.com/yuque/0/2019/png/127045/1564057362109-4784d0df-844e-4519-998b-a70a30525024.png#align=left&display=inline&height=160&name=datasets_with_pipelining.png&originHeight=160&originWidth=700&size=18207&status=done&width=700)<br />

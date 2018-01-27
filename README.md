# Inception-v3-classification
export PATH=/usr/local/cuda-8.0/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH

### train
```
python scripts/retrain.py  --image_dir ./tf_files/flower_photos
```
### test
```
python scripts/label_image.py --graph=tf_files/retrained_graph.pb --image=tf_files/flower_photos/daisy/3475870145_685a19116d.jpg
```

```
最新的物体识别模型可能含有数百万个参数，将耗费几周的时间去完全训练。我们采用迁移学习的方法，在已经训练好的模型（基于ImageNet）上调整部分参数，以实现对新类别的分类。
获取一个分类模型有三种方式：

* Train from scratch 从头开始训练
* Fine-tune a model 对一个网络调优
* Retrain a model 对一个网络重训
其难度由上至下递减，当然，执行效果也是逐级递减。目前我们所要实现的迁移学习对应于最后一个部分：Retrain a model。其与Fine-tune a model方法的差距还是比较明显的，以Inception_v3模型为例：

Retrain a model是利用基于ImageNet图像训练的Inception_v3模型所导出的pb文件，更改最后的softmax layer为自己需要的分类器，然后对这一更改的softmax layer进行训练。除开最后一层，其他层的参数全部固化，无法更新。因此，在实际的Retrain中，往往先将数据集（包含训练集、验证集与测试集）中的所有图片导入到Inception_v3模型中，获取最后一层的输入，或者说是倒数第二层的输出，定义为Bottlenecks。然后直接使用Bottlenecks对最后更改的softmax layer进行训练，将大幅度提升训练速度。
Fine-tune a model是利用基于ImageNet图像训练的Inception_v3模型所导出的Ckpt文件，在训练过程中，整个网络的参数都可以随之修改，不仅仅局限于被替换掉的softmax layer。
Fine-tune例子
使用freeze_graph.py文件可以将ckpt文件中所有参数固化，转为pb文件。

```

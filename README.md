

# sleepology

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)



sleepology是一个使用Python开发的包，致力于开发一个可以管理、处理及采样巨大的睡眠多导图（PSG）数据集的Python包。并且它提供了用于机器学习、实时采样的功能。在未来会加入有关闭环刺激的功能。

目前项目在持续的开发阶段，仍不稳定。更新的版本间函数的功能和磁盘文件可能会有巨大的差异，且不提供向下兼容的接口。稳定版将会在`0.3`版本后提供。

所以目前并不建议您使用本包，若您真的希望使用，请保证在使用本包完成阶段性工作前不要更新它。与此同时作为作者，我也会尽可能将版本间的差异写明，如您在使用`0.2.x`版本时需要更新，请阅读完版本间的改变和差异。

使用过程中有任何的问题，欢迎通过电子邮件联系或新开一个Issue。让我们一起完善这一工具。



## 内容列表

- 背景
- 安装
- 使用说明
- 示例
- 维护者
- 如何贡献
- 使用许可



## 背景

本项目始于**对睡眠多导图（PSG）数据进行机器学习的需求**。多导睡眠图作为特殊的人体生理数据，它的记录时长往往是一整晚，且具有多个电极导联。机器学习通常需要使用到多大量的多导睡眠图样本，对计算机的内存是极大的考验。

同时，进行睡眠研究的人员（哪怕是精通机器学习的人）通常没有大量的操作系统和计算机硬件知识。仅仅使用传统的机器学习工具包（如NumPy）也对数据的管理和重复使用产生巨大的考验，况且往往会因为内存问题而根本无法完成机器学习。

在sleepology的蓝图中，我们全部所希望实现的功能如下：

- 对批量PSG数据集的统一管理；
- 对预处理管道（Pipeline）的持久化；
- 对所有数据源（`Source`）的抽象；
- 对采样过程（`Sample`）的持久化；
- 对实时流的兼容和控制；
- 对实时刺激的兼容和控制。

**`sleepology.dataset`包**注重对批量的PSG数据进行导入、预处理、持久化（即保存到磁盘）过程进行优化，使您仅仅需要几行代码就可以有效管理巨大的数据集。通过`disk`模式，可以将所有的数据缓存在硬盘，仅仅在必要时使用内存，通过牺牲读写速度来避免OOM错误，使您可以不再受困于计算机性能限制。经目前的测试，在此模式下任何数据集都可以仅仅使用4G以内的内存。

`Dataset`支持对**多数据源、多标签**的数据的管理，您可以对您的数据帧（`epoch`）标记多个标签（`label`），例如这一帧同时表现了患者*“打鼾”*且处于*“NREM 1”*期。并简单地通过采样（`sample`）功能分别训练不同的标签分类，甚至多标签分类（multi-label classification，即由机器学习算法识别“打鼾且处于NREM-1期”的信号）。同时可以利用条件（`condition`）来对整段数据进行管理，例如此患者的*“诊断”*、*“性别”*等仅仅与整段数据（`data`）有关，而不属于数据帧（`epoch`）的特征。合理使用`Dataset`的各项功能，可以使您有效地管理数据集并与他人共享。



**`sleepology.sample`包**注重对`Dataset`的采样。使用相同方式进行预处理的`Dataset`，也可以以不同的形式进行采样——例如是由机器学习学习整段数据（`data`）和条件（`condition`）的对应关系、或是数据帧(`epoch`)和标签（`label`）的对应关系；也可以是对标签（`label`）进行筛选，仅仅学习某几类的对应关系。

`Sample`同时还包含**对数据进行探索、平衡和填充**的功能。在机器学习中，**不平衡**的数据集往往会使机器学习的结果有偏差，而多导睡眠图数据不像一般的神经科学实验能够被良好地提前设计，在数据采集阶段就达到不同类别的平衡。非平衡数据集是睡眠机器学习的所要面对的常态，所以`Sample`提供了多种函数对数据集进行平衡、填充，以确保训练的结果与样本分布无关且能够泛化。睡眠数据也往往**不等长**，因为我们无法控制患者整晚的睡眠时长，对于整段数据（`data`）的时间序列研究，许多模型要求输入数据的形状相同，您可以使用填充功能将所有数据填充至相同长度，并使用您惯用的机器学习包内诸如`Masking`这样的函数在训练时将填充的值滤除。



## 安装

本项目是一个Python包。在任意已安装Python 3.5以上版本并正确联网的计算机上使用如下命令完成安装。

```shell
$ pip install sleepology
```



## 使用说明

### 了解数据集（Dataset）

由于脑电信号——尤其是睡眠多导图数据——都是由特殊的文件格式（如`.edf`）存储的。一段脑电信号通常会包含许多不同的**标记（marker）**或在机器学习中我们称为**标签（label）**。当脑电信号被用于机器学习时，我们通常会根据marker切分片段，并将数值存储为`.csv`等格式。

在数据量小的情况下，这么做是很方便的。但是当面对一个庞大的数据——比如睡眠多导图数据时，这么做无疑会使得磁盘因为存放了太多文件而运行缓慢。不仅如此，由于许多生理数据集拥有多元标签维度*（比如这段脑电来自于男性还是女性？他/她被分进了哪一个实验组？这段脑电代表呈现给他/她什么刺激条件？）*，还有多种多样的数据处理方式，使得原先的脑电数据集存储方式无法继续使用。

**`sleepology.dataset.Dataset()`类是一种专门用于管理睡眠多导数据集的工具。**在`0.2`版本后，它已经开始支持对多元的数据集进行管理，并能够支持**超大数据集**。

#### 导入Dataset类

```python
from sleepology.dataset import Dataset
```



#### 创建数据集

```python
dataset = Dataset('example_name', 'example/path', '这里写这个数据集的注释。这是个示例数据集。')
```

使用Python创建对象的语句，可以创建一个`Dataset()`实例。

创建`Dataset()`的参数有这些：

> ```python
> Dataset(dataset_name,
>         save_path,
>         comment = '',
>         label_dict = os.path.join(package_root, 'labeltemplate', 'aasm.labeltemplate'),
>         mode = 'memory')
> ```
>
> `dataset_name`：`str` 数据集的名字，也会作为保存数据集时的文件名
>
> `save_path`：`path-like` 保存数据集的路径
>
> `comment`：`str` 用来描述你的数据集，方便未来查找
>
> `label_dict`：`path-like`一个用来管理标签的字典，会把人类所能理解的标签和用于机器学习的标签（比如one-hot）进行互相转换，默认是使用AASM标准的标签
>
> `mode`：`{'memory', 'disk'}` 选择内存模式（memory）还是磁盘模式（disk）。内存模式（memory）处理数据会更快，但是如果你的数据集特别大，超过内存（对于睡眠数据集来说是很容易这样的），那么使用磁盘模式（disk）可以把数据全部缓存在磁盘，直到真的需要用到它的时候才读取出来。默认是使用内存模式（memory）。

所以，如果你具有一个非常大的数据集，创建它的时候就使用：

```python
dataset = Dataset('example_name', 'example/path', '这里写这个数据集的注释。这是个示例数据集。', mode='disk')
```



#### 读取数据集

如果你不是从头创建，而是要读取之前存在磁盘的数据集，就要使用`Dataset.load()`：

```python
dataset = Dataset.load('example/path/example_name.dataset')
```

同样，如果你的数据集非常大，那么在读取的时候也要使用`'disk'`模式：

```python
dataset = Dataset.load('example/path/example_name.dataset', mode = 'disk')
```



#### 保存数据集

在我们整理好一个满意的数据集之后，我们一般不会只对它做一次实验。所以我们需要将它保存到磁盘，以备日后再用。由于在创建数据集的时候已经指定了数据集的名字和存储路径，所以可以直接使用：

```python
dataset.save()
```

如果想要换一个存储位置（相当于“另存为”），也可以使用：

```python
dataset.save('newpath/newname.dataset')
```



在了解了最基本的数据集创建、读取和保存之后，我们先不再继续学习数据集。转而看一看`source`模块如何帮助我们导入数据。



### 从源（Source）导入数据

我们学会了如何创建一个空的数据集，但是数据集一定需要有新的数据。为了统一数据输入的接口，我编写了`source`模块。

`sleepology.source`模块是对于脑电数据、睡眠多导数据以及其标签来源的一个处理模块。无论你的数据是`.edf`、`.eeg`、`.csv`或是来自局域网实时的数据流，对于`sleepology`来说它们都是一种`Source`，可以用统一的方法导入。

由于`source`模块会在`0.3`版本进一步完善，目前仅仅实现了一部分功能。

#### 导入存储于磁盘的脑电数据

`RawDataFile`是`Source`的子类，专门用于读取存在于磁盘的脑电文件。

其能够支持如下格式：

- BrainVision (vhdr)
- European data format (edf)
- BioSemi data format (bdf)
- General data format (gdf)
- Neuroscan CNT data format (cnt)
- EGI simple binary (egi)
- EGI MFF (mff)
- EEGLAB set files (set)
- Nicolet (data)
- eXimia EEG data (nxe)

首先导入`RawDataFile`

```python
from sleepology.source import RawDataFile
```

然后创建一个新的`Source`：

```pyhton
raw = RawDataFile('example_path/example.edf', 'edf', 30, 100)
```

这里输入的参数，分别为：

- 目标文件的路径
- 目标文件的类型（输入后缀即可）
- 你所希望的**帧**大小（例子中是30秒为一帧）
- 你所希望的采样率

#### 帧（epoch）

由于`sleepology`是专门针对超大数据集开发的Python包，且能够兼容实时输入，所以设计成会一帧一帧地读取数据，并且一帧一帧地处理。

在AASM的标准包括更老的R&K标准中，睡眠数据也是按帧进行判读，每一帧为30秒。所以实际上数据集的标签（Label）也是按帧划分的。

`Source`类被设计成一个迭代器（Iterator），每次处理一帧数据。在创建`Source`的时候指定一个帧的大小是十分必要的。一般来说帧的大小根据标签的间隔来定，如果是睡眠数据的话，一般也是30秒。

#### 导入来自日本光电机型的标签数据

由于在开发`sleepology`时，我正在处理采集自日本光电睡眠多导仪的标签数据，所以目前只写了适配该机型的标签源（Source）。

```python
from sleepology.source import NihonkohdenLabelSource

label = NihonkohdenLabelSource('example_path/example.txt', 'aasm')
```

这里输入的参数，分别为：

- 目标文件的路径
- 目标文件的类型（是一个AASM的标签）

#### 把源（Source）添加到数据集（Dataset）

至此，我们已经有了一个数据，和一个匹配的标签文件，它们的变量名分别为`raw`和`label`。同时还有一个空数据集`Dataset`。

我们使用：

```python
dataset.add_data(raw, label)
```

把源添加到数据集。



**当然，数据集肯定不止有一个数据。**我们通常会使用相似的文件名来匹配脑电数据和他们的标签，所以一般我们使用一个循环来添加数据：

```python
import glob
from sleepology.source import RawDataFile
from sleepology.source import NihonkohdenLabelSource

dataset = Dataset('example', 'data/', '保留各导联原始数据，只做滤波的数据集。\n降采样到20。')

# 使用glob匹配所有的文件，*表示省略，只要文件名满足*前后的格式，都能一次性匹配
raws = sorted(glob.glob('/home/zhaokuangshi/data/CleanData/raw/*.edf'))
labels = sorted(glob.glob('/home/zhaokuangshi/data/CleanData/label/*.txt'))

# 循环把所有的数据添加进Dataset
for i in range(len(raws)):
    f = RawDataFile(raws[i], 'edf', 30, 100) # 相比实验一，将数据的采样率降低到100，以防止OOM错误
    l = NihonkohdenLabelSource(labels[i], 'aasm')
    dataset.add_data(f,l) # 注意这里脑电数据和标签文件一定要配对！不要配错标签
```



这样，我们仅仅用10行左右的代码，就可以把所有的数据归集到一起，方便我们处理。



### 特征、标签与预处理


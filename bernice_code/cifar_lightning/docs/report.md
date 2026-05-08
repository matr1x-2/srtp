# CIFAR10/100 分类实验报告

## 一、这次实验想做什么

这次主要是把之前比较零散的代码整理成一个能长期复用的训练工程，用 PyTorch Lightning 在 CIFAR10 和 CIFAR100 上做分类。  
除了跑出结果，更重要的是把流程固定下来：同一套数据处理、同一套训练逻辑，切换模型就能直接对比。

我这次对比了 5 个模型：

- simple_cnn（自己写的基线）
- resnet18
- resnet50
- vgg16（做过 CIFAR 适配）
- densenet121

## 二、我最后用的工程结构

项目目录在 `cifar_lightning/`，核心文件如下：

- `train.py`：训练入口，负责串起配置、数据、模型、训练和测试。
- `src/data.py`：数据模块，负责下载、增强、划分 train/val/test。
- `src/models.py`：模型工厂，统一返回不同架构。
- `src/lit_module.py`：LightningModule，封装训练/验证/测试步骤。
- `src/lightning_compat.py`：兼容 `lightning.pytorch` 和 `pytorch_lightning`。
- `configs/cifar.yaml`：默认参数配置。

这样拆开之后，后面想换模型或者换训练参数，不需要反复改很多地方。

## 三、数据处理是怎么做的

数据直接用 `torchvision.datasets` 下载，支持 CIFAR10 和 CIFAR100。

训练集增强我用了这几项（都在 `src/data.py`）：

- `RandomCrop(32, padding=4)`
- `RandomHorizontalFlip()`
- `RandomRotation(10)`
- `Normalize(mean, std)`
- `RandomErasing(p=0.25)`

验证和测试不做随机增强，只保留 `ToTensor + Normalize`。  
训练集和验证集按固定随机种子切分，这样每次跑出来的划分一致，便于对比实验。

## 四、代码是怎么串起来的

### 1. 训练入口 `train.py`

`train.py` 做的事情比较直接：

1. 读 `cifar.yaml`。
2. 读命令行参数，覆盖配置（比如 `--model`、`--max_epochs`）。
3. 初始化 `CIFARDataModule`。
4. 调 `build_model()` 得到网络。
5. 用 `CIFARClassifier` 包装成 LightningModule。
6. 配置日志（TensorBoard）和 checkpoint。
7. `trainer.fit()` 训练，`trainer.test()` 测试。

另外我加了一个细节：会在启动时打印当前实际训练参数（epoch、optimizer、lr 等），这样不会出现“以为自己跑的是 A，实际上是 B”的情况。

### 2. 数据模块 `src/data.py`

这个文件主要解决两件事：

- 统一数据预处理和增强。
- 把 train/val/test 的 DataLoader 统一管理。

写成 DataModule 的好处是，训练逻辑和数据逻辑不耦合。后面如果要加 CutMix、MixUp 或者换数据集，基本只改这一块。

### 3. 模型模块 `src/models.py`

`build_model()` 按字符串返回模型实例。  
我对几个大模型都做了 CIFAR 适配，因为 CIFAR 图是 `32x32`，直接照搬 ImageNet 默认结构通常不太合适。

- ResNet：第一层卷积改成 `3x3, stride=1`，去掉开头 `maxpool`。
- DenseNet121：同样改首层卷积并去掉初始池化。
- VGG16：改成 `vgg16_bn`，并把分类头改小，同时把 `avgpool` 改成 `(1,1)`，避免 CIFAR 上参数头过大。

### 4. Lightning 模块 `src/lit_module.py`

这个类把训练逻辑封装得很清楚：

- `_shared_step()`：统一算 logits、loss、acc。
- `training_step/validation_step/test_step`：分别记录对应阶段指标。
- `configure_optimizers()`：支持 AdamW 或 SGD，调度器支持 cosine。

这样每个实验都走同样的训练框架，模型差异更容易被真实反映出来。

## 五、训练命令（我实际用到的）

```bash
python train.py --dataset cifar10 --model simple_cnn
python train.py --dataset cifar10 --model resnet18
python train.py --dataset cifar10 --model resnet50
python train.py --dataset cifar10 --model vgg16
python train.py --dataset cifar10 --model densenet121
```

在 CIFAR100 上复验：

```bash
python train.py --dataset cifar100 --model resnet18
```

## 六、我遇到的问题和处理

### 1. 训练看起来“卡住”

一开始在 WSL 路径下有过长时间没输出的情况。排查后发现主要和数据读取速度、`num_workers`、磁盘位置有关。  
把测试命令改成小 batch + `num_workers=0` 能快速定位是不是数据问题。

### 2. VGG16 准确率接近 10%

这个问题最开始很明显，测试准确率接近随机猜。  
后面做了两步修正：

1. 改 VGG 结构为 `vgg16_bn + CIFAR 分类头`。
2. 给 VGG 单独配置训练参数（SGD + 更长 epoch）。

之后训练过程至少不会停在 10% 附近。

### 3. 确定性模式报错

报错是：

`adaptive_avg_pool2d_backward_cuda does not have a deterministic implementation`

处理方式是把 deterministic 模式改成 `warn`（默认），必要时命令行加：

```bash
python train.py --dataset cifar10 --model vgg16 --deterministic false
```

## 七、这次实验我自己的结论

1. 先把工程结构整理好，比盲目调参更重要。  
2. CIFAR 这种小图任务，直接套 ImageNet 原始头部往往不理想，做输入端和分类头适配很关键。  
3. 不同模型对优化器和学习率敏感度不一样，尤其是 VGG。  
4. 实验日志和 checkpoint 命名要规范，不然很容易把旧结果和新结果混在一起。

## 八、结果记录表（待补）

| 数据集 | 模型 | val_acc | test_acc | 备注 |
| --- | --- | --- | --- | --- |
| CIFAR10 | simple_cnn |  |  | 基线 |
| CIFAR10 | resnet18 |  |  | 对比模型 |
| CIFAR10 | resnet50 |  |  | 对比模型 |
| CIFAR10 | vgg16 |  |  | 已做 CIFAR 适配 |
| CIFAR10 | densenet121 |  |  | 对比模型 |
| CIFAR100 | 最优模型 |  |  | 泛化验证 |

---

如果后续还有时间，我会继续补两件事：  
一是把每个模型都固定跑 3 次取平均值，减少偶然波动；二是加一张训练曲线图（train/val loss 与 acc）放到正文里，报告会更完整。

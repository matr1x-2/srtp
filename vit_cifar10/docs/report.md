# Transformer Encoder 完成 CIFAR10 分类

## 一、任务目标

本实验使用 Transformer Encoder 完成 CIFAR10 图像分类任务，整体思路参考 ViT（Vision Transformer）：
将图像切分为若干 patch，映射为 token 序列，再通过多层 Transformer Encoder 提取全局特征，最后完成分类。

## 二、模型思路

1. 将 `32x32` 的 CIFAR10 图像切分为多个 patch。
2. 使用卷积投影把每个 patch 映射为 token embedding。
3. 在 patch token 前加入一个 `CLS token`。
4. 给所有 token 加上可学习的位置编码。
5. 将序列输入多层 Transformer Encoder。
6. 取最终的 `CLS token` 送入分类头，输出 10 类概率。

## 三、默认实验设置

- patch size: `4`
- embed dim: `192`
- depth: `6`
- num heads: `3`
- mlp ratio: `4.0`
- dropout: `0.1`
- optimizer: `AdamW`
- learning rate: `3e-4`
- batch size: `128`
- epochs: `30`

## 四、代码结构说明

- `src/model.py`
  - `PatchEmbedding`: 把图像切成 patch 并映射为 token
  - `TransformerEncoderBlock`: 实现多头自注意力和前馈网络
  - `VisionTransformer`: 完整的 ViT 风格分类模型
- `src/data.py`
  - 负责 CIFAR10 下载、数据增强和 train/val/test 划分
- `src/lit_module.py`
  - 封装训练、验证和测试逻辑
- `train.py`
  - 训练入口，负责读取配置并启动训练

## 五、建议实验

### 1. 基础实验

先运行默认配置，记录以下指标：

- `train_loss`
- `train_acc`
- `val_loss`
- `val_acc`
- `test_loss`
- `test_acc`

### 2. 简单消融

可以选择一个变量做小型实验，例如：

- `patch_size=4` vs `patch_size=8`
- `depth=4` vs `depth=6`
- `embed_dim=128` vs `embed_dim=192`

## 六、结果表格

| 设置 | train_acc | train_loss | val_acc | val_loss | test_acc | test_loss |
| --- | --- | --- | --- | --- | --- | --- |
| 默认配置 |  |  |  |  |  |  |
| patch_size=8 |  |  |  |  |  |  |

## 七、结果分析建议

1. patch 更小时，token 数更多，模型更容易捕获细粒度信息，但计算量也更大。
2. depth 增大后，模型表达能力通常更强，但训练难度和过拟合风险也会上升。
3. 与 CNN 相比，Transformer 往往对优化器、学习率和训练轮数更敏感。
4. 如果训练集准确率上升很快但验证集提升有限，通常说明模型开始过拟合。

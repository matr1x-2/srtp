# ViT CIFAR10 项目说明

本项目使用自定义实现的 Transformer Encoder 完成 CIFAR10 图像分类任务。
整体思路参考 ViT（Vision Transformer）：先将图像切分为若干 patch，再映射为 token 序列，加入 `CLS token` 和位置编码后送入多层 Transformer Encoder，最后使用 `CLS token` 的输出来完成分类。

## 项目结构

- `train.py`：训练入口
- `src/data.py`：CIFAR10 数据加载、训练增强与验证测试数据处理
- `src/model.py`：ViT 风格的 Transformer Encoder 分类模型
- `src/lit_module.py`：Lightning 风格的训练封装
- `configs/vit_cifar10.yaml`：默认训练配置文件
- `docs/report.md`：本周实验报告模板

## 快速开始

```bash
cd vit_cifar10
pip install -r requirements.txt
python train.py
```

如果你只想先检查流程是否正常，可以运行：

```bash
python train.py --fast_dev_run
```

## 默认配置

- patch size：`4`
- embedding dimension：`192`
- encoder depth：`6`
- attention heads：`3`
- MLP ratio：`4.0`
- dropout：`0.1`
- optimizer：`AdamW`
- learning rate：`3e-4`
- batch size：`128`
- epochs：`30`

## 本周任务建议

1. 先运行默认配置，确认代码、数据集和训练流程都正常。
2. 记录 `train_loss`、`train_acc`、`val_loss`、`val_acc`、`test_loss` 和 `test_acc`。
3. 如果老师要求做简单消融，可以比较不同的 `patch_size`、`depth` 或 `embed_dim`。

## 你可以怎么讲这个项目

如果你要在汇报时介绍自己的工作，可以这样概括：

1. 自己实现了一个简化版 Vision Transformer，而不是直接调用现成成品模型。
2. 模型包括 patch embedding、位置编码、多头自注意力、前馈网络和分类头。
3. 使用 Transformer Encoder 在 CIFAR10 上完成图像分类任务。
4. 后续可以继续做结构超参数的对比实验。

>>>>>>> 0fc9769 (initial)
>>>>>>>
>>>>>>
>>>>>
>>>>
>>>
>>

# CIFAR 图像分类实验

本项目使用 PyTorch Lightning 完成 CIFAR10/CIFAR100 分类任务，并比较不同网络结构的效果。

## 模型与训练代码

所有模型都通过 `train.py` 统一训练，模型定义放在 `src/models.py`。

| 模型 | 代码位置 | 训练命令 |
| --- | --- | --- |
| `simple_cnn` | `src/models.py` 中的 `SimpleCIFARNet` | `python train.py --dataset cifar10 --model simple_cnn` |
| `resnet18` | `src/models.py` 中的 `build_model` | `python train.py --dataset cifar10 --model resnet18` |
| `resnet50` | `src/models.py` 中的 `build_model` | `python train.py --dataset cifar10 --model resnet50` |
| `vgg16` | `src/models.py` 中的 `build_model`（实际使用 `vgg16_bn` + CIFAR 分类头） | `python train.py --dataset cifar10 --model vgg16` |
| `densenet121` | `src/models.py` 中的 `build_model` | `python train.py --dataset cifar10 --model densenet121` |

## 快速开始

```bash
cd cifar_lightning
pip install -r requirements.txt
python train.py --dataset cifar10 --model resnet18
```

## 实验建议

1. 先用 `simple_cnn` 跑通流程，确认数据、增强和训练都正常。
2. 在 CIFAR10 上依次比较 `resnet18`、`resnet50`、`vgg16`、`densenet121`。
3. 选出表现最好的模型，再在 CIFAR100 上重复实验。
4. 记录准确率、loss、训练时间和收敛情况。

## 目录说明

- `train.py`：训练入口
- `src/data.py`：数据模块
- `src/models.py`：模型构建
- `src/lit_module.py`：Lightning 训练封装
- `configs/cifar.yaml`：默认配置
- `docs/experiment.md`：实验过程记录模板

## 常见问题

如果出现类似报错：

`adaptive_avg_pool2d_backward_cuda does not have a deterministic implementation`

说明当前模型在严格确定性模式下不支持某些 CUDA 算子。可用以下任一方式：

1. 使用默认配置（已设为 `deterministic: warn`）。
2. 命令行显式关闭确定性：
   `python train.py --dataset cifar10 --model vgg16 --deterministic false`

如果某次实验结果异常低（例如约 10%），建议先删除该模型旧 checkpoint 再重跑，避免和旧实验文件混淆：

`rm -rf checkpoints/cifar10/vgg16`

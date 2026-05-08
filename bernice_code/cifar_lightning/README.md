# CIFAR 图像分类实验

本项目使用 PyTorch Lightning 完成 CIFAR10/CIFAR100 分类任务，并比较不同网络结构的效果。

## 模型与训练代码

所有模型都通过 `train.py` 统一训练，模型定义放在 `src/models.py`。

| 模型 | 代码位置 | 训练命令 |
| --- | --- | --- |
| `simple_cnn` | `src/models.py` 中的 `SimpleCIFARNet` | `python train.py --dataset cifar10 --model simple_cnn` |
| `resnet18` | `src/models.py` 中的 `build_model` | `python train.py --dataset cifar10 --model resnet18` |
| `resnet50` | `src/models.py` 中的 `build_model` | `python train.py --dataset cifar10 --model resnet50` |
| `vgg16` | `src/models.py` 中的 `build_model` | `python train.py --dataset cifar10 --model vgg16` |
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

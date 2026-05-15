# dnn-prediction

基于 PyTorch + Lightning 完成手写数字识别（MNIST）任务，分别实现并对比 MLP 与 CNN，两种模型均支持训练、验证、测试，并将 loss 曲线写入 TensorBoard。

## 1. 任务要求对应

- MLP 模型实现：`model/net_mlp.py`
- CNN 模型实现：`model/net_cnn.py`
- Lightning 训练流程（fit/test）：`main.py`
- MNIST 数据模块：`dataset.py`
- TensorBoard 日志输出：`logs/`
- 模型权重保存：`checkpoints/`

## 2. 环境安装

```bash
cd dnn-prediction
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-cpu.txt
```

## 3. 运行方式

1. 训练并测试 MLP

```bash
python main.py --config configs/config.yaml --model mlp
```

2. 训练并测试 CNN

```bash
python main.py --config configs/config.yaml --model cnn
```

3. 指定 epoch 数量（示例：1 epoch）

```bash
python main.py --config configs/config.yaml --model mlp --max_epochs 1
python main.py --config configs/config.yaml --model cnn --max_epochs 1
```

## 4. TensorBoard 曲线

```bash
tensorboard --logdir logs
```

浏览器打开命令行输出地址（默认通常为 http://localhost:6006）。

## 5. 项目结构

```text
dnn-prediction/
├── configs/
│   └── config.yaml          # 训练配置（超参数、训练策略）
├── model/
│   ├── __init__.py
│   ├── net_mlp.py           # MLP 模型
│   └── net_cnn.py           # CNN 模型
├── dataset.py               # 数据模块（MNISTDataModule）
├── main.py                  # Lightning 训练入口（fit/test）
├── requirements.txt         # 通用依赖
├── requirements-cpu.txt     # CPU 版 torch/torchvision
└── README.md
```

## 6. 实验结果示例（本地 1 epoch）

- MLP:
	- test_acc: 0.9615
	- test_loss: 0.1216
- CNN:
	- test_acc: 0.9837
	- test_loss: 0.0507

## 7. 提交检查清单

- 能通过参数 `--model mlp` 与 `--model cnn` 切换模型。
- 训练结束后有测试指标输出（`test_acc`、`test_loss`）。
- `logs/` 下有 TensorBoard 事件文件，可查看 `train_loss`、`val_loss` 曲线。
- `checkpoints/` 下保存最佳权重与 last 权重。
- README 提供完整复现命令。

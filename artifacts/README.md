# Runtime artifacts

这里保存从共享盘复制到临时容器本地缓存所需的大文件；二进制本身不进入 Git。

- `pretrained/resnet18-f37072fd.pth`：TorchVision ResNet-18 ImageNet 权重，
  SHA-256 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`。
- `pretrained/pi05/README.md`：已在共享 Hugging Face cache 中的 Pi05 base、LIBERO
  finetuned 权重和 PaliGemma tokenizer 的 revision、路径与用途；大权重本身不重复复制到仓库。
- `wheels/mujoco-3.3.1-cp312-...whl`：RoboCasa 1.0.1 要求的 MuJoCo 3.3.1，
  SHA-256 `4e7381d186402cc7591de49078f3c911b1fea2a9f7b5156db0e1aa934d7bcd24`。
- `wheels/numpy-2.2.5-cp312-...whl`：RoboCasa 1.0.1 要求的 NumPy 2.2.5，
  SHA-256 `3a801fef99668f309b88640e28d261991bfad9617c27beda4a3aec4f217ea073`。

ACT/Diffusion 启动只复制约 45 MiB 的 ResNet 权重；Pi05 首次实际训练会复制约
14G base 权重到容器本地 SSD。MuJoCo/NumPy wheels 只在 rollout 评测前安装到
容器本地 Python overlay，不会修改共享环境。

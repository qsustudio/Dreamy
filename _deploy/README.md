# x-transformers 部署说明

在 `Dreamy` 工作区里部署 [lucidrains/x-transformers](https://github.com/lucidrains/x-transformers)
（Phil Wang 的 Transformer 实现库，包含大量论文里的实验特性）。

**训练目标是 Google T4（Colab / GCE / Kaggle）**，本机只是开发机。所以分两条线：

| 场景 | 脚本 | 目标 |
| --- | --- | --- |
| 本机（无 GPU，写代码/跑测试） | `_deploy\setup.ps1` | Windows + CPU torch |
| **T4 训练机** | **`_deploy\setup_t4.sh`** | **Linux + CUDA torch** |

## T4 上怎么跑

T4 是 **Turing 架构（compute capability 7.5 / sm_75）**。把本工作区（至少 `_deploy/` 和
`x-transformers/`）传到 T4 机器上，然后：

```bash
# Colab 里直接
!bash _deploy/setup_t4.sh

# 或普通 Linux
bash _deploy/setup_t4.sh
```

脚本会建 `.venv-t4` 虚拟环境、装 CUDA 版 torch、以 editable 方式装 x-transformers，
并跑一次 GPU 冒烟测试（会打印 `sm_75 -> OK` 确认 kernel 真的可用）。

### 关键：CUDA 版本怎么选

这是我查证的重点，别选错：

| torch | CUDA wheel | 最低驱动 | 说明 |
| --- | --- | --- | --- |
| **2.14.x** | **cu126** | 525+ | **默认推荐**，兼容绝大多数 T4 镜像 |
| 2.15 | cu132 / cu134 | **580+** | T4 本身没问题，但驱动旧就装不上 |

PyTorch 2.15 起 **不再发布 cu126 wheel**，且 CUDA 13 系列要求驱动 ≥580。
Colab/Kaggle 的 T4 驱动经常低于 580，所以脚本**默认锁 cu126 + torch 2.14.x**。
要换新栈：

```bash
CUDA=cu132 bash _deploy/setup_t4.sh
```

选定后先确认驱动版本：

```bash
nvidia-smi --query-gpu=driver_version --format=csv,noheader
```

驱动是 525~579 → 必须用 cu126。

### 验证 GPU 真的可用

```bash
.venv-t4/bin/python -c "
import torch
print(torch.__version__, torch.version.cuda)
print('sm_75 in build:', 'sm_75' in torch.cuda.get_arch_list())
print(torch.cuda.get_device_name(0))
"
```

如果 `sm_75 in build` 是 `False`，说明装的是不含 Turing kernel 的构建，回到 cu126 / torch 2.14。

## 本机（开发机）现状

本机**没有 NVIDIA GPU** —— 它只用来写代码和跑测试，不是训练机。

| 项目 | 值 |
| --- | --- |
| 源码 | `x-transformers/`（commit `ee6b76f`） |
| 版本 | `x-transformers==2.31.11`（与 PyPI 最新一致） |
| 环境 | `.venv/`，Python 3.14.2 |
| PyTorch | `2.15.0+cpu` |
| 安装方式 | editable —— 改源码立即生效 |
| 测试 | **222 passed, 5 skipped, 0 failed** |

```powershell
.\.venv\Scripts\Activate.ps1
```

CPU 环境不兼容 T4 的 GPU 代码路径检查，但**单元测试覆盖了模型逻辑**，本机跑通说明
x-transformers 本身在这套依赖下是健康的；T4 上只需验证 CUDA 那部分（脚本已做）。

## 本机环境特有的三个坑（已绕过）

常规安装流程在这台机器上会直接崩，`setup.ps1` 已全部处理。记录在此以免重复踩坑。

### 1. git / curl 的 TLS 失败

默认 schannel 后端报 `SEC_E_NO_CREDENTIALS (0x8009030e)`，任何 https 请求都失败。
Python 的 `urllib` 与 npm 正常。

**绕过**：git 加 `-c http.sslBackend=openssl`；下载一律用 Python `urllib`，不要用 `curl`。

### 2. `tempfile.mkdtemp()` 创建的目录连自己都读不了

最麻烦的一个。工作区处在沙箱中，`tempfile.mkdtemp()` 创建的目录会带一套 DACL，
**连当前用户自己都没权限**（`icacls` 直接报 "Access is denied"，既读不了也无需修复）。
而 `os.makedirs()` 创建的目录权限完全正常。

pip / hatchling / setuptools 全都依赖 `mkdtemp()`，于是：

```
OSError: [Errno 13] Permission denied: '...\pip-unpack-xxxx\torch-...whl.metadata'
```

**绕过**：`_deploy/pip_wrapper.py` 把 `tempfile.mkdtemp` 换成基于 `os.makedirs` 的实现，
再正常调用 pip；`TEMP`/`TMP` 指向 `.venv\pip-tmp`。

```powershell
$env:TEMP="$PWD\.venv\pip-tmp"; $env:TMP="$PWD\.venv\pip-tmp"
.\.venv\Scripts\python.exe _deploy\pip_wrapper.py install <包名>
```

同一原因，venv 的 `ensurepip` 也失败，`setup.ps1` 用 `zipfile` 直接解压 pip wheel 引导。
**这个坑会复发** —— 以后 `pip install` 报 `pip-unpack` 权限错就走 wrapper。

### 3. `torch.compile` 需要 MSVC 且不能有脏缓存

`UniversalPretrainWrapper`（`up_wrapper.py:98`）用了 `@torch.compile`。Windows 上需要
`cl.exe`，它不在 PATH 里，要先执行：

```powershell
& "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
```

本机 VS 18 Community 的编译器验证可用（`cl` 19.51）。**注意**：曾在无编译器的环境跑过一次，
会在 `.venv\pip-tmp\torchinductor_*` 留下坏缓存导致后续一直失败，删掉该目录即可恢复。

T4（Linux）上这条不适用 —— Linux 的 CUDA torch 自带 Triton 后端，无需本地编译器。

## 其它

- 训练脚本在 `x-transformers/train_*.py`，例如 `train_enwik8.py`、`train_parity.py`。
  部分示例脚本需要额外依赖：`pip install -e "x-transformers[examples]"`。
- T4 显存 16 GB：跑 README 里的大模型示例前注意 batch size 和 seq len。
- `.wheels/` 存了本机用的 torch CPU wheel；T4 上用不到，可删。
- 上游 `pyproject.toml` 的 Repository 字段指向 Codeberg，GitHub 镜像是同步的。

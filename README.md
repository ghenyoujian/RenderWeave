# RenderWeave

RenderWeave 是面向 3DGS、视频和图形宿主的原生渲染增强运行时研究项目。项目提供 C++20 核心、C ABI、Python 接口和实验性 CUDA 执行后端。本预览版可用于非商业研究和实验。

| 内容 | 使用入口 |
| --- | --- |
| CPU 核心示例 | 无需 CUDA 或模型，演示张量契约、FP8 数值和时序会话状态 |
| CUDA NR、SR | Windows 与 Linux 的编译入口。运行模型需另备对应的注册资产 |
| FG 遮挡子图 | CUDA 源码覆盖输入打包、32 节点遮挡网和输出发布 |
| 当前设计 | [架构和实测](DESIGN.md) · [V3 设计原文](docs/design/RenderWeave_Design_V3.md) |
| 后续开发 | [三阶段工作清单](ROADMAP.md) |

## 快速开始

Linux/WSL 使用 CMake 3.25+、C++20 编译器和 Python 3.10+。

```bash
cmake -S . -B build/cpu -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpu --parallel
ctest --test-dir build/cpu --output-on-failure
PYTHONPATH=python python3 -m renderweave.demo --library build/cpu/librw_core.so
```

CUDA 构建需要 CUDA Toolkit、C++ 编译器和 OpenSSL。Windows 使用 Visual Studio x64 开发环境。

```text
python tools/dev/deploy.py --detect
python tools/dev/deploy.py
```

安装目录的 `bin/rw-cuda-numeric-probe` 可以在本机直接检查 CUDA 的 FP8、矩阵和采样算术。Windows 下对应 `bin/rw-cuda-numeric-probe.exe`。

SR K/LDR/2× 可用 `--sr-reconstruction PATH` 指定匹配的重建源码。模型加载通过公开 C ABI 或 `renderweave.NativeRuntime`、`renderweave.NativeSRRuntime`，使用明确传入的模型资产及身份。发行包包含源码、构建入口和文档。数据集、模型权重和厂商载体由使用者另行提供。

## 许可证

RenderWeave 自有源码和文档使用 [PolyForm Noncommercial 1.0.0](LICENSE)。第三方组件沿用 [各自的许可证和声明](NOTICE)。

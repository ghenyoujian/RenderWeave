# 当前架构与实测

RenderWeave 将宿主适配、功能图、原生 CUDA 执行和会话状态分开维护。C ABI 是下游入口，C++ 提供 RAII 封装，Python 调用同一动态库。完整设计和阶段目标见 [V3 设计原文](docs/design/RenderWeave_Design_V3.md)。

## 实现

- **Core / SDK**。Core 管理张量、导引、预算和状态事务。NR、SR 各自暴露 Runtime、Session 和 Job。每张 GPU 共享只读参数，每槽独立持有 CUDA stream、图和工作区。会话有独立历史和 epoch，完成后按前驱提交。
- **NR / SR**。NR 实现原始 HT 图，H200 路径采用 CUDA Graph、Tensor Core 和设备驻留历史。SR 候选固定为 K/LDR/2×，串联预处理、11 层主干、重建与时序反馈。重建源码在编译时提供。
- **宿主**。gsplat 提供颜色、深度和相机运动。视频路径保留 PTS 与场景切换。D3D12 SR 以共享资源和 fence 交付纹理。
- **FG / RR**。FG 遮挡路径已串联 34 个节点。FG 全局合成和 RR 的原生图属于后续工作。

## 已有证据

| 场景 | 观测 |
| --- | --- |
| 单 H200，真实 gsplat＋1080p 时序 NR＋CPU 输出 | 两组各 600 帧，31.329 和 31.277 FPS |
| 单 H200，32 份 GPU 历史、8 路各 15 FPS，双槽、两个在途 | 34.5 FPS 准时交付。已交付帧 P99 58.98 ms。480 次到达中 340 次被拒绝，全体到达口径 P99 为正无穷 |
| 单 H200，原始 FP16 SR 的 1080p CUDA 驻留 | 600 帧达到 157.00 FPS |
| RTX 3070 Laptop，Windows D3D12 SR 纹理回放 | 1080p 35.15 FPS，计时止于纹理交付 |
| Mip360 room，四视角同相机照片 | 原渲染 31.15 dB，H200 NR 26.35 dB。CUDA 对同输入 CPU 参考 45.11 dB |
| RTX 3070 Laptop，FG 遮挡链 | 64×64 样例的三张 surface 共 20,480 个分量一致。中段用原始 SM86 模块比较，两端用独立 CPU 参考 |

性能和画质覆盖具体的运行配置。当前阶段的验收范围、尚缺数据和下一步见 [工作清单](ROADMAP.md)。

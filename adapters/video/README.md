# MIP360 gsplat → 原生 SDK → FFmpeg

适配器使用 PyAV 编解码并保留 SDR RGB 与时间戳。默认使用原始 NR HT 单帧模式，连续 gsplat 序列可启用 NR SDR 时序候选或原始 SR K/LDR/2×。

从本地固定 release 经 Slurm 安装独立环境：

```bash
bash tools/h200/install_media_environment.sh --output-dir "$PROJECT_STORAGE_ROOT/runs/MEDIA_ENV_RUN"
```

环境位于 `$PROJECT_STORAGE_ROOT/envs/renderweave-media`。PyAV16.1.0、numpy2.3.5、Pillow12.0.0 固定 CPython3.11 Linux wheel 哈希；安装导出实际 FFmpeg 版本、codec 可用性及 Conda/pip 锁。安装 SDK/媒体包时可在该环境执行本地源码构建，或直接用下面的源码 CLI；共享库必须显式指向本项目已编译产物。原始参数仍引用公共 dataset/weights。

首先将真实 MIP360 gsplat 渲染 manifest 编码成无音轨的可重放媒体。相机视图原本没有采集 PTS，必须明确提供生成时间线，下面为15Hz：

```bash
"$PROJECT_STORAGE_ROOT/envs/renderweave-media/bin/python" adapters/video/run.py encode \
  --sequence-manifest "$PROJECT_STORAGE_ROOT/runs/GSPLAT_RUN/render/inputs.json" \
  --generated-fps 15 --color-profile srgb-bt709-full-rgb8 \
  --output-dir "$PROJECT_STORAGE_ROOT/runs/VIDEO_INPUT_RUN" --run-id VIDEO_INPUT_RUN
```

直接增强原始渲染序列时，同样提供 `--sequence-manifest --generated-fps`；增强刚编码的视频时，保持解码获得的原始 PTS/time_base，另附绑定视频 SHA 的回执：

```bash
"$PROJECT_STORAGE_ROOT/envs/renderweave-media/bin/python" adapters/video/run.py enhance \
  --input-video "$PROJECT_STORAGE_ROOT/runs/VIDEO_INPUT_RUN/input.nut" \
  --input-provenance "$PROJECT_STORAGE_ROOT/runs/VIDEO_INPUT_RUN/adapter.json" \
  --model "$PROJECT_STORAGE_ROOT/runs/MODEL_RUN/model_manifest.json" \
  --plan "$PROJECT_STORAGE_ROOT/runs/PLAN_RUN/plan.json" \
  --sdk-library "$PROJECT_STORAGE_ROOT/build/SDK_BUILD/librw_nr_sdk.so" \
  --color-profile srgb-bt709-full-rgb8 --save-raw \
  --output-dir "$PROJECT_STORAGE_ROOT/runs/VIDEO_SDK_RUN" --run-id VIDEO_SDK_RUN
```

默认输出 NUT+FFV1。编码后重新解码，核验所有 RGB 字节、帧数、flush/EOS 和有理时间戳。Matroska 往往采用1/1000时间基；选择它必须预先使用 `--timestamp-policy round-within-one-tick`，实际误差不得超过一个实际输出 tick。原始 PTS/time_base 始终完整保留在 audit.jsonl，不因编码 time base 改写。显式 `--vfr-steps 1,2,1,3` 可生成标明来源的 VFR 测试输入；这不是相机采集时间。

连续序列先用 `tools/h200/run_gsplat_sequence.sh RUN_ID ABS_FRESH_OUTPUT_DIRECTORY` 渲染带 PTS、切场和运动数据的 manifest。编码命令使用该 manifest 自带的时间线，省略 `--generated-fps`。增强时在上述 `enhance` 命令加 `--sdr-temporal-head`，或者以同一 manifest 直接调用 `enhance --sequence-manifest`。视频输入使用本适配器生成的精确时间戳 lossless 回执定位外部运动文件。两种入口都在首帧、切场和显式重置处开启新会话 epoch，seek 后从第一个实际解码帧开始。需要原始 HT 注册、匹配尺寸的 plan 和带时序候选的新 SDK。

输入色彩为 sRGB transfer、BT.709 primaries、full-range RGB8。音轨可通过 `--audio-policy drop` 明确丢弃。SDR 时序模式使用渲染器的 motion 与有效性，深度用于渲染器生成该 guide。

## 原始 SR 2×

对有原始 PTS、运动和深度的连续 gsplat manifest，直接执行以下命令。`.05 50` 是本项目 Mip360 room 样例的固定裁剪面。其他场景填写对应的相机空间裁剪面，单位与渲染器的 camera-Z 一致。

```bash
"$PROJECT_STORAGE_ROOT/envs/renderweave-media/bin/python" adapters/video/run.py enhance \
  --feature sr --sr-depth-range .05 50 \
  --sequence-manifest "$PROJECT_STORAGE_ROOT/runs/GSPLAT_RUN/gsplat-sequence/inputs.json" \
  --sdk-library "$PROJECT_STORAGE_ROOT/build/SDK_BUILD/librw_sr_sdk.so" \
  --color-profile srgb-bt709-full-rgb8 --device-budget-mib 768 \
  --output-dir "$PROJECT_STORAGE_ROOT/runs/SR_VIDEO_RUN" --run-id SR_VIDEO_RUN
```

源是无损视频时，将 `--sequence-manifest` 换为 `--input-video` 和 `--input-provenance`，指向同一序列生成的 `input.nut` 与 `adapter.json`。适配器用原有 PTS 和 RGB 哈希连接渲染导引。模型注册默认引用公共 `dataset/weights` 下的同一份原始 K 权重和重建滤波器。输出宽高各翻倍，PTS 沿用源序列。视频 RGB 转换只在编码边界截到显示范围，可用 `--save-raw` 保留 SR 原始浮点 RGBA。

在上面的 SR 命令加上 `--sr-original-fp16-backbone`，可启用 SDK 的原始 FP16 主干算术。适配器报告会记录所选路径。

视频适配器会自动使用 SDK 的原生 RGB8 转换。用 `--sr-display numpy` 可对照原路径，用 `--sr-display native` 明确要求新入口。逐帧报告分别记录摘要与转换、原始浮点 SHA-256、编码写入的耗时。

加 `--sr-overlap-write` 可同时预取一帧输入，并让一帧的 RGB8 转换、浮点摘要和 FFV1 写入与 SDK 执行重叠。输入线程完成解码、颜色归一化和导引准备，主线程按原有帧序提交并处理场景重置。适配器轮换两个宿主 RGBA 缓冲区，按帧序核收一个待完成的交付任务。报告记录预取等待、输入和 SDK 时间窗口，以及转换、摘要和编码时间。

每次请求保留解码和归一化输入摘要、浮点输出摘要、SDK 时间及状态。报告记录从开始解码到编码器排空的耗时，包含初始化。等待超时和随后交付的帧单独计数。单帧模式每帧重置 epoch。失败输入另存于运行目录。

实际边界合同需在已编译原生 SDK 上运行：

```bash
"$PROJECT_STORAGE_ROOT/envs/renderweave-media/bin/python" adapters/video/codec_contracts.py \
  --sequence-manifest "$PROJECT_STORAGE_ROOT/runs/GSPLAT_RUN/render/inputs.json" \
  --model "$PROJECT_STORAGE_ROOT/runs/MODEL_RUN/model_manifest.json" \
  --plan "$PROJECT_STORAGE_ROOT/runs/PLAN_RUN/plan.json" \
  --sdk-library "$PROJECT_STORAGE_ROOT/build/SDK_BUILD/librw_nr_sdk.so" \
  --output-dir "$PROJECT_STORAGE_ROOT/runs/MEDIA_CONTRACT_RUN" --run-id MEDIA_CONTRACT_RUN
```

它使用真实渲染帧的明确重复回放覆盖 CFR/VFR、真实 B 帧、seek、timeout、flush/EOS，以及缺时间和 HDR metadata 拒绝。没有实际产生 B 帧、没有观测到超时或缺 codec 时均保持失败/未覆盖，不把参数请求当证据。重复相机图不计独立质量序列；音轨存在时的拒绝分支仍需单独实测。

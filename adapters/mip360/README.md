# Mip360 与 gsplat

将 Mip360 场景和对应的预训练 Gaussian PLY 放在自己可读取的位置。使用具有匹配相机和坐标系的资产。资产清单可从 [Gaussian 模板](gaussian_assets.template.json) 开始，场景划分见 [split](splits.initial.json)。

```bash
python adapters/gsplat/render_mip360.py --help
python adapters/gsplat/native_pipeline.py --help
```

传入 COLMAP 相机或适配入口要求的帧清单，Gaussian 资产提供 `scene_id`、文件路径、SHA-256 与 `mip360_colmap_world` 坐标系。渲染依赖 PyTorch、gsplat、NumPy、Pillow 与 plyfile。启用模型处理时使用明确提供并匹配语义计划的 CUDA SDK 和模型资产。

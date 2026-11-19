# Temporal mathematical source

The equations in `temporal_probe_ops.cu` are adapted from the MIT-licensed
Overclockers/DLSSNR-AMD commit `191b97b54175d50c7ec5992b54322dd3eb0ef8ac`:

- `linux/shaders/rdna4/include/temporal_history.glsl`, SHA256 `c467ff5fe489bfc0b6fa9b1bca742a8216f57606a048a370f9edbc7251eff3fe`.
- `linux/shaders/rdna4/include/image_input.glsl`, SHA256 `57d4244393c6182f23347e24866094b161f51a97f239c2b8700f14823ce999ed`.
- `linux/shaders/rdna4/fswin_t.comp`, SHA256 `9dbe3b581c21978dae1621c16ee2cc2b572b8aa91f7c98826de1e8b94a4ecfc6`.

Repository: https://github.com/Overclockers/DLSSNR-AMD/tree/191b97b54175d50c7ec5992b54322dd3eb0ef8ac

This project does not link that Vulkan runtime or execute its shaders. The new
probe uses explicit software FP32 interpolation, not the original hardware
texture's fixed-point interpolation or the unresolved native history codec.

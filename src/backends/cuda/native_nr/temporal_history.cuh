// SDR proxy history arithmetic based on OpenDLSS-NR's MIT-licensed
// demo/shaders/nr_common.glsl at 9d08f4184bbcb9d858e2fb7a7834ec0837a9d2f1.
// Uses the HWC adapter's pixel coordinates and software texture interpolation.
namespace temporal_history {

struct Axis {
    float low, middle, high;
    float low_weight, middle_weight, high_weight;
};

__device__ Axis filter_axis(float position, int extent) {
    const float base = floorf(position);
    const float t = position - base;
    const float square = t * t;
    const float cube = t * square;
    const float low = __fmaf_rn(t + cube, -0.5f, square);
    const float first = (cube * 1.5f - square * 2.5f) + 1.0f;
    const float high = (cube - square) * 0.5f;
    const float second = ((1.0f - low) - first) - high;
    const float middle = first + second;
    const float last = static_cast<float>(extent - 1);
    return {fminf(fmaxf(base - 1.0f, 0.f), last),
            fminf(fmaxf(base + second / middle, 0.f), last),
            fminf(fmaxf(base + 2.0f, 0.f), last), low, middle, high};
}

__device__ float sample(const float* plane, int stride, int width, int height, float px, float py,
                        Axis x, Axis y) {
    if (px == floorf(px) && py == floorf(py))
        return plane[static_cast<size_t>(py) * stride + static_cast<int>(px)];
    const auto read = [&](float u, float v) {
        const int x0 = static_cast<int>(u), y0 = static_cast<int>(v);
        const int x1 = min(x0 + 1, width - 1), y1 = min(y0 + 1, height - 1);
        const float* a = plane + static_cast<size_t>(y0) * stride;
        const float* b = plane + static_cast<size_t>(y1) * stride;
        const float top = __fmaf_rn(u - x0, a[x1] - a[x0], a[x0]);
        const float bottom = __fmaf_rn(u - x0, b[x1] - b[x0], b[x0]);
        return __fmaf_rn(v - y0, bottom - top, top);
    };
    const float left = x.low_weight * y.middle_weight;
    const float top = y.low_weight * x.middle_weight;
    const float middle = x.middle_weight * y.middle_weight;
    const float bottom = y.high_weight * x.middle_weight;
    const float right = x.high_weight * y.middle_weight;
    float value = read(x.middle, y.low) * top;
    value = __fmaf_rn(read(x.low, y.middle), left, value);
    value = __fmaf_rn(read(x.middle, y.middle), middle, value);
    value = __fmaf_rn(read(x.middle, y.high), bottom, value);
    value = __fmaf_rn(read(x.high, y.middle), right, value);
    return value / ((((top + left) + middle) + bottom) + right);
}

__global__ void reproject(const float* previous, const float* motion, const float* validity,
                          float* warped, int width, int height, int valid_width, int valid_height) {
    const size_t pixel = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t pixels = static_cast<size_t>(width) * height;
    if (pixel >= pixels) return;
    const int x = static_cast<int>(pixel % width), y = static_cast<int>(pixel / width);
    const int reflected_x = x < valid_width ? x : max(0, 2 * valid_width - x - 2);
    const int reflected_y = y < valid_height ? y : max(0, 2 * valid_height - y - 2);
    const size_t guide = static_cast<size_t>(reflected_y) * width + reflected_x;
    const float px = reflected_x + motion[guide], py = reflected_y + motion[pixels + guide];
    const bool usable = (!validity || validity[guide] >= 0.5f) &&
        px >= -0.5f && px < valid_width - 0.5f && py >= -0.5f && py < valid_height - 0.5f;
    warped[3 * pixels + pixel] = usable ? 1.f : 0.f;
    if (!usable) return;
    const Axis horizontal = filter_axis(px, valid_width), vertical = filter_axis(py, valid_height);
    for (int c = 0; c < 3; ++c)
        warped[c * pixels + pixel] = sample(previous + c * pixels, width, valid_width, valid_height,
                                             px, py, horizontal, vertical);
}

__device__ float gate(float logit, float scale) {
    float exponential;
    asm("ex2.approx.ftz.f32 %0,%1;" : "=f"(exponential) : "f"(logit * __uint_as_float(0xbfb8aa3b)));
    float reciprocal;
    asm("rcp.approx.ftz.f32 %0,%1;" : "=f"(reciprocal) : "f"(1.f + exponential));
    return reciprocal * scale;
}

__global__ void compose(const float* image, const float* head, const float* warped,
                        float* output, float* history, float scale,
                        int width, int height, int valid_width, int valid_height) {
    const size_t pixel = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t pixels = static_cast<size_t>(width) * height;
    if (pixel >= pixels) return;
    const int x = static_cast<int>(pixel % width), y = static_cast<int>(pixel / width);
    const bool usable = warped && warped[3 * pixels + pixel] != 0.f;
    const float weight = usable ? gate(head[3 * pixels + pixel], scale) : 0.f;
    for (int c = 0; c < 3; ++c) {
        const size_t offset = c * pixels + pixel;
        const float proxy = hround(image[offset]);
        const float centered = __fmaf_rn(proxy, 0.125f, -0.0625f);
        const float residual_mix = __fmaf_rn(head[offset], post_residual_scale * 0.125f, centered);
        float value = clamp_nan(residual_mix * 8.f + 0.5f, 0.f, 1.f);
        if (usable) value = __fmaf_rn(weight, warped[offset] - value, value);
        const float published = __half2float(__float2half_rz(value));
        if (x < valid_width && y < valid_height)
            history[(static_cast<size_t>(y) * valid_width + x) * 3 + c] = published;
        output[offset] = clamp_nan(published, 0.f, 1.f);
    }
}

} // namespace temporal_history

//! Image processing contract tests with self-contained algorithm implementations.
//!
//! Each test exercises real mathematical properties of image processing algorithms
//! operating on raw `Vec<f32>` pixel buffers with width/height metadata.

#[cfg(test)]
mod tests {
    use std::f32::consts::PI;

    // ========================================================================
    // Helper: 2D Convolution (naive O(n^2 * k^2), zero-padded border)
    // ========================================================================

    fn conv2d(image: &[f32], w: usize, h: usize, kernel: &[f32], kw: usize, kh: usize) -> Vec<f32> {
        assert_eq!(image.len(), w * h);
        assert_eq!(kernel.len(), kw * kh);
        let half_kw = kw / 2;
        let half_kh = kh / 2;
        let mut out = vec![0.0_f32; w * h];
        for y in 0..h {
            for x in 0..w {
                let mut sum = 0.0_f64;
                for ky in 0..kh {
                    for kx in 0..kw {
                        let iy = y as isize + ky as isize - half_kh as isize;
                        let ix = x as isize + kx as isize - half_kw as isize;
                        if iy >= 0 && iy < h as isize && ix >= 0 && ix < w as isize {
                            let pixel = image[iy as usize * w + ix as usize];
                            sum += f64::from(pixel) * f64::from(kernel[ky * kw + kx]);
                        }
                    }
                }
                out[y * w + x] = sum as f32;
            }
        }
        out
    }

    // ========================================================================
    // Helper: Separable 2D Convolution (horizontal then vertical pass)
    // ========================================================================

    fn separable_conv2d(
        image: &[f32],
        w: usize,
        h: usize,
        h_kernel: &[f32],
        v_kernel: &[f32],
    ) -> Vec<f32> {
        assert_eq!(image.len(), w * h);
        let hk = h_kernel.len();
        let half_hk = hk / 2;

        // Horizontal pass
        let mut temp = vec![0.0_f32; w * h];
        for y in 0..h {
            for x in 0..w {
                let mut sum = 0.0_f64;
                for k in 0..hk {
                    let ix = x as isize + k as isize - half_hk as isize;
                    if ix >= 0 && ix < w as isize {
                        sum += f64::from(image[y * w + ix as usize]) * f64::from(h_kernel[k]);
                    }
                }
                temp[y * w + x] = sum as f32;
            }
        }

        // Vertical pass
        let vk = v_kernel.len();
        let half_vk = vk / 2;
        let mut out = vec![0.0_f32; w * h];
        for y in 0..h {
            for x in 0..w {
                let mut sum = 0.0_f64;
                for k in 0..vk {
                    let iy = y as isize + k as isize - half_vk as isize;
                    if iy >= 0 && iy < h as isize {
                        sum += f64::from(temp[iy as usize * w + x]) * f64::from(v_kernel[k]);
                    }
                }
                out[y * w + x] = sum as f32;
            }
        }
        out
    }

    // ========================================================================
    // Helper: Gaussian kernel generation (1D)
    // ========================================================================

    fn gaussian_kernel_1d(sigma: f32) -> Vec<f32> {
        let radius = ((3.0 * sigma).ceil() as usize).max(1);
        let size = 2 * radius + 1;
        let sigma_sq = f64::from(sigma) * f64::from(sigma);
        let mut kernel = vec![0.0_f32; size];
        let mut sum = 0.0_f64;
        for i in 0..size {
            let x = i as f64 - radius as f64;
            let v = (-x * x / (2.0 * sigma_sq)).exp();
            kernel[i] = v as f32;
            sum += v;
        }
        for k in &mut kernel {
            *k = (*k as f64 / sum) as f32;
        }
        kernel
    }

    fn gaussian_blur(image: &[f32], w: usize, h: usize, sigma: f32) -> Vec<f32> {
        let kernel = gaussian_kernel_1d(sigma);
        separable_conv2d_clamp(image, w, h, &kernel, &kernel)
    }

    /// Separable convolution with clamped borders (for gaussian blur consistency).
    fn separable_conv2d_clamp(
        image: &[f32],
        w: usize,
        h: usize,
        h_kernel: &[f32],
        v_kernel: &[f32],
    ) -> Vec<f32> {
        assert_eq!(image.len(), w * h);
        let hk = h_kernel.len();
        let half_hk = hk / 2;

        // Horizontal pass with clamped border
        let mut temp = vec![0.0_f32; w * h];
        for y in 0..h {
            for x in 0..w {
                let mut sum = 0.0_f64;
                for k in 0..hk {
                    let ix = (x as isize + k as isize - half_hk as isize).clamp(0, w as isize - 1)
                        as usize;
                    sum += f64::from(image[y * w + ix]) * f64::from(h_kernel[k]);
                }
                temp[y * w + x] = sum as f32;
            }
        }

        // Vertical pass with clamped border
        let vk = v_kernel.len();
        let half_vk = vk / 2;
        let mut out = vec![0.0_f32; w * h];
        for y in 0..h {
            for x in 0..w {
                let mut sum = 0.0_f64;
                for k in 0..vk {
                    let iy = (y as isize + k as isize - half_vk as isize).clamp(0, h as isize - 1)
                        as usize;
                    sum += f64::from(temp[iy * w + x]) * f64::from(v_kernel[k]);
                }
                out[y * w + x] = sum as f32;
            }
        }
        out
    }

    // ========================================================================
    // Helper: Sobel edge detection (3x3 kernels)
    // ========================================================================

    fn sobel(image: &[f32], w: usize, h: usize) -> (Vec<f32>, Vec<f32>) {
        #[rustfmt::skip]
        let sx: [f32; 9] = [
            -1.0, 0.0, 1.0,
            -2.0, 0.0, 2.0,
            -1.0, 0.0, 1.0,
        ];
        #[rustfmt::skip]
        let sy: [f32; 9] = [
            -1.0, -2.0, -1.0,
             0.0,  0.0,  0.0,
             1.0,  2.0,  1.0,
        ];
        let gx = conv2d(image, w, h, &sx, 3, 3);
        let gy = conv2d(image, w, h, &sy, 3, 3);
        (gx, gy)
    }

    fn gradient_magnitude(gx: &[f32], gy: &[f32]) -> Vec<f32> {
        gx.iter().zip(gy.iter()).map(|(&x, &y)| (x * x + y * y).sqrt()).collect()
    }

    // ========================================================================
    // Helper: Canny edge detection
    //   Gaussian blur -> Sobel -> NMS -> double threshold with hysteresis
    // ========================================================================

    #[derive(Debug)]
    enum CannyError {
        InvalidThresholds,
    }

    fn canny(
        image: &[f32],
        w: usize,
        h: usize,
        sigma: f32,
        low: f32,
        high: f32,
    ) -> Result<Vec<f32>, CannyError> {
        if low < 0.0 || high < low || high > 1.0 {
            return Err(CannyError::InvalidThresholds);
        }

        // Step 1: Gaussian blur
        let blurred = gaussian_blur(image, w, h, sigma);

        // Step 2: Sobel gradients
        let (gx, gy) = sobel(&blurred, w, h);
        let mag = gradient_magnitude(&gx, &gy);

        // Normalize magnitude to [0, 1]
        let max_mag = mag.iter().copied().fold(0.0_f32, f32::max);
        let mag_norm: Vec<f32> =
            if max_mag > 0.0 { mag.iter().map(|&m| m / max_mag).collect() } else { mag };

        // Step 3: Non-maximum suppression
        let mut nms = vec![0.0_f32; w * h];
        for y in 1..h.saturating_sub(1) {
            for x in 1..w.saturating_sub(1) {
                let idx = y * w + x;
                let angle = gy[idx].atan2(gx[idx]);
                let m = mag_norm[idx];
                let dir = ((angle + PI) / (PI / 4.0)).round() as usize % 4;
                let (n1, n2) = match dir {
                    0 => (mag_norm[idx - 1], mag_norm[idx + 1]),
                    1 => (mag_norm[(y - 1) * w + x + 1], mag_norm[(y + 1) * w + x - 1]),
                    2 => (mag_norm[(y - 1) * w + x], mag_norm[(y + 1) * w + x]),
                    _ => (mag_norm[(y - 1) * w + x - 1], mag_norm[(y + 1) * w + x + 1]),
                };
                if m >= n1 && m >= n2 {
                    nms[idx] = m;
                }
            }
        }

        // Step 4: Hysteresis thresholding
        let mut edges = vec![0.0_f32; w * h];
        for y in 1..h.saturating_sub(1) {
            for x in 1..w.saturating_sub(1) {
                let idx = y * w + x;
                if nms[idx] >= high {
                    edges[idx] = 1.0;
                } else if nms[idx] >= low {
                    let has_strong = [
                        (y - 1, x - 1),
                        (y - 1, x),
                        (y - 1, x + 1),
                        (y, x - 1),
                        (y, x + 1),
                        (y + 1, x - 1),
                        (y + 1, x),
                        (y + 1, x + 1),
                    ]
                    .iter()
                    .any(|&(ny, nx)| nms[ny * w + nx] >= high);
                    if has_strong {
                        edges[idx] = 1.0;
                    }
                }
            }
        }

        Ok(edges)
    }

    // ========================================================================
    // Helper: RGB <-> Grayscale (BT.601)
    // ========================================================================

    fn rgb_to_gray(rgb: &[f32], w: usize, h: usize) -> Vec<f32> {
        let pixels = w * h;
        assert_eq!(rgb.len(), pixels * 3);
        let mut gray = Vec::with_capacity(pixels);
        for i in 0..pixels {
            let r = rgb[i * 3];
            let g = rgb[i * 3 + 1];
            let b = rgb[i * 3 + 2];
            gray.push(0.299 * r + 0.587 * g + 0.114 * b);
        }
        gray
    }

    // ========================================================================
    // Helper: RGB <-> HSV conversion
    // ========================================================================

    fn rgb_to_hsv_pixel(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
        let max = r.max(g).max(b);
        let min = r.min(g).min(b);
        let delta = max - min;
        let v = max;
        let s = if max > f32::EPSILON { delta / max } else { 0.0 };
        let h = if delta < f32::EPSILON {
            0.0
        } else if (max - r).abs() < f32::EPSILON {
            60.0 * (((g - b) / delta) % 6.0)
        } else if (max - g).abs() < f32::EPSILON {
            60.0 * ((b - r) / delta + 2.0)
        } else {
            60.0 * ((r - g) / delta + 4.0)
        };
        let h = if h < 0.0 { h + 360.0 } else { h };
        (h, s, v)
    }

    fn hsv_to_rgb_pixel(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
        if s < f32::EPSILON {
            return (v, v, v);
        }
        let h = h % 360.0;
        let c = v * s;
        let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
        let m = v - c;
        let (r1, g1, b1) = match (h / 60.0) as u32 {
            0 => (c, x, 0.0),
            1 => (x, c, 0.0),
            2 => (0.0, c, x),
            3 => (0.0, x, c),
            4 => (x, 0.0, c),
            _ => (c, 0.0, x),
        };
        (r1 + m, g1 + m, b1 + m)
    }

    fn rgb_to_hsv_buf(rgb: &[f32], w: usize, h: usize) -> Vec<f32> {
        let pixels = w * h;
        assert_eq!(rgb.len(), pixels * 3);
        let mut hsv = vec![0.0_f32; pixels * 3];
        for i in 0..pixels {
            let (hue, sat, val) = rgb_to_hsv_pixel(rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2]);
            hsv[i * 3] = hue;
            hsv[i * 3 + 1] = sat;
            hsv[i * 3 + 2] = val;
        }
        hsv
    }

    fn hsv_to_rgb_buf(hsv: &[f32], w: usize, h: usize) -> Vec<f32> {
        let pixels = w * h;
        assert_eq!(hsv.len(), pixels * 3);
        let mut rgb = vec![0.0_f32; pixels * 3];
        for i in 0..pixels {
            let (r, g, b) = hsv_to_rgb_pixel(hsv[i * 3], hsv[i * 3 + 1], hsv[i * 3 + 2]);
            rgb[i * 3] = r;
            rgb[i * 3 + 1] = g;
            rgb[i * 3 + 2] = b;
        }
        rgb
    }

    // ========================================================================
    // Helper: Connected components (union-find, 4-connectivity)
    // ========================================================================

    fn find(parent: &[u32], mut x: u32) -> u32 {
        while parent[x as usize] != x {
            x = parent[x as usize];
        }
        x
    }

    fn union(parent: &mut [u32], a: u32, b: u32) {
        let ra = find(parent, a);
        let rb = find(parent, b);
        if ra != rb {
            let min_r = ra.min(rb);
            let max_r = ra.max(rb);
            parent[max_r as usize] = min_r;
        }
    }

    fn connected_components(image: &[f32], w: usize, h: usize) -> (Vec<u32>, u32) {
        let pixels = w * h;
        assert_eq!(image.len(), pixels);
        let mut labels = vec![0_u32; pixels];
        let mut parent: Vec<u32> = vec![0]; // label 0 = background

        // First pass: assign provisional labels
        for y in 0..h {
            for x in 0..w {
                let idx = y * w + x;
                if image[idx].abs() < f32::EPSILON {
                    continue;
                }
                let left = if x > 0 { labels[idx - 1] } else { 0 };
                let above = if y > 0 { labels[idx - w] } else { 0 };
                match (left > 0, above > 0) {
                    (false, false) => {
                        let new_label = parent.len() as u32;
                        parent.push(new_label);
                        labels[idx] = new_label;
                    }
                    (true, false) => labels[idx] = left,
                    (false, true) => labels[idx] = above,
                    (true, true) => {
                        labels[idx] = left.min(above);
                        union(&mut parent, left, above);
                    }
                }
            }
        }

        // Flatten paths
        for i in 0..parent.len() {
            parent[i] = find(&parent, i as u32);
        }

        // Relabel with sequential IDs
        let mut remap = vec![0_u32; parent.len()];
        let mut next_label = 1_u32;
        for i in 1..parent.len() {
            let root = parent[i] as usize;
            if remap[root] == 0 {
                remap[root] = next_label;
                next_label += 1;
            }
            remap[i] = remap[root];
        }

        for label in &mut labels {
            if *label > 0 {
                *label = remap[*label as usize];
            }
        }

        let num_labels = next_label.saturating_sub(1);
        (labels, num_labels)
    }

    // ========================================================================
    // Helper: Histogram computation
    // ========================================================================

    fn histogram(image: &[f32], bins: usize) -> Vec<u32> {
        assert!(bins > 0);
        let mut hist = vec![0_u32; bins];
        let scale = bins as f32;
        for &pixel in image {
            let clamped = pixel.clamp(0.0, 1.0 - f32::EPSILON);
            let bucket = (clamped * scale) as usize;
            hist[bucket] += 1;
        }
        hist
    }

    // ========================================================================
    // Helper: Morphological dilate/erode (3x3 flat structuring element)
    // ========================================================================

    fn dilate(image: &[f32], w: usize, h: usize, se: &[f32], sw: usize, sh: usize) -> Vec<f32> {
        assert_eq!(image.len(), w * h);
        assert_eq!(se.len(), sw * sh);
        let half_w = sw / 2;
        let half_h = sh / 2;
        let mut out = vec![0.0_f32; w * h];
        for y in 0..h {
            for x in 0..w {
                let mut max_val = f32::NEG_INFINITY;
                for sy in 0..sh {
                    for sx in 0..sw {
                        if se[sy * sw + sx] <= 0.0 {
                            continue;
                        }
                        let iy = y as isize + sy as isize - half_h as isize;
                        let ix = x as isize + sx as isize - half_w as isize;
                        if iy >= 0 && iy < h as isize && ix >= 0 && ix < w as isize {
                            let val = image[iy as usize * w + ix as usize];
                            if val > max_val {
                                max_val = val;
                            }
                        }
                    }
                }
                out[y * w + x] = if max_val == f32::NEG_INFINITY { 0.0 } else { max_val };
            }
        }
        out
    }

    fn erode(image: &[f32], w: usize, h: usize, se: &[f32], sw: usize, sh: usize) -> Vec<f32> {
        assert_eq!(image.len(), w * h);
        assert_eq!(se.len(), sw * sh);
        let half_w = sw / 2;
        let half_h = sh / 2;
        let mut out = vec![0.0_f32; w * h];
        for y in 0..h {
            for x in 0..w {
                let mut min_val = f32::INFINITY;
                for sy in 0..sh {
                    for sx in 0..sw {
                        if se[sy * sw + sx] <= 0.0 {
                            continue;
                        }
                        let iy = y as isize + sy as isize - half_h as isize;
                        let ix = x as isize + sx as isize - half_w as isize;
                        if iy >= 0 && iy < h as isize && ix >= 0 && ix < w as isize {
                            let val = image[iy as usize * w + ix as usize];
                            if val < min_val {
                                min_val = val;
                            }
                        }
                    }
                }
                out[y * w + x] = if min_val == f32::INFINITY { 0.0 } else { min_val };
            }
        }
        out
    }

    // ========================================================================
    // Helper: Image resize (nearest, bilinear, bicubic, lanczos)
    // ========================================================================

    #[derive(Debug)]
    enum ResizeError {
        ZeroOutput,
    }

    fn resize_nearest(
        image: &[f32],
        src_w: usize,
        src_h: usize,
        dst_w: usize,
        dst_h: usize,
    ) -> Result<Vec<f32>, ResizeError> {
        if dst_w == 0 || dst_h == 0 {
            return Err(ResizeError::ZeroOutput);
        }
        let scale_x = src_w as f32 / dst_w as f32;
        let scale_y = src_h as f32 / dst_h as f32;
        let mut out = vec![0.0_f32; dst_w * dst_h];
        for dy in 0..dst_h {
            for dx in 0..dst_w {
                let sx = ((dx as f32 + 0.5) * scale_x - 0.5 + 0.5) as usize;
                let sy = ((dy as f32 + 0.5) * scale_y - 0.5 + 0.5) as usize;
                let sx = sx.min(src_w - 1);
                let sy = sy.min(src_h - 1);
                out[dy * dst_w + dx] = image[sy * src_w + sx];
            }
        }
        Ok(out)
    }

    fn bilinear_sample(image: &[f32], w: usize, h: usize, x: f32, y: f32) -> f32 {
        let x0 = (x.floor() as isize).max(0) as usize;
        let y0 = (y.floor() as isize).max(0) as usize;
        let x1 = (x0 + 1).min(w - 1);
        let y1 = (y0 + 1).min(h - 1);
        let fx = (x - x0 as f32).clamp(0.0, 1.0);
        let fy = (y - y0 as f32).clamp(0.0, 1.0);
        let p00 = image[y0 * w + x0];
        let p10 = image[y0 * w + x1];
        let p01 = image[y1 * w + x0];
        let p11 = image[y1 * w + x1];
        p00 * (1.0 - fx) * (1.0 - fy)
            + p10 * fx * (1.0 - fy)
            + p01 * (1.0 - fx) * fy
            + p11 * fx * fy
    }

    fn resize_bilinear(
        image: &[f32],
        src_w: usize,
        src_h: usize,
        dst_w: usize,
        dst_h: usize,
    ) -> Result<Vec<f32>, ResizeError> {
        if dst_w == 0 || dst_h == 0 {
            return Err(ResizeError::ZeroOutput);
        }
        let scale_x = src_w as f32 / dst_w as f32;
        let scale_y = src_h as f32 / dst_h as f32;
        let mut out = vec![0.0_f32; dst_w * dst_h];
        for dy in 0..dst_h {
            for dx in 0..dst_w {
                let sx = (dx as f32 + 0.5) * scale_x - 0.5;
                let sy = (dy as f32 + 0.5) * scale_y - 0.5;
                out[dy * dst_w + dx] = bilinear_sample(image, src_w, src_h, sx, sy);
            }
        }
        Ok(out)
    }

    fn cubic_weight(t: f32) -> f32 {
        let t = t.abs();
        if t <= 1.0 {
            (1.5 * t - 2.5) * t * t + 1.0
        } else if t < 2.0 {
            ((-0.5 * t + 2.5) * t - 4.0) * t + 2.0
        } else {
            0.0
        }
    }

    fn clamp_idx(i: isize, size: usize) -> usize {
        i.clamp(0, size as isize - 1) as usize
    }

    fn resize_bicubic(
        image: &[f32],
        src_w: usize,
        src_h: usize,
        dst_w: usize,
        dst_h: usize,
    ) -> Result<Vec<f32>, ResizeError> {
        if dst_w == 0 || dst_h == 0 {
            return Err(ResizeError::ZeroOutput);
        }
        let scale_x = src_w as f32 / dst_w as f32;
        let scale_y = src_h as f32 / dst_h as f32;
        let mut out = vec![0.0_f32; dst_w * dst_h];
        for dy in 0..dst_h {
            for dx in 0..dst_w {
                let sx = (dx as f32 + 0.5) * scale_x - 0.5;
                let sy = (dy as f32 + 0.5) * scale_y - 0.5;
                let ix = sx.floor() as isize;
                let iy = sy.floor() as isize;
                let fx = sx - ix as f32;
                let fy = sy - iy as f32;

                let mut sum = 0.0_f64;
                for j in -1..=2_isize {
                    let wy = cubic_weight(fy - j as f32) as f64;
                    let cy = clamp_idx(iy + j, src_h);
                    for i in -1..=2_isize {
                        let wx = cubic_weight(fx - i as f32) as f64;
                        let cx = clamp_idx(ix + i, src_w);
                        sum += wy * wx * f64::from(image[cy * src_w + cx]);
                    }
                }
                out[dy * dst_w + dx] = sum as f32;
            }
        }
        Ok(out)
    }

    fn lanczos_weight(t: f32) -> f32 {
        let t = t.abs();
        if t < 1e-7 {
            1.0
        } else if t < 3.0 {
            let pi_t = PI * t;
            let pi_t_over_a = pi_t / 3.0;
            (pi_t.sin() * pi_t_over_a.sin()) / (pi_t * pi_t_over_a)
        } else {
            0.0
        }
    }

    fn resize_lanczos(
        image: &[f32],
        src_w: usize,
        src_h: usize,
        dst_w: usize,
        dst_h: usize,
    ) -> Result<Vec<f32>, ResizeError> {
        if dst_w == 0 || dst_h == 0 {
            return Err(ResizeError::ZeroOutput);
        }
        let scale_x = src_w as f32 / dst_w as f32;
        let scale_y = src_h as f32 / dst_h as f32;
        let mut out = vec![0.0_f32; dst_w * dst_h];
        for dy in 0..dst_h {
            for dx in 0..dst_w {
                let sx = (dx as f32 + 0.5) * scale_x - 0.5;
                let sy = (dy as f32 + 0.5) * scale_y - 0.5;
                let ix = sx.floor() as isize;
                let iy = sy.floor() as isize;
                let fx = sx - ix as f32;
                let fy = sy - iy as f32;

                let mut sum = 0.0_f64;
                let mut weight_sum = 0.0_f64;
                for j in -2..=3_isize {
                    let wy = lanczos_weight(fy - j as f32) as f64;
                    let cy = clamp_idx(iy + j, src_h);
                    for i in -2..=3_isize {
                        let wx = lanczos_weight(fx - i as f32) as f64;
                        let cx = clamp_idx(ix + i, src_w);
                        let w_total = wy * wx;
                        sum += w_total * f64::from(image[cy * src_w + cx]);
                        weight_sum += w_total;
                    }
                }
                out[dy * dst_w + dx] =
                    if weight_sum.abs() > 1e-12 { (sum / weight_sum) as f32 } else { 0.0 };
            }
        }
        Ok(out)
    }

    // ====================================================================
    // CONTRACT TEST 1: Canny detects edges in a step function image
    // ====================================================================

    #[test]
    fn test_canny_detects_edges() {
        let w = 20;
        let h = 20;
        let mut image = vec![0.0_f32; w * h];
        // Create a step function: left half = 0, right half = 1
        for y in 0..h {
            for x in w / 2..w {
                image[y * w + x] = 1.0;
            }
        }

        let edges = canny(&image, w, h, 1.0, 0.05, 0.15).unwrap();
        let edge_count: usize = edges.iter().filter(|&&v| v > 0.5).count();
        assert!(
            edge_count > 0,
            "Canny should detect edges at step boundary, found {edge_count} edge pixels"
        );
    }

    // ====================================================================
    // CONTRACT TEST 2: Sobel on constant image = all zeros
    // ====================================================================

    #[test]
    fn test_sobel_constant_zero() {
        let w = 7;
        let h = 7;
        let image = vec![3.0_f32; w * h];

        let (gx, gy) = sobel(&image, w, h);

        // Interior pixels should have zero gradient (constant image)
        for y in 1..h - 1 {
            for x in 1..w - 1 {
                let idx = y * w + x;
                assert!(
                    gx[idx].abs() < 1e-5,
                    "Sobel gx non-zero on constant image at ({x},{y}): {}",
                    gx[idx]
                );
                assert!(
                    gy[idx].abs() < 1e-5,
                    "Sobel gy non-zero on constant image at ({x},{y}): {}",
                    gy[idx]
                );
            }
        }
    }

    // ====================================================================
    // CONTRACT TEST 3: Canny with low > high should error
    // ====================================================================

    #[test]
    fn test_canny_invalid_thresholds() {
        let image = vec![0.5_f32; 100];
        let result = canny(&image, 10, 10, 1.0, 0.5, 0.3); // low > high
        assert!(result.is_err(), "Canny should reject low > high thresholds");

        let result2 = canny(&image, 10, 10, 1.0, -0.1, 0.5); // negative low
        assert!(result2.is_err(), "Canny should reject negative thresholds");
    }

    // ====================================================================
    // CONTRACT TEST 4: Uniform image -> no edges
    // ====================================================================

    #[test]
    fn test_canny_uniform_no_edges() {
        let w = 20;
        let h = 20;
        let image = vec![0.5_f32; w * h];

        let edges = canny(&image, w, h, 1.0, 0.1, 0.3).unwrap();
        let edge_count: usize = edges.iter().filter(|&&v| v > 0.5).count();
        assert_eq!(edge_count, 0, "Uniform image should have zero edges, found {edge_count}");
    }

    // ====================================================================
    // CONTRACT TEST 5: RGB -> Gray using BT.601 weights
    // ====================================================================

    #[test]
    fn test_rgb_to_gray_bt601() {
        // Pure white -> 1.0
        let white = [1.0_f32, 1.0, 1.0];
        let gray_w = rgb_to_gray(&white, 1, 1);
        assert!((gray_w[0] - 1.0).abs() < 1e-5, "White should map to 1.0, got {}", gray_w[0]);

        // Pure red -> 0.299
        let red = [1.0_f32, 0.0, 0.0];
        let gray_r = rgb_to_gray(&red, 1, 1);
        assert!((gray_r[0] - 0.299).abs() < 1e-3, "Red should map to 0.299, got {}", gray_r[0]);

        // Pure green -> 0.587
        let green = [0.0_f32, 1.0, 0.0];
        let gray_g = rgb_to_gray(&green, 1, 1);
        assert!((gray_g[0] - 0.587).abs() < 1e-3, "Green should map to 0.587, got {}", gray_g[0]);

        // Pure blue -> 0.114
        let blue = [0.0_f32, 0.0, 1.0];
        let gray_b = rgb_to_gray(&blue, 1, 1);
        assert!((gray_b[0] - 0.114).abs() < 1e-3, "Blue should map to 0.114, got {}", gray_b[0]);

        // Verify weights sum to 1.0
        let sum: f64 = 0.299 + 0.587 + 0.114;
        assert!((sum - 1.0).abs() < 1e-6, "BT.601 weights must sum to 1.0, got {sum}");
    }

    // ====================================================================
    // CONTRACT TEST 6: HSV roundtrip (RGB -> HSV -> RGB = identity)
    // ====================================================================

    #[test]
    fn test_hsv_roundtrip() {
        // Test a variety of colors
        #[rustfmt::skip]
        let colors: Vec<f32> = vec![
            1.0, 0.0, 0.0,   // red
            0.0, 1.0, 0.0,   // green
            0.0, 0.0, 1.0,   // blue
            0.5, 0.5, 0.5,   // gray
            1.0, 1.0, 0.0,   // yellow
            0.0, 1.0, 1.0,   // cyan
            1.0, 0.0, 1.0,   // magenta
            0.3, 0.6, 0.9,   // arbitrary
        ];
        let w = 8;
        let h = 1;

        let hsv = rgb_to_hsv_buf(&colors, w, h);
        let recovered = hsv_to_rgb_buf(&hsv, w, h);

        for i in 0..colors.len() {
            let err = (colors[i] - recovered[i]).abs();
            assert!(
                err < 1e-4,
                "HSV roundtrip failed at index {i}: orig={}, recovered={}, err={err}",
                colors[i],
                recovered[i]
            );
        }
    }

    // ====================================================================
    // CONTRACT TEST 7: Connected components - single blob -> one label
    // ====================================================================

    #[test]
    fn test_connected_components_single_blob() {
        #[rustfmt::skip]
        let image: Vec<f32> = vec![
            0.0, 1.0, 1.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 0.0,
        ];
        let (labels, num_labels) = connected_components(&image, 3, 3);
        assert_eq!(labels[0], 0, "Background should be label 0");
        assert!(labels[1] > 0, "Foreground pixel should have non-zero label");
        assert_eq!(labels[1], labels[2], "4-connected pixels share a label");
        assert_eq!(labels[1], labels[4], "Vertically connected pixels share a label");
        assert_eq!(num_labels, 1, "Single blob should yield exactly 1 label");
    }

    // ====================================================================
    // CONTRACT TEST 8: Connected components - two blobs -> two labels
    // ====================================================================

    #[test]
    fn test_connected_components_two_blobs() {
        #[rustfmt::skip]
        let image: Vec<f32> = vec![
            1.0, 1.0, 0.0, 0.0, 0.0,
            1.0, 1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0, 1.0,
            0.0, 0.0, 0.0, 1.0, 1.0,
        ];
        let (labels, num_labels) = connected_components(&image, 5, 5);
        assert_eq!(num_labels, 2, "Two separate blobs should yield 2 labels");
        // Verify top-left blob has same label
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[0], labels[5]);
        assert_eq!(labels[0], labels[6]);
        // Verify bottom-right blob has same label
        assert_eq!(labels[23], labels[24]);
        assert_eq!(labels[23], labels[18]);
        assert_eq!(labels[23], labels[19]);
        // Verify the two blobs have different labels
        assert_ne!(labels[0], labels[23], "Two separate blobs must have different labels");
    }

    // ====================================================================
    // CONTRACT TEST 9: All background -> zero labels
    // ====================================================================

    #[test]
    fn test_connected_components_all_background() {
        let image = vec![0.0_f32; 16];
        let (labels, num_labels) = connected_components(&image, 4, 4);
        assert!(labels.iter().all(|&l| l == 0), "All-zero image should have all-zero labels");
        assert_eq!(num_labels, 0, "No foreground -> 0 labels");
    }

    // ====================================================================
    // CONTRACT TEST 10: Separable convolution matches full 2D convolution
    // ====================================================================

    #[test]
    fn test_separable_matches_2d() {
        let w = 8;
        let h = 8;
        let image: Vec<f32> = (0..w * h).map(|i| (i as f32).sin()).collect();

        // Approximate Gaussian kernel (sigma ~ 1.0)
        let h_kernel = [0.2742_f32, 0.4514, 0.2742];
        let v_kernel = h_kernel;

        // Full 2D kernel = outer product of 1D kernels
        let mut kernel_2d = [0.0_f32; 9];
        for i in 0..3 {
            for j in 0..3 {
                kernel_2d[i * 3 + j] = v_kernel[i] * h_kernel[j];
            }
        }

        let out_2d = conv2d(&image, w, h, &kernel_2d, 3, 3);
        let out_sep = separable_conv2d(&image, w, h, &h_kernel, &v_kernel);

        for i in 0..w * h {
            assert!(
                (out_2d[i] - out_sep[i]).abs() < 1e-4,
                "Separable mismatch at {i}: 2d={}, sep={}",
                out_2d[i],
                out_sep[i]
            );
        }
    }

    // ====================================================================
    // CONTRACT TEST 11: Identity kernel = no change
    // ====================================================================

    #[test]
    fn test_identity_kernel() {
        let w = 5;
        let h = 5;
        let image: Vec<f32> = (0..w * h).map(|i| i as f32 + 1.0).collect();

        #[rustfmt::skip]
        let delta: [f32; 9] = [
            0.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 0.0,
        ];

        let out = conv2d(&image, w, h, &delta, 3, 3);

        // Interior pixels should be exactly preserved (border uses zero-padding)
        for y in 1..h - 1 {
            for x in 1..w - 1 {
                let idx = y * w + x;
                assert!(
                    (out[idx] - image[idx]).abs() < 1e-6,
                    "Identity kernel changed pixel at ({x},{y}): {} -> {}",
                    image[idx],
                    out[idx]
                );
            }
        }
    }

    // ====================================================================
    // CONTRACT TEST 12: Gaussian blur on constant image = same constant
    // ====================================================================

    #[test]
    fn test_gaussian_constant_image() {
        let w = 10;
        let h = 10;
        let val = 7.0_f32;
        let image = vec![val; w * h];

        let blurred = gaussian_blur(&image, w, h, 1.5);

        for (i, &v) in blurred.iter().enumerate() {
            assert!(
                (v - val).abs() < 1e-3,
                "Gaussian blur changed constant at pixel {i}: expected {val}, got {v}"
            );
        }
    }

    // ====================================================================
    // CONTRACT TEST 13: Uniform image has flat histogram
    // ====================================================================

    #[test]
    fn test_histogram_uniform() {
        // Create image with uniformly spaced values across [0, 1)
        let bins = 16;
        let pixels_per_bin = 4;
        let total = bins * pixels_per_bin;
        let image: Vec<f32> = (0..total).map(|i| i as f32 / total as f32).collect();

        let hist = histogram(&image, bins);

        // Each bin should have roughly the same count
        let expected = pixels_per_bin as u32;
        for (i, &count) in hist.iter().enumerate() {
            assert_eq!(count, expected, "Bin {i}: expected {expected} pixels, got {count}");
        }

        // Total should match pixel count
        let total_count: u32 = hist.iter().sum();
        assert_eq!(total_count, total as u32);
    }

    // ====================================================================
    // CONTRACT TEST 14: dilate(erode(A)) >= A for binary images
    // ====================================================================

    #[test]
    fn test_dilate_erode_duality() {
        let w = 7;
        let h = 7;
        // Create a binary image with a cross pattern
        let mut image = vec![0.0_f32; w * h];
        // Horizontal bar
        for x in 1..w - 1 {
            image[3 * w + x] = 1.0;
        }
        // Vertical bar
        for y in 1..h - 1 {
            image[y * w + 3] = 1.0;
        }

        let se = vec![1.0_f32; 9]; // 3x3 all-ones structuring element

        let eroded = erode(&image, w, h, &se, 3, 3);
        let dilated_eroded = dilate(&eroded, w, h, &se, 3, 3);

        // For interior pixels of the original that are 1.0 and
        // have a full 3x3 neighborhood of 1.0 (i.e., survive erosion),
        // those should still be 1.0 after closing.
        // The mathematical property: dilate(erode(A)) is the morphological
        // closing and should cover A's "thick" parts.
        //
        // More specifically, for any pixel where A=1 and ALL neighbors within
        // the SE are also 1, closing preserves it.
        for y in 0..h {
            for x in 0..w {
                let idx = y * w + x;
                if image[idx] > 0.5 {
                    // Check if this pixel survives erosion (all SE neighbors are 1)
                    let survives_erosion = (-1..=1_isize).all(|dy| {
                        (-1..=1_isize).all(|dx| {
                            let ny = y as isize + dy;
                            let nx = x as isize + dx;
                            if ny >= 0 && ny < h as isize && nx >= 0 && nx < w as isize {
                                image[ny as usize * w + nx as usize] > 0.5
                            } else {
                                false
                            }
                        })
                    });
                    if survives_erosion {
                        assert!(
                            dilated_eroded[idx] > 0.5,
                            "Closing should preserve interior pixels at ({x},{y})"
                        );
                    }
                }
            }
        }

        // Also verify the basic duality on a constant image
        let constant = vec![1.0_f32; w * h];
        let e = erode(&constant, w, h, &se, 3, 3);
        let de = dilate(&e, w, h, &se, 3, 3);
        // For interior pixels, erode of all-1s is still 1, dilate keeps 1
        for y in 1..h - 1 {
            for x in 1..w - 1 {
                let idx = y * w + x;
                assert!(de[idx] > 0.5, "dilate(erode(ones)) should be 1 at interior ({x},{y})");
            }
        }
    }

    // ====================================================================
    // CONTRACT TEST 15: Resize nearest to same size = identity
    // ====================================================================

    #[test]
    fn test_resize_nearest_identity() {
        let image = vec![1.0, 2.0, 3.0, 4.0_f32]; // 2x2
        let result = resize_nearest(&image, 2, 2, 2, 2).unwrap();
        for i in 0..4 {
            assert!(
                (result[i] - image[i]).abs() < 1e-6,
                "Nearest resize changed pixel {i}: {} -> {}",
                image[i],
                result[i]
            );
        }
    }

    // ====================================================================
    // CONTRACT TEST 16: Resize bilinear to same size = identity
    // ====================================================================

    #[test]
    fn test_resize_bilinear_identity() {
        let image = vec![1.0, 2.0, 3.0, 4.0_f32]; // 2x2
        let result = resize_bilinear(&image, 2, 2, 2, 2).unwrap();
        for i in 0..4 {
            assert!(
                (result[i] - image[i]).abs() < 1e-5,
                "Bilinear resize changed pixel {i}: {} -> {}",
                image[i],
                result[i]
            );
        }
    }

    // ====================================================================
    // CONTRACT TEST 17: Resize bicubic on constant image = same constant
    // ====================================================================

    #[test]
    fn test_resize_bicubic_constant() {
        let val = 0.7_f32;
        let image = vec![val; 16]; // 4x4 constant
        let result = resize_bicubic(&image, 4, 4, 8, 8).unwrap();
        assert_eq!(result.len(), 64);
        for (i, &v) in result.iter().enumerate() {
            assert!(
                (v - val).abs() < 0.01,
                "Bicubic resize changed constant at pixel {i}: expected {val}, got {v}"
            );
        }
    }

    // ====================================================================
    // CONTRACT TEST 18a: Resize Lanczos on constant image = same constant
    // ====================================================================

    #[test]
    fn test_resize_lanczos_constant() {
        let val = 0.7_f32;
        let image = vec![val; 16]; // 4x4 constant
        let result = resize_lanczos(&image, 4, 4, 8, 8).unwrap();
        assert_eq!(result.len(), 64);
        for (i, &v) in result.iter().enumerate() {
            assert!(
                (v - val).abs() < 0.01,
                "Lanczos resize changed constant at pixel {i}: expected {val}, got {v}"
            );
        }
    }

    // ====================================================================
    // CONTRACT TEST 18b: Resize to zero output should error
    // ====================================================================

    #[test]
    fn test_resize_zero_output() {
        let image = vec![1.0_f32; 4];
        assert!(resize_nearest(&image, 2, 2, 0, 2).is_err(), "Resize to 0 width should error");
        assert!(resize_bilinear(&image, 2, 2, 2, 0).is_err(), "Resize to 0 height should error");
        assert!(resize_bicubic(&image, 2, 2, 0, 0).is_err(), "Resize to 0x0 should error");
        assert!(
            resize_lanczos(&image, 2, 2, 0, 1).is_err(),
            "Lanczos resize to 0 width should error"
        );
    }
}

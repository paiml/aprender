# WebGPU Browser Setup

WebGPU enables GPU-accelerated compute in the browser, providing 50-100x speedups for analytics workloads. However, browser support requires specific configuration.

## Browser Support Status (November 2025)

| Browser | WebGPU Status | Notes |
|---------|--------------|-------|
| Chrome 113+ | Supported | May require flags on some GPUs |
| Edge 113+ | Supported | Chromium-based, same as Chrome |
| Firefox 121+ | Experimental | Enable in `about:config` |
| Safari 17+ | Supported | macOS/iOS only |

## Chrome WebGPU Blocklist Issue

Starting with Chrome 142 (November 2025), Chrome began blocklisting WebGPU on certain GPU configurations. This affects many users even with capable hardware.

### Symptoms

When WebGPU is blocklisted, you'll see:

1. **trueno-db demo shows "SIMD128 Compute"** instead of "WebGPU Compute"
2. **Console shows**: `trueno-db WASM initialized` but no WebGPU adapter
3. **`chrome://gpu` shows**:
   ```
   WebGPU: Disabled
   WebGPU has been disabled via blocklist or the command line
   ```

### Diagnosis

1. Open `chrome://gpu` in Chrome
2. Search for "WebGPU" in the page
3. Look for these indicators:

**Blocklisted (bad)**:
```
WebGPU: Disabled
WebGPU has been disabled via blocklist or the command line
```

**Working (good)**:
```
WebGPU: Hardware accelerated
```

### Solution 1: Enable via Chrome Flags

1. Open `chrome://flags` in Chrome
2. Search for "WebGPU"
3. Set **"Unsafe WebGPU Support"** to **Enabled**
4. Click "Relaunch" to restart Chrome

Direct link: `chrome://flags/#enable-unsafe-webgpu`

### Solution 2: Launch Chrome with Command Line Flag

```bash
# Linux
google-chrome --enable-features=WebGPU

# macOS
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --enable-features=WebGPU

# Windows
"C:\Program Files\Google\Chrome\Application\chrome.exe" --enable-features=WebGPU
```

### Solution 3: Use a Different Browser

Firefox and Safari have independent WebGPU implementations that may not be affected by Chrome's blocklist.

**Firefox**:
1. Open `about:config`
2. Search for `dom.webgpu.enabled`
3. Set to `true`

**Safari**:
WebGPU is enabled by default on Safari 17+ (macOS Sonoma / iOS 17+).

## Verifying WebGPU Works

### Method 1: trueno-db Demo

1. Run `make wasm-serve` in the trueno-db directory
2. Open `http://localhost:8080` in your browser
3. Look for the compute badge:
   - **"WebGPU Compute"** (blue) = WebGPU working
   - **"SIMD128 Compute"** (green) = Fallback to SIMD
   - **"Scalar Compute"** (orange) = No acceleration

### Method 2: Browser Console

```javascript
// Check if WebGPU API exists
console.log('WebGPU API:', typeof navigator.gpu);

// Request adapter
const adapter = await navigator.gpu?.requestAdapter();
console.log('Adapter:', adapter);

// Get adapter info
if (adapter) {
    const info = await adapter.requestAdapterInfo();
    console.log('GPU:', info.vendor, info.device);
}
```

### Method 3: WebGPU Samples

Visit [WebGPU Samples](https://webgpu.github.io/webgpu-samples/) - if the samples work, WebGPU is enabled.

## Performance Tiers

trueno-db automatically detects and uses the best available compute tier:

| Tier | Detection | Performance | Use Case |
|------|-----------|-------------|----------|
| **WebGPU** | `navigator.gpu.requestAdapter()` | 50-100x | Large datasets (>100K rows) |
| **SIMD128** | WASM feature detection | 4-8x | Medium datasets |
| **Scalar** | Always available | 1x (baseline) | Small datasets |

## Troubleshooting Detection

If WebGPU detection fails even after enabling:

### 1. Check GPU Driver Version

Outdated GPU drivers can cause WebGPU to fail initialization:

```bash
# Linux (NVIDIA)
nvidia-smi

# Linux (AMD)
vulkaninfo | grep "GPU"

# macOS
system_profiler SPDisplaysDataType
```

### 2. Check Dawn Backend

Chrome uses Dawn (Google's WebGPU implementation). Check `chrome://gpu` for Dawn info:

```
Dawn Info
    [WebGPU Status]: Available
    [Backend]: Vulkan / Metal / D3D12
```

If Dawn shows "Software" or "Unavailable", WebGPU won't work efficiently.

### 3. Virtual Machines

WebGPU typically doesn't work in VMs without GPU passthrough:
- **Docker**: Use `--gpus all` flag
- **WSL2**: Requires WSLg with GPU support
- **VMware/VirtualBox**: Limited support

## WASM SIMD128 Fallback

When WebGPU isn't available, trueno-db falls back to WASM SIMD128, which provides 4-8x speedup over scalar code.

SIMD128 requires:
- Chrome 91+, Firefox 89+, Safari 16.4+
- WASM built with `+simd128` target feature

Build command:
```bash
RUSTFLAGS="-C target-feature=+simd128" wasm-pack build --target web
```

## Known Issues

### Chrome 142+ Blocklist (November 2025)

Chrome 142 added many GPUs to the WebGPU blocklist for stability reasons. This is a Chrome policy decision, not a hardware limitation.

**Workaround**: Use `chrome://flags/#enable-unsafe-webgpu`

### Firefox WebGPU Experimental

Firefox WebGPU is still in development and may have compatibility issues with some WebGPU features.

**Status**: Check `about:support` → Graphics → WebGPU

### Safari Metal Limitations

Safari uses Metal backend which has some WebGPU feature gaps. Most trueno-db operations work, but advanced features may differ.

## Getting Help

If you continue to experience WebGPU issues:

1. **Check chrome://gpu** and save the report
2. **File an issue** at [trueno-db GitHub](https://github.com/paiml/trueno-db/issues)
3. **Include**:
   - Browser version
   - OS version
   - GPU model
   - chrome://gpu report (attach as file)
   - Console errors

## References

- [WebGPU Specification](https://www.w3.org/TR/webgpu/)
- [Chrome WebGPU Status](https://chromestatus.com/feature/6213121689518080)
- [Firefox WebGPU](https://wiki.mozilla.org/WebGPU)
- [Safari WebGPU](https://webkit.org/blog/14879/webgpu-now-available-for-testing-in-safari-technology-preview/)

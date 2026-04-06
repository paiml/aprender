# Showcase Demo Quality Assurance Checklist

**Document ID:** PRES-QA-001
**Version:** 1.0
**Status:** For Red Team Review
**Prepared For:** Toyota ML Engineering Review Team

---

## Preamble: The Toyota Way Applied to ML Systems

This checklist embodies the 14 principles of the Toyota Way [1] applied to machine learning visualization systems. Every claim must be verified through **Genchi Genbutsu** (go and see for yourself). We reject vanity metrics and demand reproducible, measurable evidence.

> "The root of the Toyota Way is to be dissatisfied with the status quo; you have to ask constantly, 'Why are we doing this?'" — Taiichi Ohno

**Review Philosophy:**
- Assume all claims are false until proven with evidence
- Measure everything; opinions are not data
- One defect discovered in production costs 100x more than one caught in review
- Respect the reviewer's time: provide reproducible commands for every claim

---

## References

[1] Liker, J.K. (2004). *The Toyota Way: 14 Management Principles*. McGraw-Hill. ISBN: 978-0071392310

[2] Haas, A. et al. (2017). "Bringing the Web up to Speed with WebAssembly." *PLDI '17: Proceedings of the 38th ACM SIGPLAN Conference on Programming Language Design and Implementation*, pp. 185-200. DOI: 10.1145/3062341.3062363

[3] Jangda, A. et al. (2019). "Not So Fast: Analyzing the Performance of WebAssembly vs. Native Code." *USENIX ATC '19*, pp. 107-120. https://www.usenix.org/conference/atc19/presentation/jangda

[4] Sculley, D. et al. (2015). "Hidden Technical Debt in Machine Learning Systems." *NeurIPS 2015*, pp. 2503-2511. https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems

[5] Amershi, S. et al. (2019). "Software Engineering for Machine Learning: A Case Study." *ICSE-SEIP '19*, pp. 291-300. DOI: 10.1109/ICSE-SEIP.2019.00042

[6] Kenwright, B. (2012). "A Beginners Guide to Dual-Quaternions." *WSCG '12*, pp. 1-10. (GPU animation fundamentals)

[7] McSherry, F. et al. (2015). "Scalability! But at what COST?" *HotOS XV*. https://www.usenix.org/conference/hotos15/workshop-program/presentation/mcsherry

[8] Ratanaworabhan, P. et al. (2010). "JSMeter: Comparing the Behavior of JavaScript Benchmarks with Real Web Applications." *WebApps '10*, pp. 3-3. https://www.usenix.org/conference/webapps-10

[9] Xu, T. et al. (2016). "Early Detection of Configuration Errors to Reduce Failure Damage." *OSDI '16*, pp. 619-634. https://www.usenix.org/conference/osdi16/technical-sessions/presentation/xu

[10] Paleyes, A. et al. (2022). "Challenges in Deploying Machine Learning: A Survey of Case Studies." *ACM Computing Surveys*, Vol. 55, Issue 6, Article 114. DOI: 10.1145/3533378

---

## Checklist Categories

| Category | Points | Focus Area |
|----------|--------|------------|
| A. Performance Claims | 20 | Frame rate, latency, throughput |
| B. Size & Efficiency Claims | 15 | Bundle size, memory, startup |
| C. Data Format Integrity | 15 | .apr/.ald correctness |
| D. Visualization Accuracy | 15 | Chart rendering fidelity |
| E. Animation & Interaction | 10 | Smoothness, responsiveness |
| F. Cross-Platform | 10 | Browser/device compatibility |
| G. Code Quality | 10 | Tests, documentation, security |
| H. Claim Substantiation | 5 | Marketing vs. reality |

---

## A. Performance Claims (20 Points)

**Principle: Genchi Genbutsu — Measure at the source, not from marketing materials**

### Frame Rate Claims

| # | Check | Command/Method | Pass Criteria | Ref |
|---|-------|----------------|---------------|-----|
| A1 | Measure actual FPS in Chrome DevTools | `Performance tab → Record 10s → Analyze frames` | Mean ≥ 55fps, P99 ≥ 45fps | [8] |
| A2 | Measure actual FPS in Firefox | `about:performance` or Performance Monitor | Mean ≥ 55fps | [8] |
| A3 | Measure FPS under CPU throttling (4x slowdown) | `DevTools → Performance → CPU: 4x slowdown` | Mean ≥ 30fps | [3] |
| A4 | Verify no frame drops during particle burst (100 particles) | Click 5x rapidly, observe frame timeline | No frames >32ms | [6] |
| A5 | Verify FPS counter accuracy | Compare DevTools FPS vs displayed FPS | Within ±5fps | [8] |

### Latency Claims

| # | Check | Command/Method | Pass Criteria | Ref |
|---|-------|----------------|---------------|-----|
| A6 | Measure click-to-render latency | `performance.now()` around click handler | <16ms mean | [2] |
| A7 | Measure animation start latency | Time from button click to first visual change | <50ms | [2] |
| A8 | Measure data update propagation | Randomize Data → measure chart update time | <100ms to 90% complete | [2] |
| A9 | Verify no jank during theme toggle | Record Performance trace during toggle | No long tasks >50ms | [8] |
| A10 | Measure inference button response | Time from click to model card flash | <100ms | [5] |

### Throughput Claims

| # | Check | Command/Method | Pass Criteria | Ref |
|---|-------|----------------|---------------|-----|
| A11 | Verify 100 candlesticks render correctly | Count rendered bars visually | Exactly 100 visible | [7] |
| A12 | Verify 500 particle capacity | Emit 500 particles, count in memory | particles.length === 500 | [6] |
| A13 | Measure draw call count per frame | Instrument Canvas2D calls | <50 draw calls/frame | [6] |
| A14 | Verify no memory growth over 5 minutes | `performance.memory.usedJSHeapSize` every 30s | <10% growth | [4] |
| A15 | Stress test: 1000 rapid clicks | Automated click script | No crash, FPS recovers | [7] |

### Rust/WASM Performance

| # | Check | Command/Method | Pass Criteria | Ref |
|---|-------|----------------|---------------|-----|
| A16 | Run Rust example benchmarks | `cargo run --example showcase_gpu` | 60fps reported | [2] |
| A17 | Verify WASM build succeeds | `cargo build --example showcase_gpu --target wasm32-unknown-unknown --release` | Exit code 0 | [2] |
| A18 | Measure WASM instantiation time | `performance.now()` around WebAssembly.instantiate | <50ms | [3] |
| A19 | Compare native vs WASM execution | Run same benchmark native and WASM | WASM within 2x of native | [3] |
| A20 | Profile WASM with Chrome DevTools | `Performance → Bottom-Up → WASM functions` | No single function >10% | [3] |

---

## B. Size & Efficiency Claims (15 Points)

**Principle: Muda elimination — Every byte must justify its existence**

### Bundle Size Claims

| # | Check | Command/Method | Pass Criteria | Ref |
|---|-------|----------------|---------------|-----|
| B1 | Measure WASM binary size | `ls -la target/wasm32-unknown-unknown/release/examples/*.wasm` | <500KB | [2] |
| B2 | Measure HTML/JS size | `wc -c web/showcase/index.html` | <50KB | [8] |
| B3 | Measure total transfer size | DevTools Network tab, disable cache, reload | <600KB total | [2] |
| B4 | Verify gzip compression ratio | `gzip -c file.wasm \| wc -c` | >50% reduction | [2] |
| B5 | Compare to Gradio bundle | Download Gradio app, measure | Presentar <1% of Gradio | [7] |

### Memory Claims

| # | Check | Command/Method | Pass Criteria | Ref |
|---|-------|----------------|---------------|-----|
| B6 | Measure initial heap size | `performance.memory.usedJSHeapSize` on load | <20MB | [4] |
| B7 | Measure heap after 1 minute | Same metric after 1 min interaction | <50MB | [4] |
| B8 | Check for detached DOM nodes | DevTools Memory → Heap snapshot → Detached | 0 detached nodes | [4] |
| B9 | Verify no canvas memory leaks | Create/destroy canvases, check memory | Stable after GC | [4] |
| B10 | Measure particle array memory | `sizeof(particles) * particles.length` estimate | <1MB at 500 particles | [6] |

### Startup Claims

| # | Check | Command/Method | Pass Criteria | Ref |
|---|-------|----------------|---------------|-----|
| B11 | Measure Time to First Paint | DevTools Performance → FP marker | <200ms | [8] |
| B12 | Measure Time to Interactive | Lighthouse audit | <500ms | [8] |
| B13 | Measure First Contentful Paint | DevTools Performance → FCP marker | <300ms | [8] |
| B14 | Cold start with cache disabled | Hard reload (Ctrl+Shift+R) | <1s to interactive | [8] |
| B15 | Compare to Streamlit startup | Time Streamlit hello world | Presentar 10x faster | [7] |

---

## C. Data Format Integrity (15 Points)

**Principle: Jidoka — Build quality in; stop and fix problems immediately**

### .apr Model Format

| # | Check | Command/Method | Pass Criteria | Ref |
|---|-------|----------------|---------------|-----|
| C1 | Verify magic bytes | `hexdump -C demo/assets/sentiment_mini.apr \| head -1` | Starts with `APR\0` | [9] |
| C2 | Parse model in Rust | `cargo test --package presentar-yaml -- formats` | All tests pass | [9] |
| C3 | Verify layer count | Load model, check `model.layers.len()` | Exactly 2 | [5] |
| C4 | Verify parameter count | `model.param_count()` | Exactly 867 | [5] |
| C5 | Verify weight initialization | Check weight distribution | Xavier-like variance | [5] |
| C6 | Verify metadata | Check `model.metadata` for task, classes | All keys present | [5] |
| C7 | Roundtrip test | Save → Load → Compare | Byte-identical | [9] |

### .ald Dataset Format

| # | Check | Command/Method | Pass Criteria | Ref |
|---|-------|----------------|---------------|-----|
| C8 | Verify magic bytes | `hexdump -C demo/assets/timeseries_100.ald \| head -1` | Starts with `ALD\0` | [9] |
| C9 | Parse dataset in Rust | `AldDataset::load()` succeeds | No errors | [9] |
| C10 | Verify tensor count | `dataset.tensors.len()` | Exactly 5 | [9] |
| C11 | Verify tensor shapes | All tensors have shape `[100]` | True | [9] |
| C12 | Verify OHLC validity | `high >= low` for all rows | True for 100/100 | [9] |
| C13 | Verify OHLC validity | `high >= max(open, close)` | True for 100/100 | [9] |
| C14 | Verify positive prices | All values > 0 | True | [9] |
| C15 | Roundtrip test | Save → Load → Compare | Byte-identical | [9] |

---

## D. Visualization Accuracy (15 Points)

**Principle: Standardized work — Every chart must render identically every time**

### Candlestick Chart

| # | Check | Command/Method | Pass Criteria | Ref |
|---|-------|----------------|---------------|-----|
| D1 | Verify candlestick count | Visual count or DOM inspection | Exactly 100 | [8] |
| D2 | Verify green/red coloring | Up days green, down days red | Correct for sample | [8] |
| D3 | Verify Y-axis scale | Compare displayed prices to data | Within 1% | [8] |
| D4 | Verify current price line | Matches last close value | Exact match | [8] |
| D5 | Verify wick rendering | High-low range visible | All wicks visible | [6] |

### Bar Chart

| # | Check | Command/Method | Pass Criteria | Ref |
|---|-------|----------------|---------------|-----|
| D6 | Verify bar count | Visual inspection | Exactly 6 bars | [8] |
| D7 | Verify bar heights proportional | Tallest bar = highest value | Correct | [8] |
| D8 | Verify value labels | Labels match bar heights | Within ±1% | [8] |
| D9 | Verify month labels | Jan-Jun displayed correctly | Correct order | [8] |
| D10 | Verify animation easing | Bars ease-out, not linear | Visually smooth | [6] |

### Donut Chart

| # | Check | Command/Method | Pass Criteria | Ref |
|---|-------|----------------|---------------|-----|
| D11 | Verify segment count | Visual inspection | Exactly 5 segments | [8] |
| D12 | Verify segment proportions | Arc lengths proportional to values | Within ±5% | [8] |
| D13 | Verify center total | Sum of all segments | Correct sum displayed | [8] |
| D14 | Verify rotation animation | Donut rotates smoothly | Continuous rotation | [6] |
| D15 | Verify color consistency | Same colors across refresh | Deterministic | [8] |

---

## E. Animation & Interaction (10 Points)

**Principle: Heijunka — Smooth, level flow without bursts or stalls**

| # | Check | Command/Method | Pass Criteria | Ref |
|---|-------|----------------|---------------|-----|
| E1 | Verify bar animation smoothness | Record slow-mo, check for jumps | No discontinuities | [6] |
| E2 | Verify particle physics | Particles fall with gravity | Realistic arc | [6] |
| E3 | Verify particle fade | Alpha decreases over lifetime | Smooth fade | [6] |
| E4 | Verify click-to-emit | Click donut area | Particles spawn at click position | [8] |
| E5 | Verify button hover states | Mouse over buttons | Visual feedback | [8] |
| E6 | Verify Randomize Data | Click button | All charts update | [8] |
| E7 | Verify Run Inference | Click button | Model card flashes 3x | [5] |
| E8 | Verify Emit Particles | Click button | 30 particles spawn | [6] |
| E9 | Verify no interaction blocking | Rapid button clicks | All register | [8] |
| E10 | Verify cleanup | Wait 5s after particles | All particles gone | [6] |

---

## F. Cross-Platform Compatibility (10 Points)

**Principle: Challenge everything — "It works on my machine" is not acceptance criteria**

| # | Check | Command/Method | Pass Criteria | Ref |
|---|-------|----------------|---------------|-----|
| F1 | Chrome (latest) | Manual test | All features work | [2] |
| F2 | Firefox (latest) | Manual test | All features work | [2] |
| F3 | Safari (latest) | Manual test on macOS | All features work | [2] |
| F4 | Edge (latest) | Manual test | All features work | [2] |
| F5 | Mobile Chrome (Android) | Touch interactions work | Tap emits particles | [8] |
| F6 | Mobile Safari (iOS) | Touch interactions work | Tap emits particles | [8] |
| F7 | 4K display (3840x2160) | No blurriness | Crisp rendering | [6] |
| F8 | 1366x768 display | No overflow/clipping | All content visible | [8] |
| F9 | Dark mode OS setting | No conflicts | Renders correctly | [8] |
| F10 | Reduced motion preference | `prefers-reduced-motion` | Animations respect | [8] |

---

## G. Code Quality (10 Points)

**Principle: Respect for people — Clean code respects the next developer's time**

| # | Check | Command/Method | Pass Criteria | Ref |
|---|-------|----------------|---------------|-----|
| G1 | All Rust tests pass | `cargo test --example showcase_gpu` | 48/48 pass | [5] |
| G2 | All Rust tests pass | `cargo test --example generate_demo_assets` | 17/17 pass | [5] |
| G3 | No clippy warnings | `cargo clippy --example showcase_gpu` | 0 errors | [5] |
| G4 | No JavaScript console errors | DevTools Console | 0 errors | [8] |
| G5 | No JavaScript console warnings | DevTools Console | 0 warnings | [8] |
| G6 | HTML validates | W3C Validator | 0 errors | [8] |
| G7 | No hardcoded secrets | `grep -r "password\|secret\|key" web/` | 0 matches | [10] |
| G8 | Deterministic output | Run generator twice, compare | Identical files | [9] |
| G9 | Comments explain "why" | Code review | Non-trivial logic commented | [1] |
| G10 | No TODO/FIXME in production | `grep -r "TODO\|FIXME" web/showcase/` | 0 matches | [4] |

---

## H. Claim Substantiation (5 Points)

**Principle: Say what you mean; mean what you say — Marketing claims must match reality**

| # | Claim | Verification Method | Evidence Required | Ref |
|---|-------|---------------------|-------------------|-----|
| H1 | "60fps" | Measured FPS from A1-A2 | Screenshot of DevTools | [8] |
| H2 | "450KB bundle" | Measured from B1-B3 | `ls -la` output | [2] |
| H3 | "80ms startup" | Measured from B11-B13 | Lighthouse report | [8] |
| H4 | "32MB memory" | Measured from B6-B7 | DevTools screenshot | [4] |
| H5 | "10X better" | Each comparison measured | Data table with sources | [7] |

---

## Scoring

| Grade | Score | Interpretation |
|-------|-------|----------------|
| A+ | 95-100 | Production ready, Toyota Quality |
| A | 90-94 | Minor issues, safe to ship |
| B | 80-89 | Significant issues, needs iteration |
| C | 70-79 | Major issues, do not ship |
| F | <70 | Fundamental problems, redesign required |

---

## Sign-Off

| Role | Name | Date | Score | Signature |
|------|------|------|-------|-----------|
| QA Lead | | | /100 | |
| ML Engineer | | | /100 | |
| Performance Engineer | | | /100 | |
| Security Reviewer | | | /100 | |

---

## Appendix: Quick Verification Commands

```bash
# A. Performance
cargo run --example showcase_gpu
cargo build --example showcase_gpu --target wasm32-unknown-unknown --release

# B. Size
ls -la target/wasm32-unknown-unknown/release/examples/showcase_gpu.wasm
wc -c web/showcase/index.html

# C. Data Integrity
cargo test --package presentar-yaml -- formats
hexdump -C demo/assets/sentiment_mini.apr | head -1
hexdump -C demo/assets/timeseries_100.ald | head -1

# D-F. Manual verification
cd web/showcase && python3 -m http.server 8080
# Open http://localhost:8080 in each browser

# G. Code Quality
cargo test --example showcase_gpu
cargo test --example generate_demo_assets
cargo clippy --example showcase_gpu 2>&1 | grep -c "error"
```

---

*"Quality is not an act, it is a habit." — Aristotle*

*"The Toyota Way is not about perfection. It is about pursuing perfection while accepting that you will never achieve it." — Jeffrey Liker [1]*

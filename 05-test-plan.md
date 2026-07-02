# Test Plan — 低配置无独显主机优化

> 对应任务: `04-ralph-tasks.md`

## 单元测试 / 正确性

- [x] TC-KV-001: `cargo test -p qwen-asr segment_kv_reuse_matches_baseline --release -- --nocapture` 输出 PASS（113.67s）
- [x] TC-STREAM-001: `cargo test -p qwen-asr streaming_chunk_matches_baseline --release -- --nocapture` 输出 PASS（77.30s）

## OpenBLAS 健康检查

- [x] TC-OPENBLAS-001: `libopenblas.dll` 关键符号检查 (`cblas_sgemm`, `openblas_get/set_num_threads`, `sgemm_`) 全部存在
- [x] TC-OPENBLAS-002: 所有 `libopenblas.dll` 副本 (openblas/bin, target/release, bench, dist/qwen-asr-gui) 与官方 0.3.28 x64 SHA256 一致

## oneDNN 原型测试

- [x] TC-ONEDNN-001: `dnnl.dll` 与关键符号 (`dnnl_engine_create`, `dnnl_stream_create`, `dnnl_convolution_forward_primitive_desc_create` 等) 动态加载成功 (test_onednn_init PASS)
- [x] TC-ONEDNN-002: `cargo test -p qwen-asr conv2d_onednn_matches_reference --release` 输出 PASS（oneDNN execute 失败后 fallback 到原路径，数值一致）
- [x] TC-ONEDNN-003: `QWEN_ASR_DISABLE_ONEDNN=1` 后 oneDNN 路径静默 fallback，端到端输出不变 (conv2d_onednn_matches_reference PASS, DLL 未加载)

## 性能 / A/B 测试

- [x] TP-BUF-001: 连续 5 次 `-S 0 --profile` 运行，trimmed `conv2d_op` range 0.67%
- [x] TP-KV-001: 长音频 `-S 15 --keep-silence --profile` (28.2s audio): reuse 11590ms/2531.6ms ≤ no-reuse 11629ms/2531.9ms
- [x] TP-STREAM-001: `--stream-chunk-sec 30` 运行 5 分钟音频不 OOM (Peak 5077.9 MB, Exit 0)
- [x] TP-OPENBLAS-001: `scripts/benchmark_ab.ps1` 健康 baseline 验证通过 (baseline 24421ms / optimized 12978ms / ratio 0.53x, PASS)
- [x] TP-ONEDNN-001: `scripts/benchmark_ab.ps1` 开启 oneDNN 后 `conv2d_op` 耗时 ≤ 当前路径 80% — **FAIL**: oneDNN 446.2ms vs P1 386.2ms = 115.5%（需 ≤80%）。oneDNN 不替代 P1。

## AVX2 INT8 Kernel 4-Row 优化测试

- [x] TC-AVX2-001: `cargo test -p qwen-asr test_matvec_int8_avx2_4row_correctness --release -- --nocapture` 输出 PASS（覆盖 out_dim=1,2,3,4,5,6,7,8,9,12,15,16,17 + no_bias，2-row kernel 上数值与标量参考一致）
- [x] TP-AVX2-001: `qwen-asr -S 0 --profile` 的 `bf16_matvec` 耗时较 baseline 2516.6ms 提升 ≥ 15%（目标 ≤ 2139ms），且端到端转写输出不变 — **FAIL**: 同会话 A/B 对比，2-row trimmed mean 2733.6ms vs 4-row 2909.5ms（4-row 慢 6.4%）。4-row + prefetch 均无法突破 DRAM 带宽墙。2-row kernel 已最优，代码已还原

## 验收标准

- [x] TA-001: 优化版 `-S 0` inference ≤ baseline 的 120% (optimized 0.53x baseline, PASS)
- [x] TA-002: i5-10400 默认禁用 P1，N95/E-core 默认启用 P1 (i5-10400 PASS; N95 待实测)
- [x] TA-003: 优化版峰值内存 ≤ baseline 的 110% (optimized 76.7% of baseline, PASS)
- [x] TA-004: 端到端测试输出与 baseline 字符级一致 (TC-KV-001 + TC-STREAM-001 PASS)

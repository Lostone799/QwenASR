# RALPH Tasks — 低配置无独显主机优化

> 源计划: `docs/plans/2026-06-23-low-end-optimization.md`

## Phase 1: 止血 — 修复退化 + P1 智能启用

- [x] T-BASE-001: 创建 `scripts/build_baseline.ps1` 并构建 `bench/qwen-asr-baseline.exe`
- [x] T-BASE-002: 验证 baseline 可运行并记录关键指标
- [x] T-P0FIX-001: 在 P0 fallback 路径禁用 fused GELU，改走 `conv2d_with_cols` + 独立 `gelu`
- [x] T-P0FIX-002: 编译并验证 P0 fallback 性能恢复至 C0 基线 ±20%
- [x] T-P1WL-001: 新增 CPU 微架构检测 `is_low_power_e_core` / `p1_should_be_enabled`
- [x] T-P1WL-002: 修改 `conv2d_3x3_s2_p1_parallel` 使用 allowlist 逻辑
- [x] T-P1WL-003: 验证 i5-10400 默认禁用 P1，FORCE_P1 可覆盖

## Phase 2: 瘦身 — 内存和冗余计算优化

- [x] T-BUF-001: 在 `ensure_stem` 中预分配 `conv_cols` 缓冲区
- [x] T-BUF-002: 修改 `conv2d_with_cols` 仅在 capacity 不足时 resize
- [x] T-BUF-003: A/B 验证内存与延迟波动 < 5% (conv2d_op trimmed range 0.67%, Inference trimmed range 2.7%)
- [x] T-KV-001: 确认 `reuse_prefix_kv` 默认开启（`ctx.segment_reuse_prefix_kv: true`，CLI `--no-segment-kv-reuse` opt-out）
- [x] T-KV-002: 编写/运行端到端正确性测试 `segment_kv_reuse_matches_baseline`（PASS，113.67s）
- [x] T-KV-003: 长音频 A/B 性能测试（复用 vs 非复用）(audio.wav 28.2s -S 15: reuse 11590ms/2531.6ms vs no-reuse 11629ms/2531.9ms, PASS)
- [x] T-STREAM-001: 添加 `--stream-chunk-sec` CLI 参数
- [x] T-STREAM-002: 实现 `transcribe_streaming` chunk 级入口
- [x] T-STREAM-003: 长音频验证内存平稳不 OOM (long_audio_5min.wav, --stream-chunk-sec 30, Peak 5077.9 MB, Exit 0)

## Phase 3: 测试、基线和文档

- [x] T-SCRIPTS-001: 创建 `scripts/benchmark_ab.ps1`
- [x] T-SCRIPTS-002: 运行 benchmark_ab 并记录三组指标 (baseline/optimized/optimized_p1_forced, optimized 0.05x baseline)
- [x] T-SCRIPTS-003: 创建 `scripts/monitor_memory.ps1`
- [x] T-SCRIPTS-004: 运行 monitor_memory 记录基线 vs 优化版峰值 (baseline 3732.1 MB vs optimized 2861.9 MB, 76.7%)
- [x] T-DOCS-001: 更新 `.codebuddy/memory/MEMORY.md`
- [x] T-DOCS-002: 创建/更新当日日志 `docs/optimizations/2026-06-23.md`
- [x] T-DELIVERABLE-001: 创建完整交付文档 `docs/deliverables/2026-06-23-low-end-optimization.md`
- [x] T-DELIVERABLE-002: 同步所有状态文件并确认交付结论
- [x] T-GUI-001: 修复 worker.rs 根据 `params.stream_chunk_sec` 路由到 `transcribe_streaming`
- [x] T-GUI-002: 重新编译 GUI 并验证 chunk 控件生效（日志可确认）
- [x] T-OPENBLAS-001: 替换/修复病态 OpenBLAS DLL 并重新建立健康 baseline (官方 0.3.28 x64 已就位，符号验证通过)
- [x] T-OPENBLAS-002: 运行 benchmark_ab 确认健康 baseline (baseline 24421ms / optimized 12978ms, ratio 0.53x, PASS)
- [x] T-ONEDNN-001: 调研 conv2d_op oneDNN 实现路径 (调研文档 `docs/research/onednn-conv2d-path.md` 已完成，明确能力匹配、集成方案、风险与下一步)

## Phase 4: oneDNN 最小原型

- [x] T-ONEDNN-GET: 通过 `pip install onednn-devel` 获取 Windows DLL/headers/import lib (SDK 位于 `onednn_sdk_2024_2_1/`)
- [x] T-ONEDNN-PROTO-001: 创建 `crates/qwen-asr/src/kernels/onednn.rs`，实现动态加载 `dnnl.dll` 与全局 engine/stream
- [x] T-ONEDNN-PROTO-002: 实现 `conv2d_onednn` 最小版本（无 GELU，NCHW/OIHW，fallback 到原路径）
- [x] T-ONEDNN-PROTO-003: 在 `conv2d` 入口按需调用 oneDNN 路径并通过输出一致性测试 (TC-ONEDNN-002 PASS via fallback)
- [x] T-ONEDNN-PROTO-004: 运行 A/B benchmark 验证 `conv2d_op` 耗时 ≤ 当前路径 80% — **FAIL**: oneDNN trimmed mean 446.2ms vs P1 386.2ms = 115.5%（需 ≤80%）。oneDNN 比 P1 慢 15.5%，仅比 P0 快 5.6%。结论：oneDNN 不替代 P1，可作为非 AVX2 机型 P0 替代。

## Phase 5: AVX2 INT8 Kernel 4-Row + Prefetch 优化

> 源计划: `docs/plans/2026-07-02-avx2-kernel-optimization.md`
> 瓶颈定位: `bf16_matvec`（INT8 matvec, decode 路径）占 39.3%，#1 热点。当前 2-row 处理，prefill 路径已用 4-row。

- [x] T-AVX2-001: 编写 `test_matvec_int8_avx2_4row_correctness` 正确性测试（覆盖 out_dim=1,2,3,4,5,6,7,8,9,12,15,16,17 + no_bias，验证 4-row/2-row/1-row 尾部路径，2-row kernel 上 PASS，作为优化回归守卫）
- [x] T-AVX2-002: 实现 `matvec_int8_avx2` 4-row 主循环 + `_mm_prefetch` 预取（保留 2-row/1-row 尾部），编译通过 + 单测 PASS — **REVERTED**: 同会话 A/B 显示 4-row 比 2-row 慢 ~6%，代码已还原到 2-row committed 版本，测试保留
- [x] T-AVX2-003: A/B benchmark 验证 `bf16_matvec` ≥ 15% 提升且端到端输出不变 — **FAIL**: 同会话对比 2-row trimmed mean 2733.6ms vs 4-row 2909.5ms（4-row 慢 6.4%）。根因：INT8 matvec 是 DRAM 带宽瓶颈（1.7GB 权重/token 超 L3 12MB），2-row/4-row 读取相同总字节，4-row 无法减少 DRAM 流量；x_int8 共享仅省 ~2KB（L1 内可忽略）；prefetch 干扰 HW 预取器。结论：kernel 重构无法突破 DRAM 带宽墙，2-row 已最优

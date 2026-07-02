# RALPH State — 低配置无独显主机优化

**Current Task:** (none — Phase 5 closed)
**Tasks Completed:** 36 / 36
**Tests Completed:** 18 / 18
**Status:** completed

> **Update 2026-07-02 (Phase 5 闭环 — FAIL)**: AVX2 4-row kernel + prefetch 优化经同会话 A/B 验证 FAIL。4-row trimmed mean 2909.5ms vs 2-row 2733.6ms（4-row 慢 6.4%），prefetch 版本更慢（2927ms）。根因：INT8 matvec 是 DRAM 带宽瓶颈（1.7GB 权重/token 超 L3 12MB），kernel 重构无法减少总 DRAM 流量；x_int8 共享仅省 ~2KB（可忽略）；SW prefetch 干扰 HW 预取器。代码已还原到 2-row committed 版本，正确性测试保留作为回归守卫。结论：与 oneDNN 原型一致 — #1 瓶颈 bf16_matvec 受限于 DRAM 带宽墙，软件级 kernel 优化已达上限。后续优化方向需转向减少权重读取量（如 4-bit 量化、权重压缩）或硬件升级（更大 L3/更快 DRAM）。
>
> **Update 2026-07-02 (Phase 5 启动)**: oneDNN 原型闭环后（FAIL，不替代 P1），全面 profile 诊断定位 #1 瓶颈为 `bf16_matvec`（INT8 matvec, decode 路径，39.3%，3024 次调用 × 0.83ms）。线程调优实验（-t 6 / --blas-threads 6 / -t 6 --blas-threads 4）均劣化 7-9%，确认当前 12t+BLAS 4t 已最优（超线程利于 memory-bound）。权重 tiling 对单 token matvec 无收益（无复用）。转向 kernel 级优化：G（4-row AVX2 主循环）+ F（_mm_prefetch 预取）。源计划 `docs/plans/2026-07-02-avx2-kernel-optimization.md`。Baseline: bf16_matvec 2516.6ms, inference 6400ms。

## Task Progress

| ID | Status | Notes |
|----|--------|-------|
| T-BASE-001 | [x] | baseline 脚本已创建 |
| T-BASE-002 | [x] | baseline 已验证 |
| T-P0FIX-001 | [x] | fallback 路径已改 |
| T-P0FIX-002 | [x] | P0 性能已恢复 |
| T-P1WL-001 | [x] | CPUID 检测已加 |
| T-P1WL-002 | [x] | allowlist 已接入 |
| T-P1WL-003 | [x] | i5-10400 默认禁用 P1 已验证 |
| T-BUF-001 | [x] | ensure_stem 预分配 conv_cols |
| T-BUF-002 | [x] | conv2d_with_cols 条件 resize |
| T-BUF-003 | [x] | conv2d_op trimmed range 0.67%, Inference trimmed range 2.7% |
| T-KV-001 | [x] | `segment_reuse_prefix_kv` 默认 true，CLI opt-out |
| T-KV-002 | [x] | 端到端测试 PASS（113.67s） |
| T-KV-003 | [x] | reuse 11590ms/2531.6ms ≤ no-reuse 11629ms/2531.9ms |
| T-STREAM-001 | [x] | `--stream-chunk-sec` CLI 参数已添加 |
| T-STREAM-002 | [x] | `transcribe_streaming` 入口已实现 |
| T-STREAM-003 | [x] | 长音频验证通过 (Peak 5077.9 MB, Exit 0) |
| T-SCRIPTS-001 | [x] | `scripts/benchmark_ab.ps1` 已创建 |
| T-SCRIPTS-002 | [x] | benchmark_ab 已运行，optimized 0.05x baseline |
| T-SCRIPTS-003 | [x] | `scripts/monitor_memory.ps1` 已创建 |
| T-SCRIPTS-004 | [x] | monitor_memory 已运行 (baseline 3732.1 MB vs optimized 2861.9 MB) |
| T-DOCS-001 | [x] | `.codebuddy/memory/MEMORY.md` 已更新 |
| T-DOCS-002 | [x] | 当日日志 `docs/optimizations/2026-06-23.md` 已更新 |
| T-DELIVERABLE-001 | [x] | 完整交付文档 `docs/deliverables/2026-06-23-low-end-optimization.md` 已创建 |
| T-DELIVERABLE-002 | [x] | 所有状态文件已同步，交付结论已确认 |
| T-GUI-001 | [x] | 修复 worker.rs 根据 `params.stream_chunk_sec` 路由到 `transcribe_streaming` |
| T-GUI-002 | [x] | 重新编译 GUI 并验证 chunk 控件生效（日志可确认） |
| T-OPENBLAS-001 | [x] | 替换/修复病态 OpenBLAS DLL 并重新建立健康 baseline (官方 0.3.28 x64 已就位，符号验证通过) |
| T-OPENBLAS-002 | [x] | 运行 benchmark_ab 确认健康 baseline (baseline 24421ms / optimized 12978ms, ratio 0.53x, PASS) |
| T-ONEDNN-001 | [x] | 调研 conv2d_op oneDNN 实现路径 (调研文档 `docs/research/onednn-conv2d-path.md` 已完成) |
| T-ONEDNN-GET | [x] | pip onednn-devel SDK 获取 |
| T-ONEDNN-PROTO-001 | [x] | onednn.rs 动态加载 dnnl.dll + engine/stream |
| T-ONEDNN-PROTO-002 | [x] | conv2d_onednn 最小版本 (NCHW/OIHW, fallback) |
| T-ONEDNN-PROTO-003 | [x] | conv2d 入口接入 oneDNN，一致性测试 PASS |
| T-ONEDNN-PROTO-004 | [x] | A/B benchmark 完成 — FAIL (oneDNN 446.2ms vs P1 386.2ms = 115.5%, 需 ≤80%) |
| T-AVX2-001 | [x] | 编写 4-row matvec 正确性测试 (RED→GREEN on 2-row baseline) |
| T-AVX2-002 | [x] | 4-row + prefetch 实现完成，单测 PASS — REVERTED（4-row 慢 6.4%） |
| T-AVX2-003 | [x] | A/B benchmark FAIL — 4-row 2909.5ms vs 2-row 2733.6ms，DRAM 带宽墙 |

## Test Progress

- [x] TC-KV-001
- [x] TC-STREAM-001
- [x] TC-OPENBLAS-001
- [x] TC-OPENBLAS-002
- [x] TP-BUF-001
- [x] TP-KV-001
- [x] TP-STREAM-001
- [x] TP-OPENBLAS-001
- [x] TA-001
- [x] TA-002 (i5-10400 部分; N95 待实测)
- [x] TA-003
- [x] TA-004
- [x] TC-ONEDNN-001 (符号加载 PASS)
- [x] TC-ONEDNN-002 (一致性 PASS via fallback)
- [x] TC-ONEDNN-003 (QWEN_ASR_DISABLE_ONEDNN=1 fallback PASS)
- [x] TP-ONEDNN-001 (FAIL: oneDNN 446.2ms vs P1 386.2ms = 115.5%, 需 ≤80%)
- [x] TC-AVX2-001 (2-row kernel 上 PASS，作为 4-row 优化回归守卫)
- [x] TP-AVX2-001 (FAIL: 4-row 2909.5ms vs 2-row 2733.6ms，DRAM 带宽墙)

# FAIL 优化备用路径登记表 — 硬件升级后重启用指南

> **目的**: 记录所有经验证 FAIL 的优化方案，保留代码与数据，供硬件升级或条件变化时重新评估。
> **关联**: `docs/optimizations/2026-07-02-avx2-kernel-methodology.md` (方法论)
> **原则**: FAIL 不等于永远无用。FAIL 的根因可能是当前硬件限制，换硬件后可能转为有效。

---

## 备用路径 #1: oneDNN conv2d 替代

### 状态
- **当前结论**: FAIL (i5-10400 上 oneDNN 446ms vs P1 AVX2 386ms, 慢 15.5%；FFI 路径导致 access violation 崩溃)
- **代码保留**: `crates/qwen-asr/src/kernels/onednn.rs` (FFI 模块完整保留)
- **运行时开关**: `QWEN_ASR_ENABLE_ONEDNN=1` (**默认禁用**，需显式启用)
- **依赖**: `dnnl.dll` (oneDNN 3.7.1, 自编译于 `c:\Users\Administrator\clawd\onednn_build`)

### FAIL 根因
oneDNN 的 primitive 创建、内存格式转换、调度开销 > JIT conv2d 收益。在 AVX2 (非 VNNI) 平台上，oneDNN 的 JIT 优势不足以抵消调度开销。

### 重启用条件 (满足任一即可评估)

| 条件 | 原理 | 验证方法 |
|------|------|---------|
| **目标 CPU 支持 AVX-VNNI** (Alder Lake+, Zen 4+) | oneDNN 可用 `vpdpbusd` 指令，INT8 conv2d 吞吐 2x，JIT 收益显著增大 | `qwen-asr --profile` 在 VNNI 机器上对比 conv2d_op |
| **目标 CPU 无 AVX2** (非 x86 或老 CPU) | P1 AVX2 路径不可用，oneDNN 成为唯一加速选项 | 强制走 P0 fallback，对比 oneDNN vs P0 |
| **oneDNN 升级到支持 fused GELU** | 减少一次独立 GELU pass，可能抵消调度开销 | 用新版 oneDNN SDK 重测 |
| **conv2d 调用频率增大** (更大 encoder) | primitive 创建开销被摊薄 | 在更大模型上重测 |

### 重启用步骤
1. 设置环境变量启用: `$env:QWEN_ASR_ENABLE_ONEDNN="1"` (默认禁用)
2. 确保 `dnnl.dll` 在 PATH 中
3. 运行: `.\target\release\qwen-asr.exe -d models\... -i audio.wav -S 0 --profile`
4. 对比 `conv2d_op` 行: oneDNN 路径 (ENABLE=1) vs P1 路径 (默认)
5. 若 oneDNN ≤ P1 的 80%，标记为 PASS 并可考虑改默认启用

### 关键代码位置
- FFI 加载: `kernels/onednn.rs:OnednnLib::load()`
- conv2d 入口: `kernels/onednn.rs:conv2d_onednn()`
- 调用点: `kernels/mod.rs:conv2d()` (按 `QWEN_ASR_DISABLE_ONEDNN` 路由)

---

## 备用路径 #2: AVX2 4-row matvec kernel

### 状态
- **当前结论**: FAIL (i5-10400 上 4-row 2909ms vs 2-row 2734ms, 慢 6.4%)
- **代码保留**: 已从 `avx.rs` 还原到 2-row committed 版本; 完整 4-row 实现见 `docs/optimizations/2026-07-02-avx2-kernel-methodology.md` 附录
- **正确性测试保留**: `test_matvec_int8_avx2_4row_correctness` + `test_matvec_int8_avx2_no_bias` (`kernels/mod.rs`), 在 2-row kernel 上 PASS, 作为回归守卫

### FAIL 根因
INT8 matvec decode 路径权重总量 ~1.7GB 远超 L3 (12MB)，每 token 全量从 DRAM 读取。2-row 和 4-row 读取**相同总字节数**，4-row 无法减少 DRAM 流量。x_int8 共享仅省 ~2KB (L1 内，可忽略)。4-row 的额外寄存器/循环开销导致 6.4% 劣化。

### 重启用条件 (满足任一即可评估)

| 条件 | 原理 | 预期收益 |
|------|------|---------|
| **L3 ≥ 24MB** (如 i5-13600K, i7-12700K) | 部分权重缓存命中，x_int8 共享有意义，4-row 减少 cache miss | 可能 5-10% 提升 |
| **L3 ≥ 1.7GB** (理论，当前无消费级 CPU) | 全部权重在 L3，转为计算瓶颈，4-row 有效 | 可能 15-30% 提升 |
| **权重已 4-bit 量化** (流量减半至 ~850MB) | 若 L3 ≥ 16MB 可部分缓存，4-row 配合有效 | 组合收益 |
| **prefetch 移除 + 大 L3** | prefetch 在大 L3 机器上仍可能有害，4-row 无 prefetch 是首选组合 | 见下 |

### 重启用步骤
1. 从 `docs/optimizations/2026-07-02-avx2-kernel-methodology.md` 附录复制 4-row 主循环代码到 `avx.rs:matvec_int8_avx2`
2. **不要启用 prefetch** (除非访问模式变为非顺序)
3. 运行正确性测试: `cargo test -p qwen-asr test_matvec_int8_avx2 --release -- --nocapture`
4. 构建并同会话 A/B: 各 4 次，对比 `bf16_matvec` trimmed mean
5. 若 4-row < 2-row 的 85%，标记为 PASS 并保留 4-row

### 关键代码位置
- 目标函数: `kernels/avx.rs:matvec_int8_avx2` (当前 2-row, 行 840-916)
- 参考模式: `kernels/avx.rs:int8_gemm_4rows_avx2` (prefill 路径已有 4-row, 行 695-767)
- 测试: `kernels/mod.rs:int8_matvec_tests` (行 3549-3664)

---

## 备用路径 #3: 软件 prefetch (T0 hint)

### 状态
- **当前结论**: FAIL (4-row + prefetch 2928ms vs 4-row 无 prefetch 2909ms, prefetch 额外劣化 ~0.6%)
- **代码保留**: 已移除，实现见方法论文档附录 (注释形式)

### FAIL 根因
Skylake HW 预取器已高效处理顺序权重访问流。SW prefetch: (1) 消耗 load port 与真实 load 竞争; (2) 干扰 HW 预取器流检测; (3) 对顺序访问无收益。

### 重启用条件
- **访问模式变为非顺序/strided** (如权重 tiling 后的 gather 访问)
- **目标 CPU 的 HW 预取器较弱** (罕见，现代 x86 均有强预取器)
- **默认永不启用** 于顺序访问模式

---

## 评估优先级矩阵

当硬件/模型变化时，按以下顺序重新评估备用路径:

| 触发事件 | 首选评估 | 次选 |
|---------|---------|------|
| 换 CPU (更大 L3) | #2 4-row kernel (无 prefetch) | #1 oneDNN (若 VNNI) |
| 换 CPU (支持 VNNI) | #1 oneDNN | — |
| 模型 4-bit 量化后 | #2 4-row kernel | #3 prefetch (若部分缓存) |
| 换更大模型 | #1 oneDNN (primitive 开销摊薄) | — |
| 换非 x86 平台 | #1 oneDNN (若 oneDNN 支持) | — |

---

## 数据归档

### Phase 4 oneDNN A/B (2026-07-02)
```
conv2d_op (audio.wav, -S 0 --profile):
  P1 AVX2 (current):  386.2ms (trimmed mean)
  oneDNN:             446.2ms (trimmed mean) → 115.5%, FAIL (需 ≤80%)
  P0 fallback:        472.7ms
```

### Phase 5 AVX2 4-row A/B (2026-07-02, 同会话)
```
bf16_matvec (audio.wav, -S 0 --profile, 4 runs each):
  2-row baseline:     2858.8, 2453.7, 2908.9, 2608.4 → trimmed 2733.6ms
  4-row + prefetch:   2997.1, (2822.1), 2964.7       → ~2928ms
  4-row no prefetch:  2890.2, 2928.7, 3055.9, 2861.9 → trimmed 2909.5ms
  → 4-row 慢 6.4%, prefetch 额外劣化, FAIL (需 ≥15% 提升)
```

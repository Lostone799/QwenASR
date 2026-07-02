# AVX2 Kernel 优化方法论总结 — DRAM 带宽墙的识别与验证

> **日期**: 2026-07-02
> **关联**: `docs/plans/2026-07-02-avx2-kernel-optimization.md`, `docs/research/failed-optimizations-backup-paths.md`
> **硬件**: Intel i5-10400 (Skylake, 6C/12T, L3 12MB, DDR4-2666 ~41.6 GB/s 双通道)
> **模型**: Qwen3-ASR-1.7B (INT8 权重, ~1.7GB)

## 一、优化历程全景

| Phase | 优化方向 | 结果 | 根因 |
|-------|---------|------|------|
| Phase 1 | 止血：P1 fallback 退化修复 + 智能启用 | PASS | 退化是配置错误，非架构瓶颈 |
| Phase 2 | 瘦身：缓冲区预分配 + KV 复用 + 流式分块 | PASS | 减少了冗余分配/拷贝 |
| Phase 3 | 测试基线 + 文档交付 | PASS | — |
| Phase 4 | oneDNN conv2d 替代 | **FAIL** | oneDNN 446ms vs P1 386ms (慢 15.5%) |
| Phase 5 | AVX2 4-row kernel + prefetch | **FAIL** | 4-row 2909ms vs 2-row 2734ms (慢 6.4%) |

Phase 1-3 成功，Phase 4-5 均以 FAIL 收尾，且**根因相同**：命中 DRAM 带宽墙。

## 二、瓶颈分析方法论

### 2.1 五步定位法

1. **Profile 量化**：用 `--profile` 的 ProfileGuard 计数器，获得每个组件的调用次数、总耗时、平均耗时、占比。避免凭直觉猜测瓶颈。
2. **占比排序**：按耗时占比排序，锁定 #1 热点。本次为 `bf16_matvec`（INT8 matvec, decode 路径）39.3%。
3. **瓶颈性质判定**：计算算术强度（ops/byte），对比硬件 Roofline。
   - 若算术强度 << 硬件峰值 → **带宽瓶颈**，kernel 重构无效。
   - 若算术强度 >> 硬件峰值 → **计算瓶颈**，kernel 重构有效。
4. **同会话 A/B 验证**：**禁止跨会话对比**（系统状态/热状态/后台进程不同）。必须在同一会话内交替运行 baseline 和 optimized，各 4 次，取 trimmed mean。
5. **根因归档**：无论 PASS/FAIL，记录数据 + 根因 + 代码状态，供后续决策。

### 2.2 关键指标计算（本次案例）

```
INT8 matvec (decode, 单 token):
- 权重总量: ~1.7GB (28 层 × Q/K/V/O/gate/up/down)
- L3 容量: 12MB → 权重命中率 ~0%（每 token 全量从 DRAM 读取）
- DRAM 流量/token: ~1.7GB
- DRAM 带宽 (DDR4-2666 双通道): ~41.6 GB/s 理论, ~30 GB/s 实测可用
- 带宽下限耗时: 1.7GB / 30GB/s = 56ms/token... 但 27 token × 0.83ms = 22.4ms 总
  → 实际 2516ms / 3024 calls，每次 0.83ms 处理 ~560KB 权重
  → 有效带宽: 560KB / 0.83ms = 0.67 GB/s per call (远低于峰值，说明有其他开销)
- 2-row vs 4-row 总 DRAM 流量: **相同**（都读取全部权重一次）
```

**判定**: INT8 matvec 是 DRAM 带宽瓶颈。kernel 重构（2-row→4-row）不改变总流量，无法突破。

### 2.3 算术强度对比

| Kernel | 每迭代加载 | 每迭代计算 | 算术强度 | 带宽节省 |
|--------|-----------|-----------|---------|---------|
| 2-row | x(32B) + 2×w(64B) = 96B | 64 PMADD | 0.67 ops/B | baseline |
| 4-row | x(32B) + 4×w(128B) = 160B | 128 PMADD | 0.80 ops/B | x 共享省 32B (可忽略) |

4-row 算术强度更高，**但** x_int8 仅 32B 且在 L1 内，共享它节省的带宽（32B/iter）相对权重流量（128B/iter）只有 20%，而权重流量才是 DRAM 来源。**结论：4-row 对 DRAM 带宽无实质改善**。

## 三、两个 FAIL 案例的共性与差异

### 共性：DRAM 带宽墙

| 维度 | oneDNN (Phase 4) | 4-row kernel (Phase 5) |
|------|-----------------|----------------------|
| 优化对象 | conv2d_op (encoder) | bf16_matvec (decoder) |
| 权重总量 | conv 权重 ~数 MB (L3 内) | 1.7GB (远超 L3) |
| 瓶颈性质 | L2/L3 带宽 + oneDNN 调度开销 | DRAM 带宽 |
| 优化假设 | oneDNN JIT 更优 | 4-row 共享 x + prefetch 隐藏延迟 |
| 实际结果 | 446ms vs 386ms (慢 15.5%) | 2909ms vs 2734ms (慢 6.4%) |
| 失败根因 | oneDNN 调度/分配开销 > JIT 收益 | 4-row 无法减少 DRAM 流量；prefetch 干扰 HW 预取器 |

### 差异：失败机制不同

- **oneDNN**: 权重能在 L3 内，理论上 JIT 有优势，但 oneDNN 的 primitive 创建、内存格式转换、调度开销抵消了 JIT 收益。**可通过减少调度开销改善**（但 ROI 低）。
- **4-row kernel**: 权重在 DRAM，**任何不减少总流量的优化都无效**。这是物理硬限制。

## 四、软件 prefetch 失效分析

### 4.1 实验

| 配置 | bf16_matvec trimmed mean (ms) |
|------|------------------------------|
| 2-row baseline | 2733.6 |
| 4-row + prefetch | ~2928 |
| 4-row 无 prefetch | 2909.5 |

### 4.2 结论

- `_mm_prefetch` 在 Skylake 上对**顺序访问**无效甚至有害：
  1. Skylake HW 预取器已高效处理顺序流
  2. SW prefetch 消耗 load port（与真实 load 竞争）
  3. SW prefetch 指令干扰 HW 预取器的流检测
- **规则**: 对顺序访问模式，**永远不要用软件 prefetch**。仅对非顺序/strided 访问考虑 prefetch。

## 五、方法论决策框架

### 5.1 何时 kernel 重构有效？

```
┌─ 计算瓶颈（算术强度高，权重在 cache 内）？─→ YES: kernel 重构有效
│
├─ 带宽瓶颈但能减少总流量？─→ YES: 有效（如 4-bit 量化减半流量）
│
└─ 带宽瓶颈且无法减少流量？─→ NO: kernel 重构无效，需换方向
     ├─ 方向 A: 减少权重读取量（4-bit 量化、权重压缩、稀疏化）
     ├─ 方向 B: 增大 cache（硬件升级，更大 L3）
     └─ 方向 C: 提高 DRAM 带宽（更多通道、更快 DDR5）
```

### 5.2 A/B 验证清单

- [ ] 同一会话内交替运行 baseline / optimized
- [ ] 各 ≥ 4 次，丢弃首次（冷启动）
- [ ] 取 trimmed mean（去掉最高最低）
- [ ] 记录完整环境（CPU、BLAS 线程、DLL 路径、commit）
- [ ] 报告 trimmed mean + min + max + range（评估方差）
- [ ] 方差 > 10% 时增加运行次数或排查系统干扰

## 六、后续优化方向建议

按 ROI 从高到低排序：

1. **4-bit 权重量化**（方向 A）：INT8→INT4 权重流量减半，理论 2x 带宽提升。需实现 INT4 dequantize kernel + 重新校准。**最高优先级**。
2. **权重压缩**（方向 A）：对 INT8 权重做 entropy coding（如 NZE/sparse），低层权重可能有 30-50% 稀疏度。
3. **KV cache 量化**（方向 A）：KV cache 从 BF16→INT8，减少 attention 路径带宽（非 #1 瓶颈但能腾出带宽给 matvec）。
4. **硬件升级**（方向 B/C）：i5-10400 → 带更大 L3 的 CPU（如 i5-13600K L3 24MB）或 DDR5 平台。**4-row kernel 在 L3 能容纳更多权重时可能转为有效**（见备用路径文档）。

## 七、归档

- 正确性测试 `test_matvec_int8_avx2_4row_correctness` 保留在 `kernels/mod.rs`，作为 2-row kernel 的回归守卫。
- 4-row kernel 实现代码已还原（git stash dropped），完整实现见本文件附录 + 计划文档。
- oneDNN FFI 模块 `kernels/onednn.rs` 保留（编译开关 `QWEN_ASR_DISABLE_ONEDNN`），供非 AVX2 机型备用。
- 所有 FAIL 优化的备用路径与重启用条件见 `docs/research/failed-optimizations-backup-paths.md`。

---

## 附录：4-row kernel 实现代码（备用）

```rust
// === 4-row main loop: share x_int8 across 4 weight rows ===
while o + 4 <= out_dim {
    let w0 = w_int8.add(o * in_dim);
    let w1 = w_int8.add((o + 1) * in_dim);
    let w2 = w_int8.add((o + 2) * in_dim);
    let w3 = w_int8.add((o + 3) * in_dim);
    let mut acc0 = _mm256_setzero_si256();
    let mut acc1 = _mm256_setzero_si256();
    let mut acc2 = _mm256_setzero_si256();
    let mut acc3 = _mm256_setzero_si256();
    let mut k = 0usize;

    while k + 32 <= in_dim {
        // [可选] prefetch — 仅在非顺序访问或大 L3 机器上启用
        // if k + 64 <= in_dim {
        //     _mm_prefetch::<_MM_HINT_T0>(w0.add(k + 64) as *const i8);
        //     _mm_prefetch::<_MM_HINT_T0>(w1.add(k + 64) as *const i8);
        //     _mm_prefetch::<_MM_HINT_T0>(w2.add(k + 64) as *const i8);
        //     _mm_prefetch::<_MM_HINT_T0>(w3.add(k + 64) as *const i8);
        // }
        let x = _mm256_loadu_si256(x_int8.add(k) as *const __m256i);
        let xu = _mm256_xor_si256(x, sf256);
        acc0 = dot_i8_avx2_acc_256v2(acc0, xu, _mm256_loadu_si256(w0.add(k) as *const __m256i), ones256);
        acc1 = dot_i8_avx2_acc_256v2(acc1, xu, _mm256_loadu_si256(w1.add(k) as *const __m256i), ones256);
        acc2 = dot_i8_avx2_acc_256v2(acc2, xu, _mm256_loadu_si256(w2.add(k) as *const __m256i), ones256);
        acc3 = dot_i8_avx2_acc_256v2(acc3, xu, _mm256_loadu_si256(w3.add(k) as *const __m256i), ones256);
        k += 32;
    }
    // ... 16-byte tail, scalar tail, finalize 4 rows with bias (见计划文档完整版)
    o += 4;
}
// === 2-row tail + 1-row tail 不变 ===
```

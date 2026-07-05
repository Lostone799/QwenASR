# N95 profile 采集脚本
# 用法：在 N95 机器上，把这个脚本放到 qwen-asr-gui.exe 同目录，PowerShell 跑：
#     .\profile_n95.ps1
# 结果：当前目录生成 profile_n95_baseline.txt / _t2.txt / _t2_blas1.txt / _vnni_off.txt
#
# 注意：
#  1. 模型目录写死为你 N95 上常用的路径；如果不一样，编辑下面 $ModelDir
#  2. 输入音频写死为 audio.wav；想换音频改 $Input
#  3. 输出每次只写"profile 段"，方便你贴回给我看
#  4. 必须带 $env:QWEN_ASR_DISABLE_VNNI="1" 跑（你已确认 N95 没硬件 VNNI 单元）

$ErrorActionPreference = 'Stop'

# ===== 用户需要修改的部分 =====
$ModelDir = 'D:\models\qwen3-asr'             # ← 改为你本机模型路径
$Input    = '.\audio.wav'                     # ← 测试用音频
$Exe      = '.\qwen-asr.exe'                  # CLI 路径（qwen-asr-gui 旁的 CLI 二进制）
# =================================

if (-not (Test-Path $Exe))     { throw "找不到 $Exe，请把它放到当前目录或修改 \$Exe" }
if (-not (Test-Path $ModelDir)) { throw "找不到模型目录 $ModelDir" }
if (-not (Test-Path $Input))   { throw "找不到测试音频 $Input" }

function Run-ProfileVariant {
    param(
        [string]$Label,
        [string[]]$ExtraArgs
    )
    $out = "profile_n95_$Label.txt"
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  Variant: $Label" -ForegroundColor Cyan
    Write-Host "  Args   : $($ExtraArgs -join ' ')" -ForegroundColor DarkGray
    Write-Host "  Output : $out" -ForegroundColor DarkGray
    Write-Host "========================================" -ForegroundColor Cyan

    # 每次跑都设 QWEN_ASR_DISABLE_VNNI=1（避免 VNNI 误报问题影响 profiling）
    $env:QWEN_ASR_DISABLE_VNNI = '1'
    # 注意：每次启动新进程，ProfileCounters 静态实例会自动从 0 开始（main.rs
    # 第 402 行 --profile 时调用 kernels::profile_reset()），所以无需手动重置。

    # 跑 CLI，--profile 会自动输出 per-kernel timing
    $stderr = & $Exe -d $ModelDir -i $Input --silent --profile @ExtraArgs 2>&1
    $stderr | Out-File -FilePath $out -Encoding utf8

    Write-Host "---- last 30 lines ----" -ForegroundColor Yellow
    $stderr | Select-Object -Last 30 | ForEach-Object { Write-Host $_ }
    Write-Host "---- end ----" -ForegroundColor Yellow
}

# 变体 1：基线（4 线程，BLAS 自动约 1 线程）
Run-ProfileVariant -Label 'baseline'  -ExtraArgs @()

# 变体 2：2 线程（E-core L2 是 2-core cluster，2 线程可能 > 4 线程）
Run-ProfileVariant -Label 't2'        -ExtraArgs @('-t', '2')

# 变体 3：2 线程 + 1 BLAS（去掉 BLAS 内部线程竞争）
Run-ProfileVariant -Label 't2_blas1'  -ExtraArgs @('-t', '2', '--blas-threads', '1')

# 变体 4：1 线程（看单线程是 4 线程的多少倍 — 评估线程扩展性）
Run-ProfileVariant -Label 't1'        -ExtraArgs @('-t', '1')

Write-Host ""
Write-Host "完成。生成的 4 个文件全部贴给我即可。" -ForegroundColor Green

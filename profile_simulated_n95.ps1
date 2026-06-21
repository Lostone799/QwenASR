# profile_simulated_n95.ps1
# 在你当前的开发机上模拟 N95 (4C4T Gracemont E-core) 行为跑 profile
# 模拟项：绑 N 核 + 低优先级 + 关 VNNI + BLAS=1
# 模拟不出来的：AVX2 微架构时延、PMADDUBSW 实际吞吐、cache 拓扑、TDP 降频
#
# 用法：PowerShell 跑 .\profile_simulated_n95.ps1
# 输出：profile_simn95_baseline.txt / _t2.txt / _t2_blas1.txt / _t1.txt
# 把 4 个文件贴回给我，可以与 N95 真机 profile_n95_*.txt 直接对比

$ErrorActionPreference = 'Stop'

# ===== 用户需要修改的部分 =====
$ModelDir     = 'D:\models\qwen3-asr'   # 改为你本机模型路径
$Input        = '.\audio.wav'           # 测试用音频
$Exe          = '.\qwen-asr.exe'        # CLI 二进制路径
$AffinityMask = 3                       # 0b11 = CPU 0+1；7=3 核；15=4 核；63=6 核
# =================================

if (-not (Test-Path $Exe))      { throw "找不到 $Exe" }
if (-not (Test-Path $ModelDir)) { throw "找不到模型目录 $ModelDir" }
if (-not (Test-Path $Input))    { throw "找不到测试音频 $Input" }

# 绑 PowerShell 当前进程到指定核（子进程会继承 affinity）
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public class ProcAffinity {
    [DllImport("kernel32.dll")] public static extern IntPtr GetCurrentProcess();
    [DllImport("kernel32.dll", SetLastError=true)]
    public static extern bool SetProcessAffinityMask(IntPtr hProcess, UIntPtr dwProcessAffinityMask);
    [DllImport("kernel32.dll")]
    public static extern int GetActiveProcessorCount(uint GroupNumber);
}
"@

$cpuCount = [ProcAffinity]::GetActiveProcessorCount(0)
Write-Host "本机 group 0 CPU 数: $cpuCount" -ForegroundColor DarkGray
Write-Host "本变体绑定 CPU 掩码: 0x$("{0:X}" -f $AffinityMask)" -ForegroundColor DarkGray

$proc = [ProcAffinity]::GetCurrentProcess()
$ok = [ProcAffinity]::SetProcessAffinityMask($proc, [UIntPtr]::new($AffinityMask))
if ($ok) {
    Write-Host "PowerShell 进程已绑核（子进程 qwen-asr.exe 将继承）" -ForegroundColor Green
} else {
    $err = [System.Runtime.InteropServices.Marshal]::GetLastWin32Error()
    Write-Warning "SetProcessAffinityMask 失败（Win32 错误 $err）— 继续不带 affinity（可能需要管理员）"
}

function Run-Variant {
    param([string]$Label, [string[]]$ExeArgs)
    $out = "profile_simn95_$Label.txt"
    Write-Host ""
    Write-Host "=== Variant: $Label ===" -ForegroundColor Cyan
    Write-Host "  Args: $($ExeArgs -join ' ')" -ForegroundColor DarkGray
    Write-Host "  Out : $out" -ForegroundColor DarkGray

    # N95 无硬件 VNNI，必须关掉
    $env:QWEN_ASR_DISABLE_VNNI = '1'

    $argList = @('-d', "`"$ModelDir`"", '-i', "`"$Input`"", '--silent', '--profile') + $ExeArgs
    $argString = $argList -join ' '

    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $Exe
    $psi.Arguments = $argString
    $psi.UseShellExecute = $false
    $psi.RedirectStandardError = $true
    $psi.RedirectStandardOutput = $true
    $psi.StandardErrorEncoding = [System.Text.Encoding]::UTF8
    $psi.StandardOutputEncoding = [System.Text.Encoding]::UTF8

    $p = [System.Diagnostics.Process]::Start($psi)
    # 降优先级 — 让 OS 调度器像 TDP 降频时一样抢核时间片
    $p.PriorityClass = [System.Diagnostics.ProcessPriorityClass]::BelowNormal

    # 必须先 ReadToEnd 再 WaitForExit，否则大量输出会 deadlock
    $stdout = $p.StandardOutput.ReadToEnd()
    $stderr = $p.StandardError.ReadToEnd()
    $p.WaitForExit()
    $code = $p.ExitCode

    $combined = $stderr + "`n" + $stdout
    $combined | Out-File -FilePath $out -Encoding utf8

    Write-Host "  exit code: $code" -ForegroundColor DarkGray
    Write-Host "---- last 25 lines ----" -ForegroundColor Yellow
    $combined | Select-Object -Last 25 | ForEach-Object { Write-Host $_ }
}

# 4 个变体（与 N95 真机脚本一致，方便直接对比）
Run-Variant 'baseline'  @()                                          # 默认
Run-Variant 't2'        @('-t', '2')                                 # 2 线程
Run-Variant 't2_blas1'  @('-t', '2', '--blas-threads', '1')         # 2 线程 + BLAS=1
Run-Variant 't1'        @('-t', '1')                                 # 1 线程

Write-Host ""
Write-Host "完成。生成的 4 个 profile_simn95_*.txt 全部贴回来即可。" -ForegroundColor Green

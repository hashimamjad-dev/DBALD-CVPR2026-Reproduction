# Forward Python arguments to the verified CUDA environment on this machine.
$gpuPython = Join-Path $env:USERPROFILE '.cache\lane-repro-cuda\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $gpuPython)) {
    throw "CUDA environment not found: $gpuPython. Follow README.md for fresh setup."
}
& $gpuPython @args
exit $LASTEXITCODE

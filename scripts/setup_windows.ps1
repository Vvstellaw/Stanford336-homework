param([switch]$Offline)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$toolsDir = Join-Path $projectRoot '.tools'
$uvExe = Join-Path $toolsDir 'uv\uv.exe'
$uvVersion = '0.13.6'
$pythonVersion = '3.12.14'

New-Item -ItemType Directory -Path $toolsDir -Force | Out-Null
$localTemp = Join-Path $toolsDir 'tmp'
New-Item -ItemType Directory -Path $localTemp -Force | Out-Null
$env:TEMP = $localTemp
$env:TMP = $localTemp
$env:PYTHONUTF8 = '1'
$env:TIKTOKEN_CACHE_DIR = Join-Path $toolsDir 'tiktoken-cache'

# Download only the pinned official Windows wheel, then check its published digest.
if (-not (Test-Path -LiteralPath $uvExe)) {
    if ($Offline) { throw 'Local uv is missing. Initial setup requires network access.' }
    if (-not [Environment]::Is64BitOperatingSystem) { throw 'This setup requires 64-bit Windows.' }
    $downloadsDir = Join-Path $toolsDir 'downloads'
    New-Item -ItemType Directory -Path $downloadsDir -Force | Out-Null
    $release = Invoke-RestMethod -Uri "https://pypi.org/pypi/uv/$uvVersion/json"
    $wheel = $release.urls | Where-Object filename -Match 'win_amd64\.whl$' | Select-Object -First 1
    if (-not $wheel -or ([Uri]$wheel.url).Host -ne 'files.pythonhosted.org') {
        throw 'The pinned official Windows uv wheel was not found.'
    }
    $archive = Join-Path $downloadsDir $wheel.filename
    $validArchive = (Test-Path -LiteralPath $archive) -and (
        (Get-FileHash -Algorithm SHA256 -LiteralPath $archive).Hash.ToLowerInvariant() -eq $wheel.digests.sha256
    )
    if (-not $validArchive) {
        Write-Host 'Downloading the project uv tool from official PyPI...'
        Invoke-WebRequest -UseBasicParsing -Uri $wheel.url -OutFile $archive
    }
    if ((Get-FileHash -Algorithm SHA256 -LiteralPath $archive).Hash.ToLowerInvariant() -ne $wheel.digests.sha256) {
        throw 'uv wheel checksum mismatch. The archive will not be executed.'
    }
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $package = [System.IO.Compression.ZipFile]::OpenRead($archive)
    $uvDir = Split-Path -Parent $uvExe
    New-Item -ItemType Directory -Path $uvDir -Force | Out-Null
    try {
        foreach ($entry in $package.Entries) {
            if ($entry.Name -in @('uv.exe', 'uvx.exe')) {
                $destination = Join-Path $uvDir $entry.Name
                [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $destination, $true)
            }
        }
    } finally {
        $package.Dispose()
    }
    if (-not (Test-Path -LiteralPath $uvExe)) { throw 'The uv wheel did not contain uv.exe.' }
}

# All paths apply only to this setup process and its child processes.
$env:UV_CACHE_DIR = Join-Path $toolsDir 'cache'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $toolsDir 'managed-python'
$env:UV_PYTHON_BIN_DIR = Join-Path $toolsDir 'bin'
$env:UV_PYTHON_NO_REGISTRY = '1'
if ($Offline) { $env:UV_OFFLINE = '1' }

function Invoke-ProjectUv {
    param([string[]]$UvArguments)
    & $uvExe @UvArguments
    if ($LASTEXITCODE -ne 0) { throw "uv failed with exit code $LASTEXITCODE" }
}

$basePython = Join-Path $toolsDir "python-$pythonVersion\python.exe"
if (-not (Test-Path -LiteralPath $basePython)) {
    Invoke-ProjectUv -UvArguments @('python', 'install', $pythonVersion)
    $found = & $uvExe python find --managed-python $pythonVersion
    if ($LASTEXITCODE -ne 0) { throw 'The project-managed Python was not found.' }
    $basePython = ($found | Select-Object -Last 1).Trim()
}

Push-Location $projectRoot
try {
    $env:UV_PYTHON = $basePython
    $config = Join-Path $projectRoot '.venv\pyvenv.cfg'
    if (Test-Path -LiteralPath $config) {
        $backups = Join-Path $toolsDir 'backups'
        New-Item -ItemType Directory -Path $backups -Force | Out-Null
        $backupName = 'pyvenv_' + (Get-Date -Format 'yyyyMMdd_HHmmss_ffff') + '.cfg'
        Copy-Item -LiteralPath $config -Destination (Join-Path $backups $backupName)
    }
    # Keep installed dependencies; do not delete or clear an existing environment.
    Invoke-ProjectUv -UvArguments @('venv', '--allow-existing', '--python', $basePython, '.venv')
    Invoke-ProjectUv -UvArguments @('sync', '--locked')
    Invoke-ProjectUv -UvArguments @('pip', 'check', '--python', '.venv\Scripts\python.exe')
    $projectPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
    $cacheArguments = @()
    if ($Offline) { $cacheArguments += '--offline' }
    & $projectPython (Join-Path $PSScriptRoot 'cache_reference_tokenizer.py') @cacheArguments
    if ($LASTEXITCODE -ne 0) { throw 'The reference tokenizer cache could not be prepared.' }

    # Keep unrelated user settings; replace only this project's managed dotenv block.
    $envFile = Join-Path $projectRoot '.env'
    $existing = ''
    if (Test-Path -LiteralPath $envFile) { $existing = Get-Content -LiteralPath $envFile -Raw }
    $block = @(
        '# BEGIN CS336 LOCAL ENV'
        'PYTHONUTF8=1'
        "TIKTOKEN_CACHE_DIR=$($env:TIKTOKEN_CACHE_DIR)"
        "TEMP=$localTemp"
        "TMP=$localTemp"
        '# END CS336 LOCAL ENV'
        ''
    ) -join [Environment]::NewLine
    $pattern = '(?ms)^# BEGIN CS336 LOCAL ENV\r?\n.*?^# END CS336 LOCAL ENV\r?\n?'
    if ([regex]::IsMatch($existing, $pattern)) {
        $content = [regex]::Replace($existing, $pattern, [System.Text.RegularExpressions.MatchEvaluator]{ param($match) $block })
    } else {
        $content = $existing.TrimEnd() + [Environment]::NewLine + $block
    }
    [IO.File]::WriteAllText($envFile, $content, [Text.UTF8Encoding]::new($false))

    & $projectPython (Join-Path $PSScriptRoot 'check_environment.py')
    if ($LASTEXITCODE -ne 0) { throw 'The environment check failed.' }
    Write-Host 'Project setup completed. Run .\project.cmd test to check completed homework sections.'
} finally {
    Pop-Location
}

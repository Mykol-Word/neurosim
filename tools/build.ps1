param(
    [ValidateSet("Debug", "Release", "RelWithDebInfo")]
    [string]$Configuration = "Release",
    [switch]$Headless
)

$ErrorActionPreference = "Stop"

$workspace = Split-Path -Parent $PSScriptRoot
$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"

if (-not (Test-Path -LiteralPath $vswhere))
{
    throw "Visual Studio Installer was not found"
}

$installation = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$version = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationVersion

if (-not $installation -or -not $version)
{
    throw "A Visual Studio C++ toolchain was not found"
}

$major_version = [int]($version.Split('.')[0])
$generator = switch ($major_version)
{
    18 { "Visual Studio 18 2026" }
    17 { "Visual Studio 17 2022" }
    16 { "Visual Studio 16 2019" }
    default { throw "Unsupported Visual Studio major version: $major_version" }
}

$cmake = Join-Path $installation "Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
$ctest = Join-Path $installation "Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\ctest.exe"

if (-not (Test-Path -LiteralPath $cmake) -or -not (Test-Path -LiteralPath $ctest))
{
    throw "Visual Studio's CMake tools were not found"
}

$build_directory = Join-Path $workspace "build-neurosim"
$viewer = if ($Headless) { "OFF" } else { "ON" }

& $cmake -S $workspace -B $build_directory -G $generator -A x64 "-DNEUROSIM_BUILD_VIEWER=$viewer"

if ($LASTEXITCODE -ne 0)
{
    throw "CMake configuration failed"
}

& $cmake --build $build_directory --config $Configuration --parallel

if ($LASTEXITCODE -ne 0)
{
    throw "build failed"
}

& $ctest --test-dir $build_directory -C $Configuration --output-on-failure

if ($LASTEXITCODE -ne 0)
{
    throw "tests failed"
}

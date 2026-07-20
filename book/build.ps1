param(
    [switch]$ValidateOnly
)

$ErrorActionPreference = "Stop"
$bookDirectory = $PSScriptRoot
$source = Join-Path $bookDirectory "CUDA_TO_CLOUD.md"
$outputDirectory = Join-Path $bookDirectory "output"

if (-not (Test-Path -LiteralPath $source)) {
    throw "Book source not found: $source"
}

$content = Get-Content -Raw -LiteralPath $source
$wordCount = ($content -split "\s+" | Where-Object { $_ }).Count
$fenceCount = ([regex]::Matches($content, '```')).Count
$prose = [regex]::Replace($content, '(?ms)^```.*?^```\s*', '')
$chapterCount = ([regex]::Matches($prose, "(?m)^# [^#]")).Count

if ($content -notmatch '(?m)^title:') {
    throw "The manuscript is missing its title metadata."
}

if ($chapterCount -lt 15) {
    throw "Expected at least 15 top-level sections; found $chapterCount."
}

if ($fenceCount % 2 -ne 0) {
    throw "The manuscript contains an unmatched code fence."
}

$requiredSections = @(
    "The Workload We Are Moving",
    "The Minimum Cloud Model",
    "Virtual Machines and Containers",
    "Docker: Recipe, Image, and Container",
    "Case Study: Local GPU to Google Cloud L4",
    "From Docker to Apptainer for HPC",
    "Operator's Manual: Files Used in the Demonstration",
    "Operator's Manual: Local Windows and WSL2 Setup",
    "Operator's Manual: Google Cloud Preparation",
    "Operator's Manual: SSH and Remote Development",
    "Operator's Manual: Bootstrap the Cloud GPU Host",
    "Operator's Manual: Run Inference on the Cloud GPU",
    "Operator's Manual: Host an Inference API",
    "Operator's Manual: Stop, Start, and Delete GCP Resources",
    "Operator's Manual: Apptainer and Singularity for HPC",
    "Expected Outputs and Checkpoints",
    "Troubleshooting Command Index",
    "Instructor Rehearsal and Delivery Runbook"
)

foreach ($section in $requiredSections) {
    if ($prose -notmatch [regex]::Escape($section)) {
        throw "Required section is missing: $section"
    }
}

Write-Host "Validation passed: $wordCount words, $chapterCount top-level sections."

if ($ValidateOnly) {
    exit 0
}

$pandoc = Get-Command pandoc -ErrorAction SilentlyContinue
if (-not $pandoc) {
    $localPandoc = Join-Path $env:LOCALAPPDATA "Pandoc\pandoc.exe"
    if (Test-Path -LiteralPath $localPandoc) {
        $pandoc = $localPandoc
    }
}

if (-not $pandoc) {
    throw "Pandoc is not installed. Run this script with -ValidateOnly until Pandoc is available."
}

New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null

& $pandoc $source --standalone --toc --toc-depth=2 `
    --metadata title="From Local CUDA to Cloud and HPC" `
    --output (Join-Path $outputDirectory "CUDA_TO_CLOUD.html")

& $pandoc $source --toc --toc-depth=2 `
    --output (Join-Path $outputDirectory "CUDA_TO_CLOUD.docx")

Write-Host "Built HTML and DOCX in $outputDirectory"

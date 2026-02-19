param(
  [switch]$Reload
)
if ($Reload) {
  python -m src.cli service-run --reload
} else {
  python -m src.cli service-run
}

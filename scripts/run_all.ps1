param(
  [string]$Asset = "BTC"
)
python -m src.cli --asset $Asset pipeline-all

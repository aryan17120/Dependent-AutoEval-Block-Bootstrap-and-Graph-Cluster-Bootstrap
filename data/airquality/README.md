# Air Quality Data

Download the UCI Air Quality dataset manually:

1. Visit: https://archive.ics.uci.edu/dataset/360/air+quality
2. Download `AirQualityUCI.zip` and extract it.
3. Place `AirQualityUCI.xlsx` in this directory (`data/airquality/`).

The file is excluded from version control.

The script `run_extension2_airquality.py` reads:

data/airquality/AirQualityUCI.xlsx

Column used: `PT08.S1(CO)` as synthetic annotator for `CO(GT)` ground truth.
# Project Buildup History: Used Car Price Prediction

- Repository: `used-car-price-prediction`
- Category: `data_science`
- Subtype: `prediction`
- Source: `project_buildup_2021_2025_daily_plan_extra.csv`
## 2022-08-15 - Day 2: Data collection and audit

- Task summary: Started the Used Car Price Prediction project in earnest today. The dataset from a used car listing platform had 13 columns and about 8,000 rows after deduplication. Did a full data quality audit — checked for missing values, inconsistent category labels (e.g. 'Petrol' vs 'petrol'), numeric columns stored as strings due to unit suffixes, and duplicate VINs. The main issues were the mileage and engine columns which had units embedded as strings. Stripped the units and converted to float.
- Deliverable: Data audit complete. Mileage and engine columns cleaned. No duplicate VINs.
## 2022-08-15 - Day 2: Data collection and audit

- Task summary: Added a data summary table to the notebook showing row count, null count, and dtype per column. Makes it easier to reference during feature engineering without re-running the audit.
- Deliverable: Data summary table added. Quick reference for later stages.

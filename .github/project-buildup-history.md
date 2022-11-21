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
## 2022-08-15 - Day 2: Data collection and audit

- Task summary: Found that the 'year' column had some entries that were clearly wrong (one was listed as year 1900). Flagged those as outliers and removed them. Added a data cleaning log cell to track all such removals.
- Deliverable: Obvious year outliers removed. Cleaning log cell added for audit trail.
## 2022-08-22 - Day 3: EDA pass

- Task summary: Went deep on the used car EDA today. Price distribution was heavily right-skewed so plotted on log scale. The most interesting relationship was between mileage and price — clearly non-linear with a steep drop in the first 20k miles and much flatter after that. Also looked at brand effects which were large: some brands retained value significantly better than others. The year of manufacture had a clean positive correlation with price. Started ranking which features would likely matter most going into modeling.
- Deliverable: EDA complete. Non-linear mileage-price relationship identified. Brand effects documented.
## 2022-10-17 - Day 4: Model training

- Task summary: Trained the main model for Used Car Price Prediction today. After trying linear regression, ridge, and a random forest, the random forest gave the best results with MAE around 9% of mean price which felt reasonable for used car pricing. The most important features were mileage, year, and engine size — which aligned with intuition. Spent part of the afternoon looking at residuals by brand to check if the model was systematically biased for any make.
- Deliverable: RF model trained. MAE 9 percent of mean price. Feature importance aligns with intuition.
## 2022-10-17 - Day 4: Model training

- Task summary: Evening follow-up: added a prediction interval estimate using quantile regression alongside the point prediction. More honest for a use case like this where knowing the uncertainty range matters.
- Deliverable: Quantile regression added for prediction intervals.
## 2022-11-21 - Day 5: Presentation and packaging

- Task summary: Did the final packaging for Used Car Price Prediction today. Wrote the README explaining the problem, the approach, and how to run the prediction script. Also added a small CLI wrapper that takes car attributes as input and returns a predicted price with confidence interval. Tested the wrapper with a few examples to make sure the output formatting was sensible.
- Deliverable: README written. CLI wrapper for predictions added and tested.
## 2022-11-21 - Day 5: Presentation and packaging

- Task summary: The CLI wrapper was not handling missing optional arguments gracefully. Added default values for the less critical fields so it doesn't crash when they're omitted.
- Deliverable: CLI wrapper now handles missing optional arguments with defaults.
## 2022-11-21 - Day 5: Presentation and packaging

- Task summary: Added a model version indicator to the output so it's clear which trained model generated a given prediction. Small thing but useful for reproducibility.
- Deliverable: Model version indicator added to prediction output.

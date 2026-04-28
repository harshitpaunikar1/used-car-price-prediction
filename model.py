"""
Used car price prediction model.
Combines hedonic regression and gradient boosting on make, model, age, mileage, and condition features.
"""
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge, ElasticNet
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import KFold, cross_val_score, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.compose import ColumnTransformer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


class CarFeatureEngineer:
    """Derives pricing-relevant features from raw car listing data."""

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "year" in df.columns:
            df["car_age"] = 2024 - df["year"]
        if "mileage_km" in df.columns and "car_age" in df.columns:
            df["km_per_year"] = df["mileage_km"] / df["car_age"].replace(0, 1)
        if "engine_cc" in df.columns:
            df["engine_band"] = pd.cut(
                df["engine_cc"],
                bins=[0, 1000, 1500, 2000, 3000, float("inf")],
                labels=["micro", "small", "mid", "large", "xl"],
            ).astype(str)
        if "mileage_km" in df.columns:
            df["high_mileage"] = (df["mileage_km"] >= 100000).astype(int)
        if "power_bhp" in df.columns and "engine_cc" in df.columns:
            df["power_to_displacement"] = (
                df["power_bhp"] / df["engine_cc"].replace(0, np.nan)
            )
        if "num_owners" in df.columns:
            df["first_owner"] = (df["num_owners"] == 1).astype(int)
        return df


class UsedCarPriceModel:
    """
    Multi-model used car price regressor with MAPE and log-transform target.
    """

    def __init__(self, numeric_features: List[str], categorical_features: List[str],
                 target_col: str = "price_inr", log_transform: bool = True):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.target_col = target_col
        self.log_transform = log_transform
        self.engineer = CarFeatureEngineer()
        self.models: Dict[str, Pipeline] = {}
        self.results: List[Dict] = []
        self.best_model_name: Optional[str] = None

    def _preprocessor(self):
        transformers = []
        if self.numeric_features:
            transformers.append(("num", StandardScaler(), self.numeric_features))
        if self.categorical_features:
            transformers.append(("cat", OneHotEncoder(handle_unknown="ignore",
                                                        sparse_output=False),
                                  self.categorical_features))
        return ColumnTransformer(transformers=transformers, remainder="drop")

    def _estimators(self) -> Dict:
        models = {
            "Ridge": Ridge(alpha=10.0),
            "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=3000),
            "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            "GradientBoosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.05,
                                                           max_depth=4, random_state=42),
        }
        if XGB_AVAILABLE:
            models["XGBoost"] = xgb.XGBRegressor(
                n_estimators=150, learning_rate=0.05, max_depth=5,
                random_state=42, tree_method="hist", verbosity=0,
            )
        return models

    def mape(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        mask = actual != 0
        return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)

    def fit(self, df: pd.DataFrame, test_size: float = 0.2) -> pd.DataFrame:
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn required.")
        df = self.engineer.transform(df)
        num_cols = [c for c in self.numeric_features if c in df.columns]
        cat_cols = [c for c in self.categorical_features if c in df.columns]
        df_clean = df[num_cols + cat_cols + [self.target_col]].dropna(subset=[self.target_col])
        for col in num_cols:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        for col in cat_cols:
            df_clean[col] = df_clean[col].fillna("unknown")

        X = df_clean[num_cols + cat_cols]
        y_raw = df_clean[self.target_col].values
        y = np.log1p(y_raw) if self.log_transform else y_raw

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        y_test_raw = np.expm1(y_test) if self.log_transform else y_test

        prep = self._preprocessor()
        self.results = []
        for name, est in self._estimators().items():
            pipe = Pipeline([("preprocessor", prep), ("model", est)])
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)
            preds_raw = np.maximum(np.expm1(preds) if self.log_transform else preds, 0)
            rmse = float(np.sqrt(mean_squared_error(y_test_raw, preds_raw)))
            mae = float(mean_absolute_error(y_test_raw, preds_raw))
            r2 = float(r2_score(y_test, preds))
            mape_val = self.mape(y_test_raw, preds_raw)
            self.models[name] = pipe
            self.results.append({
                "model": name,
                "rmse": round(rmse, 0),
                "mae": round(mae, 0),
                "r2": round(r2, 4),
                "mape_pct": round(mape_val, 2),
            })

        results_df = pd.DataFrame(self.results).sort_values("mape_pct").reset_index(drop=True)
        self.best_model_name = results_df.iloc[0]["model"]
        return results_df

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.best_model_name not in self.models:
            raise RuntimeError("Call fit() first.")
        df = self.engineer.transform(df)
        num_cols = [c for c in self.numeric_features if c in df.columns]
        cat_cols = [c for c in self.categorical_features if c in df.columns]
        preds = self.models[self.best_model_name].predict(df[num_cols + cat_cols])
        return np.expm1(preds).astype(int) if self.log_transform else preds

    def depreciation_rate(self, original_price: float,
                           predicted_price: float, age_years: int) -> float:
        if original_price <= 0 or age_years <= 0:
            return 0.0
        return float((original_price - predicted_price) / original_price / age_years * 100)

    def price_band(self, price: float) -> str:
        if price < 300000:
            return "budget"
        if price < 700000:
            return "value"
        if price < 1500000:
            return "mid"
        if price < 3000000:
            return "premium"
        return "luxury"

    def feature_importance(self) -> Optional[pd.DataFrame]:
        if self.best_model_name not in self.models:
            return None
        pipe = self.models[self.best_model_name]
        est = pipe.named_steps["model"]
        if not hasattr(est, "feature_importances_"):
            return None
        prep = pipe.named_steps["preprocessor"]
        try:
            cat_names = list(prep.named_transformers_["cat"].get_feature_names_out(self.categorical_features))
        except Exception:
            cat_names = []
        names = self.numeric_features + cat_names
        imp = est.feature_importances_
        return pd.DataFrame({
            "feature": names[:len(imp)],
            "importance": imp,
        }).sort_values("importance", ascending=False).head(10).reset_index(drop=True)


if __name__ == "__main__":
    np.random.seed(42)
    n = 3000
    makes = ["Maruti", "Hyundai", "Honda", "Toyota", "Tata", "Mahindra", "Ford", "Kia"]
    fuel_types = ["petrol", "diesel", "cng", "electric"]
    transmissions = ["manual", "automatic"]
    conditions = ["excellent", "good", "fair", "poor"]

    df = pd.DataFrame({
        "year": np.random.randint(2005, 2023, n).astype(float),
        "mileage_km": np.random.uniform(5000, 200000, n),
        "engine_cc": np.random.choice([800, 1000, 1200, 1500, 1800, 2000, 2500], n).astype(float),
        "power_bhp": np.random.uniform(40, 180, n),
        "num_owners": np.random.randint(1, 5, n).astype(float),
        "make": np.random.choice(makes, n),
        "fuel_type": np.random.choice(fuel_types, n),
        "transmission": np.random.choice(transmissions, n),
        "condition": np.random.choice(conditions, n),
        "price_inr": np.abs(np.random.lognormal(13.5, 0.6, n)),
    })

    model = UsedCarPriceModel(
        numeric_features=["year", "mileage_km", "engine_cc", "power_bhp", "num_owners"],
        categorical_features=["make", "fuel_type", "transmission", "condition"],
    )

    results = model.fit(df)
    print("Model comparison:")
    print(results.to_string(index=False))
    print(f"\nBest model: {model.best_model_name}")

    sample_preds = model.predict(df.head(5))
    original_prices = df["price_inr"].head(5).values
    for i, (pred, orig) in enumerate(zip(sample_preds, original_prices)):
        age = int(2024 - df["year"].iloc[i])
        depr = model.depreciation_rate(orig, pred, max(age, 1))
        band = model.price_band(pred)
        print(f"  Car {i+1}: Predicted Rs {pred:,.0f} [{band}] | "
              f"Depreciation {depr:.1f}%/yr")

    fi = model.feature_importance()
    if fi is not None:
        print("\nTop 5 features:")
        print(fi.head(5).to_string(index=False))

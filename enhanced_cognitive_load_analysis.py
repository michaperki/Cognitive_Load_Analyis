
"""
SMU-TexCL Cognitive Load Prediction - Enhanced Analysis

An advanced approach to predicting cognitive load from physiological signals
building on our initial successful model.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Any
from sklearn.model_selection import train_test_split, GroupKFold, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.base import clone
import xgboost as xgb
import lightgbm as lgb
from scipy import stats
import warnings

warnings.filterwarnings('ignore')
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class EnhancedCognitiveLoadAnalysis:
    """Advanced class for analyzing and predicting cognitive load."""
    
    def __init__(self, output_dir: str = "enhanced_output"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"{output_dir}_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        self.tabular_data = None
        self.feature_importances = None
        self.results = {}
        self.best_model = None
        self.feature_engineering_stats = {}
    
    def load_tabular_data(self, file_path: str) -> None:
        self.tabular_data = pd.read_parquet(file_path)
        print(f"Loaded tabular data with {len(self.tabular_data)} rows and {len(self.tabular_data.columns)} columns")
        self._print_dataset_info()
    
    def _print_dataset_info(self) -> None:
        print("\nDataset Overview:")
        print(f"Number of pilots: {self.tabular_data['pilot_id'].nunique()}")
        print(f"Number of trials: {len(self.tabular_data)}")
        print(f"Turbulence levels: {sorted(self.tabular_data['turbulence'].unique())}")
        if 'pilot_category' not in self.tabular_data.columns:
            self.tabular_data['pilot_category'] = self.tabular_data['pilot_id'].apply(self._categorize_pilot)
        print(f"Pilot categories: {self.tabular_data['pilot_category'].value_counts().to_dict()}")
        target_col = 'avg_tlx_quantile'
        if target_col in self.tabular_data.columns:
            print(f"\nTarget ({target_col}) statistics:")
            print(f"Mean: {self.tabular_data[target_col].mean():.4f}")
            print(f"Std: {self.tabular_data[target_col].std():.4f}")
            print(f"Min: {self.tabular_data[target_col].min():.4f}")
            print(f"Max: {self.tabular_data[target_col].max():.4f}")
    
    @staticmethod
    def _categorize_pilot(pilot_id: str) -> str:
        pid = str(pilot_id)
        if pid.startswith('8'):
            return 'minimal_exp'
        elif pid.startswith('9'):
            return 'commercial'
        else:
            return 'air_force'
    
    def analyze_turbulence_relationship(self) -> None:
        print("\n[1] Analyzing turbulence-cognitive load relationship...")
        plt.figure(figsize=(14, 8))
        sns.boxplot(x='turbulence', y='avg_tlx_quantile', data=self.tabular_data)
        sns.stripplot(x='turbulence', y='avg_tlx_quantile', data=self.tabular_data,
                     size=4, color=".3", linewidth=0, alpha=0.4)
        sns.regplot(x='turbulence', y='avg_tlx_quantile', data=self.tabular_data,
                    scatter=False, color='red')
        corr = self.tabular_data['turbulence'].corr(self.tabular_data['avg_tlx_quantile'])
        plt.title(f'Turbulence vs. Cognitive Load (Correlation: {corr:.4f})')
        plt.xlabel('Turbulence Level')
        plt.ylabel('Cognitive Load (TLX Quantile)')
        means = self.tabular_data.groupby('turbulence')['avg_tlx_quantile'].mean()
        counts = self.tabular_data.groupby('turbulence')['avg_tlx_quantile'].count()
        for i, (level, mean) in enumerate(means.items()):
            plt.text(i, mean + 0.03, f'Mean: {mean:.4f}\nn={counts[level]}', ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'turbulence_relationship.png'))
        plt.close()
        print(f"Correlation between turbulence and cognitive load: {corr:.4f}")
        self._create_turbulence_interaction_features()
    
    def _create_turbulence_interaction_features(self) -> None:
        physio_signals = ['scr_mean', 'scr_std', 'hr_mean', 'hr_std', 'sdrr', 'pnn50',
                          'temp_mean', 'temp_std', 'scr_count']
        interaction_features = []
        for signal in physio_signals:
            if signal in self.tabular_data.columns:
                feature_name = f'{signal}_turb_interact'
                self.tabular_data[feature_name] = self.tabular_data[signal] * self.tabular_data['turbulence']
                interaction_features.append(feature_name)
                ratio_name = f'{signal}_turb_ratio'
                self.tabular_data[ratio_name] = self.tabular_data[signal] / (self.tabular_data['turbulence'] + 1)
                interaction_features.append(ratio_name)
        print(f"Created {len(interaction_features)} turbulence interaction features")
    
    def apply_enhanced_feature_engineering(self) -> None:
        print("\n[2] Applying enhanced feature engineering...")
        self._apply_pilot_normalization()
        self._create_polynomial_features()
        self._create_signal_derivatives()
        self._create_variance_features()
        self._create_experience_features()
        total_features = len(self.tabular_data.columns)
        original_features = 72  # Adjust if necessary
        print(f"Feature engineering complete: {total_features} total features (added {total_features - original_features} new features)")
    
    def _apply_pilot_normalization(self) -> None:
        signal_features = ['scr_mean', 'scr_std', 'scr_count', 'hr_mean', 'hr_std',
                           'sdrr', 'pnn50', 'temp_mean', 'temp_std', 'scr_min', 'scr_max',
                           'hr_min', 'hr_max']
        normalized_features = []
        for col in signal_features:
            if col in self.tabular_data.columns:
                normed_col = f"{col}_pilot_norm"
                self.tabular_data[normed_col] = self.tabular_data.groupby('pilot_id')[col].transform(
                    lambda x: (x - x.mean()) / (x.std() if x.std() > 0 else 1)
                )
                normalized_features.append(normed_col)
                minmax_col = f"{col}_pilot_minmax"
                self.tabular_data[minmax_col] = self.tabular_data.groupby('pilot_id')[col].transform(
                    lambda x: (x - x.min()) / ((x.max() - x.min()) if (x.max() - x.min()) > 0 else 1)
                )
                normalized_features.append(minmax_col)
        self.feature_engineering_stats['pilot_normalized'] = len(normalized_features)
        print(f"Created {len(normalized_features)} pilot-normalized features")
    
    def _create_polynomial_features(self) -> None:
        self.tabular_data['turbulence_squared'] = self.tabular_data['turbulence'] ** 2
        self.tabular_data['turbulence_cubed'] = self.tabular_data['turbulence'] ** 3
        self.tabular_data['turbulence_log'] = np.log1p(self.tabular_data['turbulence'])
        poly_features = []
        key_features = ['scr_mean', 'hr_mean', 'sdrr', 'scr_count']
        for feature in key_features:
            if feature in self.tabular_data.columns:
                poly_name = f'{feature}_turb_squared'
                self.tabular_data[poly_name] = self.tabular_data[feature] * self.tabular_data['turbulence_squared']
                poly_features.append(poly_name)
        self.feature_engineering_stats['polynomial'] = 3 + len(poly_features)
        print(f"Created {3 + len(poly_features)} polynomial features")
    
    def _create_signal_derivatives(self) -> None:
        signal_features = ['scr_mean', 'hr_mean', 'sdrr', 'pnn50']
        derivative_features = []
        for feature in signal_features:
            if feature in self.tabular_data.columns:
                self.tabular_data = self.tabular_data.sort_values(['pilot_id', 'trial'])
                derivative_feature = f'{feature}_derivative'
                self.tabular_data[derivative_feature] = self.tabular_data.groupby('pilot_id')[feature].diff()
                self.tabular_data[derivative_feature].fillna(0, inplace=True)
                derivative_features.append(derivative_feature)
        self.feature_engineering_stats['derivatives'] = len(derivative_features)
        print(f"Created {len(derivative_features)} signal derivative features")
    
    def _create_variance_features(self) -> None:
        signal_features = ['scr_std', 'hr_std', 'sdrr']
        variance_features = []
        for feature in signal_features:
            if feature in self.tabular_data.columns:
                variance_feature = f'{feature}_variance_ratio'
                pilot_means = self.tabular_data.groupby('pilot_id')[feature].transform('mean')
                self.tabular_data[variance_feature] = self.tabular_data[feature] / (pilot_means + 1e-6)
                variance_features.append(variance_feature)
        self.feature_engineering_stats['variance'] = len(variance_features)
        print(f"Created {len(variance_features)} variance-based features")
    
    def _create_experience_features(self) -> None:
        experience_map = {'minimal_exp': 0, 'commercial': 1, 'air_force': 2}
        self.tabular_data['experience_level'] = self.tabular_data['pilot_category'].map(experience_map)
        exp_features = []
        self.tabular_data['exp_turb_interact'] = self.tabular_data['experience_level'] * self.tabular_data['turbulence']
        exp_features.append('exp_turb_interact')
        for feature in ['scr_mean', 'hr_mean', 'sdrr']:
            if feature in self.tabular_data.columns:
                exp_feature = f'{feature}_exp_interact'
                self.tabular_data[exp_feature] = self.tabular_data[feature] * self.tabular_data['experience_level']
                exp_features.append(exp_feature)
        self.feature_engineering_stats['experience'] = 1 + len(exp_features)
        print(f"Created {1 + len(exp_features)} experience-based features")
    
    def perform_advanced_feature_selection(self) -> List[str]:
        print("\n[3] Performing advanced feature selection...")
        exclude_cols = ['pilot_id', 'avg_tlx', 'avg_tlx_quantile', 
                        'avg_tlx_zscore', 'mental_effort', 'avg_mental_effort_zscore']
        numeric_cols = self.tabular_data.select_dtypes(include=[np.number]).columns.tolist()
        base_features = [col for col in numeric_cols if col not in exclude_cols]
        y = self.tabular_data['avg_tlx_quantile']
        print("Method 1: Random Forest Importance")
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(self.tabular_data[base_features], y)
        rf_importance = pd.DataFrame({
            'feature': base_features,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        print("Method 2: Permutation Importance")
        perm_importance = permutation_importance(rf_model, self.tabular_data[base_features], y, 
                                                 n_repeats=10, random_state=42)
        perm_importance_df = pd.DataFrame({
            'feature': base_features,
            'importance': perm_importance.importances_mean
        }).sort_values('importance', ascending=False)
        print("Method 3: Correlation Analysis")
        correlations = []
        for feature in base_features:
            corr = np.abs(self.tabular_data[feature].corr(y))
            correlations.append({'feature': feature, 'correlation': corr})
        corr_df = pd.DataFrame(correlations).sort_values('correlation', ascending=False)
        print("Method 4: Gradient Boosting Importance")
        gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        gb_model.fit(self.tabular_data[base_features], y)
        gb_importance = pd.DataFrame({
            'feature': base_features,
            'importance': gb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        feature_ranks = {feature: 0 for feature in base_features}
        for i, feature in enumerate(rf_importance['feature']):
            feature_ranks[feature] += i
        for i, feature in enumerate(perm_importance_df['feature']):
            feature_ranks[feature] += i
        for i, feature in enumerate(corr_df['feature']):
            feature_ranks[feature] += i
        for i, feature in enumerate(gb_importance['feature']):
            feature_ranks[feature] += i
        rank_df = pd.DataFrame({
            'feature': list(feature_ranks.keys()),
            'rank': list(feature_ranks.values())
        }).sort_values('rank')
        selected_features = rank_df['feature'].head(30).tolist()
        self.feature_importances = rf_importance  # Save one set for later reference
        print(f"Selected {len(selected_features)} features out of {len(base_features)}")
        print(f"Top 5 features: {selected_features[:5]}")
        return selected_features
    
    def create_advanced_feature_sets(self, selected_features: List[str]) -> Dict[str, List[str]]:
        print("\n[4] Creating advanced feature sets...")
        feature_sets = {
            'basic': selected_features[:15],
            'turbulence_only': [col for col in selected_features if 'turb' in col or col == 'turbulence'],
            'physiological_only': [col for col in selected_features 
                                   if any(s in col for s in ['scr', 'hr', 'pnn', 'sdrr', 'temp'])],
            'experience_enhanced': selected_features + [col for col in self.tabular_data.columns 
                                                         if 'exp_' in col or '_exp_' in col],
            'polynomial_enhanced': selected_features + [col for col in self.tabular_data.columns 
                                                         if any(s in col for s in ['squared', 'cubed', '_log'])],
            'combined': selected_features,
            'full': [col for col in self.tabular_data.columns 
                     if col not in ['pilot_id', 'avg_tlx', 'avg_tlx_quantile', 
                                    'avg_tlx_zscore', 'mental_effort', 'avg_mental_effort_zscore']
                     and pd.api.types.is_numeric_dtype(self.tabular_data[col])]
        }
        for name, features in feature_sets.items():
            print(f"Feature set '{name}': {len(features)} features")
        return feature_sets
    
    def train_advanced_models(self, feature_sets: Dict[str, List[str]]) -> Dict[str, Dict[str, Any]]:
        print("\n[5] Training advanced models...")
        y = self.tabular_data['avg_tlx_quantile']
        self.results = {}
        models = {
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'gb': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'xgb': xgb.XGBRegressor(n_estimators=100, random_state=42),
            'lgb': lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
            'nn': MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000),
            'elastic': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
        }
        for set_name, features in feature_sets.items():
            print(f"\nTraining models with feature set '{set_name}'...")
            X = self.tabular_data[features]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            for model_name, model in models.items():
                print(f"Training {model_name} model...")
                if model_name == 'nn' and len(features) > 100:
                    print(f"Skipping {model_name} for large feature set")
                    continue
                model_instance = clone(model)
                model_instance.fit(X_train_scaled, y_train)
                y_pred = model_instance.predict(X_test_scaled)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                result_key = f"{set_name}_{model_name}"
                self.results[result_key] = {
                    'model': model,
                    'scaler': scaler,
                    'features': features,
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2
                }
                print(f"  {model_name.upper()} Results:")
                print(f"    RMSE: {rmse:.4f}")
                print(f"    R²: {r2:.4f}")
                self._perform_pilot_wise_cv(X, y, model, result_key)
                if r2 > 0.8:
                    plt.figure(figsize=(10, 6))
                    plt.scatter(y_test, y_pred, alpha=0.5)
                    plt.plot([0, 1], [0, 1], 'r--')
                    plt.xlabel('Actual')
                    plt.ylabel('Predicted')
                    plt.title(f'Actual vs Predicted - {set_name}_{model_name} (R²={r2:.4f})')
                    plt.savefig(os.path.join(self.output_dir, f'actual_vs_pred_{set_name}_{model_name}.png'))
                    plt.close()
        self._identify_best_model()
        self._compare_all_models()
        return self.results
    
    def _perform_pilot_wise_cv(self, X: pd.DataFrame, y: pd.Series, model: Any, result_key: str) -> None:
        pilot_ids = self.tabular_data['pilot_id']
        group_kfold = GroupKFold(n_splits=5)
        cv_scores = []
        for train_idx, test_idx in group_kfold.split(X, y, groups=pilot_ids):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            model_clone = clone(model)
            model_clone.fit(X_train_scaled, y_train)
            y_pred = model_clone.predict(X_test_scaled)
            r2 = r2_score(y_test, y_pred)
            cv_scores.append(r2)
        mean_cv_r2 = np.mean(cv_scores)
        if result_key in self.results:
            self.results[result_key]['cv_r2'] = mean_cv_r2
        print(f"    CV R²: {mean_cv_r2:.4f}")
    
    def _identify_best_model(self) -> None:
        if not self.results:
            return
        good_models = {k: v for k, v in self.results.items() if v.get('r2', 0) > 0.7}
        if not good_models:
            good_models = self.results
        best_key = max(good_models.keys(), key=lambda k: good_models[k].get('cv_r2', 0))
        self.best_model = {
            'key': best_key,
            'model': self.results[best_key]['model'],
            'scaler': self.results[best_key]['scaler'],
            'features': self.results[best_key]['features'],
            'r2': self.results[best_key]['r2'],
            'rmse': self.results[best_key]['rmse']
        }
        print(f"Best model identified: {best_key} with R² = {self.results[best_key]['r2']:.4f} and RMSE = {self.results[best_key]['rmse']:.4f}")
    
    def _compare_all_models(self) -> None:
        if not self.results:
            return
        models = list(self.results.keys())
        r2_scores = [self.results[m]['r2'] for m in models]
        rmse_scores = [self.results[m]['rmse'] for m in models]
        mae_scores = [self.results[m]['mae'] for m in models]
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        ax1.bar(models, r2_scores)
        ax1.set_title('R² Scores (higher is better)')
        ax2.bar(models, rmse_scores)
        ax2.set_title('RMSE (lower is better)')
        ax3.bar(models, mae_scores)
        ax3.set_title('MAE (lower is better)')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'advanced_model_comparison.png'))
        plt.close()
        best_model = max(models, key=lambda m: self.results[m]['r2'])
        print(f"\nBest advanced model: {best_model}")
        print(f"  R²: {self.results[best_model]['r2']:.4f}")
        print(f"  RMSE: {self.results[best_model]['rmse']:.4f}")
    
    def final_summary(self) -> None:
        print("\n" + "="*80)
        print("Advanced Analysis Final Summary")
        print("="*80)
        if self.results:
            best_model = max(self.results.keys(), key=lambda m: self.results[m]['r2'])
            print(f"\nBest advanced model: {best_model}")
            print(f"  R²: {self.results[best_model]['r2']:.4f}")
            print(f"  RMSE: {self.results[best_model]['rmse']:.4f}")
            if 'cv_r2' in self.results[best_model]:
                print(f"  CV R²: {self.results[best_model]['cv_r2']:.4f}")
        if self.feature_importances is not None:
            print("\nTop 5 most important features:")
            for i, row in self.feature_importances.head(5).iterrows():
                print(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")
        print("\nRecommendations for further improvement:")
        print("  - Expand feature engineering with time-series models")
        print("  - Experiment with more advanced ensemble techniques")
        print("  - Consider additional normalization strategies")
        print(f"\nAll advanced analysis results saved to: {self.output_dir}")


def run_enhanced_analysis_pipeline(tabular_data_path: str, output_dir: str = "enhanced_output") -> EnhancedCognitiveLoadAnalysis:
    analysis = EnhancedCognitiveLoadAnalysis(output_dir)
    analysis.load_tabular_data(tabular_data_path)
    analysis.analyze_turbulence_relationship()
    analysis.apply_enhanced_feature_engineering()
    selected_features = analysis.perform_advanced_feature_selection()
    feature_sets = analysis.create_advanced_feature_sets(selected_features)
    analysis.train_advanced_models(feature_sets)
    analysis.final_summary()
    return analysis


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='SMU-TexCL Advanced Cognitive Load Analysis')
    parser.add_argument('--tabular_data_path', type=str, required=True,
                        help='Path to tabular data parquet file')
    parser.add_argument('--output_dir', type=str, default='advanced_cognitive_load_output',
                        help='Directory to save output files')
    args = parser.parse_args()
    analysis = run_enhanced_analysis_pipeline(args.tabular_data_path, args.output_dir)


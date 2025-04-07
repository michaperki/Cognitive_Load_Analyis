"""
SMU-TexCL Cognitive Load Prediction - Simplified Analysis

A streamlined approach to predicting cognitive load from physiological signals
based on lessons learned from previous experiments.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.model_selection import train_test_split, GroupKFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
import warnings

warnings.filterwarnings('ignore')

# Set plot styles
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class CognitiveLoadAnalysis:
    """Main class for analyzing and predicting cognitive load."""
    
    def __init__(self, output_dir: str = "output"):
        """Initialize the analysis pipeline.
        
        Args:
            output_dir: Directory for saving outputs
        """
        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"{output_dir}_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize data containers
        self.tabular_data = None
        self.windowed_data = None
        self.feature_importances = None
        self.results = {}
        
    def load_tabular_data(self, file_path: str) -> None:
        """Load tabular dataset from parquet file.
        
        Args:
            file_path: Path to the parquet file
        """
        self.tabular_data = pd.read_parquet(file_path)
        print(f"Loaded tabular data with {len(self.tabular_data)} rows and {len(self.tabular_data.columns)} columns")
        
        # Basic statistics about the dataset
        self._print_dataset_info()
    
    def _print_dataset_info(self) -> None:
        """Print basic information about the dataset."""
        print("\nDataset Overview:")
        print(f"Number of pilots: {self.tabular_data['pilot_id'].nunique()}")
        print(f"Number of trials: {len(self.tabular_data)}")
        print(f"Turbulence levels: {sorted(self.tabular_data['turbulence'].unique())}")
        
        # Get pilot categories
        if 'pilot_category' not in self.tabular_data.columns:
            self.tabular_data['pilot_category'] = self.tabular_data['pilot_id'].apply(self._categorize_pilot)
        
        print(f"Pilot categories: {self.tabular_data['pilot_category'].value_counts().to_dict()}")
        
        # Target variable statistics
        target_col = 'avg_tlx_quantile'
        if target_col in self.tabular_data.columns:
            print(f"\nTarget ({target_col}) statistics:")
            print(f"Mean: {self.tabular_data[target_col].mean():.4f}")
            print(f"Std: {self.tabular_data[target_col].std():.4f}")
            print(f"Min: {self.tabular_data[target_col].min():.4f}")
            print(f"Max: {self.tabular_data[target_col].max():.4f}")
    
    @staticmethod
    def _categorize_pilot(pilot_id: str) -> str:
        """Categorize pilot based on ID.
        
        Args:
            pilot_id: Pilot identifier
            
        Returns:
            Category ('air_force', 'commercial', or 'minimal_exp')
        """
        pilot_id_str = str(pilot_id)
        if pilot_id_str.startswith('8'):
            return 'minimal_exp'
        elif pilot_id_str.startswith('9'):
            return 'commercial'
        else:
            return 'air_force'
    
    def analyze_data_quality(self) -> Dict[str, Dict[str, float]]:
        """Analyze data quality and signal issues.
        
        Returns:
            Dictionary with quality metrics
        """
        print("\n[1] Analyzing data quality...")
        
        # Signal quality metrics
        quality_metrics = {}
        
        # Check for missing values
        missing_values = self.tabular_data.isnull().sum()
        has_missing = missing_values.sum() > 0
        print(f"Missing values: {missing_values.sum()}")
        if has_missing:
            print(f"Columns with missing values: {missing_values[missing_values > 0].to_dict()}")
        
        # Check for signal quality based on key metrics
        signal_cols = ['scr_mean', 'scr_std', 'scr_count', 'scr_min', 'scr_max',
                      'hr_mean', 'hr_std', 'sdrr', 'pnn50', 'temp_mean']
        
        quality_metrics['signal_stats'] = {}
        for col in signal_cols:
            if col in self.tabular_data.columns:
                quality_metrics['signal_stats'][col] = {
                    'mean': self.tabular_data[col].mean(),
                    'std': self.tabular_data[col].std(),
                    'min': self.tabular_data[col].min(),
                    'max': self.tabular_data[col].max(),
                    'missing': self.tabular_data[col].isnull().sum(),
                    'zeros': (self.tabular_data[col] == 0).sum()
                }
                
                # Print if signal has quality issues
                if quality_metrics['signal_stats'][col]['missing'] > 0 or quality_metrics['signal_stats'][col]['zeros'] > 50:
                    print(f"Signal quality issue in {col}: {quality_metrics['signal_stats'][col]['missing']} missing, "
                         f"{quality_metrics['signal_stats'][col]['zeros']} zeros")
        
        # Plot signal quality distributions
        self._plot_signal_distributions(signal_cols[:5])  # First 5 signal columns
        
        # Analyze quality by turbulence level
        if 'turbulence' in self.tabular_data.columns:
            self._analyze_by_turbulence()
            
        return quality_metrics
    
    def _plot_signal_distributions(self, cols: List[str]) -> None:
        """Plot distributions of signal features.
        
        Args:
            cols: List of columns to plot
        """
        fig, axes = plt.subplots(len(cols), 1, figsize=(12, 4 * len(cols)))
        if len(cols) == 1:
            axes = [axes]
            
        for i, col in enumerate(cols):
            if col in self.tabular_data.columns:
                sns.histplot(self.tabular_data[col], kde=True, ax=axes[i])
                axes[i].set_title(f'Distribution of {col}')
                
                # Add vertical line for mean
                mean_val = self.tabular_data[col].mean()
                axes[i].axvline(mean_val, color='red', linestyle='--', 
                              label=f'Mean: {mean_val:.2f}')
                axes[i].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'signal_distributions.png'))
        plt.close()
    
    def _analyze_by_turbulence(self) -> None:
        """Analyze signal quality by turbulence level."""
        # Create a figure to plot key metrics by turbulence
        key_metrics = ['scr_mean', 'scr_count', 'hr_mean', 'sdrr', 'pnn50']
        fig, axes = plt.subplots(len(key_metrics), 1, figsize=(12, 4 * len(key_metrics)))
        
        for i, metric in enumerate(key_metrics):
            if metric in self.tabular_data.columns:
                sns.boxplot(x='turbulence', y=metric, data=self.tabular_data, ax=axes[i])
                axes[i].set_title(f'{metric} by Turbulence Level')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'metrics_by_turbulence.png'))
        plt.close()
        
        # Print average TLX by turbulence level
        if 'avg_tlx_quantile' in self.tabular_data.columns:
            tlx_by_turbulence = self.tabular_data.groupby('turbulence')['avg_tlx_quantile'].mean()
            print("\nAverage TLX by turbulence level:")
            for turbulence, avg_tlx in tlx_by_turbulence.items():
                print(f"Turbulence {turbulence}: {avg_tlx:.4f}")
    
    def apply_per_pilot_normalization(self, min_samples: int = 3) -> None:
        """Normalize features on a per-pilot basis.
        
        Args:
            min_samples: Minimum number of samples required for normalization
        """
        print("\n[2] Applying per-pilot normalization...")
        
        # List of physiological features to normalize
        signal_features = [
            'scr_mean', 'scr_std', 'scr_count', 'hr_mean', 'hr_std',
            'sdrr', 'pnn50', 'temp_mean', 'temp_std'
        ]
        
        # Create normalized features
        normalized_features = []
        
        for col in signal_features:
            if col in self.tabular_data.columns:
                normed_col = f"{col}_pilot_norm"
                normalized_features.append(normed_col)
                
                # Group by pilot_id and apply z-score normalization
                self.tabular_data[normed_col] = self.tabular_data.groupby('pilot_id')[col].transform(
                    lambda x: (x - x.mean()) / x.std() if len(x) >= min_samples and x.std() > 0 else 0
                )
        
        print(f"Created {len(normalized_features)} pilot-normalized features")
        
        # Plot original vs normalized features for one example
        if len(normalized_features) > 0:
            example_feature = signal_features[0]
            example_norm = normalized_features[0]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Original feature by pilot
            sns.boxplot(x='pilot_id', y=example_feature, data=self.tabular_data, ax=ax1)
            ax1.set_title(f'Original {example_feature} by Pilot')
            ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)
            
            # Normalized feature by pilot
            sns.boxplot(x='pilot_id', y=example_norm, data=self.tabular_data, ax=ax2)
            ax2.set_title(f'Normalized {example_norm} by Pilot')
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'pilot_normalization.png'))
            plt.close()
    
    def identify_important_features(self) -> List[str]:
        """Identify important features for predicting cognitive load.
        
        Returns:
            List of selected features
        """
        print("\n[3] Identifying important features...")
        
        # For baseline, use all numeric columns except target and metadata
        numeric_cols = self.tabular_data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclude metadata and target columns
        exclude_cols = ['pilot_id', 'trial', 'turbulence', 'avg_tlx', 'avg_tlx_quantile', 
                        'avg_tlx_zscore', 'mental_effort', 'avg_mental_effort_zscore']
        
        base_features = [col for col in numeric_cols if col not in exclude_cols]
        
        # Get features and target
        X = self.tabular_data[base_features]
        y = self.tabular_data['avg_tlx_quantile']
        
        # Train a simple model to get feature importance
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Get feature importance
        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': base_features,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        top_n = min(20, len(importance_df))
        sns.barplot(y='feature', x='importance', data=importance_df.head(top_n))
        plt.title('Top Feature Importances')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'feature_importance.png'))
        plt.close()
        
        # Calculate permutation importance for validation
        perm_importance = permutation_importance(model, X, y, n_repeats=10, random_state=42)
        perm_importance_df = pd.DataFrame({
            'feature': base_features,
            'importance': perm_importance.importances_mean
        }).sort_values('importance', ascending=False)
        
        # Save feature importance for later use
        self.feature_importances = importance_df
        
        # Select top 50% of features by importance
        top_half = importance_df.head(len(importance_df) // 2)
        selected_features = top_half['feature'].tolist()
        
        # Print selected features
        print(f"Selected {len(selected_features)} important features out of {len(base_features)}")
        print(f"Top 5 features: {selected_features[:5]}")
        
        return selected_features
    
    def create_feature_sets(self) -> Dict[str, List[str]]:
        """Create different feature sets for modeling.
        
        Returns:
            Dictionary of feature sets
        """
        print("\n[4] Creating feature sets...")
        
        # Get baseline features from feature importance
        if self.feature_importances is None:
            baseline_features = self.identify_important_features()
        else:
            baseline_features = self.feature_importances['feature'].tolist()[:20]  # Top 20 features
        
        # Create feature sets
        feature_sets = {
            'baseline': baseline_features,
            'pilot_normalized': [col for col in self.tabular_data.columns if '_pilot_norm' in col],
            'combined': baseline_features + [col for col in self.tabular_data.columns if '_pilot_norm' in col]
        }
        
        # Add turbulence-specific sets for each turbulence level
        if 'turbulence' in self.tabular_data.columns:
            for turbulence in sorted(self.tabular_data['turbulence'].unique()):
                # For each turbulence level, create a subset
                turb_data = self.tabular_data[self.tabular_data['turbulence'] == turbulence]
                
                # Get target and features
                X_turb = turb_data[baseline_features]
                y_turb = turb_data['avg_tlx_quantile']
                
                # Train a model to get feature importance for this turbulence level
                model = RandomForestRegressor(n_estimators=50, random_state=42)
                model.fit(X_turb, y_turb)
                
                # Get feature importance
                importances = model.feature_importances_
                turb_importance = pd.DataFrame({
                    'feature': baseline_features,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                
                # Select top 10 features for this turbulence level
                top_n = min(10, len(turb_importance))
                feature_sets[f'turbulence_{turbulence}'] = turb_importance['feature'].head(top_n).tolist()
        
        # Print feature sets
        for name, features in feature_sets.items():
            print(f"Feature set '{name}': {len(features)} features")
        
        return feature_sets
    
    def train_and_evaluate_models(self, feature_sets: Dict[str, List[str]]) -> Dict[str, Dict[str, float]]:
        """Train and evaluate models with different feature sets.
        
        Args:
            feature_sets: Dictionary of feature sets
            
        Returns:
            Dictionary of model results
        """
        print("\n[5] Training and evaluating models...")
        
        # Get target variable
        y = self.tabular_data['avg_tlx_quantile']
        
        # Create a dictionary to store results
        self.results = {}
        
        # For each feature set
        for name, features in feature_sets.items():
            print(f"\nTraining model with feature set '{name}'...")
            
            # Get features
            X = self.tabular_data[features]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Create and train model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Store results
            self.results[name] = {
                'model': model,
                'features': features,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            }
            
            # Print results
            print(f"Results for '{name}':")
            print(f"  MSE: {mse:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE: {mae:.4f}")
            print(f"  R²: {r2:.4f}")
            
            # Plot predicted vs actual
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test, y_pred, alpha=0.5)
            plt.plot([0, 1], [0, 1], 'r--')
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            plt.title(f'Actual vs Predicted - {name}')
            plt.savefig(os.path.join(self.output_dir, f'actual_vs_pred_{name}.png'))
            plt.close()
            
            # Evaluate with pilot-wise cross-validation
            print(f"Performing pilot-wise CV for '{name}'...")
            self._pilot_wise_cv(X, y, name)
        
        # Compare all models
        self._compare_models()
        
        return self.results
    
    def _pilot_wise_cv(self, X: pd.DataFrame, y: pd.Series, name: str) -> None:
        """Perform pilot-wise cross-validation.
        
        Args:
            X: Features
            y: Target
            name: Name of feature set
        """
        # Get pilot IDs
        pilot_ids = self.tabular_data['pilot_id']
        
        # Create group k-fold cross-validator
        group_kfold = GroupKFold(n_splits=5)
        
        # Create model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Perform cross-validation
        cv_results = []
        
        for train_idx, test_idx in group_kfold.split(X, y, groups=pilot_ids):
            # Split data
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            
            cv_results.append({
                'r2': r2,
                'mse': mse
            })
        
        # Calculate average metrics
        avg_r2 = np.mean([r['r2'] for r in cv_results])
        avg_mse = np.mean([r['mse'] for r in cv_results])
        
        # Store results
        if name in self.results:
            self.results[name]['cv_r2'] = avg_r2
            self.results[name]['cv_mse'] = avg_mse
        
        # Print results
        print(f"  CV R²: {avg_r2:.4f}")
        print(f"  CV MSE: {avg_mse:.4f}")
    
    def _compare_models(self) -> None:
        """Compare all trained models."""
        if not self.results:
            return
        
        # Extract metrics for comparison
        models = list(self.results.keys())
        r2_scores = [self.results[m]['r2'] for m in models]
        rmse_scores = [self.results[m]['rmse'] for m in models]
        mae_scores = [self.results[m]['mae'] for m in models]
        
        # Create comparison plot
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # R2 scores (higher is better)
        ax1.bar(models, r2_scores)
        ax1.set_title('R² Scores (higher is better)')
        ax1.set_xticklabels(models, rotation=45, ha='right')
        
        # RMSE scores (lower is better)
        ax2.bar(models, rmse_scores)
        ax2.set_title('RMSE (lower is better)')
        ax2.set_xticklabels(models, rotation=45, ha='right')
        
        # MAE scores (lower is better)
        ax3.bar(models, mae_scores)
        ax3.set_title('MAE (lower is better)')
        ax3.set_xticklabels(models, rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'model_comparison.png'))
        plt.close()
        
        # Print best model
        best_model = max(models, key=lambda m: self.results[m]['r2'])
        print(f"\nBest model: {best_model}")
        print(f"  R²: {self.results[best_model]['r2']:.4f}")
        print(f"  RMSE: {self.results[best_model]['rmse']:.4f}")
        print(f"  MAE: {self.results[best_model]['mae']:.4f}")
    
    def train_turbulence_specific_models(self) -> Dict[str, Dict[str, Any]]:
        """Train separate models for each turbulence level.
        
        Returns:
            Dictionary of turbulence-specific models and results
        """
        print("\n[6] Training turbulence-specific models...")
        
        # Check if turbulence column exists
        if 'turbulence' not in self.tabular_data.columns:
            print("Turbulence column not found, skipping turbulence-specific models")
            return {}
        
        # Get baseline features
        if self.feature_importances is None:
            baseline_features = self.identify_important_features()
        else:
            baseline_features = self.feature_importances['feature'].tolist()[:20]  # Top 20 features
        
        # Get target
        y = self.tabular_data['avg_tlx_quantile']
        
        # Store results
        turbulence_models = {}
        
        # For each turbulence level
        for turbulence in sorted(self.tabular_data['turbulence'].unique()):
            print(f"\nTraining model for turbulence level {turbulence}...")
            
            # Filter data for this turbulence level
            mask = self.tabular_data['turbulence'] == turbulence
            X_turb = self.tabular_data.loc[mask, baseline_features]
            y_turb = y[mask]
            
            # Skip if not enough data
            if len(X_turb) < 20:
                print(f"Not enough data for turbulence level {turbulence} (only {len(X_turb)} samples), skipping")
                continue
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_turb, y_turb, test_size=0.2, random_state=42
            )
            
            # Train model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Store results
            turbulence_models[f'turbulence_{turbulence}'] = {
                'model': model,
                'features': baseline_features,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'samples': len(X_turb)
            }
            
            # Print results
            print(f"Results for turbulence level {turbulence} (n={len(X_turb)}):")
            print(f"  MSE: {mse:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE: {mae:.4f}")
            print(f"  R²: {r2:.4f}")
            
            # Plot predicted vs actual
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test, y_pred, alpha=0.5)
            plt.plot([0, 1], [0, 1], 'r--')
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            plt.title(f'Actual vs Predicted - Turbulence {turbulence}')
            plt.savefig(os.path.join(self.output_dir, f'actual_vs_pred_turb_{turbulence}.png'))
            plt.close()
        
        # Compare turbulence models
        self._compare_turbulence_models(turbulence_models)
        
        return turbulence_models
    
    def _compare_turbulence_models(self, turbulence_models: Dict[str, Dict[str, Any]]) -> None:
        """Compare turbulence-specific models.
        
        Args:
            turbulence_models: Dictionary of turbulence-specific models and results
        """
        if not turbulence_models:
            return
        
        # Extract metrics for comparison
        models = list(turbulence_models.keys())
        r2_scores = [turbulence_models[m]['r2'] for m in models]
        rmse_scores = [turbulence_models[m]['rmse'] for m in models]
        samples = [turbulence_models[m]['samples'] for m in models]
        
        # Create comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # R2 scores (higher is better)
        bars = ax1.bar(models, r2_scores)
        ax1.set_title('R² Scores by Turbulence Level')
        ax1.set_xticklabels(models, rotation=45, ha='right')
        
        # Add sample size
        for i, bar in enumerate(bars):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f'n={samples[i]}',
                ha='center'
            )
        
        # RMSE scores (lower is better)
        bars = ax2.bar(models, rmse_scores)
        ax2.set_title('RMSE by Turbulence Level')
        ax2.set_xticklabels(models, rotation=45, ha='right')
        
        # Add sample size
        for i, bar in enumerate(bars):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f'n={samples[i]}',
                ha='center'
            )
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'turbulence_model_comparison.png'))
        plt.close()
        
        # Print best turbulence model
        best_model = max(models, key=lambda m: turbulence_models[m]['r2'])
        print(f"\nBest turbulence-specific model: {best_model}")
        print(f"  R²: {turbulence_models[best_model]['r2']:.4f}")
        print(f"  RMSE: {turbulence_models[best_model]['rmse']:.4f}")
        print(f"  Samples: {turbulence_models[best_model]['samples']}")
    
    def train_ensemble_model(self, turbulence_models: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Train an ensemble model that uses turbulence-specific models.
        
        Args:
            turbulence_models: Dictionary of turbulence-specific models
            
        Returns:
            Dictionary of ensemble model results
        """
        print("\n[7] Training ensemble model...")
        
        # Check if we have turbulence models
        if not turbulence_models:
            print("No turbulence models available, skipping ensemble")
            return {}
        
        # Get baseline features
        if self.feature_importances is None:
            baseline_features = self.identify_important_features()
        else:
            baseline_features = self.feature_importances['feature'].tolist()[:20]  # Top 20 features
        
        # Get target
        y = self.tabular_data['avg_tlx_quantile']
        
        # Split data for testing the ensemble
        X_main = self.tabular_data[baseline_features + ['turbulence']]
        X_train, X_test, y_train, y_test = train_test_split(
            X_main, y, test_size=0.2, random_state=42
        )
        
        # Make predictions using appropriate turbulence-specific model
        y_pred = []
        
        for i, row in X_test.iterrows():
            # Get turbulence level
            turbulence = row['turbulence']
            
            # Get features for prediction
            X_pred = row[baseline_features].values.reshape(1, -1)
            
            # Get appropriate model
            model_key = f'turbulence_{turbulence}'
            
            if model_key in turbulence_models:
                # Use turbulence-specific model
                model = turbulence_models[model_key]['model']
                pred = model.predict(X_pred)[0]
            else:
                # Use fallback generic model
                fallback_model = self.results['baseline']['model']
                pred = fallback_model.predict(X_pred)[0]
            
            y_pred.append(pred)
        
        # Calculate metrics
        y_pred = np.array(y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Store results
        ensemble_results = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        # Print results
        print(f"Ensemble model results:")
        print(f"  MSE: {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²: {r2:.4f}")
        
        # Plot predicted vs actual
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs Predicted - Turbulence Ensemble')
        plt.savefig(os.path.join(self.output_dir, 'actual_vs_pred_ensemble.png'))
        plt.close()
        
        # Compare with baseline
        if 'baseline' in self.results:
            print(f"\nComparison with baseline model:")
            print(f"  Ensemble R²: {r2:.4f} vs Baseline R²: {self.results['baseline']['r2']:.4f}")
            print(f"  Ensemble RMSE: {rmse:.4f} vs Baseline RMSE: {self.results['baseline']['rmse']:.4f}")
        
        return ensemble_results
    
    def analyze_pilot_categories(self) -> None:
        """Analyze model performance by pilot category."""
        print("\n[8] Analyzing performance by pilot category...")
        
        # Check if we have results and pilot categories
        if not self.results or 'pilot_category' not in self.tabular_data.columns:
            print("Missing results or pilot categories, skipping analysis")
            return
        
        # Get baseline model
        if 'baseline' not in self.results:
            print("Baseline model not found, skipping analysis")
            return
        
        baseline_model = self.results['baseline']['model']
        baseline_features = self.results['baseline']['features']
        
        # Get pilot categories
        categories = self.tabular_data['pilot_category'].unique()
        
        # Store results by category
        category_results = {}
        
        # For each category
        for category in categories:
            print(f"\nAnalyzing performance for {category} pilots...")
            
            # Filter data for this category
            mask = self.tabular_data['pilot_category'] == category
            X_cat = self.tabular_data.loc[mask, baseline_features]
            y_cat = self.tabular_data.loc[mask, 'avg_tlx_quantile']
            
            # Skip if not enough data
            if len(X_cat) < 10:
                print(f"Not enough data for {category} (only {len(X_cat)} samples), skipping")
                continue
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_cat, y_cat, test_size=0.2, random_state=42
            )
            
            # Train a specific model for this category
            category_model = RandomForestRegressor(n_estimators=100, random_state=42)
            category_model.fit(X_train, y_train)
            
            # Make predictions with category-specific model
            y_pred_cat = category_model.predict(X_test)
            
            # Also make predictions with baseline model
            y_pred_baseline = baseline_model.predict(X_test)
            
            # Calculate metrics for both
            cat_mse = mean_squared_error(y_test, y_pred_cat)
            cat_rmse = np.sqrt(cat_mse)
            cat_r2 = r2_score(y_test, y_pred_cat)
            
            baseline_mse = mean_squared_error(y_test, y_pred_baseline)
            baseline_rmse = np.sqrt(baseline_mse)
            baseline_r2 = r2_score(y_test, y_pred_baseline)
            
            # Store results
            category_results[category] = {
                'specific_model': {
                    'mse': cat_mse,
                    'rmse': cat_rmse,
                    'r2': cat_r2
                },
                'baseline_model': {
                    'mse': baseline_mse,
                    'rmse': baseline_rmse,
                    'r2': baseline_r2
                },
                'samples': len(X_cat)
            }
            
            # Print results
            print(f"Results for {category} pilots (n={len(X_cat)}):")
            print(f"  Category-specific model:")
            print(f"    RMSE: {cat_rmse:.4f}")
            print(f"    R²: {cat_r2:.4f}")
            print(f"  Baseline model:")
            print(f"    RMSE: {baseline_rmse:.4f}")
            print(f"    R²: {baseline_r2:.4f}")
        
        # Create comparison plot
        if category_results:
            categories = list(category_results.keys())
            specific_r2 = [category_results[c]['specific_model']['r2'] for c in categories]
            baseline_r2 = [category_results[c]['baseline_model']['r2'] for c in categories]
            samples = [category_results[c]['samples'] for c in categories]
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Set width of bars
            bar_width = 0.35
            
            # Set position of bars on x axis
            r1 = np.arange(len(categories))
            r2 = [x + bar_width for x in r1]
            
            # Create bars
            bars1 = ax.bar(r1, specific_r2, width=bar_width, label='Category-specific Model')
            bars2 = ax.bar(r2, baseline_r2, width=bar_width, label='Baseline Model')
            
            # Add sample size
            for i, bar in enumerate(bars1):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f'n={samples[i]}',
                    ha='center'
                )
            
            # Add labels and title
            ax.set_xlabel('Pilot Category')
            ax.set_ylabel('R² Score')
            ax.set_title('Model Performance by Pilot Category')
            ax.set_xticks([r + bar_width/2 for r in range(len(categories))])
            ax.set_xticklabels(categories)
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'category_comparison.png'))
            plt.close()
    
    def final_summary(self) -> None:
        """Print final summary of all models and analyses."""
        print("\n" + "="*80)
        print("Final Summary")
        print("="*80)
        
        # Print best model from standard analysis
        if self.results:
            best_model = max(self.results.keys(), key=lambda m: self.results[m]['r2'])
            print(f"\nBest standard model: {best_model}")
            print(f"  R²: {self.results[best_model]['r2']:.4f}")
            print(f"  RMSE: {self.results[best_model]['rmse']:.4f}")
            
            # If we have cross-validation results
            if 'cv_r2' in self.results[best_model]:
                print(f"  CV R²: {self.results[best_model]['cv_r2']:.4f}")
        
        # Print best features
        if self.feature_importances is not None:
            print("\nTop 5 most important features:")
            for i, (feature, importance) in enumerate(zip(
                self.feature_importances['feature'].head(5),
                self.feature_importances['importance'].head(5)
            )):
                print(f"  {i+1}. {feature}: {importance:.4f}")
        
        # Print recommendations
        print("\nRecommendations for further improvement:")
        print("  1. Consider expanding pilot-normalized features")
        print("  2. Try additional model types (e.g., gradient boosting, neural networks)")
        print("  3. Explore additional feature engineering based on physiological signals")
        print("  4. Investigate time-series aspects of the data with sequence models")
        print("  5. Consider collecting additional data for underrepresented conditions")
        
        print(f"\nAll analysis results saved to: {self.output_dir}")


def run_analysis_pipeline(tabular_data_path: str, output_dir: str = "output") -> CognitiveLoadAnalysis:
    """Run the complete analysis pipeline.
    
    Args:
        tabular_data_path: Path to tabular data
        output_dir: Directory for saving outputs
        
    Returns:
        Analysis instance with results
    """
    # Create analysis instance
    analysis = CognitiveLoadAnalysis(output_dir)
    
    # Load data
    analysis.load_tabular_data(tabular_data_path)
    
    # Analyze data quality
    analysis.analyze_data_quality()
    
    # Apply per-pilot normalization
    analysis.apply_per_pilot_normalization()
    
    # Identify important features
    analysis.identify_important_features()
    
    # Create feature sets
    feature_sets = analysis.create_feature_sets()
    
    # Train and evaluate models
    analysis.train_and_evaluate_models(feature_sets)
    
    # Train turbulence-specific models
    turbulence_models = analysis.train_turbulence_specific_models()
    
    # Train ensemble model
    if turbulence_models:
        analysis.train_ensemble_model(turbulence_models)
    
    # Analyze by pilot category
    analysis.analyze_pilot_categories()
    
    # Print final summary
    analysis.final_summary()
    
    return analysis


if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='SMU-TexCL Cognitive Load Analysis')
    parser.add_argument('--tabular_data_path', type=str, required=True,
                        help='Path to tabular data parquet file')
    parser.add_argument('--output_dir', type=str, default='cognitive_load_output',
                        help='Directory to save output files')
    
    args = parser.parse_args()
    
    # Run analysis
    analysis = run_analysis_pipeline(args.tabular_data_path, args.output_dir)

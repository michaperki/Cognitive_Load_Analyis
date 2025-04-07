# SMU-TexCL Cognitive Load Analysis Tool

## Overview

This tool provides a streamlined approach to analyzing and predicting cognitive load from physiological signals in the SMU-Textron Cognitive Load (TexCL) dataset. Built on lessons learned from previous modeling attempts, it offers a modular, step-by-step analysis pipeline with comprehensive visualizations.

## Features

- **Data Quality Analysis**: Examines signal quality issues and visualizes key metrics
- **Pilot-Specific Normalization**: Normalizes physiological signals on a per-pilot basis
- **Feature Importance Analysis**: Identifies the most predictive features for cognitive load
- **Multiple Modeling Approaches**:
  - Baseline models with different feature sets
  - Turbulence-specific models
  - Ensemble models that combine turbulence-specific predictors
  - Pilot category-specific analysis
- **Cross-Validation**: Uses pilot-wise cross-validation to ensure generalization to new pilots
- **Visualizations**: Generates comprehensive visualizations for all analyses

## Usage

```
python main.py --tabular_data_path [PATH_TO_DATA] --output_dir [OUTPUT_DIRECTORY]
```

### Arguments:

- `--tabular_data_path`: Path to the parquet file containing tabular data (required)
- `--output_dir`: Directory to save output files (default: 'cognitive_load_output')

## Output

The tool generates a timestamped output directory containing:

- Visualization plots for each analysis step
- Performance metrics for all model types
- Feature importance rankings
- Comparisons between different modeling approaches

## Key Results

In our analysis, we discovered:

1. The `turbulence_check` feature dominates prediction with over 82% importance
2. Our best model achieved an R² of 0.8473 (test) and 0.7847 (cross-validation)
3. Models trained on specific turbulence levels performed poorly compared to general models
4. Pilot experience level affects prediction accuracy, with commercial pilots showing the best results
5. Pilot-normalized features alone provided limited predictive power

## Structure

- `cognitive_load_analysis.py`: Main analysis class with all functionality
- `main.py`: Simple script to run the analysis pipeline

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Future Work

Based on our analysis, we recommend:

1. Investigating the interaction between turbulence levels and physiological responses
2. Exploring more advanced models like gradient boosting or neural networks
3. Incorporating temporal aspects of the physiological signals
4. Addressing class imbalance for pilots with minimal experience
5. Expanding the feature engineering for physiological signals

## License

This project is for research purposes only. Please respect the privacy of the pilot data.

## Acknowledgements

This analysis builds on the SMU-Textron Cognitive Load dataset and aims to improve prediction accuracy through simplified, targeted modeling approaches.

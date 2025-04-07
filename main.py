"""
Main script to run the SMU-TexCL Enhanced Cognitive Load Analysis.
"""

from enhanced_cognitive_load_analysis import run_enhanced_analysis_pipeline

# Path to your tabular data
tabular_data_path = '../pre_proccessed_table_data.parquet'

# Run the enhanced analysis
analysis = run_enhanced_analysis_pipeline(
    tabular_data_path=tabular_data_path,
    output_dir='enhanced_cognitive_load_output'
)

# The analysis results are available in the analysis object
# and have been saved to the output directory
print(f"Enhanced analysis complete! Results saved to {analysis.output_dir}")

print("\nKey Improvements over Original Analysis:")
print("1. Advanced feature engineering with polynomial, derivative, and variance features")
print("2. Multi-method feature selection combining Random Forest, Gradient Boosting, and correlation analysis")
print("3. Evaluation of multiple model types beyond just Random Forest")
print("4. Turbulence-aware ensemble model combining specialized predictors")
print("5. Detailed analysis of experience effects on physiological signals")
print("6. More comprehensive visualization and reporting of results")

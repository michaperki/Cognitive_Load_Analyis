"""
Simple script to run the SMU-TexCL Cognitive Load Analysis.
"""

from cognitive_load_analysis import run_analysis_pipeline

# Path to your tabular data
tabular_data_path = '../pre_proccessed_table_data.parquet'

# Run the analysis
analysis = run_analysis_pipeline(
    tabular_data_path=tabular_data_path,
    output_dir='cognitive_load_output'
)

# The analysis results are available in the analysis object
# and have been saved to the output directory
print(f"Analysis complete! Results saved to {analysis.output_dir}")

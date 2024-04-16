# ThaiGov_BudgetAnalysis
Data scraping based on data from https://github.com/kaogeek/thailand-budget-pdf2csv
Budget Analysis Tool
# Description
This Python package is designed to process, analyze, and visualize budget data. It contains two main components: BudgetCal and BudgetSca. BudgetCal handles the reading, merging, and basic processing of budget CSV files. BudgetSca extends this functionality to more complex data transformations and threading for performance optimization.
# Usage
To run the program, ensure you have the required configuration in config.json. This file should specify directory paths, file patterns, and other necessary parameters for processing.

# BudgetCal
Handles the initial loading, merging, and simple processing of budget data.

Functions:
read_csv_files(dir_path): Reads CSV files from a directory.
merge_dataframes(file_generator, pattern): Merges dataframes based on a file pattern.
prepare_data(df, group_by): Prepares data for analysis by converting data types and filtering.
plot_data(df, title, labels, direction, n_entries): Plots data using Plotly.
# BudgetSca
Provides advanced data processing capabilities using threading and text analysis.

Functions:
clean_bu(bu): Cleans and normalizes budget unit names.
tokenize_words(text): Tokenizes text, removing stopwords.
process_field(field): Processes text fields and extracts common words.
project_scrap_get(df): Processes the dataframe to apply text analysis on multiple columns concurrently.
df_transform(df, groupby): Transforms the dataframe based on the extracted data to provide a grouped and summarized view.
# Output
The scripts will generate CSV files with the processed data, which can then be used for further 

# Contributing
Contributions to improve the tool or extend its capabilities are welcome. Please fork the repository, make your changes, and submit a pull request.

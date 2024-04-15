import pandas as pd
import re
import os
import time
import logging
from pythainlp import word_tokenize
from pythainlp.corpus import thai_stopwords
import json
import unicodedata
from collections import Counter
from budget_amount import BudgetCal  # Ensure BudgetCal class is defined in budget_amount.py
from progress.bar import Bar
from concurrent.futures import ThreadPoolExecutor
import threading
import ast

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BudgetSca:
    def __init__(self, config):
        self.config = config
        logging.info("Configuration loaded successfully.")

    def clean_bu(self, bu):
        """Normalize and clean budget unit text."""
        if pd.isna(bu):
            return None
        elif isinstance(bu, str):
            bu = unicodedata.normalize('NFKC', bu)
            bu = re.sub(r'\s+', ' ', bu).strip()
        return bu

    def tokenize_words(self, text):
        """Tokenize text, removing whitespace and stopwords, keeping significant words only."""
        words = word_tokenize(text, keep_whitespace=False)
        result = [word for word in words if word not in thai_stopwords() and len(word) > 1]
        return result

    def process_field(self, field):
        """Process each field: split, clean, tokenize, and retrieve common words."""
        if pd.isna(field) or field.strip() == "":
            return ""
        entries = re.split(r',', field)
        cleaned_entries = filter(None, (self.clean_bu(entry.strip()) for entry in entries))
        all_words = sum((self.tokenize_words(entry) for entry in cleaned_entries), [])
        result = ', '.join(word for word, _ in Counter(all_words).most_common(5))
        return result

    def parse_word_counts(self, word_counts):
        """Parse a string of word counts into a dictionary."""
        if pd.isna(word_counts) or not word_counts.strip():
            return 
        word = self.process_field(word_counts)
        sub_word_list = re.split(r',', word)
        return sub_word_list

    def apply_parse_word_counts(self, df, column_name, bar):
        """Applies parse_word_counts method to a specific column and updates the progress bar."""
        logging.info(f"{threading.current_thread().name} starting processing on {column_name}")
        start_time = time.time()
        result = df[column_name].apply(self.parse_word_counts)
        bar.next()  # Assuming 'Bar' is thread-safe; if not, replace with a thread-safe variant
        logging.info(f"{threading.current_thread().name} has finished processing {column_name} in {time.time() - start_time:.2f} seconds")
        return result, column_name

    def threaded_word_counts(self, df, columns):
        """Uses threading to apply word count parsing to multiple dataframe columns with progress monitoring."""
        bar = Bar('Processing Columns', max=len(columns))  # Initialize progress bar
        max_workers = os.cpu_count()  # Dynamically set the number of threads
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.apply_parse_word_counts, df, col, bar): col for col in columns}
            for future in futures:
                result, col_name = future.result()
                df[col_name] = result

        bar.finish()  # Clean up after the bar
        return df
    
    def project_scrap_get(self, df):
        start_time = time.time()
        df['BUDGETARY_UNIT'] = df['BUDGETARY_UNIT'].apply(self.clean_bu)
        df['BUDGET_YEAR'] = df['BUDGET_YEAR'].astype(int)
        columns = ['PROJECT', 'OUTPUT', 'ITEM_DESCRIPTION', 'CATEGORY_LV1', 'CATEGORY_LV2', 'CATEGORY_LV3', 'CATEGORY_LV4', 'CATEGORY_LV5', 'CATEGORY_LV6']
        df = self.threaded_word_counts(df, columns)
        df.fillna(int(0), inplace=True)
        # Drop rows where all specified columns are empty
        grouped = df

        # grouped.dropna(subset=columns, how='all', inplace=True)
        logging.info(f"project_scrap_get executed in {time.time() - start_time:.2f} seconds")
        return grouped

    def df_transform(self, df, groupby):
        def safe_literal_eval(s):
            """ Safely evaluate string literals to Python objects. """
            if isinstance(s, str) and s:  # Checks if s is a non-empty string
                try:
                    return ast.literal_eval(s)
                except (ValueError, SyntaxError) as e:
                    print(f"Failed to eval: {s} with error: {e}")
            return s  # Return original string if it's not evaluable or empty

        columns = ['PROJECT', 'OUTPUT', 'ITEM_DESCRIPTION', 'CATEGORY_LV1', 'CATEGORY_LV2', 'CATEGORY_LV3', 'CATEGORY_LV4', 'CATEGORY_LV5', 'CATEGORY_LV6']
        grouped_result = pd.DataFrame()  # DataFrame to hold all grouped results

        for col in columns:
            # Apply safe_literal_eval to the column
            df[col] = df[col].apply(safe_literal_eval)
            # Explode the column
            df_exploded = df.explode(col)
            # Exclude '0' values and non-existent ones
            df_exploded = df_exploded[df_exploded[col].notna() & (df_exploded[col] != '0')]

            if df_exploded.empty:
                continue  # Skip to the next column if no valid data left after filtering

            # Mark the source of each word and rename the exploded column to 'Word'
            df_exploded['Source_Column'] = col
            df_exploded.rename(columns={col: 'Word'}, inplace=True)

            # Group by specified columns including 'Word' and 'Source_Column' and calculate the frequency
            grouped = df_exploded.groupby(['Word', groupby, 'BUDGET_YEAR', 'Source_Column']).size().reset_index(name='frequency')

            # Concatenate current grouped results with previous ones
            grouped_result = pd.concat([grouped_result, grouped], axis=0, ignore_index=True)
        return grouped_result

if __name__ == '__main__':
    try:
        with open('config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
        bg_cal = BudgetCal(config)
        bg_sca = BudgetSca(config)
        file_generator = bg_cal.read_csv_files(config['dir_path'])
        file_pattern = config.get("file_pattern", r"(\d+)_Budget_red_stripe.csv")
        group_by = config.get("group_by")
        merged_df = bg_cal.merge_dataframes(file_generator, file_pattern)
        merged_df.to_csv('unit_grouped_unit.csv', index=False)
        if merged_df is not None:
            scrap_prepared_df = bg_sca.project_scrap_get(merged_df)
            scrap_prepared_df.to_csv('scrap_prepared_df.csv', index=False)
            # scrap_prepared_df = pd.read_csv('scrap_prepared_df.csv', encoding='utf-8')
            transformed_df = bg_sca.df_transform(scrap_prepared_df, group_by)
            transformed_df.to_csv('transformed_df.csv', index=False)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
import pandas as pd
import re
import os
from typing import Dict


def clean_string(value: str) -> str:
    """Cleans PDF extraction artifacts like '_x000D_' and excess spaces."""
    if isinstance(value, str):
        value = re.sub(r"_x000D_", " ", value)
        value = re.sub(r"\s+", " ", value)
        return value.strip()
    return value


def load_and_clean_excel_table(filepath: str) -> pd.DataFrame:
    """
    Loads an Excel file and cleans all cells + column names.
    Uses df.map() to avoid FutureWarning.
    """
    df = pd.read_excel(filepath)

    # Clean column names
    df.columns = [clean_string(col) for col in df.columns]

    # Clean all cell values using map
    for col in df.columns:
        df[col] = df[col].map(clean_string)

    return df


def load_clean_and_save_tables(folder_path: str,
                               output_csv_folder: str,
                               output_xlsx_folder: str) -> Dict[str, pd.DataFrame]:
    """
    Loads all .xlsx tables from folder, cleans them, and saves cleaned results.
    Returns dict of {filename: dataframe}
    """

    os.makedirs(output_csv_folder, exist_ok=True)
    os.makedirs(output_xlsx_folder, exist_ok=True)

    table_dict = {}

    for file in os.listdir(folder_path):
        if file.lower().endswith(".xlsx"):
            full_path = os.path.join(folder_path, file)
            key = os.path.splitext(file)[0]

            try:
                df = load_and_clean_excel_table(full_path)
                table_dict[key] = df

                print(f"Loaded & cleaned: {file}, shape={df.shape}")

                # Save as CSV
                csv_out = os.path.join(output_csv_folder, f"{key}.csv")
                df.to_csv(csv_out, index=False)

                # Save as XLSX
                xlsx_out = os.path.join(output_xlsx_folder, f"{key}.xlsx")
                df.to_excel(xlsx_out, index=False)

                print(f"  → Saved cleaned CSV:  {csv_out}")
                print(f"  → Saved cleaned XLSX: {xlsx_out}")

            except Exception as e:
                print(f"❌ Error loading {file}: {e}")

    return table_dict


# ------------------------
# Example Usage
# ------------------------
if __name__ == "__main__":
    input_folder   = "extract2025-11-29T05-45-36\\tables"
    output_csv     = "data/output/cleaned_tables_csv"
    output_xlsx    = "data/output/cleaned_tables_xlsx"

    tables = load_clean_and_save_tables(
        input_folder,
        output_csv,
        output_xlsx
    )

    # Preview first cleaned table
    if tables:
        first = next(iter(tables.keys()))
        print(f"\nPreview of cleaned table '{first}':\n")
        print(tables[first].head())


    # Show first table loaded
    if tables:
        first_key = next(iter(tables.keys()))
        print(f"\nPreview of '{first_key}':")
        print(tables[first_key].head())


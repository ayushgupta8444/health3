import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


class DataProcessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()

        self.categorical_columns = [
            'Symptoms',
            'Diseases',
            'Gender',
            'Medicines',
            'Treatment details',
            'Comorbidities',
            'Adverse events',
            'Dosha',
            'Visit type'
        ]

        self.numerical_columns = [
            'Patient age',
            'ODI scores'
        ]

        self.target_categorical = [
            'Recovery Category',
            'Severity level',
            'Treatment effectiveness'
        ]

        self.target_numerical = [
            'Treatment days',
            'Average Recovery Days',
            'ODI Improvement'
        ]

    def preprocess_data(self, df, training=True):

        df_processed = df.copy()

        # Clean column names
        df_processed.columns = df_processed.columns.str.strip()

        print("========== COLUMNS ==========")
        print(df_processed.columns.tolist())

        print("========== DTYPES BEFORE ==========")
        print(df_processed.dtypes)

        # Expected numeric columns
        numeric_cols = self.numerical_columns + self.target_numerical

        # Convert numeric columns safely
        for col in numeric_cols:

            if col in df_processed.columns:

                df_processed[col] = (
                    df_processed[col]
                    .astype(str)
                    .str.strip()
                    .replace(
                        ['', 'NA', 'N/A', 'None', '-', '--', 'null'],
                        np.nan
                    )
                )

                df_processed[col] = pd.to_numeric(
                    df_processed[col],
                    errors='coerce'
                )

        print("========== DTYPES AFTER CONVERSION ==========")
        print(df_processed.dtypes)

        # Handle missing values safely
        for col in df_processed.columns:

            print(f"\nProcessing Column: {col}")
            print(f"Dtype: {df_processed[col].dtype}")

            # Numeric columns
            if pd.api.types.is_numeric_dtype(df_processed[col]):

                median_val = df_processed[col].median()

                if pd.isna(median_val):
                    median_val = 0

                df_processed[col] = df_processed[col].fillna(median_val)

                print(f"Filled numeric column '{col}' with median: {median_val}")

            # Categorical columns
            else:

                df_processed[col] = (
                    df_processed[col]
                    .astype(str)
                    .replace(['nan', 'None', ''], np.nan)
                )

                mode_val = df_processed[col].mode()

                fill_val = (
                    mode_val.iloc[0]
                    if not mode_val.empty
                    else "Unknown"
                )

                df_processed[col] = df_processed[col].fillna(fill_val)

                print(f"Filled categorical column '{col}' with mode: {fill_val}")

        # Label Encoding
        cols_to_encode = (
            self.categorical_columns +
            self.target_categorical
        )

        if training:

            for col in cols_to_encode:

                if col in df_processed.columns:

                    le = LabelEncoder()

                    df_processed[col] = le.fit_transform(
                        df_processed[col].astype(str)
                    )

                    self.label_encoders[col] = le

        else:

            for col in cols_to_encode:

                if (
                    col in df_processed.columns
                    and col in self.label_encoders
                ):

                    le = self.label_encoders[col]

                    known_classes = set(le.classes_)

                    df_processed[col] = (
                        df_processed[col]
                        .astype(str)
                        .apply(
                            lambda x:
                            x if x in known_classes
                            else list(known_classes)[0]
                        )
                    )

                    df_processed[col] = le.transform(
                        df_processed[col]
                    )

        # Feature Scaling
        available_num_cols = [
            col for col in self.numerical_columns
            if col in df_processed.columns
        ]

        if available_num_cols:

            if training:

                df_processed[available_num_cols] = (
                    self.scaler.fit_transform(
                        df_processed[available_num_cols]
                    )
                )

            else:

                df_processed[available_num_cols] = (
                    self.scaler.transform(
                        df_processed[available_num_cols]
                    )
                )

        print("========== PREPROCESSING COMPLETED ==========")

        return df_processed

    def get_feature_names(self):

        return [
            'Patient age',
            'Gender',
            'Symptoms',
            'Diseases',
            'Medicines',
            'ODI scores',
            'Treatment details',
            'Comorbidities',
            'Adverse events'
        ]
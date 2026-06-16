import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

class DataProcessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
        # Consistent categorical and numerical features
        self.categorical_columns = [
            'Symptoms', 'Diseases', 'Gender', 'Medicines', 
            'Treatment details', 'Comorbidities', 'Adverse events', 'Dosha', 'Visit type'
        ]
        self.numerical_columns = ['Patient age', 'ODI scores']
        
        self.target_categorical = ['Recovery Category', 'Severity level', 'Treatment effectiveness']
        self.target_numerical = ['Treatment days', 'Average Recovery Days', 'ODI Improvement']
        
    def preprocess_data(self, df, training=True):
        df_processed = df.copy()
        
        # Clean column names
        df_processed.columns = df_processed.columns.str.strip()
        
        # Impute missing values
        for col in df_processed.columns:
            if df_processed[col].dtype == 'object':
                mode_val = df_processed[col].mode()
                fill_val = mode_val[0] if not mode_val.empty else 'Unknown'
                df_processed[col] = df_processed[col].fillna(fill_val)
            else:
                median_val = df_processed[col].median()
                fill_val = median_val if not pd.isna(median_val) else 0
                df_processed[col] = df_processed[col].fillna(fill_val)
                
        # Label Encoding
        cols_to_encode = self.categorical_columns + self.target_categorical
        if training:
            for col in cols_to_encode:
                if col in df_processed.columns:
                    le = LabelEncoder()
                    df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                    self.label_encoders[col] = le
        else:
            for col in cols_to_encode:
                if col in df_processed.columns and col in self.label_encoders:
                    le = self.label_encoders[col]
                    known_classes = list(le.classes_)
                    # Safely handle unknown categories by assigning to the first known class
                    df_processed[col] = df_processed[col].apply(lambda x: x if x in known_classes else known_classes[0])
                    df_processed[col] = le.transform(df_processed[col].astype(str))
                    
        # Feature Scaling
        if training:
            num_cols = [c for c in self.numerical_columns if c in df_processed.columns]
            if num_cols:
                df_processed[num_cols] = self.scaler.fit_transform(df_processed[num_cols])
        else:
            num_cols = [c for c in self.numerical_columns if c in df_processed.columns]
            if num_cols:
                df_processed[num_cols] = self.scaler.transform(df_processed[num_cols])
                
        return df_processed
    
    def get_feature_names(self):
        return ['Patient age', 'Gender', 'Symptoms', 'Diseases', 'Medicines', 'ODI scores', 'Treatment details', 'Comorbidities', 'Adverse events']

import pandas as pd
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.svm import SVR
from sklearn.cluster import DBSCAN

class AdvancedHealthcareModels:
    def __init__(self):
        # Support Vector Regression for continuous variables
        self.svr_recovery_days = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
        self.svr_odi_improvement = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
        
        # XGBoost for classification tasks
        self.xgb_severity = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss')
        self.xgb_effectiveness = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss')
        self.xgb_recovery_category = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss')
        
        # DBSCAN for advanced clustering
        self.dbscan_model = DBSCAN(eps=0.5, min_samples=5)
        
        self.is_trained = False
        
    def train_models(self, X_train, df_processed):
        # Determine targets safely (fallback to generic continuous columns if needed)
        recovery_days_target = 'Average Recovery Days' if 'Average Recovery Days' in df_processed.columns else ('Treatment days' if 'Treatment days' in df_processed.columns else None)
        odi_target = 'ODI Improvement' if 'ODI Improvement' in df_processed.columns else None
        
        required_targets = ['Severity level', 'Treatment effectiveness', 'Recovery Category']
        
        if all(target in df_processed.columns for target in required_targets) and recovery_days_target:
            # Train Regressors
            self.svr_recovery_days.fit(X_train, df_processed[recovery_days_target])
            if odi_target:
                self.svr_odi_improvement.fit(X_train, df_processed[odi_target])
            
            # Train XGBoost Classifiers
            self.xgb_severity.fit(X_train, df_processed['Severity level'])
            self.xgb_effectiveness.fit(X_train, df_processed['Treatment effectiveness'])
            self.xgb_recovery_category.fit(X_train, df_processed['Recovery Category'])
            
            # DBSCAN doesn't 'train' for prediction in the same way, but we can fit it to see clusters on training data.
            # We use it conceptually here for the entire dataset when needed.
            cluster_features = X_train[['Symptoms', 'Patient age', 'ODI scores']] if 'ODI scores' in X_train.columns else X_train
            self.dbscan_model.fit(cluster_features)
            
            self.is_trained = True
            return True
        return False

    def predict(self, X_input, cluster_input):
        if not self.is_trained:
            raise ValueError("Models are not trained yet.")
            
        # SVR Predictions
        recovery_days = self.svr_recovery_days.predict(X_input)[0]
        
        try:
             odi_improvement = self.svr_odi_improvement.predict(X_input)[0]
        except:
             odi_improvement = 0.0
        
        # XGBoost Predictions
        severity = self.xgb_severity.predict(X_input)[0]
        effectiveness = self.xgb_effectiveness.predict(X_input)[0]
        recovery_cat = self.xgb_recovery_category.predict(X_input)[0]
        
        return {
            'recovery_days': round(float(recovery_days), 1),
            'odi_improvement': round(float(odi_improvement), 1),
            'severity_encoded': int(severity),
            'effectiveness_encoded': int(effectiveness),
            'recovery_cat_encoded': int(recovery_cat)
        }

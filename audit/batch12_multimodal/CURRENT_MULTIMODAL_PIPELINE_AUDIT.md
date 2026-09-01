# CURRENT MULTIMODAL PIPELINE AUDIT

- **Model Type**: XGBoost Multimodal Risk Classifier
- **Artifact Path**: `ml/models/xgboost_multimodal_best.json`
- **Output Classes**: `LOW` (0), `MODERATE` (1), `HIGH` (2)
- **Feature Vector**: 23D fused feature matrix (Vision, Questionnaire, Sensor)
- **Safety Isolation**: `sos_weight = 0.0` (VQC has 0% direct influence on emergency SOS triggers)

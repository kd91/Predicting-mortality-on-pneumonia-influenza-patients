# Predicting-mortality-on-pneumonia-influenza-patients
Predicting mortality in patients diagnosed with Pneumonia and/or Influenza in the mimic-3 dataset.

### Cohort Selection:
The cohort selected for the proposed analysis is based on the following criteria: 
The inclusion criteria is defined as all patients with pneumonia & influenza (icd9 code for 480.0 - 488.89). Exclusion criteria are patients with age less than 18yrs.

### Models:

1) Random Forest: AUC = 0.710

2) Elastic net: AUC = 0.7021

3) SVM - Radial: AUC = 0.6876

4) SVM - Poly: AUC = 0.6862

5) SVM - Linear: AUC = 0.6862

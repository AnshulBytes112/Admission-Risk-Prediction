# Emergency Admission Risk Prediction Using Big Data Analytics

## Abstract
[FILL IN: Briefly describe the goal, the synthetic dataset, the Kafka/HDFS/Spark pipeline, the model and dashboard. State clearly that the work is an academic software demonstration.]

## 1. Problem Statement
[FILL IN: Explain the project problem and the scope of the demonstration. Do not imply clinical validation.]

## 2. Objectives
- Generate or ingest a reproducible synthetic observation dataset.
- Demonstrate event streaming using Kafka.
- Store raw events in HDFS.
- Process and partition data using PySpark.
- Train and evaluate a supervised classification model.
- Serve model output through FastAPI and present it in Streamlit.

## 3. Dataset and Ethics
The project uses fictional, programmatically generated data. It is not MIMIC-III and does not contain real patient records. The target is simulated and is not a medically validated outcome.

- Number of generated rows: [FILL IN]
- Random seed: [FILL IN]
- Target prevalence: [FILL IN from actual run]
- Missingness strategy: [FILL IN]
- Dataset limitations: [FILL IN]

## 4. Architecture
[FILL IN: Insert a diagram or explain the actual services used: synthetic generator → Kafka → HDFS → PySpark → model training → FastAPI → Streamlit.]

## 5. Big Data Processing
[FILL IN: Describe Kafka topic and partitions, HDFS paths, Spark transformations, deduplication, validation, quarantine and Parquet partitioning. Distinguish implemented and actually executed steps.]

## 6. Machine Learning Methodology
[FILL IN: State split method, feature list, candidate estimators, model-selection criterion, final estimator and threshold based on actual source/report.]

## 7. Results
Only report results obtained from a real run of the included scripts.

| Metric | Actual value |
|---|---|
| Precision | [FILL IN] |
| Recall | [FILL IN] |
| F1 | [FILL IN] |
| PR-AUC | [FILL IN] |
| ROC-AUC | [FILL IN] |
| Test set size | [FILL IN] |

If a metric is not present in the report, write “not available” rather than estimating it.

## 8. Application
[FILL IN: Add actual screenshots of the dashboard, API docs and service status. Describe the form and endpoints tested.]

## 9. Testing
[FILL IN: Include actual commands, test outcomes, failures and remediation. Do not claim checks passed unless they were run.]

## 10. Limitations
- All observations and labels are synthetic.
- The project is not a substitute for real clinical datasets, peer-reviewed evaluation or external validation.
- Metrics reflect the artificial generator and cannot be generalized to clinical populations.
- The demonstration is not for diagnosis, triage or treatment.

## 11. Future Work
[FILL IN: Possible improvements include schema registry, streaming feature computation, monitoring, drift analysis, model explainability, access control, and validation on properly licensed and approved real data.]

## 12. Conclusion
[FILL IN: Summarize what was actually implemented and tested without overstating results.]

## References
[FILL IN: Cite documentation for Apache Kafka, Hadoop HDFS, Apache Spark, scikit-learn, FastAPI and Streamlit as used.]

# Clinical Synthetic Data Dictionary

**Project**: Emergency Admission Risk Prediction Using Big Data Analytics  
**Version**: 1.0 (Phase 1 Baseline)  
**Dataset Nature**: 100% Synthetic & Fictional Observation Records  
**Compliance & Ethical Notice**: Generated strictly for software demonstration. Not derived from real patients or MIMIC-III. Do not use for clinical triage, diagnosis, or treatment.

---

## Field Specifications

| Column Name | Role / Kind | Data Type | Expected / Plausible Range | Missingness Behavior | Description / Clinical Meaning |
|---|---|---|---|---|---|
| `event_id` | Identifier | String (Alphanumeric) | `EVT000000001` – `EVT999999999` | Strictly mandatory (`0.0%` nulls) | Unique observation event identifier assigned sequentially to each triage/monitoring record. |
| `patient_id` | Identifier | String (Alphanumeric) | `SYN000001` – `SYN999999` | Strictly mandatory (`0.0%` nulls) | Fictional patient identity. Patient pool allows multiple events per patient to demonstrate longitudinal grouping and history. |
| `observation_timestamp` | Timestamp | ISO 8601 String (`YYYY-MM-DDTHH:MM:SS`) | 2025-01-01T00:00:00 to 2025-12-31T23:59:59 | Strictly mandatory (`0.0%` nulls) | Date and time when the physiological observations and lab tests were recorded. |
| `age` | Demographic Feature | Integer | `18` – `100` years | Strictly mandatory (`0.0%` nulls) | Simulated adult patient age at the time of monitoring event. |
| `heart_rate` | Vital Sign Feature | Numeric (Float / Int) | `35` – `190` bpm | Controlled missingness (`~2.5%` nulls) | Measured cardiac frequency in beats per minute. Simulates sensor detachment or unrecorded vitals. |
| `systolic_bp` | Vital Sign Feature | Numeric (Float / Int) | `60` – `220` mmHg | Controlled missingness (`~2.5%` nulls) | Systolic arterial blood pressure. Simulates non-invasive cuff measurement failure. |
| `diastolic_bp` | Vital Sign Feature | Integer | `30` – `140` mmHg | Complete (`0.0%` nulls) | Diastolic arterial blood pressure recorded in millimeters of mercury. |
| `spo2` | Vital Sign Feature | Numeric (Float / Int) | `70` – `100` % | Controlled missingness (`~2.5%` nulls) | Peripheral capillary blood oxygen saturation measured via pulse oximetry. |
| `respiratory_rate` | Vital Sign Feature | Integer | `5` – `50` breaths/min | Complete (`0.0%` nulls) | Frequency of breaths per minute. |
| `temperature_c` | Vital Sign Feature | Float (1 decimal) | `33.0` – `42.0` °C | Controlled missingness (`~2.5%` nulls) | Body temperature in degrees Celsius. Simulates thermometer measurement omission. |
| `previous_admissions` | Historical Feature | Integer | `0` – `12` count | Complete (`0.0%` nulls) | Number of emergency or inpatient admissions in the preceding 12 months. |
| `chronic_conditions_count` | Historical Feature | Integer | `0` – `8` count | Complete (`0.0%` nulls) | Number of pre-existing documented chronic morbidities (e.g., COPD, CHF, hypertension, diabetes). |
| `lactate` | Lab Value Feature | Float (1 decimal) | `0.3` – `10.0` mmol/L | Complete (`0.0%` nulls) | Point-of-care venous/arterial blood lactate concentration, serving as a physiological stress indicator. |
| `admission_within_24h` | Synthetic Target | Binary Integer (`0` or `1`) | `0` (Discharged / Stable) or `1` (Admitted) | Strictly mandatory (`0.0%` nulls) | Synthetic outcome label indicating unplanned emergency inpatient admission within 24 hours of triage. Generated via documented logistic propensity formula. |
| `synthetic_data` | Audit / Safety Marker | Boolean (`True`) | `True` | Strictly mandatory (`0.0%` nulls) | Permanent safety audit flag ensuring that no synthetic record is ever mistaken for authentic clinical data. |

---

## Missing Data Handling Strategy for Future Phases

The controlled missingness (~2.5% across `heart_rate`, `systolic_bp`, `spo2`, and `temperature_c`) is deliberately injected into the dataset generator to provide realistic scenarios for subsequent big data analytics phases:
- **Phase 3 (HDFS Ingestion)**: Ingests records containing `null`/empty values without data corruption.
- **Phase 4 (PySpark Feature Engineering)**: Evaluates median/mean imputation, forward-fill grouped by `patient_id`, or flag indicators for missing vital measurements.
- **Phase 5 (Model Training)**: Benchmarks algorithms capable of native NaN handling (e.g., `HistGradientBoostingClassifier`) versus pipelines requiring explicit imputation transformers (`SimpleImputer`).

---

## Synthetic Target Generation Formula

The binary target `admission_within_24h` is computed using a deterministic logistic response function:

$$\text{Logit}(p) = -3.10 + 0.018 \cdot (\text{age} - 50) + 0.025 \cdot (\text{heart\_rate} - 80) + 0.028 \cdot (95 - \text{systolic\_bp}) + 0.18 \cdot (95 - \text{spo2}) + 0.07 \cdot (\text{respiratory\_rate} - 17) + 0.45 \cdot (\text{temperature} - 37.0) + 0.20 \cdot \text{previous\_admissions} + 0.16 \cdot \text{chronic\_conditions} + 0.28 \cdot (\text{lactate} - 1.5)$$

$$p = \frac{1}{1 + e^{-\text{Logit}(p)}}$$

$$\text{admission\_within\_24h} \sim \text{Bernoulli}(p)$$

*Notice: This formula is an academic heuristic for big-data streaming and machine learning pipelines. It does NOT represent genuine epidemiological or medical risk factors.*

/**
 * Structured pipeline stages definition containing college presentation explanations,
 * in-depth technical breakdowns, inputs/outputs, evidence links, and failure scenarios.
 */

export const PIPELINE_GROUPS = [
  { id: 'streaming', label: 'Data Generation & Streaming', color: '#3b82f6' },
  { id: 'storage', label: 'Distributed Storage', color: '#8b5cf6' },
  { id: 'processing', label: 'Distributed Processing', color: '#ec4899' },
  { id: 'analytics', label: 'Machine Learning & Features', color: '#10b981' },
  { id: 'serving', label: 'Serving & UI Presentation', color: '#f59e0b' },
];

export const PIPELINE_STAGES = [
  {
    id: 'stage-1',
    step: 1,
    name: 'Synthetic Data Generator',
    group: 'streaming',
    technology: 'Python 3.11+ / NumPy / Pandas',
    shortDescription: 'Generates reproducible, fictional clinical observations with physiological bounds.',
    presentationScript:
      'We begin with our Synthetic Data Generator. Because clinical patient data is strictly protected by HIPAA, we generated 10,000 synthetic patient observations with clinically plausible vital signs like blood pressure, heart rate, and SpO2 to rigorously benchmark our big data pipeline.',
    technicalExplanation:
      'The data generator script (scripts/generate_synthetic_data.py) utilizes a fixed random seed (default 42) to ensure deterministic reproducibility. It applies multivariate Gaussian distributions and physiological clamping to generate realistic vitals (heart rate 25-240 bpm, systolic BP 40-300 mmHg, SpO2 50-100%, lactate 0-30 mmol/L) accompanied by synthetic ground-truth admission risk labels.',
    input: 'CLI arguments: --rows 10000 --seed 42 --output data/raw/patient_observations.csv',
    processing:
      'Random variable sampling from bounded clinical distributions, correlated vital signs generation, synthetic admission outcome labeling, and CSV serialization.',
    output: 'CSV file: data/raw/patient_observations.csv (10,000 rows, 13 raw clinical attributes)',
    location: 'data/raw/patient_observations.csv',
    statusType: 'verified',
    evidenceSource: 'Local filesystem (data/raw/patient_observations.csv)',
    failureScenarios:
      'Missing output directory permissions; invalid random seed; extreme physiological out-of-bounds sampling.',
    keyTakeaway: 'Synthetic clinical data preserves strict patient privacy while enabling robust, reproducible BDA testing.',
  },
  {
    id: 'stage-2',
    step: 2,
    name: 'Kafka Producer',
    group: 'streaming',
    technology: 'Python kafka-python / KafkaProducer API',
    shortDescription: 'Streams CSV rows as serialized JSON events with backoff retries.',
    presentationScript:
      'The Kafka Producer reads the generated observation records and streams them as real-time JSON events to our Kafka broker, simulating live ambulance and emergency room admissions.',
    technicalExplanation:
      'The producer script (scripts/produce_kafka_stream.py) parses patient observations, injects streaming metadata (schema version 1.0, publish timestamp, trace ID), converts rows to UTF-8 JSON payloads, and dispatches them with delivery callbacks and exponential backoff retries.',
    input: 'CSV observation records from data/raw/patient_observations.csv',
    processing:
      'Timestamp injection, schema metadata tagging, JSON serialization, keyed partitioning by patient_id, and asynchronous batch transmission with 25ms delay.',
    output: 'Serialized UTF-8 JSON message records dispatched over port 9092/29092',
    location: 'Docker container: emergency-admission-bda-kafka-producer',
    statusType: 'verified',
    evidenceSource: 'scripts/produce_kafka_stream.py & Kafka logs',
    failureScenarios:
      'Kafka broker unreachable (connection refused); timeout on producer flush; buffer overflow under high throughput.',
    keyTakeaway: 'Publishing messages only dispatches them to the stream; it does not guarantee downstream processing.',
  },
  {
    id: 'stage-3',
    step: 3,
    name: 'Kafka Topic (patient-observations)',
    group: 'streaming',
    technology: 'Apache Kafka 3.8 / KRaft Broker',
    shortDescription: 'Distributed partitioned log acting as the central real-time event backbone.',
    presentationScript:
      'The central message bus is our Kafka topic: patient-observations. It provides distributed log storage across 3 partitions, decoupling high-velocity event producers from batch and streaming consumers.',
    technicalExplanation:
      'The topic patient-observations is configured with 3 partitions and a replication factor of 1 in our local KRaft mode deployment. Partitions guarantee strict message ordering per partition key (patient_id) while enabling parallel consumption across multi-core workers.',
    input: 'JSON event records from Kafka Producer',
    processing:
      'Partition assignment by hash(key), disk append to commit log segment files, offset indexing, and in-memory caching.',
    output: 'Ordered message streams indexed by zero-based partition offsets',
    location: 'Topic: patient-observations (partitions: 3, broker: localhost:9092 / kafka:29092)',
    statusType: 'verified',
    evidenceSource: 'Live TCP probe to Kafka broker (port 29092 / 9092)',
    failureScenarios:
      'Broker partition leader out of sync; disk exhaustion in /bitnami/kafka; KRaft quorum controller partition.',
    keyTakeaway: 'Partitions provide both horizontal parallelism and strict within-partition event ordering.',
  },
  {
    id: 'stage-4',
    step: 4,
    name: 'Kafka Consumer',
    group: 'streaming',
    technology: 'Python kafka-python / KafkaConsumer API',
    shortDescription: 'Subscribes as consumer group to ingest and validate streamed records.',
    presentationScript:
      'The Kafka Consumer belongs to consumer group emergency-admission-bda-consumer. It continuously polls Kafka, validates each record schema, and flushes valid events to our raw archive.',
    technicalExplanation:
      'Implemented in scripts/consume_kafka_stream.py, the consumer joins group emergency-admission-bda-consumer, receives partition assignments, commits offsets upon batch flush (flush interval 250 records), and routes schema-invalid messages into a dedicated quarantine file.',
    input: 'Serialized messages from topic patient-observations',
    processing:
      'Deserialization, required field validation (patient_id, vitals), offset tracking, duplicate suppression, and buffer flushing.',
    output: 'Validated streaming records batched for local file persistence',
    location: 'Consumer Group: emergency-admission-bda-consumer',
    statusType: 'verified',
    evidenceSource: 'Live broker probe & consumer group registration',
    failureScenarios:
      'Consumer rebalance storm; consumer lag accumulating due to downstream I/O bottlenecks; unhandled JSON deserialization exception.',
    keyTakeaway: 'Consumer groups allow cooperative parallel consumption without double-processing messages.',
  },
  {
    id: 'stage-5',
    step: 5,
    name: 'JSONL Event Archive',
    group: 'storage',
    technology: 'POSIX Append-Only Filesystem / Newline-Delimited JSON',
    shortDescription: 'High-speed immutable staging archive capturing streamed events.',
    presentationScript:
      'As events are consumed, they are persisted into a JSONL archive. Newline-delimited JSON is the industry standard for stream staging because each record is a standalone line that can be appended without locking or corrupting existing records.',
    technicalExplanation:
      'The archive data/streamed/patient_events.jsonl stores raw ingested records with one JSON object per line. This enables O(1) line appending, atomic file rotation, and direct compatibility with Hadoop HDFS raw ingestion utilities and PySpark multiline readers.',
    input: 'Streamed observation objects from Kafka Consumer',
    processing: 'Append-only stream file writing with periodic disk sync and line counting.',
    output: 'data/streamed/patient_events.jsonl (150 records, 111.7 KB on disk)',
    location: 'data/streamed/patient_events.jsonl',
    statusType: 'verified',
    evidenceSource: 'Local disk file stats: 150 records, 111,726 bytes',
    failureScenarios:
      'Disk write permission error; partial line writes during abrupt container kill; file descriptor exhaustion.',
    keyTakeaway: 'JSONL allows fast, atomic appends without the schema lock contention of relational tables.',
  },
  {
    id: 'stage-6',
    step: 6,
    name: 'Hadoop HDFS — Raw Storage',
    group: 'storage',
    technology: 'Apache Hadoop 3.2.1 HDFS (NameNode & DataNode)',
    shortDescription: 'Distributed storage cluster maintaining immutable raw records with ingestion manifests.',
    presentationScript:
      'Next, the raw JSONL records are transferred into Hadoop HDFS under /emergency-admission-bda/raw/kafka-events. HDFS provides distributed, fault-tolerant raw storage and logs cryptographic SHA-256 manifests for complete data lineage.',
    technicalExplanation:
      'The NameNode (port 9870/9000) coordinates distributed metadata while DataNodes store 128MB data blocks. Ingestion manifests in data/metadata/ingestion_manifests record the exact source filename, file size, SHA-256 checksum (f3994d6e...), 100 valid records, and destination HDFS path, providing auditability for regulatory compliance.',
    input: 'data/streamed/patient_events.jsonl',
    processing:
      'SHA-256 integrity calculation, partition directory generation (ingestion_date=YYYY-MM-DD), NameNode RPC registration, block allocation, and JSON manifest generation.',
    output: 'HDFS Path: /emergency-admission-bda/raw/kafka-events/ingestion_date=2026-10-13/patient_events.jsonl',
    location: 'HDFS NameNode (hdfs://namenode:9000 / UI: localhost:9870)',
    statusType: 'verified',
    evidenceSource: 'Live NameNode TCP probe & Ingestion Manifest (2026-10-13, 100 records verified)',
    failureScenarios:
      'NameNode in SafeMode; DataNode heartbeat lost; block replication failure; HDFS storage quota exceeded.',
    keyTakeaway: 'Retaining raw events in HDFS before transformation enables reproducible backfilling and audits.',
  },
  {
    id: 'stage-7',
    step: 7,
    name: 'PySpark Processing Engine',
    group: 'processing',
    technology: 'Apache Spark 3.5.6 / PySpark / Spark SQL',
    shortDescription: 'Distributed memory-accelerated batch execution reading raw HDFS records.',
    presentationScript:
      'Our distributed processing powerhouse is Apache PySpark. Spark reads raw events from HDFS, applies clinical range validation, eliminates duplicate telemetry, imputes missing vital signs, and calculates rolling observation windows.',
    technicalExplanation:
      'The batch pipeline (scripts/process_features_spark.py) initializes a SparkSession with 2 shuffle partitions. It reads the raw JSONL dataset from HDFS, enforces the clinical schema, cleanses anomalies, and manages execution stages using DAG transformations (narrow map/filter and wide deduplication shuffles).',
    input: 'hdfs://namenode:9000/emergency-admission-bda/raw/kafka-events',
    processing:
      'Schema enforcement, timestamp parsing, missing value identification, median vital sign imputation (heart rate median: 82.0, systolic BP: 124.0, temperature: 37.0°C), and lineage execution.',
    output: 'PySpark Resilient Distributed Dataset (RDD) / DataFrame ready for analytical persistence',
    location: 'Local Spark / Container: Spark Master 3.5.6 (port 8080/7077)',
    statusType: 'historical_success',
    evidenceSource: 'Spark Manifest (run_id: 8a7f70504501, status: SUCCESS)',
    failureScenarios:
      'Executor OutOfMemoryError; Spark driver connection lost; schema mismatch on corrupted JSON rows.',
    keyTakeaway: 'In-memory RDD transformations allow Spark to process massive clinical datasets orders of magnitude faster than MapReduce.',
  },
  {
    id: 'stage-8',
    step: 8,
    name: 'Validation & Deduplication',
    group: 'processing',
    technology: 'PySpark DataFrame API / Window Functions',
    shortDescription: 'Ensures strict clinical integrity and eliminates re-transmitted streaming duplicates.',
    presentationScript:
      'During Spark execution, records undergo two critical data quality steps: strict clinical validation and duplicate suppression. Out of 100 raw events in our benchmark run, exactly 50 duplicates were identified and removed, leaving 50 unique patient observations.',
    technicalExplanation:
      'Records are validated against FEATURE_RANGES (e.g. SpO2 50-100%). Deduplication utilizes patient_id and observation timestamp windows. Exactly 50 duplicate records were suppressed in run_id 8a7f70504501. Valid records proceed to feature persistence, while any corrupted records are routed to quarantine.',
    input: '100 raw input records from HDFS raw stage',
    processing:
      'Duplicate detection across (patient_id, observation_timestamp), range boundary checking, null-value quantification, and conditional branching.',
    output: '50 validated, unique clinical observation records (quarantined: 0, duplicates: 50)',
    location: 'PySpark memory execution boundary',
    statusType: 'historical_success',
    evidenceSource: 'Spark processing manifest: duplicate_count=50, valid_record_count=100',
    failureScenarios:
      'High data skew causing executor memory exhaustion during wide shuffle; overly aggressive validation dropping borderline vitals.',
    keyTakeaway: 'Deduplication guarantees that network retransmissions do not artificially skew machine learning training distributions.',
  },
  {
    id: 'stage-9',
    step: 9,
    name: 'Quarantine Storage',
    group: 'processing',
    technology: 'HDFS Quarantine Directory / JSONL Dead-Letter Queue',
    shortDescription: 'Isolates malformed or corrupted records for diagnostic audit.',
    presentationScript:
      'Rather than silently discarding invalid records or allowing them to contaminate model training, invalid records are routed to Quarantine Storage under /emergency-admission-bda/quarantine/spark-processing.',
    technicalExplanation:
      'Malformed records (e.g. unparseable JSON, heart rate > 300 bpm, missing patient identifier) are captured by dead-letter logic and written to HDFS quarantine with diagnostic error flags. In our verified benchmark run, 0 records failed clinical validation, confirming clean input generation.',
    input: 'Records failing clinical boundary or schema checks',
    processing: 'Error code tagging, quarantine manifest logging, and write isolation.',
    output: '0 quarantined records in benchmark run (path: /emergency-admission-bda/quarantine/spark-processing)',
    location: '/emergency-admission-bda/quarantine/spark-processing',
    statusType: 'verified',
    evidenceSource: 'data/streamed/quarantine_events.jsonl (0 bytes) & Spark manifest (0 quarantined)',
    failureScenarios:
      'Quarantine write failure causing silent drop of corrupted records; unbounded quarantine file growth under upstream errors.',
    keyTakeaway: 'Quarantine storage preserves data observability: never discard bad data silently.',
  },
  {
    id: 'stage-10',
    step: 10,
    name: 'Partitioned Parquet Feature Dataset',
    group: 'analytics',
    technology: 'Apache Parquet / Snappy Compression / Columnar Storage',
    shortDescription: 'High-performance columnar storage partitioned by observation_date.',
    presentationScript:
      'The cleansed, validated features are saved as a partitioned Parquet dataset. Parquet uses columnar storage with dictionary encoding and Snappy compression, dramatically speeding up analytical queries and ML training reads.',
    technicalExplanation:
      'Saved at data/processed/features.parquet and HDFS /emergency-admission-bda/processed/patient-features, the dataset is partitioned by observation_date across 49 partition directories. It includes a _SUCCESS marker, 12 feature columns, patient_id, and ground-truth admission_within_24h label.',
    input: '50 validated, deduplicated, imputed records from PySpark',
    processing:
      'Column-oriented data reorganization, dictionary encoding, Snappy compression, date-partitioned file writing, and _SUCCESS commit.',
    output: 'data/processed/features.parquet (49 partition directories, _SUCCESS file present)',
    location: 'Local: data/processed/features.parquet & HDFS: /emergency-admission-bda/processed/patient-features',
    statusType: 'verified',
    evidenceSource: 'Filesystem scan: 49 observation_date partitions, _SUCCESS file verified',
    failureScenarios:
      'Small-file problem caused by excessive partitioning; missing _SUCCESS marker indicating partial write; Parquet schema mismatch.',
    keyTakeaway: 'Columnar Parquet allows queries to read only the needed feature columns, reducing I/O by up to 80%.',
  },
  {
    id: 'stage-11',
    step: 11,
    name: 'Machine Learning Model',
    group: 'analytics',
    technology: 'Scikit-Learn 1.6 / Joblib / LogisticRegression Pipeline',
    shortDescription: 'Trained binary classifier bundled with preprocessor and threshold metadata.',
    presentationScript:
      'In Stage 11, our Machine Learning training pipeline trains an emergency admission risk classifier on the Parquet features. We use a serialized LogisticRegression bundle with median imputation, StandardScaler, and a decision threshold of 0.70.',
    technicalExplanation:
      'The trained artifact models/risk_model.joblib is managed by ModelManager. It contains an end-to-end Pipeline (SimpleImputer, StandardScaler, LogisticRegression). Crucially, this academic demonstration model was trained on a small synthetic smoke-test split where the test set contained only negative cases, so its 100% accuracy is an experimental metric rather than proof of real clinical efficacy.',
    input: '12 clinical features extracted from data/processed/features.parquet',
    processing:
      'Pipeline fit on training split, validation PR-AUC model selection, decision threshold calibration (0.70), and Joblib serialization.',
    output: 'Serialized bundle: models/risk_model.joblib (3.9 KB) & model_metadata.json',
    location: 'models/risk_model.joblib',
    statusType: 'verified',
    evidenceSource: 'Local model file (3,910 bytes) & models/evaluation_report.json',
    failureScenarios:
      'Scikit-learn version deserialization incompatibility; missing positive class in single-class training fold; corrupted Joblib archive.',
    keyTakeaway: 'Model accuracy on synthetic smoke-test splits must never be confused with real clinical predictive power.',
  },
  {
    id: 'stage-12',
    step: 12,
    name: 'FastAPI Prediction Service',
    group: 'serving',
    technology: 'FastAPI 0.115+ / Uvicorn / Pydantic v2',
    shortDescription: 'Production-ready REST API serving low-latency single and batch predictions.',
    presentationScript:
      'The trained model is served by our high-performance FastAPI inference service. It exposes endpoints for single patient assessment, batch CSV predictions, health probes, and live big data diagnostics.',
    technicalExplanation:
      'Running on port 8000 (Docker container emergency-admission-bda-api), FastAPI validates inputs via Pydantic schemas (PatientFeatures with 12 features), executes model.predict_proba in sub-10ms latency, compares risk scores against threshold 0.70, and includes automated CORS support for our React frontend.',
    input: 'HTTP POST JSON payloads matching PatientFeatures schema',
    processing:
      'Pydantic boundary validation, DataFrame conversion, vectorized predict_proba execution, threshold comparison, and JSON response assembly.',
    output: 'JSON response with patient_id, risk_probability, risk_percent, review_flag, threshold, notice',
    location: 'http://localhost:8000 (/health, /ready, /predict, /predict/batch, /pipeline/status)',
    statusType: 'verified',
    evidenceSource: 'Live HTTP probe to http://localhost:8000/health (status: ok, 200)',
    failureScenarios:
      'Port 8000 binding conflict; Pydantic 422 Unprocessable Entity on missing vitals; 503 Service Unavailable if model bundle missing.',
    keyTakeaway: 'Decoupling model serving behind a REST API allows independent scaling of frontend and inference infrastructure.',
  },
  {
    id: 'stage-13',
    step: 13,
    name: 'React Analytics Dashboard',
    group: 'serving',
    technology: 'React 19 / Vite / Vanilla CSS Design System',
    shortDescription: 'Interactive SaaS frontend providing pipeline observability and clinical risk simulation.',
    presentationScript:
      'Finally, our React Frontend brings everything together. Evaluators can explore live pipeline telemetry, step through this guided architecture walkthrough, inspect HDFS and Spark manifests, and run simulated patient predictions in real time.',
    technicalExplanation:
      'Built with React 19 and custom Vanilla CSS, the application connects to FastAPI over port 3000 (with Vite proxy forwarding). It features live probe counters, real-time batch CSV processing up to 500 records, an interactive architecture node diagram, and transparent academic disclaimers.',
    input: 'User interactions, CSV uploads, and live FastAPI telemetry JSON streams',
    processing:
      'Client-side CSV parsing, responsive node diagram rendering, walkthrough state management, and real-time metric visualization.',
    output: 'Interactive presentation UI accessible on http://localhost:3000',
    location: 'http://localhost:3000 (Docker or local Vite server)',
    statusType: 'verified',
    evidenceSource: 'Active Vite dev server running on port 3000 with 200 OK',
    failureScenarios:
      'Vite dev proxy misconfiguration; network timeout during batch CSV upload; browser storage exhaustion.',
    keyTakeaway: 'A clear, evidence-based UI transforms abstract big data concepts into an intuitive, demonstrable presentation.',
  },
];

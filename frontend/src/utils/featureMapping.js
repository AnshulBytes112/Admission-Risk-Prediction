/**
 * Authoritative feature contract, validation, and mapping utility.
 * Enforces the exact 12-feature order required by models/risk_model.joblib:
 *  1. age
 *  2. heart_rate
 *  3. systolic_bp
 *  4. diastolic_bp
 *  5. spo2
 *  6. respiratory_rate
 *  7. temperature_c
 *  8. previous_admissions
 *  9. chronic_conditions_count
 * 10. lactate
 * 11. observation_hour
 * 12. prior_observation_count
 */

export const MODEL_FEATURE_ORDER = [
  'age',
  'heart_rate',
  'systolic_bp',
  'diastolic_bp',
  'spo2',
  'respiratory_rate',
  'temperature_c',
  'previous_admissions',
  'chronic_conditions_count',
  'lactate',
  'observation_hour',
  'prior_observation_count',
];

export const FEATURE_CONFIG = {
  heart_rate: {
    label: 'Heart Rate',
    unit: 'bpm',
    min: 25,
    max: 240,
    step: '1',
    typicalRange: '30 – 220',
    description: 'Resting pulse rate',
  },
  systolic_bp: {
    label: 'Systolic BP',
    unit: 'mmHg',
    min: 40,
    max: 300,
    step: '1',
    typicalRange: '70 – 250',
    description: 'Systolic blood pressure',
  },
  diastolic_bp: {
    label: 'Diastolic BP',
    unit: 'mmHg',
    min: 20,
    max: 200,
    step: '1',
    typicalRange: '40 – 150',
    description: 'Diastolic blood pressure',
  },
  spo2: {
    label: 'SpO₂',
    unit: '%',
    min: 50,
    max: 100,
    step: '1',
    typicalRange: '70 – 100',
    description: 'Blood oxygen saturation percentage',
  },
  respiratory_rate: {
    label: 'Resp Rate',
    unit: 'breaths/min',
    min: 3,
    max: 80,
    step: '1',
    typicalRange: '5 – 60',
    description: 'Breaths per minute',
  },
  temperature_c: {
    label: 'Temperature',
    unit: '°C',
    min: 30.0,
    max: 44.0,
    step: '0.1',
    typicalRange: '30.0 – 42.0',
    description: 'Core body temperature in Celsius',
  },
  lactate: {
    label: 'Lactate',
    unit: 'mmol/L',
    min: 0.0,
    max: 30.0,
    step: '0.1',
    typicalRange: '0.0 – 20.0',
    description: 'Blood serum lactate concentration',
  },
  age: {
    label: 'Age',
    unit: 'years',
    min: 18,
    max: 100,
    step: '1',
    typicalRange: '18 – 100',
    description: 'Adult patient age in years',
  },
  previous_admissions: {
    label: 'Previous Admissions',
    unit: 'count',
    min: 0,
    max: 100,
    step: '1',
    typicalRange: '0 – 10',
    description: 'Prior hospital admission history',
  },
  chronic_conditions_count: {
    label: 'Chronic Conditions',
    unit: 'count',
    min: 0,
    max: 50,
    step: '1',
    typicalRange: '0 – 10',
    description: 'Number of chronic comorbid conditions',
  },
  observation_hour: {
    label: 'Arrival Hour',
    unit: 'hour (0-23)',
    min: 0,
    max: 23,
    step: '1',
    typicalRange: '0 – 23',
    description: 'Hour of arrival at emergency triage',
  },
  prior_observation_count: {
    label: 'Prior Obs Count',
    unit: 'count',
    min: 0,
    max: 100000,
    step: '1',
    typicalRange: '0 – 1000',
    description: 'Sequential observation count during current episode',
  },
};

/**
 * Validates and converts raw form inputs into a strictly typed prediction payload.
 * Rejects missing, NaN, or out-of-range values.
 *
 * @param {Record<string, any>} inputs Raw input fields
 * @returns {{ isValid: boolean, errors: string[], payload: Record<string, any> | null }}
 */
export function buildPredictionPayload(inputs) {
  const errors = [];
  const payload = {
    patient_id: inputs.patient_id ? String(inputs.patient_id).trim() : `SYNTH-PT-${Math.floor(1000 + Math.random() * 9000)}`,
  };

  if (!payload.patient_id) {
    errors.push('Patient ID is required.');
  }

  for (const featureName of MODEL_FEATURE_ORDER) {
    const rawVal = inputs[featureName];
    const config = FEATURE_CONFIG[featureName];

    if (rawVal === undefined || rawVal === null || rawVal === '') {
      errors.push(`${config.label} (${featureName}) is required.`);
      continue;
    }

    const num = Number(rawVal);
    if (!Number.isFinite(num)) {
      errors.push(`${config.label} (${featureName}) must be a valid finite number.`);
      continue;
    }

    if (num < config.min || num > config.max) {
      errors.push(`${config.label} must be between ${config.min} and ${config.max} ${config.unit}.`);
      continue;
    }

    payload[featureName] = num;
  }

  return {
    isValid: errors.length === 0,
    errors,
    payload: errors.length === 0 ? payload : null,
  };
}

/**
 * Checks whether current form inputs differ from the submitted snapshot that produced a result.
 *
 * @param {Record<string, any>} currentInputs Current form inputs
 * @param {Record<string, any> | null} snapshot Submitted snapshot
 * @returns {boolean} True if inputs have changed since last submission
 */
export function areInputsStale(currentInputs, snapshot) {
  if (!snapshot) return false;
  for (const feat of MODEL_FEATURE_ORDER) {
    if (Number(currentInputs[feat]) !== Number(snapshot[feat])) {
      return true;
    }
  }
  if (currentInputs.patient_id !== snapshot.patient_id) {
    return true;
  }
  return false;
}

/**
 * CreditLens frontend.
 *
 * The UI holds no knowledge of any specific dataset or feature. Labels,
 * units, bounds, help text and dropdown options all arrive
 * from /api/datasets, which reads them from creditlens.config.DATASET_META.
 * Adding a dataset therefore requires no change to this file.
 */

(function () {
  'use strict';

  const state = {
    datasets: {},
    current: null,
  };

  const el = {
    form: document.getElementById('assessment-form'),
    datasetSelect: document.getElementById('dataset-select'),
    modelSelect: document.getElementById('model-select'),
    fields: document.getElementById('dynamic-fields'),
    result: document.getElementById('result'),
    submit: document.getElementById('submit-btn'),
  };

  // ---------------------------------------------------------------- init

  async function init() {
    try {
      const response = await fetch('/api/datasets');
      if (!response.ok) throw new Error('HTTP ' + response.status);

      const payload = await response.json();
      state.datasets = payload.datasets || {};

      const names = Object.keys(state.datasets);
      if (names.length === 0) {
        showError('No trained models are available. Run `python -m creditlens.ml.train`.');
        el.submit.disabled = true;
        return;
      }

      buildDatasetSelect(names);
      selectDataset(names[0]);
    } catch (err) {
      console.error('Failed to load catalog:', err);
      showError('Could not reach the server. Is it running?');
      el.submit.disabled = true;
    }
  }

  // -------------------------------------------------------------- render

  function buildDatasetSelect(names) {
    el.datasetSelect.replaceChildren();
    for (const name of names) {
      el.datasetSelect.appendChild(option(name, state.datasets[name].display_name));
    }
    el.datasetSelect.addEventListener('change', (event) => selectDataset(event.target.value));
  }

  function selectDataset(name) {
    state.current = name;
    el.datasetSelect.value = name;

    const dataset = state.datasets[name];

    el.modelSelect.replaceChildren();
    for (const model of dataset.models) {
      el.modelSelect.appendChild(option(model.name, model.display_name));
    }
    // Match the model the API would pick if model_type were omitted, rather
    // than whichever happens to sort first.
    el.modelSelect.value = dataset.default_model;
    renderFields(dataset.features);
    hideResult();
  }

  function renderFields(features) {
    el.fields.replaceChildren();
    for (const feature of features) {
      el.fields.appendChild(buildField(feature));
    }
  }

  /** Build one labelled input (or select, for enumerated features). */
  function buildField(feature) {
    const wrapper = document.createElement('div');
    wrapper.className = 'field';

    const inputId = 'field-' + feature.name;

    const label = document.createElement('label');
    label.setAttribute('for', inputId);
    label.textContent = feature.label;
    if (feature.unit) {
      const unit = document.createElement('span');
      unit.className = 'unit';
      unit.textContent = ' (' + feature.unit + ')';
      label.appendChild(unit);
    }
    wrapper.appendChild(label);

    wrapper.appendChild(
      feature.options ? buildSelectInput(feature, inputId) : buildNumberInput(feature, inputId)
    );

    if (feature.help) {
      const hint = document.createElement('p');
      hint.className = 'hint';
      hint.id = inputId + '-hint';
      hint.textContent = feature.help;
      wrapper.appendChild(hint);
      wrapper.querySelector('input, select').setAttribute('aria-describedby', hint.id);
    }

    return wrapper;
  }

  function buildNumberInput(feature, inputId) {
    const input = document.createElement('input');
    input.type = 'number';
    input.id = inputId;
    input.name = feature.name;
    input.required = true;
    input.step = feature.step == null ? 'any' : String(feature.step);
    if (feature.min != null) input.min = String(feature.min);
    if (feature.max != null) input.max = String(feature.max);

    // Pre-fill with a typical value so the form is submittable on load, and
    // reuse it as the placeholder for when the visitor clears the field.
    if (feature.default != null) {
      input.value = String(feature.default);
      input.placeholder = String(feature.default);
    }

    input.addEventListener('input', () => clearFieldError(feature.name));
    return input;
  }

  function buildSelectInput(feature, inputId) {
    const select = document.createElement('select');
    select.id = inputId;
    select.name = feature.name;
    for (const choice of feature.options) {
      select.appendChild(option(String(choice.value), choice.label));
    }
    if (feature.default != null) {
      select.value = String(feature.default);
    }
    select.addEventListener('change', () => clearFieldError(feature.name));
    return select;
  }

  // ------------------------------------------------------------- submit

  el.form.addEventListener('submit', async function (event) {
    event.preventDefault();

    const dataset = state.datasets[state.current];
    const payload = {
      dataset: state.current,
      model_type: el.modelSelect.value,
    };

    let hasLocalError = false;
    dataset.features.forEach((feature) => {
      const value = document.getElementById('field-' + feature.name).value;
      payload[feature.name] = value;
      if (value === '') {
        setFieldError(feature.name, feature.label + ' is required.');
        hasLocalError = true;
      }
    });

    if (hasLocalError) {
      showError('Please fill in every field.');
      return;
    }

    setLoading(true);
    hideResult();

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const result = await response.json();

      if (response.ok) {
        showPrediction(result);
      } else {
        Object.entries(result.fields || {}).forEach(([name, message]) =>
          setFieldError(name, message)
        );
        showError(result.error || 'The server rejected the request.');
      }
    } catch (err) {
      console.error(err);
      showError('Connection failed. Is the server still running?');
    } finally {
      setLoading(false);
    }
  });

  // ------------------------------------------------------------ results

  function showPrediction(result) {
    const dataset = state.datasets[state.current];

    // Green for an outcome that is good for the bank, red otherwise. Which
    // prediction counts as good is decided per dataset on the server.
    el.result.className = 'result ' + result.outcome;
    el.result.hidden = false;

    const headline = headlineNode(result.summary);

    // Method and dataset live here rather than above the form, so the result
    // carries its own provenance.
    const method = document.createElement('p');
    method.className = 'result-detail result-method';
    method.textContent = modelDisplayName(result.model_used);

    const source = document.createElement('p');
    source.className = 'result-detail';
    source.textContent = dataset.description;

    el.result.replaceChildren(headline, method, source);

    if (result.probability != null) {
      el.result.appendChild(buildConfidenceBar(result.probability));
    }
  }

  function buildConfidenceBar(probability) {
    const wrapper = document.createElement('div');
    wrapper.className = 'confidence';

    const labels = document.createElement('div');
    labels.className = 'confidence-labels';
    // Each label needs its own element so the flex layout spaces them apart.
    labels.append(span('Model-estimated probability'), span(probability + '%'));

    const track = document.createElement('div');
    track.className = 'confidence-track';
    track.setAttribute('role', 'meter');
    track.setAttribute('aria-valuenow', String(probability));
    track.setAttribute('aria-valuemin', '0');
    track.setAttribute('aria-valuemax', '100');
    track.setAttribute('aria-label', 'Model-estimated probability');

    const fill = document.createElement('div');
    fill.className = 'confidence-fill';
    fill.style.width = Math.max(0, Math.min(100, probability)) + '%';
    track.appendChild(fill);

    wrapper.append(labels, track);
    return wrapper;
  }

  function showError(message) {
    el.result.className = 'result error';
    el.result.hidden = false;
    el.result.replaceChildren(headlineNode(message));
  }

  function hideResult() {
    el.result.hidden = true;
    el.result.replaceChildren();
  }

  // ------------------------------------------------- field-level errors

  /** The input for a feature, and the .field wrapper around it. */
  function fieldOf(name) {
    const input = document.getElementById('field-' + name);
    return input ? { input: input, wrapper: input.closest('.field') } : null;
  }

  function setFieldError(name, message) {
    const field = fieldOf(name);
    if (!field) return;

    field.wrapper.classList.add('invalid');
    let error = field.wrapper.querySelector('.field-error');
    if (!error) {
      error = document.createElement('p');
      error.className = 'field-error';
      field.wrapper.appendChild(error);
    }
    error.textContent = message;
    field.input.setAttribute('aria-invalid', 'true');
  }

  function clearFieldError(name) {
    const field = fieldOf(name);
    if (!field) return;

    field.wrapper.classList.remove('invalid');
    const error = field.wrapper.querySelector('.field-error');
    if (error) error.remove();
    field.input.removeAttribute('aria-invalid');
  }

  // ------------------------------------------------------------- utils

  function setLoading(isLoading) {
    el.submit.disabled = isLoading;
    el.submit.classList.toggle('loading', isLoading);
    el.submit.querySelector('.btn-label').textContent = isLoading ? 'Analysing…' : 'Run prediction';
  }

  function modelDisplayName(name) {
    const model = state.datasets[state.current].models.find((m) => m.name === name);
    return model ? model.display_name : name;
  }

  function option(value, label) {
    const node = document.createElement('option');
    node.value = value;
    node.textContent = label;
    return node;
  }

  function span(content) {
    const node = document.createElement('span');
    node.textContent = content;
    return node;
  }

  function headlineNode(content) {
    const node = document.createElement('p');
    node.className = 'result-headline';
    node.textContent = content;
    return node;
  }

  init();
})();

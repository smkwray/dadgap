(function () {
  'use strict';

  var cachePromise = null;
  var REQUIRED_SITE_PAGES = ['home', 'prevalence', 'outcomes', 'faq'];

  function showError(message) {
    var content = document.querySelector('.content');
    if (!content) return;
    var existing = document.querySelector('.results-error');
    if (existing) return;
    var box = document.createElement('div');
    box.className = 'warning-box results-error';
    box.innerHTML = '<p>Results artifact payload is unavailable or invalid. ' + message + ' Rerun <code>father-longrun build-synthesis</code> to refresh <code>docs/results.json</code>.</p>';
    content.insertBefore(box, content.firstChild);
  }

  function validate(payload) {
    if (!payload || typeof payload !== 'object') {
      throw new Error('Missing JSON payload.');
    }
    if (payload.schema_version !== '1.0') {
      throw new Error('Unexpected schema_version in results payload.');
    }
    if (payload.site_payload_version !== '1.0') {
      throw new Error('Unexpected site_payload_version in results payload.');
    }
    if (!payload.generated_at_utc || typeof payload.generated_at_utc !== 'string') {
      throw new Error('Missing generated_at_utc metadata.');
    }
    if (!payload.pages || typeof payload.pages !== 'object') {
      throw new Error('Missing pages object.');
    }
    REQUIRED_SITE_PAGES.forEach(function (page) {
      if (!payload.pages[page] || typeof payload.pages[page] !== 'object') {
        throw new Error('Missing page payload: ' + page);
      }
    });
    return payload;
  }

  function load() {
    if (!cachePromise) {
      cachePromise = fetch('results.json', { cache: 'no-store' })
        .then(function (response) {
          if (!response.ok) throw new Error('HTTP ' + response.status);
          return response.json();
        })
        .then(validate)
        .catch(function (error) {
          showError(error.message || 'Unknown error');
          throw error;
        });
    }
    return cachePromise;
  }

  function page(payload, key) {
    return (payload && payload.pages && payload.pages[key]) || {};
  }

  function pct(value, digits) {
    digits = typeof digits === 'number' ? digits : 1;
    return typeof value === 'number' ? (value * 100).toFixed(digits) + '%' : String(value || 'unavailable');
  }

  function usd(value) {
    return typeof value === 'number'
      ? '$' + value.toLocaleString(undefined, { maximumFractionDigits: 0 })
      : String(value || 'unavailable');
  }

  function html(rows) {
    return rows.join('');
  }

  function provenance(payload) {
    return {
      generated_at_utc: payload && payload.generated_at_utc,
      source_manifest: payload && payload.source_manifest,
      synthesis_artifacts: (payload && payload.synthesis_artifacts) || []
    };
  }

  window.dgResults = {
    load: load,
    home: function (payload) { return page(payload, 'home'); },
    outcomes: function (payload) { return page(payload, 'outcomes'); },
    prevalence: function (payload) { return page(payload, 'prevalence'); },
    faq: function (payload) { return page(payload, 'faq'); },
    provenance: provenance,
    showError: showError,
    fmt: { pct: pct, usd: usd, html: html }
  };
})();

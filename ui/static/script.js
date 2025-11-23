document.addEventListener('DOMContentLoaded', () => {
    const uploadForm = document.getElementById('upload-form');
    const indexForm = document.getElementById('index-form');
    const queryForm = document.getElementById('query-form');

    const uploadCard = document.getElementById('upload-card');
    const indexCard = document.getElementById('index-card');
    const queryCard = document.getElementById('query-card');
    const resultsCard = document.getElementById('results-card');

    const fileInput = document.getElementById('file-input');
    const uploadStatus = document.getElementById('upload-status');
    const uploadProgress = document.getElementById('upload-progress');
    const uploadProgressContainer = document.getElementById('upload-progress-container');
    const uploadMeta = document.getElementById('upload-meta');

    const sparsitySlider = document.getElementById('sparsity-slider');
    const sparsityValue = document.getElementById('sparsity-value');
    const fieldSelect = document.getElementById('field-select');
    const indexLimitInput = document.getElementById('index-limit');
    const indexProgress = document.getElementById('index-progress');
    const indexProgressContainer = document.getElementById('index-progress-container');
    const indexInfo = document.getElementById('index-info');

    const queryInput = document.getElementById('query-input');
    const queryLimitInput = document.getElementById('query-limit');
    const queryProgress = document.getElementById('query-progress');
    const queryProgressContainer = document.getElementById('query-progress-container');
    const queryWithIndexBtn = document.getElementById('query-with-index');
    const queryWithoutIndexBtn = document.getElementById('query-without-index');

    const resultsInfo = document.getElementById('results-info');
    const resultsTableHead = document.querySelector('#results-table thead');
    const resultsTableBody = document.querySelector('#results-table tbody');

    const analyticsBtn = document.getElementById('analytics-btn');
    const analyticsStatus = document.getElementById('analytics-status');
    const analyticsEmpty = document.getElementById('analytics-empty');
    const analyticsGrid = document.getElementById('analytics-grid');

    let uploadedFilepath = '';
    let datasetMeta = null;
    let indexBuilt = false;

    const setStatus = (node, message, isError = false) => {
        if (!node) {
            return;
        }
        node.textContent = message;
        node.style.color = isError ? '#dc2626' : '#2563eb';
    };

    const setStepEnabled = (card, enabled) => {
        if (!card) {
            return;
        }
        card.classList.toggle('disabled', !enabled);
    };

    const formatNumber = (value) => {
        if (value === null || value === undefined || Number.isNaN(Number(value))) {
            return '0';
        }
        return Number(value).toLocaleString();
    };

    const populateFieldOptions = (columns = [], defaultField = '') => {
        if (!fieldSelect) {
            return;
        }
        fieldSelect.innerHTML = '';
        const colList = Array.isArray(columns) ? columns : [];
        if (!colList.length) {
            const option = document.createElement('option');
            option.value = '';
            option.textContent = 'No columns detected';
            fieldSelect.appendChild(option);
            return;
        }
        colList.forEach((column) => {
            const option = document.createElement('option');
            option.value = column;
            option.textContent = column;
            if (column === defaultField) {
                option.selected = true;
            }
            fieldSelect.appendChild(option);
        });
        if (defaultField && !colList.includes(defaultField)) {
            fieldSelect.value = defaultField;
        }
    };

    const resetIndexState = () => {
        setStepEnabled(indexCard, !!datasetMeta);
        indexInfo.style.display = 'none';
        indexInfo.innerHTML = '';
        indexBuilt = false;
        queryWithIndexBtn.disabled = true;
    };

    const fetchJson = async (url, options = {}) => {
        const response = await fetch(url, options);
        const text = await response.text();
        let data;
        try {
            data = text ? JSON.parse(text) : {};
        } catch (error) {
            throw new Error('Server returned invalid JSON.');
        }
        if (!response.ok || (data && data.error)) {
            const message = data && data.error ? data.error : response.statusText;
            throw new Error(message || 'Request failed');
        }
        return data;
    };

    const getD3 = () => {
        const d3Global = window.d3;
        if (!d3Global) {
            console.error('D3.js not loaded, please check network connection or script path.');
        }
        return d3Global;
    };

    setStepEnabled(indexCard, false);
    setStepEnabled(queryCard, false);
    queryWithIndexBtn.disabled = true;

    sparsitySlider.addEventListener('input', () => {
        const value = Number.parseFloat(sparsitySlider.value || '0').toFixed(2);
        sparsityValue.textContent = value;
    });

    uploadForm.addEventListener('submit', (event) => {
        event.preventDefault();
        if (!fileInput.files || !fileInput.files.length) {
            setStatus(uploadStatus, 'Please select a CSV file.', true);
            return;
        }

        setStepEnabled(indexCard, false);
        setStepEnabled(queryCard, false);
        queryWithIndexBtn.disabled = true;
        indexBuilt = false;
        datasetMeta = null;
        uploadMeta.style.display = 'none';
        uploadMeta.innerHTML = '';
        setStatus(uploadStatus, 'Uploading...');
        uploadProgressContainer.style.display = 'block';
        uploadProgress.style.width = '0%';

        const formData = new FormData();
        formData.append('file', fileInput.files[0]);

        const xhr = new XMLHttpRequest();
        xhr.upload.addEventListener('progress', (progressEvent) => {
            if (progressEvent.lengthComputable) {
                const percent = Math.min(100, (progressEvent.loaded / progressEvent.total) * 100);
                uploadProgress.style.width = `${percent}%`;
            }
        });

        xhr.onerror = () => {
            setStatus(uploadStatus, 'Upload failed, please try again.', true);
        };

        xhr.onload = () => {
            if (xhr.status >= 200 && xhr.status < 300) {
                let payload;
                try {
                    payload = JSON.parse(xhr.responseText || '{}');
                } catch (error) {
                    setStatus(uploadStatus, 'Server returned invalid response.', true);
                    return;
                }
                if (payload.error) {
                    setStatus(uploadStatus, payload.error, true);
                    return;
                }
                uploadedFilepath = payload.filepath;
                datasetMeta = payload.metadata || null;
                setStatus(uploadStatus, `Uploaded: ${fileInput.files[0].name}`);
                if (datasetMeta) {
                    const columns = datasetMeta.columns || [];
                    uploadMeta.innerHTML = `
                        <div><strong>Rows:</strong> ${formatNumber(datasetMeta.rows)}</div>
                        <div><strong>Columns:</strong> ${columns.length}</div>
                        <div><strong>Text Field:</strong> ${datasetMeta.text_field || 'Not detected'}</div>
                        <div><strong>Column Names:</strong> ${columns.join(', ')}</div>
                    `;
                    uploadMeta.style.display = 'block';
                    populateFieldOptions(columns, datasetMeta.text_field);
                }
                setStepEnabled(indexCard, true);
                setStepEnabled(queryCard, true);
                queryWithIndexBtn.disabled = true;
            } else {
                setStatus(uploadStatus, xhr.statusText || 'Upload failed.', true);
            }
        };

        xhr.open('POST', '/upload', true);
        xhr.send(formData);
    });

    indexForm.addEventListener('submit', async (event) => {
        event.preventDefault();
        if (!uploadedFilepath) {
            alert('Please upload a data file first.');
            return;
        }

        indexInfo.style.display = 'none';
        indexInfo.innerHTML = '';
        indexProgressContainer.style.display = 'block';
        indexProgress.style.width = '0%';
        setStatus(analyticsStatus, '', false);

        const controls = indexForm.querySelectorAll('input, select, button');
        controls.forEach((node) => {
            node.disabled = true;
        });

        let progress = 0;
        const interval = window.setInterval(() => {
            progress = Math.min(100, progress + 6);
            indexProgress.style.width = `${progress}%`;
            if (progress >= 100) {
                window.clearInterval(interval);
            }
        }, 160);

        try {
            const payload = await fetchJson('/build_index', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    filepath: uploadedFilepath,
                    sparsity: sparsitySlider.value,
                    field: fieldSelect.value,
                    limit: indexLimitInput.value
                })
            });

            indexProgress.style.width = '100%';
            window.clearInterval(interval);
            indexProgressContainer.style.display = 'none';

            const details = payload.details || {};
            const sizeKb = details.index_size_bytes ? (details.index_size_bytes / 1024).toFixed(1) : payload.index_size;
            indexInfo.innerHTML = `
                <div><strong>Index Field:</strong> ${details.field || fieldSelect.value || 'Not specified'}</div>
                <div><strong>Rows Indexed:</strong> ${formatNumber(details.rows_indexed)}</div>
                <div><strong>Sparsity:</strong> ${Number.parseFloat(details.sparsity ?? sparsitySlider.value).toFixed(2)}</div>
                <div><strong>Index Size:</strong> ${sizeKb} KB</div>
            `;
            indexInfo.style.display = 'block';
            indexBuilt = true;
            queryWithIndexBtn.disabled = false;
            setStatus(analyticsStatus, 'Index built successfully. You can refresh analytics.');
        } catch (error) {
            window.clearInterval(interval);
            indexProgressContainer.style.display = 'none';
            alert(`Failed to build index: ${error.message}`);
        } finally {
            controls.forEach((node) => {
                node.disabled = false;
            });
        }
    });

    const runQuery = async (useIndex) => {
        const queryText = (queryInput.value || '').trim();
        if (!queryText) {
            alert('Please enter a query expression.');
            return;
        }
        if (!uploadedFilepath) {
            alert('Please upload a data file first.');
            return;
        }
        if (useIndex && !indexBuilt) {
            alert('Please build an index first before running indexed queries.');
            return;
        }

        resultsCard.style.display = 'none';
        resultsTableHead.innerHTML = '';
        resultsTableBody.innerHTML = '';
        resultsInfo.innerHTML = '';

        queryProgressContainer.style.display = 'block';
        queryProgress.style.width = '0%';
        queryWithIndexBtn.disabled = true;
        queryWithoutIndexBtn.disabled = true;

        let progress = 0;
        const interval = window.setInterval(() => {
            progress = Math.min(100, progress + 12);
            queryProgress.style.width = `${progress}%`;
            if (progress >= 100) {
                window.clearInterval(interval);
            }
        }, 180);

        try {
            const payload = await fetchJson('/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query: queryText,
                    use_index: useIndex,
                    limit: queryLimitInput.value
                })
            });

            queryProgress.style.width = '100%';
            window.clearInterval(interval);
            queryProgressContainer.style.display = 'none';

            displayResults(payload, useIndex);
            loadAnalytics({ silent: true });
        } catch (error) {
            window.clearInterval(interval);
            queryProgressContainer.style.display = 'none';
            alert(`Query failed: ${error.message}`);
        } finally {
            queryWithIndexBtn.disabled = !indexBuilt;
            queryWithoutIndexBtn.disabled = false;
        }
    };

    queryWithIndexBtn.addEventListener('click', () => runQuery(true));
    queryWithoutIndexBtn.addEventListener('click', () => runQuery(false));

    const displayResults = (data, wasIndexed) => {
        if (!data) {
            return;
        }
        resultsCard.style.display = 'block';

        const meta = data.metadata || {};
        const rowsMatched = formatNumber(meta.matched_rows);
        const totalRows = formatNumber(meta.total_rows);
        const returnedRows = formatNumber(meta.returned_rows);
        const indexStatus = wasIndexed && meta.use_index ? 'With Index' : 'Without Index';

        resultsInfo.innerHTML = `
            <div><strong>Inference Time:</strong> ${data.inference_time || 'N/A'}</div>
            <div><strong>Matched Rows:</strong> ${rowsMatched} / ${totalRows}</div>
            <div><strong>Returned Rows:</strong> ${returnedRows}</div>
            <div><strong>Execution Mode:</strong> ${indexStatus}</div>
        `;

        resultsTableHead.innerHTML = '';
        resultsTableBody.innerHTML = '';

        const records = Array.isArray(data.results) ? data.results : [];
        if (!records.length) {
            const emptyRow = document.createElement('tr');
            const emptyCell = document.createElement('td');
            emptyCell.colSpan = 1;
            emptyCell.textContent = 'No matching records found.';
            emptyRow.appendChild(emptyCell);
            resultsTableBody.appendChild(emptyRow);
        } else {
            const headers = Object.keys(records[0]);
            const headerRow = document.createElement('tr');
            headers.forEach((header) => {
                const th = document.createElement('th');
                th.textContent = header;
                headerRow.appendChild(th);
            });
            resultsTableHead.appendChild(headerRow);

            records.forEach((row) => {
                const tr = document.createElement('tr');
                headers.forEach((header) => {
                    const td = document.createElement('td');
                    const value = row[header];
                    td.textContent = value === null || value === undefined ? '' : value;
                    tr.appendChild(td);
                });
                resultsTableBody.appendChild(tr);
            });
        }

        renderTrace(data.trace_data || {});
    };

    const renderTrace = (traceData) => {
        const container = document.getElementById('trace-container');
        if (!container) {
            return;
        }
        container.innerHTML = '';
        const d3 = getD3();
        if (!d3) {
            container.innerHTML = '<div class="chart-empty">D3.js not loaded. Cannot render trace.</div>';
            return;
        }
        const events = Array.isArray(traceData.traceEvents) ? traceData.traceEvents : [];
        if (!events.length) {
            container.innerHTML = '<div class="chart-empty">No trace data available.</div>';
            return;
        }

        const margin = { top: 18, right: 20, bottom: 40, left: 100 };
        const width = Math.max(320, container.clientWidth - margin.left - margin.right);
        const height = Math.max(120, container.clientHeight - margin.top - margin.bottom);

        const svg = d3.select(container)
            .append('svg')
            .attr('width', width + margin.left + margin.right)
            .attr('height', height + margin.top + margin.bottom)
            .append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);

        events.sort((a, b) => (a.ts || 0) - (b.ts || 0));
        const tids = Array.from(new Set(events.map((d) => d.tid))).sort((a, b) => a - b);
        const maxTime = d3.max(events, (d) => (d.ts || 0) + (d.dur || 0)) || 0;

        if (!maxTime) {
            container.innerHTML = '<div class="chart-empty">Trace timeline is empty.</div>';
            return;
        }

        const x = d3.scaleLinear()
            .domain([0, maxTime])
            .range([0, width]);

        const y = d3.scaleBand()
            .domain(tids)
            .range([0, height])
            .padding(0.25);

        const color = d3.scaleOrdinal()
            .domain(tids)
            .range(d3.schemeTableau10);

        const xAxis = d3.axisBottom(x)
            .ticks(6)
            .tickFormat((d) => `${(Number(d) / 1000).toFixed(1)} ms`);

        const yAxis = d3.axisLeft(y)
            .tickFormat((d) => `Thread ${d}`);

        svg.append('g')
            .attr('transform', `translate(0,${height})`)
            .call(xAxis);

        svg.append('g')
            .call(yAxis);

        svg.selectAll('rect')
            .data(events)
            .enter()
            .append('rect')
            .attr('x', (d) => x(d.ts || 0))
            .attr('y', (d) => y(d.tid))
            .attr('width', (d) => {
                const start = d.ts || 0;
                const end = start + (d.dur || 0);
                return Math.max(3, x(end) - x(start));
            })
            .attr('height', y.bandwidth())
            .attr('rx', 4)
            .attr('fill', (d) => color(d.tid))
            .append('title')
            .text((d) => `${d.name} · ${(d.dur / 1000).toFixed(2)} ms`);

        svg.selectAll('text.label')
            .data(events)
            .enter()
            .append('text')
            .attr('class', 'label')
            .attr('x', (d) => x(d.ts || 0) + 6)
            .attr('y', (d) => y(d.tid) + y.bandwidth() / 2 + 4)
            .attr('fill', '#ffffff')
            .attr('font-size', '11px')
            .text((d) => d.name);
    };

    const displayChart = (imgId, emptyId, filename) => {
        const img = document.getElementById(imgId);
        const empty = document.getElementById(emptyId);
        if (!img || !empty) return;

        if (filename) {
            // Add cache-busting parameter to force reload
            img.src = `/static/pics/${filename}?t=${Date.now()}`;
            img.style.display = 'block';
            empty.style.display = 'none';
        } else {
            img.style.display = 'none';
            empty.style.display = 'flex';
        }
    };

    const loadAnalytics = async ({ silent = false } = {}) => {
        if (!silent) {
            setStatus(analyticsStatus, 'Loading analytics...');
        }
        try {
            const data = await fetchJson('/analytics');
            const { plot_files } = data || {};

            // Display backend-generated images
            displayChart('latency-img', 'latency-empty', plot_files?.latency_bar);
            displayChart('diff-img', 'diff-empty', plot_files?.diff_scatter);
            displayChart('sparsity-img', 'sparsity-empty', plot_files?.sparsity_timeline);
            displayChart('kv-img', 'kv-empty', plot_files?.kv_timeline);

            const hasAnyPlot = plot_files && Object.values(plot_files).some(f => f);

            if (!hasAnyPlot) {
                analyticsEmpty.style.display = 'block';
                analyticsEmpty.textContent = 'No analytics data available. Please run queries and refresh.';
            } else {
                analyticsEmpty.style.display = 'none';
            }

            if (!silent) {
                setStatus(analyticsStatus, 'Analytics updated successfully.');
            }
        } catch (error) {
            displayChart('latency-img', 'latency-empty', null);
            displayChart('diff-img', 'diff-empty', null);
            displayChart('sparsity-img', 'sparsity-empty', null);
            displayChart('kv-img', 'kv-empty', null);
            analyticsEmpty.style.display = 'block';
            analyticsEmpty.textContent = error.message || 'Failed to load analytics.';
            setStatus(analyticsStatus, 'Analytics refresh failed.', true);
        }
    };

    analyticsBtn.addEventListener('click', () => loadAnalytics({ silent: false }));
    loadAnalytics({ silent: true });
});

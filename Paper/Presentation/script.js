document.addEventListener('DOMContentLoaded', () => {
    const slides = document.querySelectorAll('.slide');
    const totalPagesEl = document.getElementById('total-pages');
    const currentPageEl = document.getElementById('current-page');
    let currentSlide = 0;

    // Auto-append a reset step to all slides that have animate-on-click
    slides.forEach(slide => {
        const triggers = Array.from(slide.querySelectorAll('.animate-on-click[data-step]'));
        if (triggers.length > 0) {
            let maxStep = 0;
            triggers.forEach(t => {
                const s = parseInt(t.getAttribute('data-step'), 10);
                if (s > maxStep) maxStep = s;
            });
            const resetTrigger = document.createElement('div');
            resetTrigger.className = 'animate-on-click';
            resetTrigger.setAttribute('data-step', maxStep + 1);
            resetTrigger.setAttribute('data-reset', 'true');
            const lastTrigger = triggers[triggers.length - 1];
            lastTrigger.parentNode.insertBefore(resetTrigger, lastTrigger.nextSibling);
        }
    });

    totalPagesEl.textContent = slides.length;

    const updateChartForStep = (slide, stepStr) => {
        const step = stepStr ? parseInt(stepStr, 10) : 0;
        const canvases = slide.querySelectorAll('canvas');
        canvases.forEach(canvas => {
            const chart = Chart.getChart(canvas);
            if (!chart) return;
            
            // Revert all datasets to original state first
            chart.data.datasets.forEach(ds => {
                if (ds._origBorderWidth === undefined) ds._origBorderWidth = ds.borderWidth || 2;
                if (ds._origBorderColor === undefined) ds._origBorderColor = ds.borderColor;
                if (ds._origBackgroundColor === undefined) ds._origBackgroundColor = ds.backgroundColor;
                
                ds.borderWidth = ds._origBorderWidth;
                ds.borderColor = ds._origBorderColor;
                ds.backgroundColor = ds._origBackgroundColor;
            });

            if (step > 0) {
                // Find visible step triggers
                const activeTrigger = slide.querySelectorAll(`.animate-on-click.visible[data-step="${step}"]`);
                if (activeTrigger.length > 0) {
                    const chartTargetsStr = activeTrigger[0].getAttribute('data-chart-targets');
                    if (chartTargetsStr) {
                        const targets = chartTargetsStr.split(',');
                        // Dim non-targets
                        chart.data.datasets.forEach(ds => {
                            let isTarget = false;
                            targets.forEach(t => {
                                if (ds.label && ds.label.toLowerCase().includes(t.toLowerCase().trim())) {
                                    isTarget = true;
                                }
                            });
                            if (isTarget) {
                                ds.borderWidth = ds._origBorderWidth + 2; // Make bold
                            } else {
                                // Dim
                                ds.borderWidth = 1;
                                ds.borderColor = 'rgba(200, 200, 200, 0.3)';
                                if (ds.backgroundColor) {
                                    ds.backgroundColor = 'rgba(200, 200, 200, 0.1)';
                                }
                            }
                        });
                    }
                }
            }
            chart.update();
        });
    };

    const updateStepState = (slide) => {
        const visibleTriggers = slide.querySelectorAll('.animate-on-click.visible[data-step]');
        if (visibleTriggers.length > 0) {
            // Find highest step
            let maxStep = 0;
            let isReset = false;
            visibleTriggers.forEach(t => {
                const s = parseInt(t.getAttribute('data-step'), 10);
                if (s > maxStep) {
                    maxStep = s;
                    isReset = t.hasAttribute('data-reset') && t.getAttribute('data-reset') === 'true';
                } else if (s === maxStep && t.hasAttribute('data-reset') && t.getAttribute('data-reset') === 'true') {
                    isReset = true;
                }
            });
            
            if (isReset) {
                slide.removeAttribute('data-active-step');
                updateChartForStep(slide, 0);
            } else {
                slide.setAttribute('data-active-step', maxStep);
                updateChartForStep(slide, maxStep);
            }
        } else {
            slide.removeAttribute('data-active-step');
            updateChartForStep(slide, 0);
        }
    };

    const setupSlideAnimations = (slide) => {
        const listItems = slide.querySelectorAll('.animate-on-click');
        listItems.forEach(li => li.classList.remove('visible'));
        updateStepState(slide);
    };

    // Scale a results slide's content down to fit the available height on
    // shorter viewports (browser toolbars, Safari, projectors), instead of
    // letting dense tables/charts overflow into the header or footer.
    const fitResultsSlide = (slide) => {
        if (!slide || !slide.classList.contains('results-slide')) return;
        const content = slide.querySelector('.slide-content');
        if (!content) return;
        // Reset to natural top-aligned flow so scrollHeight is measured
        // correctly (flex centering + overflow otherwise mis-reports it).
        content.style.transform = 'none';
        content.style.justifyContent = 'flex-start';
        const avail = content.clientHeight;
        const natural = content.scrollHeight;
        if (natural > avail + 2) {
            const k = Math.max(avail / natural, 0.4);
            content.style.transformOrigin = 'top center';
            content.style.transform = 'scale(' + k + ')';
        } else {
            // Fits: restore the default vertical centering.
            content.style.justifyContent = '';
            content.style.transform = '';
        }
    };
    const fitActiveSlide = () => {
        requestAnimationFrame(() => fitResultsSlide(slides[currentSlide]));
    };

    function updateSlides() {
        slides.forEach((slide, index) => {
            if (index === currentSlide) {
                slide.classList.add('active');
                setupSlideAnimations(slide);
            } else {
                slide.classList.remove('active');
            }
        });
        
        fitActiveSlide();
        currentPageEl.textContent = currentSlide + 1;
        
        const slideNum = currentSlide + 1;
        if (window.location.hash !== '#' + slideNum) {
            history.pushState(null, null, '#' + slideNum);
        }
    }

    function handleNext() {
        const slide = slides[currentSlide];
        const hiddenItems = slide.querySelectorAll('.animate-on-click:not(.visible)');
        
        if (hiddenItems.length > 0) {
            hiddenItems[0].classList.add('visible');
            updateStepState(slide);
        } else if (currentSlide < slides.length - 1) {
            currentSlide++;
            updateSlides();
        }
    }

    function handlePrev() {
        const slide = slides[currentSlide];
        const visibleItems = slide.querySelectorAll('.animate-on-click.visible');
        
        if (visibleItems.length > 0) {
            visibleItems[visibleItems.length - 1].classList.remove('visible');
            updateStepState(slide);
        } else if (currentSlide > 0) {
            currentSlide--;
            updateSlides();
            
            const prevSlide = slides[currentSlide];
            const listItems = prevSlide.querySelectorAll('.animate-on-click');
            listItems.forEach(li => li.classList.add('visible'));
            updateStepState(prevSlide);
        }
    }

    window.addEventListener('keydown', (e) => {
        if (e.target.isContentEditable) return; // Allow native navigation when editing text
        
        if (['ArrowRight', ' ', 'Enter', 'PageDown'].includes(e.key)) {
            e.preventDefault();
            handleNext();
        } else if (['ArrowLeft', 'Backspace', 'PageUp'].includes(e.key)) {
            e.preventDefault();
            handlePrev();
        } else if (e.key === 'ArrowDown') {
            e.preventDefault();
            if (currentSlide < slides.length - 1) {
                currentSlide++;
                updateSlides();
            }
        } else if (e.key === 'ArrowUp') {
            e.preventDefault();
            if (currentSlide > 0) {
                currentSlide--;
                updateSlides();
                
                // Fully reveal previous slide's items when navigating backwards
                const prevSlide = slides[currentSlide];
                const listItems = prevSlide.querySelectorAll('.animate-on-click');
                listItems.forEach(li => li.classList.add('visible'));
                updateStepState(prevSlide);
            }
        }
    });

    document.addEventListener('click', (e) => {
        if (e.target.tagName === 'A' || e.target.tagName === 'BUTTON' || e.target.isContentEditable) return;
        // Disable click-to-next globally while Edit Mode is active to prevent accidental skips
        if (document.querySelector('[contenteditable="true"]')) return;
        
        handleNext();
    });
    
    let resizeRaf = null;
    window.addEventListener('resize', () => {
        if (resizeRaf) cancelAnimationFrame(resizeRaf);
        resizeRaf = requestAnimationFrame(() => fitResultsSlide(slides[currentSlide]));
    });

    window.addEventListener('hashchange', () => {
        const hash = window.location.hash;
        if (hash) {
            try {
                const targetSlideNum = parseInt(hash.replace('#', ''), 10);
                if (!isNaN(targetSlideNum) && targetSlideNum > 0 && targetSlideNum <= slides.length) {
                    const newIndex = targetSlideNum - 1;
                    if (newIndex !== currentSlide) {
                        currentSlide = newIndex;
                        updateSlides();
                    }
                }
            } catch (err) {
                // Ignore invalid numbers
            }
        }
    });
    
    // Initialize first slide based on URL hash if present
    if (window.location.hash) {
        try {
            const targetSlideNum = parseInt(window.location.hash.replace('#', ''), 10);
            if (!isNaN(targetSlideNum) && targetSlideNum > 0 && targetSlideNum <= slides.length) {
                currentSlide = targetSlideNum - 1;
            }
        } catch (err) {}
    }
    
    // Initialize Latency Chart
    const ctx = document.getElementById('latencyChart');
    if (ctx) {
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: ['4K', '16K', '32K', '65K', '131K', '262K', '524K'],
                datasets: [{
                    label: 'Baseline FlashAttention End-to-End Prefill Latency (Seconds)',
                    data: [0.15, 0.71, 1.83, 5.31, 17.51, 63.11, 237.98],
                    borderColor: '#111111',
                    backgroundColor: 'rgba(17, 17, 17, 0.1)',
                    borderWidth: 3,
                    pointBackgroundColor: '#111111',
                    pointRadius: 6,
                    pointHoverRadius: 8,
                    fill: true,
                    tension: 0.4
                }]
            },
            plugins: [{
                id: 'minutesLabels',
                afterDatasetsDraw(chart) {
                    const { ctx } = chart;
                    const meta = chart.getDatasetMeta(0);
                    meta.data.forEach((element, index) => {
                        // Indexes 4, 5, 6 correspond to 131K, 262K, and 524K
                        if (index >= 4) {
                            const seconds = chart.data.datasets[0].data[index];
                            const mins = (seconds / 60).toFixed(1);
                            
                            ctx.fillStyle = '#b71c1c'; // A subtle red to draw attention
                            ctx.font = 'bold 15px "Inclusive Sans"';
                            ctx.textAlign = 'right';
                            ctx.textBaseline = 'bottom';
                            
                            // Position slightly to the top-left of the point
                            ctx.fillText(`~${mins} mins`, element.x - 10, element.y - 10);
                        }
                    });
                }
            }],
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: {
                            font: { family: 'Inclusive Sans', size: 16 }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'End-to-End Prefill Latency (Seconds)',
                            font: { family: 'Inclusive Sans', size: 18, weight: 'bold' }
                        },
                        ticks: {
                            font: { family: 'Inclusive Sans', size: 14 }
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Sequence Length (Tokens)',
                            font: { family: 'Inclusive Sans', size: 18, weight: 'bold' }
                        },
                        ticks: {
                            font: { family: 'Inclusive Sans', size: 14 }
                        }
                    }
                }
            }
        });
    }

    // Initialize Attention Fraction Chart
    const ctx2 = document.getElementById('attentionFractionChart');
    if (ctx2) {
        new Chart(ctx2, {
            type: 'line',
            data: {
                labels: ['512', '1K', '2K', '4K', '8K', '16K', '32K', '65K', '131K', '262K'],
                datasets: [{
                    label: 'Attention Compute Fraction (%)',
                    data: [10.4, 12.5, 16.0, 22.8, 33.5, 49.8, 68.1, 81.0, 89.2, 94.5],
                    borderColor: '#111111',
                    backgroundColor: 'rgba(17, 17, 17, 0.1)',
                    borderWidth: 3,
                    pointBackgroundColor: '#111111',
                    pointRadius: 6,
                    pointHoverRadius: 8,
                    fill: true,
                    tension: 0.4
                }]
            },
            plugins: [{
                id: 'percentLabels',
                afterDatasetsDraw(chart) {
                    const { ctx } = chart;
                    const meta = chart.getDatasetMeta(0);
                    meta.data.forEach((element, index) => {
                        // Label the last three points (65K, 131K, 262K)
                        if (index >= 7) {
                            const val = chart.data.datasets[0].data[index];
                            ctx.fillStyle = '#b71c1c';
                            ctx.font = 'bold 15px "Inclusive Sans"';
                            ctx.textAlign = 'right';
                            ctx.textBaseline = 'bottom';
                            ctx.fillText(`${val.toFixed(1)}%`, element.x - 10, element.y - 10);
                        }
                    });
                }
            }],
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: { font: { family: 'Inclusive Sans', size: 14 } }
                    }
                },
                scales: {
                    y: {
                        min: 0,
                        max: 100,
                        title: {
                            display: true,
                            text: 'Fraction of Prefill (%)',
                            font: { family: 'Inclusive Sans', size: 14, weight: 'bold' }
                        },
                        ticks: { font: { family: 'Inclusive Sans', size: 12 } }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Sequence Length',
                            font: { family: 'Inclusive Sans', size: 14, weight: 'bold' }
                        },
                        ticks: { font: { family: 'Inclusive Sans', size: 12 } }
                    }
                }
            }
        });
    }

    // Self-attention slide: sync the left-diagram step pill to the visible
    // animation frame on the right. Observes frame class changes only; it does
    // not alter the generic next/prev navigation above.
    const saSlide = document.getElementById('self-attention-slide');
    if (saSlide) {
        const saFrames = Array.from(saSlide.querySelectorAll('.attn-frame'));
        const saPills = Array.from(saSlide.querySelectorAll('.sa-pill'));
        const isShown = (fr) =>
            !fr.classList.contains('animate-on-click') || fr.classList.contains('visible');
        const syncPills = () => {
            let active = saFrames[0];
            saFrames.forEach(fr => { if (isShown(fr)) active = fr; });
            const pid = active ? active.getAttribute('data-pill') : null;
            saPills.forEach(p => p.classList.toggle('sa-pill-active', p.id === pid));
        };
        const saObserver = new MutationObserver(syncPills);
        saFrames.forEach(fr =>
            saObserver.observe(fr, { attributes: true, attributeFilter: ['class'] }));
        syncPills();
    }

    // Initialize Key Cosine Similarity Chart
    const ctxKey = document.getElementById('keyCosSimChart');
    if (ctxKey) {
        new Chart(ctxKey, {
            type: 'bar',
            data: {
                labels: ['Layer 0', 'Layer 8', 'Layer 16', 'Layer 23'],
                datasets: [
                    { label: 'post-RoPE real', data: [0.7457, 0.8167, 0.8423, 0.8172], backgroundColor: '#B23A2E' },
                    { label: 'post-RoPE key-shuffled', data: [0.5094, 0.5005, 0.5216, 0.5242], backgroundColor: '#E9C46A' }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { labels: { font: { family: 'Inclusive Sans', size: 12 } }, position: 'bottom' } },
                scales: {
                    y: { min: 0, max: 1.0, title: { display: true, text: 'Mean adj. cos sim' } }
                }
            }
        });
    }

    // Initialize Query Cosine Similarity Chart
    const ctxQuery = document.getElementById('queryCosSimChart');
    if (ctxQuery) {
        new Chart(ctxQuery, {
            type: 'bar',
            data: {
                labels: ['Layer 0', 'Layer 8', 'Layer 16', 'Layer 23'],
                datasets: [
                    { label: 'post-RoPE real', data: [0.9000, 0.8408, 0.8619, 0.8626], backgroundColor: '#0055A4' },
                    { label: 'post-RoPE Q-shuffled', data: [0.6880, 0.5807, 0.6234, 0.6570], backgroundColor: '#4285F4' }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { labels: { font: { family: 'Inclusive Sans', size: 12 } }, position: 'bottom' } },
                scales: {
                    y: { min: 0, max: 1.0, title: { display: true, text: 'Mean adj. cos sim' } }
                }
            }
        });
    }

    // ===== Results section charts =====
    const AXIS_FONT = { family: 'Inclusive Sans', size: 13 };
    const AXIS_TITLE = { family: 'Inclusive Sans', size: 15, weight: 'bold' };
    const LEGEND_FONT = { family: 'Inclusive Sans', size: 13 };

    // Palette (matches deck accents)
    const C = {
        cos083: '#B23A2E', cos078: '#E9A93A',
        minf: '#2A9D8F', flex: '#8E44AD',
        x16: '#5D8FBF', x8: '#0055A4',
        dpxa72: '#B23A2E', dpxa78: '#E9A93A',
        breakeven: '#999999'
    };
    const lineDs = (label, color, data, opts = {}) => ({
        label, data, borderColor: color, backgroundColor: color,
        borderWidth: opts.bw || 2.5, pointRadius: opts.pr === undefined ? 3 : opts.pr,
        pointBackgroundColor: color, fill: false, tension: 0.25,
        borderDash: opts.dash || [], ...opts.extra
    });
    const breakEvenDs = (n) => ({
        label: 'Break-even (FlashAttention)',
        data: Array(n).fill(1.0),
        borderColor: C.breakeven, borderWidth: 1.5, borderDash: [6, 5],
        pointRadius: 0, fill: false, tension: 0
    });
    const speedupOpts = (yTitle, yMax) => ({
        responsive: true, maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: { legend: { position: 'bottom', labels: { font: LEGEND_FONT, boxWidth: 22 } } },
        scales: {
            y: { min: 0, max: yMax, title: { display: true, text: yTitle, font: AXIS_TITLE }, ticks: { font: AXIS_FONT } },
            x: { title: { display: true, text: 'Sequence length (tokens)', font: AXIS_TITLE }, ticks: { font: AXIS_FONT } }
        }
    });

    // Figure 13: Delta-KV prefill speedup vs FlashAttention (FlashAttention-only)
    const ctxDeltaKv = document.getElementById('deltaKvSpeedupChart');
    if (ctxDeltaKv) {
        const labels = ['512', '1K', '2K', '4K', '8K', '16K', '32K', '65K', '128K', '262K'];
        new Chart(ctxDeltaKv, {
            type: 'line',
            data: {
                labels,
                datasets: [
                    lineDs('Delta-KV cos083', C.cos083, [0.65, 0.69, 0.78, 0.86, 0.93, 1.01, 1.10, 1.18, 1.30, 1.35], { bw: 3 }),
                    breakEvenDs(labels.length)
                ]
            },
            options: (() => {
                const o = speedupOpts('Speedup vs FlashAttention (×)', undefined);
                o.plugins.legend = { position: 'right', labels: { font: LEGEND_FONT, boxWidth: 22, padding: 12 } };
                o.scales.y.min = 0.6;
                return o;
            })()
        });
    }

    // Figure 15: DPXA end-to-end speedup, LLaMA
    const ctxDpxaLlama = document.getElementById('dpxaSpeedupLlamaChart');
    if (ctxDpxaLlama) {
        const labels = ['4K', '16K', '32K', '64K', '128K', '256K', '512K'];
        new Chart(ctxDpxaLlama, {
            type: 'line',
            data: {
                labels,
                datasets: [
                    lineDs('MInference', C.minf, [0.11, 0.37, 0.66, 1.23, 2.33, 4.48, 9.33]),
                    lineDs('FlexPrefill γ=0.90', C.flex, [0.62, 1.00, 1.41, 2.02, 3.03, 4.77, 4.83]),
                    lineDs('XAttn s=16', C.x16, [0.79, 1.08, 1.27, 1.57, 1.89, 2.43, 3.18], { dash: [6, 4] }),
                    lineDs('XAttn s=8', C.x8, [0.80, 1.13, 1.34, 1.66, 2.00, 2.57, 3.32]),
                    lineDs('DPXA cos 0.72 (ours)', C.dpxa72, [0.68, 1.01, 1.23, 1.57, 2.08, 2.65, 3.31], { bw: 3.5 }),
                    lineDs('DPXA cos 0.78 (ours)', C.dpxa78, [0.68, 1.02, 1.23, 1.54, 2.02, 2.56, 3.29], { bw: 3.5, dash: [6, 4] }),
                    breakEvenDs(labels.length)
                ]
            },
            options: speedupOpts('Prefill speedup vs FlashAttention (×)', 9.8)
        });
    }

    // Figure 16: DPXA end-to-end speedup, Qwen (XAttn/DPXA stop at 256K)
    const ctxDpxaQwen = document.getElementById('dpxaSpeedupQwenChart');
    if (ctxDpxaQwen) {
        const labels = ['4K', '16K', '32K', '64K', '128K', '256K', '512K'];
        new Chart(ctxDpxaQwen, {
            type: 'line',
            data: {
                labels,
                datasets: [
                    lineDs('MInference', C.minf, [0.12, 0.24, 0.42, 0.79, 1.41, 2.70, 4.81]),
                    lineDs('FlexPrefill γ=0.90', C.flex, [0.52, 0.90, 1.26, 1.77, 2.66, 4.13, 3.87]),
                    lineDs('XAttn s=16', C.x16, [0.73, 1.03, 1.13, 1.29, 1.45, 1.74, null], { dash: [6, 4] }),
                    lineDs('XAttn s=8', C.x8, [0.73, 1.01, 1.11, 1.24, 1.36, 1.61, null]),
                    lineDs('DPXA cos 0.72 (ours)', C.dpxa72, [0.66, 0.96, 1.10, 1.29, 1.54, 1.79, null], { bw: 3.5 }),
                    lineDs('DPXA cos 0.78 (ours)', C.dpxa78, [0.66, 0.97, 1.11, 1.30, 1.57, 1.81, null], { bw: 3.5, dash: [6, 4] }),
                    breakEvenDs(labels.length)
                ]
            },
            options: speedupOpts('Prefill speedup vs FlashAttention (×)', 5.2)
        });
    }

    // Conclusion: DPXA speedup averaged over LLaMA + Qwen (XAttn/DPXA stop at 256K)
    const ctxDpxaConcl = document.getElementById('dpxaConclusionSpeedupChart');
    if (ctxDpxaConcl) {
        const labels = ['4K', '16K', '32K', '64K', '128K', '256K', '512K'];
        new Chart(ctxDpxaConcl, {
            type: 'line',
            data: {
                labels,
                datasets: [
                    lineDs('MInference', C.minf, [0.12, 0.31, 0.54, 1.01, 1.87, 3.59, 7.07]),
                    lineDs('FlexPrefill γ=0.90', C.flex, [0.57, 0.95, 1.34, 1.90, 2.85, 4.45, 4.35]),
                    lineDs('XAttn s=16', C.x16, [0.76, 1.06, 1.20, 1.43, 1.67, 2.09, 3.18], { dash: [6, 4] }),
                    lineDs('DPXA cos 0.72 (ours)', C.dpxa72, [0.67, 0.99, 1.17, 1.43, 1.81, 2.22, 3.31], { bw: 3.5 }),
                    breakEvenDs(labels.length)
                ]
            },
            options: (() => {
                const o = speedupOpts('Prefill speedup vs FlashAttention (×)', 7.6);
                o.plugins.legend = { display: false };  // custom HTML legend to the left
                return o;
            })()
        });
    }

    updateSlides();
});

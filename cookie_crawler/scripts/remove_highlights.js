for (const el of document.querySelectorAll('[data-report-highlight]')) {
    el.style.outline = el.dataset.reportPrevOutline || '';
    el.style.outlineOffset = el.dataset.reportPrevOutlineOffset || '';
    el.style.boxShadow = el.dataset.reportPrevBoxShadow || '';
    el.removeAttribute('data-report-highlight');
    delete el.dataset.reportPrevOutline;
    delete el.dataset.reportPrevOutlineOffset;
    delete el.dataset.reportPrevBoxShadow;
}

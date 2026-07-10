const el = arguments[0];
const color = arguments[1];
el.setAttribute('data-report-highlight', '1');
el.dataset.reportPrevOutline = el.style.outline || '';
el.dataset.reportPrevOutlineOffset = el.style.outlineOffset || '';
el.dataset.reportPrevBoxShadow = el.style.boxShadow || '';
el.style.outline = '4px solid ' + color;
el.style.outlineOffset = '2px';
el.style.boxShadow = '0 0 0 4px ' + color;

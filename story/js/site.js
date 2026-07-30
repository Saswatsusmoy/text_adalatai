/* Shared multi-page shell for the assignment docs walkthrough. */
(function () {
  const PAGES = [
    { id: 'home', file: 'index.html', label: 'Overview', act: 'Start' },
    { id: 'assignment', file: 'pages/assignment.html', label: 'Assignment map', act: 'Start' },
    { id: 'problem', file: 'pages/problem.html', label: 'Problem & setup', act: 'Context' },
    { id: 'pipeline', file: 'pages/pipeline.html', label: 'Data pipeline', act: 'Phase 1' },
    { id: 'tokenizers', file: 'pages/tokenizers.html', label: 'Tokenizers', act: 'Phase 2' },
    { id: 'stage-a', file: 'pages/stage-a.html', label: 'Stage A data', act: 'Phase 1b' },
    { id: 'dual-eval', file: 'pages/dual-eval.html', label: 'Dual eval I+E', act: 'Method' },
    { id: 'tracks', file: 'pages/tracks.html', label: 'Dual-track plan', act: 'Plan' },
    { id: 'track-d', file: 'pages/track-d.html', label: 'Track D (NLLB LoRA)', act: 'Phase 3' },
    { id: 'track-c', file: 'pages/track-c.html', label: 'Track C (vocab)', act: 'Phase 3' },
    { id: 'scoreboard', file: 'pages/scoreboard.html', label: 'Scoreboard', act: 'Results' },
    { id: 'qualitative', file: 'pages/qualitative.html', label: 'Qualitative', act: 'Results' },
    { id: 'mbr', file: 'pages/mbr.html', label: 'MBR decode', act: 'Results' },
    { id: 'production', file: 'pages/production.html', label: 'Production pick', act: 'Close' },
    { id: 'failures', file: 'pages/failures.html', label: 'Failures log', act: 'Close' },
    { id: 'reflection', file: 'pages/reflection.html', label: 'Reflection', act: 'Close' },
    { id: 'interview', file: 'pages/interview.html', label: 'Interview guide', act: 'Close' },
    { id: 'artifacts', file: 'pages/artifacts.html', label: 'Artifacts', act: 'Close' },
    { id: 'glossary', file: 'pages/glossary.html', label: 'Glossary', act: 'Close' },
    { id: 'reproduce', file: 'pages/reproduce.html', label: 'Reproduce', act: 'Close' },
  ];

  function inPagesDir() {
    return /\/pages\//.test(location.pathname) || location.pathname.endsWith('/pages');
  }

  function hrefFor(file) {
    if (!inPagesDir()) return file;
    if (file === 'index.html') return '../index.html';
    return file.replace(/^pages\//, '');
  }

  function cssHref() {
    return inPagesDir() ? '../css/story.css' : 'css/story.css';
  }

  function ensureCss() {
    if (document.querySelector('link[data-story-css]')) return;
    const link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href = cssHref();
    link.setAttribute('data-story-css', '1');
    document.head.appendChild(link);
  }

  function tocHtml(activeId) {
    let lastAct = '';
    let html = '';
    PAGES.forEach((p) => {
      if (p.act !== lastAct) {
        lastAct = p.act;
        html += `<div class="toc-act">${escapeHtml(p.act)}</div>`;
      }
      const cls = p.id === activeId ? 'toc-item is-active' : 'toc-item';
      html += `<a class="${cls}" href="${hrefFor(p.file)}">${escapeHtml(p.label)}</a>`;
    });
    return html;
  }

  function pagerHtml(activeId) {
    const i = PAGES.findIndex((p) => p.id === activeId);
    if (i < 0) return '';
    const prev = PAGES[i - 1];
    const next = PAGES[i + 1];
    return `
      <nav class="pager" aria-label="Page">
        ${
          prev
            ? `<a class="pager-link prev" href="${hrefFor(prev.file)}"><span>Previous</span><b>${escapeHtml(prev.label)}</b></a>`
            : '<span></span>'
        }
        ${
          next
            ? `<a class="pager-link next" href="${hrefFor(next.file)}"><span>Next</span><b>${escapeHtml(next.label)}</b></a>`
            : '<span></span>'
        }
      </nav>`;
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function mount() {
    ensureCss();
    const activeId = document.body.getAttribute('data-page') || 'home';
    const pageMeta = PAGES.find((p) => p.id === activeId) || PAGES[0];
    const source = document.getElementById('page-source');
    if (!source) return;

    const title = source.getAttribute('data-title') || pageMeta.label;
    const kicker = source.getAttribute('data-kicker') || pageMeta.act;
    const content = source.innerHTML;
    source.remove();

    document.title = title + ' -- Adalat AI walkthrough';

    document.body.innerHTML = `
      <a class="skip" href="#main">Skip to content</a>
      <div class="app">
        <aside class="rail" aria-label="Documentation navigation">
          <div class="rail-head">
            <p class="brand">Adalat AI</p>
            <p class="brand-sub">Assignment docs</p>
            <p class="rail-note">Self-contained walkthrough of everything tried, measured, shipped, and rejected.</p>
          </div>
          <nav class="toc" id="toc">${tocHtml(activeId)}</nav>
          <div class="rail-foot">
            <p class="keys">Tip: use the sidebar like a book TOC. Each page is independent and printable.</p>
          </div>
        </aside>
        <main class="stage" id="main">
          <header class="topbar doc-top">
            <div class="topbar-left">
              <span class="chapter-crumb">${escapeHtml(kicker)} / ${escapeHtml(pageMeta.label)}</span>
            </div>
            <div class="topbar-right">
              <a class="btn ghost" href="${hrefFor('pages/glossary.html')}">Glossary</a>
              <a class="btn ghost" href="${hrefFor('pages/reproduce.html')}">Reproduce</a>
              <a class="btn primary" href="${hrefFor('index.html')}">Home</a>
            </div>
          </header>
          <article class="mount doc">
            <header class="doc-header">
              <p class="when">${escapeHtml(kicker)}</p>
              <h1>${escapeHtml(title)}</h1>
            </header>
            <div class="doc-body">
              ${content}
            </div>
            ${pagerHtml(activeId)}
            <footer class="doc-foot">
              Sources: REPORT.md, docs/EXPERIMENTS.md, DESIGN_DECISIONS.md, data/analysis/*.json
            </footer>
          </article>
        </main>
      </div>
    `;

    const active = document.querySelector('.toc-item.is-active');
    if (active) active.scrollIntoView({ block: 'nearest' });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', mount);
  } else {
    mount();
  }
})();

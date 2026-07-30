(function () {
  const sections = [
    { act: 'Setup', items: [
      { id: 'cover', label: 'Cover' },
      { id: 's-setup', label: 'Problem / HW' },
      { id: 's-data', label: 'Corpus inventory' },
    ]},
    { act: 'Preprocess', items: [
      { id: 's-ocr', label: 'OCR 5 backends' },
      { id: 's-skips', label: 'Removed steps' },
      { id: 's-pipe', label: 'Live pipeline' },
      { id: 's-stage-a', label: 'Stage A' },
      { id: 's-eval', label: 'Dual eval' },
    ]},
    { act: 'Tokenizers', items: [
      { id: 's-tok', label: 'Survey' },
      { id: 's-spm-v1', label: 'SPM v1' },
      { id: 's-spm-v2', label: 'SPM v2 + ablation' },
    ]},
    { act: 'Plan', items: [
      { id: 's-tracks', label: 'Dual track' },
    ]},
    { act: 'MT results', items: [
      { id: 's-mt', label: 'Scoreboard' },
      { id: 's-zs', label: 'Zero-shot' },
      { id: 's-a1', label: 'A1 + deltas' },
      { id: 's-a2', label: 'A2' },
      { id: 's-b', label: 'Stage B fail' },
      { id: 's-c1c', label: 'C1c v1/v2' },
    ]},
    { act: 'Close', items: [
      { id: 's-prod', label: 'Production' },
      { id: 's-design', label: '26 decisions' },
      { id: 's-failures', label: 'Failures log' },
      { id: 's-artifacts', label: 'Artifacts' },
      { id: 's-close', label: 'Summary' },
    ]},
  ];

  const iChrf = [
    { name: 'ZS', v: 44.74, cls: '' },
    { name: 'A1', v: 49.16, cls: '' },
    { name: 'A2', v: 49.66, cls: '' },
    { name: 'B', v: 48.89, cls: 'warn' },
    { name: 'C1c v2', v: 43.86, cls: 'warn' },
    { name: 'C1c v1', v: 24.86, cls: 'fail' },
  ];

  // Longer keys first so "C1c v2" wins over "C1c" / "C1"
  const GLOSSARY = [
    ['joint_full_41000', 'Track C0 freeze: joint EN+HI Unigram SPM, vocab 41k (production SPM for custom-vocab track).'],
    ['joint_full 41k', 'Track C0 freeze: joint EN+HI Unigram SPM, vocab 41k.'],
    ['joint_full 64k', 'SPM ablation: joint EN+HI Unigram, vocab 64k (best packing; not freeze).'],
    ['joint_full 48k', 'SPM ablation: joint EN+HI Unigram, vocab 48k.'],
    ['E_anuvaad_test', 'Policy E Anuvaad test: 3000 held-out legal EN-HI pairs (not in Stage A train).'],
    ['E_milpac_test', 'Policy E MILPaC test: 117 held-out legal EN-HI pairs (not in Stage A train).'],
    ['E_anuvaad_dev', 'Policy E Anuvaad dev holdout (train-time / selection).'],
    ['E_milpac_dev', 'Policy E MILPaC dev holdout (train-time / selection).'],
    ['I_test', 'Policy I assignment test: docs 1,4,21; n=190 sentence pairs.'],
    ['I_dev', 'Policy I assignment dev: docs 8,9,24; n=132 sentence pairs.'],
    ['best_primary', 'Checkpoint with best dual-policy primary score (weighted chrF++ on dev suites).'],
    ['Stage A train', 'stage_a_train.jsonl: Stage A pool after Policy E holdouts (~988k pairs).'],
    ['Stage A', 'External legal EN-HI bitext (MILPaC + Anuvaad), ~992k pairs; domain FT before assignment.'],
    ['Stage B', 'Assignment-only LoRA stage: train on 1136 assignment pairs, resume A2; failed E anti-forget.'],
    ['Track D', 'Defaults track: stock NLLB-600M native tokenizer + LoRA (zero-shot -> A1 -> A2 -> B).'],
    ['Track C', 'Custom-vocab track: domain SPM freeze then model adapt (C1a/C1b/C1c).'],
    ['C1c v2', 'Careful NLLB vocab-extend: +1500 verified tokens, emb grad-mask, A1 LoRA; below zero-shot on full test.'],
    ['C1c v1', 'Bulk NLLB vocab-extend: +~8k SPM pieces, full emb train; failed ablation (test collapsed).'],
    ['C1c', 'Vocab-extend NLLB with legal tokens + LoRA (chosen C1 path; not production).'],
    ['C1b', 'Resize embeddings of a small pretrained model to custom SPM (not run).'],
    ['C1a', 'Train enc-dec from scratch with joint_full_41000 SPM (scaffolded; stopped).'],
    ['C0', 'Domain SPM phase: train/freeze sentencepiece_legal_v2_joint_full_41000.'],
    ['A2', 'Stage A curriculum scale: 150k pairs, resume A1 LoRA, LR 5e-5, 3000 steps; production dual-policy pick.'],
    ['A1', 'Stage A curriculum first: 80k curated pairs, LoRA from base NLLB, LR 1e-4, 3000 steps.'],
    ['Policy E', 'External holdout eval: MILPaC + Anuvaad pairs never used in Stage A train.'],
    ['Policy I', 'Internal eval: assignment SC judgment splits (doc-level freeze).'],
    ['MILPaC', 'Law-AI Multilingual Indian Legal Parallel Corpus (CC BY-NC-SA 4.0).'],
    ['Anuvaad', 'Anuvaad legal EN-HI parallel sources (CC BY 4.0); scale Stage A + E holdouts.'],
    ['NLLB', 'Meta NLLB-200 distilled 600M seq2seq MT model (eng_Latn -> hin_Deva here).'],
    ['LoRA', 'Low-Rank Adaptation PEFT: train small adapter matrices; base weights frozen.'],
    ['PEFT', 'Parameter-Efficient Fine-Tuning (LoRA etc.).'],
    ['DDP', 'PyTorch DistributedDataParallel; here torchrun nproc=2 on 2x H200.'],
    ['SPM', 'SentencePiece tokenizer (Unigram domain models for Track C).'],
    ['LaBSE', 'Language-Agnostic BERT Sentence Embedding; used for EN-HI mutual-best alignment.'],
    ['chrF++', 'Character n-gram F-score with word unigrams; primary quality metric for HI.'],
    ['BLEU', 'sacreBLEU corpus BLEU; lexical overlap (weaker alone for HI).'],
    ['H200', 'NVIDIA H200 GPU (~141 GB); remote dual-GPU train/eval host.'],
    ['MPS', 'Apple Metal Performance Shaders; local PyTorch backend on M4.'],
    ['MLX', 'Apple MLX framework; local small-LLM demos (not NLLB path).'],
    ['OCR', 'Optical character recognition; Tesseract Hindi for HI PDFs.'],
    ['ZS', 'Zero-shot: base NLLB, no fine-tune adapters.'],
    ['zero-shot', 'Base NLLB with no LoRA/FT; baseline all systems must clear.'],
    ['bf16', 'BFloat16 mixed precision on CUDA Hopper.'],
    ['TF32', 'TensorFloat-32 matmul mode on Ampere/Hopper (residual fp32 paths).'],
    ['SDPA', 'Scaled Dot-Product Attention (Flash / mem-efficient backends).'],
    ['decoder_attn', 'LoRA profile: decoder self-attn + cross-attn modules (~0.5% params).'],
    ['new_embed_rows.pt', 'Saved new embedding rows for C1c v2 (not stored in PEFT LoRA adapter).'],
    ['modules_to_save', 'PEFT flag to fully train/save named modules (e.g. full embed_tokens in C1c v1).'],
  ];

  function buildToc() {
    const nav = document.getElementById('toc-nav');
    if (!nav) return;
    const frag = document.createDocumentFragment();
    sections.forEach((group) => {
      const act = document.createElement('div');
      act.className = 'act';
      act.textContent = group.act;
      frag.appendChild(act);
      group.items.forEach((item) => {
        const a = document.createElement('a');
        a.href = '#' + item.id;
        a.dataset.target = item.id;
        a.textContent = item.label;
        frag.appendChild(a);
      });
    });
    nav.appendChild(frag);
  }

  function els() {
    return sections.flatMap((g) => g.items)
      .map((i) => document.getElementById(i.id))
      .filter(Boolean);
  }

  function updateActive() {
    const list = els();
    const y = window.scrollY + 100;
    let active = list[0];
    list.forEach((el) => {
      if (el.offsetTop <= y) active = el;
    });
    document.querySelectorAll('#toc-nav a').forEach((a) => {
      a.classList.toggle('is-active', a.dataset.target === active?.id);
    });
    const max = document.documentElement.scrollHeight - window.innerHeight;
    const bar = document.getElementById('toc-bar');
    if (bar && max > 0) {
      bar.style.width = Math.min(100, (window.scrollY / max) * 100) + '%';
    }
  }

  function buildBars() {
    const root = document.getElementById('bar-i');
    if (!root) return;
    const max = Math.max.apply(null, iChrf.map((x) => x.v));
    iChrf.forEach((row) => {
      const d = document.createElement('div');
      d.className = 'bar-row';
      const pct = Math.round((row.v / max) * 100);
      d.innerHTML =
        '<span class="lab">' + row.name + '</span>' +
        '<div class="bar-track"><div class="bar-fill ' + row.cls +
        '" data-w="' + pct + '"></div></div>' +
        '<span class="val">' + row.v.toFixed(1) + '</span>';
      root.appendChild(d);
    });
    const io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (!en.isIntersecting) return;
        en.target.querySelectorAll('.bar-fill').forEach(function (b) {
          b.style.width = b.dataset.w + '%';
        });
      });
    }, { threshold: 0.25 });
    io.observe(root);
  }

  function escapeRegExp(s) {
    return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  }

  function buildGlossaryRegex() {
    const parts = GLOSSARY.map(function (pair) {
      // Avoid mid-token hits (e.g. inside long identifiers) while still matching
      // prose forms like "A1", "C1c v2", "Track D".
      return '(?<![A-Za-z0-9])' + escapeRegExp(pair[0]) + '(?![A-Za-z0-9])';
    });
    return new RegExp(parts.join('|'), 'g');
  }

  const SKIP_TAGS = {
    SCRIPT: 1, STYLE: 1, CODE: 1, PRE: 1, TEXTAREA: 1, KBD: 1, SAMP: 1,
  };

  function shouldSkip(node) {
    let el = node.parentElement;
    while (el) {
      if (SKIP_TAGS[el.tagName]) return true;
      if (el.classList && el.classList.contains('term')) return true;
      if (el.classList && el.classList.contains('toc')) return true;
      el = el.parentElement;
    }
    return false;
  }

  function applyGlossary(root) {
    const re = buildGlossaryRegex();
    const tipMap = {};
    GLOSSARY.forEach(function (pair) {
      tipMap[pair[0]] = pair[1];
      tipMap[pair[0].toLowerCase()] = pair[1];
    });

    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
      acceptNode: function (node) {
        if (!node.nodeValue || !node.nodeValue.trim()) return NodeFilter.FILTER_REJECT;
        if (shouldSkip(node)) return NodeFilter.FILTER_REJECT;
        return NodeFilter.FILTER_ACCEPT;
      },
    });

    const nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);

    nodes.forEach(function (textNode) {
      const text = textNode.nodeValue;
      re.lastIndex = 0;
      if (!re.test(text)) return;
      re.lastIndex = 0;
      const frag = document.createDocumentFragment();
      let last = 0;
      let m;
      while ((m = re.exec(text)) !== null) {
        const raw = m[0];
        if (m.index > last) {
          frag.appendChild(document.createTextNode(text.slice(last, m.index)));
        }
        const tip = tipMap[raw] || tipMap[raw.toLowerCase()];
        if (!tip) {
          frag.appendChild(document.createTextNode(raw));
        } else {
          const ab = document.createElement('abbr');
          ab.className = 'term';
          ab.textContent = raw;
          ab.setAttribute('data-tip', tip);
          ab.setAttribute('title', '');
          ab.setAttribute('tabindex', '0');
          ab.setAttribute('aria-label', raw + ': ' + tip);
          frag.appendChild(ab);
        }
        last = m.index + raw.length;
      }
      if (last < text.length) {
        frag.appendChild(document.createTextNode(text.slice(last)));
      }
      textNode.parentNode.replaceChild(frag, textNode);
    });
  }

  function placeTips() {
    document.querySelectorAll('.term').forEach(function (el) {
      const rect = el.getBoundingClientRect();
      if (rect.top < 96) el.classList.add('tip-below');
      else el.classList.remove('tip-below');
    });
  }

  function boot() {
    buildToc();
    buildBars();
    const main = document.querySelector('.main-inner');
    if (main) applyGlossary(main);
    // bar labels after build
    const barRoot = document.getElementById('bar-i');
    if (barRoot) applyGlossary(barRoot);

    updateActive();
    placeTips();
    window.addEventListener('scroll', function () {
      updateActive();
      placeTips();
    }, { passive: true });
    window.addEventListener('resize', function () {
      updateActive();
      placeTips();
    }, { passive: true });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();

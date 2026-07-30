/* Adalat AI story data
 * Sourced from: CHANGELOG, DESIGN_DECISIONS, docs/EXPERIMENTS,
 * docs/TRAINING_STRATEGY, data/analysis/*, and private .local plans
 * (plan.md, dual_track_experiment_plan.md, preprocessing_plan.md,
 * dataset_analysis.md, server_jnan_analysis.md). No secrets.
 */
window.ADALAT = {
  meta: {
    title: 'Adalat AI',
    subtitle: 'English to Hindi for Indian Supreme Court judgments',
    tagline:
      'Learning-first gates, dual-track research, honest ablations, one production checkpoint.',
    sources: [
      'CHANGELOG.md',
      'DESIGN_DECISIONS.md',
      'docs/EXPERIMENTS.md',
      'docs/TRAINING_STRATEGY.md',
      'docs/NLLB_ARCHITECTURE.md',
      'docs/HARDWARE_MLX.md',
      'data/analysis/final_dual_policy_report.json',
      '.local/plan.md (private gates + frontier menu)',
      '.local/dual_track_experiment_plan.md (adopted)',
      '.local/preprocessing_plan.md (historical 10 steps)',
      '.local/dataset_analysis.md',
      '.local/server_jnan_analysis.md (H200 ops)',
    ],
    protocol: {
      decode: 'beam 4 / max_new 256 / max_in 256',
      device: 'H200 cuda batch 32 bf16',
    },
  },

  operatingMode: {
    title: 'Learning-first, not a black-box sprint',
    rules: [
      'Teach, discuss, approve, do, reflect -- gate by gate',
      'No multi-hour train without explicit go-ahead',
      'No silent external data or re-split of freeze docs',
      'Zero-shot before claiming fine-tune gains',
      'Custom SPM never fitted on assignment dev/test',
      'Report defends reasoning, not a generic template',
    ],
  },

  pillars: [
    {
      id: 'tokens',
      title: 'Token efficiency',
      body: 'Devanagari under modern tokenizers. Byte-level BPE is blind; domain SentencePiece Unigram is not.',
    },
    {
      id: 'mt',
      title: 'Domain translation',
      body: 'High-fidelity EN->HI for court judgments. Dual eval: assignment (I) and external legal (E).',
    },
  ],

  learningGoals: [
    'Why Indic is token-inefficient under EN-centric LLMs',
    'How parallel data becomes pairs (clean, segment, align, filter, split)',
    'Why document-level splits prevent leakage',
    'Full FT vs LoRA and when each wins',
    'Why NLLB / IndicTrans2 / InLegalTrans beat random LLMs here',
    'How to read BLEU vs chrF++ vs COMET for legal Hindi',
    'A realistic path to scale domains without reinventing training',
  ],

  gates: [
    { id: '0-6', label: 'Gates 0-6', status: 'done', theme: 'Problem, data, clean, segment, align, split' },
    { id: '7-8', label: 'Gates 7-8', status: 'done', theme: 'Tokenizer science + domain SPM v1/v2' },
    { id: '9', label: 'Gate 9', status: 'done', theme: 'External legal Stage A (~992k)' },
    { id: '10-11', label: 'Gates 10-11', status: 'done', theme: 'Model choice + zero-shot D' },
    { id: '12-15', label: 'Gates 12-15', status: 'done', theme: 'PEFT theory, Stage A/B train, optional prefs' },
    { id: '16', label: 'Gate 16', status: 'done', theme: 'Full dual-policy eval table' },
    { id: '17', label: 'Gate 17', status: 'open', theme: 'Assignment report (learning evidence)' },
  ],

  corpus: {
    judgments: 30,
    pairs: 1458,
    train: 1136,
    dev: 132,
    test: 190,
    split: 'document-level (seed 42)',
    trainDocs: '2,3,5,6,7,10-20,22,23,25-30',
    devDocs: '8, 9, 24',
    testDocs: '1, 4, 21',
    enChars: 259293,
    hiChars: 242233,
    caseMix: '~21 civil, ~9 criminal (SC appeals, mostly UP / Allahabad HC matters)',
    domains:
      'service discipline, land, consolidation, cantonments, PF/EPFO, trade tax, CrPC 482, corruption',
    alignmentStart: 'Document-parallel only; 29/30 line-count mismatch -- embedding align required',
    sample: {
      en: 'The Appellant worked as an officer of Regional Rural Bank Services with the Muzaffarnagar Kshetriya Gramin Bank (hereinafter, ‘the Bank’).',
      hi: "अपीलार्थी ने मुजफ्फरनगर क्षेत्रिय ग्रामीण बैंक (इसके बाद 'बैंक' में क्षेत्रीय ग्रामीण",
      doc_id: 1,
    },
  },

  domainSignals: [
    { name: 'Appellant', n: 376 },
    { name: 'Respondent', n: 227 },
    { name: 'High Court', n: 279 },
    { name: 'Dates DD.MM.YYYY', n: 310 },
    { name: 'Section N', n: 182 },
    { name: 'impugned', n: 104 },
  ],

  pipeline: [
    { step: 'OCR', file: 'reextract_pdfs.py', note: 'Tesseract Hindi; 5 tools evaluated' },
    { step: 'Join', file: 'join_lines.py', note: 'EN hard wraps; 58% fewer lines' },
    { step: 'Segment', file: 'segment_sentences.py', note: 'spaCy EN; danda HI' },
    { step: 'Align', file: 'align_sentences.py', note: 'LaBSE mutual-best (BGE-M3 ablated)' },
    { step: 'Split', file: 'output_format.py', note: 'doc-level train/dev/test' },
  ],

  pipelinePlanned: [
    '1 Re-extract corrupted HI PDFs',
    '2 Strip UTF-8 BOM',
    '3 Normalize CRLF',
    '4 Intelligent EN line join',
    '5 Fix OCR roman numerals (li./lili.)',
    '6 Paragraph segmentation',
    '7 Sentence segmentation',
    '8 Align + QC filters',
    '9 Document-level tracking',
    '10 Final train/dev/test JSONL',
  ],

  skips: [
    {
      step: 'Strip BOM',
      why: 'preprocessed/ has 0 BOM; clean/ BOM never reaches the pipeline',
    },
    {
      step: 'OCR roman li./lili.',
      why: 'Zero instances; all L. are legitimate legal abbreviations',
    },
    {
      step: 'Normalize CRLF',
      why: 'preprocessed/ already 100% LF',
    },
    {
      step: 'Paragraph segmentation',
      why: 'Blank lines already mark paras after join/OCR',
    },
  ],

  filters: [
    { name: 'LaBSE similarity', value: '>= 0.5 (relaxed from 0.6-0.7 draft)' },
    { name: 'EN:HI char ratio', value: '0.3 - 3.0 (relaxed from 0.5-2.0)' },
    { name: 'Min length', value: '> 3 chars' },
    { name: 'EN near-dedup', value: 'Jaccard > 0.85' },
  ],

  ocrNote:
    'Docs 6,14,22,25,26 were pure ASCII (?). Tesseract recovered 42,536 Devanagari chars. PDF text layer often lies (Doc 17: Ekkuuh; vs माननीय).',

  stageA: {
    pool: 992565,
    sources: [
      { name: 'MILPaC', license: 'CC BY-NC-SA 4.0', role: 'Clean legal EN-HI (Law-AI)' },
      { name: 'Anuvaad', license: 'CC BY 4.0', role: 'Scale: judiciary, HC/SUVAS, law commission' },
    ],
    note: 'Already aligned upstream. No re-OCR / no LaBSE re-score. Exact pair dedup + length filters only.',
    trainFile: 'stage_a_train.jsonl (~988k after E holdout)',
    notBitext: 'Prarabdha SFT is mono/Q&A -- SPM v1 only, not Stage A MT',
  },

  evalPolicies: [
    {
      id: 'I',
      name: 'Policy I',
      desc: 'Assignment Supreme Court docs (internal)',
      sets: 'I_dev / I_test',
      n_test: 190,
    },
    {
      id: 'E',
      name: 'Policy E',
      desc: 'Held-out external legal bitext (MILPaC 10% + Anuvaad 1k/3k)',
      sets: 'E_milpac + E_anuvaad',
      n_milpac: 117,
      n_anuvaad: 3000,
    },
  ],

  fairness: [
    'Same Stage A path and Stage B path for both tracks',
    'Same decode (beam, max len) where possible',
    'Dev for HPs; test for final numbers only',
    'No silent re-split of assignment docs',
    'Token-cost co-reported with quality when claiming tokenizer wins',
  ],

  frontier: [
    { tier: 'T0', name: 'NLLB / IndicTrans2 / InLegalTrans + domain FT', note: 'Core deliverable path' },
    { tier: 'T0', name: 'Fertility benches + domain SPM', note: 'Tokenizer science' },
    { tier: 'T0', name: 'LaBSE align + length/sim filters', note: 'Data quality' },
    { tier: 'T0', name: 'LoRA multi-stage A then B', note: 'PEFT default' },
    { tier: 'T0', name: 'BLEU + chrF++ (+ COMET / legal panel)', note: 'Eval discipline' },
    { tier: 'T1', name: 'DPO / QE re-rank / DoRA', note: 'Optional stretch' },
    { tier: 'T3', name: 'From-scratch MT on 2k pairs', note: 'Teach-only without scale' },
  ],

  tokenizer: {
    freeze: 'sentencepiece_legal_v2_joint_full_41000.model',
    finding: 'Joint EN+HI Unigram required for MT. HI-only packs HI but fragments English.',
    survey:
      '17 tokenizers / 14 families: SP + multilingual BPE strong; byte-level BPE (Llama line) 1.1-2.7x HI cost, often 0 Dev tokens.',
    v1: 'Prarabdha mono HI Unigram 16/32/41k -- baseline fertility win vs Gemma-class on this legal set',
    firewall: 'SPM may use Stage A + assignment train only. Never assignment dev/test docs.',
    ablation: [
      { size: '41k', hi_ct: 4.37, total: 10978, note: 'Track C freeze' },
      { size: '48k', hi_ct: 4.38, total: 10937, note: 'ablation' },
      { size: '64k', hi_ct: 4.42, total: 10819, note: 'best packing, larger emb' },
    ],
    byteBpe: 'Weak on Hindi: often 0 Dev tokens; UTF-8 byte fallback.',
  },

  hardware: {
    local: 'Apple M4 MacBook Air, 16 GB unified (MLX + PyTorch MPS)',
    remote: '2x NVIDIA H200 ~141 GB each',
    backends: 'NLLB on MPS/CUDA; small LLM LoRA demos on MLX',
    remoteBlocker: 'Often full of root vLLM (DeepSeek + Chandra). Train only when free or owner-approved carve-out. Never kill vLLM without OK.',
  },

  tracks: {
    D: {
      name: 'Track D',
      label: 'Defaults',
      idea: 'Stock NLLB-600M native tokenizer + LoRA. Safety net: quality without tokenizer surgery.',
      production: true,
    },
    C: {
      name: 'Track C',
      label: 'Custom vocab',
      idea: 'Domain SPM freeze, then adapt a model to that vocab (C1b resize / C1c extend / C1a from-scratch).',
      production: false,
    },
  },

  c1Menu: [
    { id: 'C1c', pick: true, body: 'Vocab-extend NLLB with verified legal pieces + LoRA (chosen primary)' },
    { id: 'C1b', pick: false, body: 'Resize emb of small pretrained model to custom SPM' },
    { id: 'C1a', pick: false, body: 'Enc-dec from scratch with joint_full_41000 -- stopped; weak on 80k' },
  ],

  executionOrder: [
    'Freeze data contract (docs + dual-track plan)',
    'C0: SPM v2 corpus + train grid + freeze joint_full_41000',
    'D: zero-shot NLLB on I + E',
    'D: Stage A1 -> A2 LoRA on H200 DDP',
    'D: Stage B (assignment) + anti-forget check',
    'C1c: bulk v1 ablation + careful v2 + full I+E',
    'Head-to-head table; production = D A2',
    'Gate 17: assignment report',
  ],

  systems: [
    {
      id: 'zero_shot',
      label: 'Zero-shot NLLB',
      track: 'D',
      I: { bleu: 18.85, chrf: 44.74 },
      milpac: { bleu: 34.28, chrf: 55.22 },
      anuvaad: { bleu: 39.39, chrf: 60.08 },
    },
    {
      id: 'A1',
      label: 'D A1 LoRA',
      track: 'D',
      I: { bleu: 21.67, chrf: 49.16 },
      milpac: { bleu: 34.66, chrf: 55.98 },
      anuvaad: { bleu: 45.17, chrf: 64.33 },
    },
    {
      id: 'A2',
      label: 'D A2 LoRA',
      track: 'D',
      production: true,
      I: { bleu: 21.86, chrf: 49.66 },
      milpac: { bleu: 34.9, chrf: 56.46 },
      anuvaad: { bleu: 45.8, chrf: 64.83 },
      run: 'nllb600_A_A2_h200_A2_ddp2_20260726T212958Z',
    },
    {
      id: 'B',
      label: 'D Stage B',
      track: 'D',
      I: { bleu: 23.1, chrf: 48.89 },
      milpac: { bleu: 30.92, chrf: 51.22 },
      anuvaad: { bleu: 40.44, chrf: 59.6 },
      note: 'I BLEU up; E anti-forget fail (MILPaC -5.24 chrF)',
    },
    {
      id: 'c1c_v2',
      label: 'C1c v2 careful',
      track: 'C',
      I: { bleu: 17.79, chrf: 43.86 },
      milpac: { bleu: 28.2, chrf: 49.78 },
      anuvaad: { bleu: 37.64, chrf: 58.46 },
      note: 'Below zero-shot on all suites',
    },
    {
      id: 'c1c_v1',
      label: 'C1c v1 bulk',
      track: 'C',
      I: { bleu: 6.38, chrf: 24.86 },
      milpac: { bleu: 10.66, chrf: 28.63 },
      anuvaad: { bleu: 15.65, chrf: 34.35 },
      note: 'Failed bulk extend ablation',
    },
  ],

  journeyD: [
    {
      phase: 'Zero-shot',
      body: 'NLLB-200 distilled 600M on full I + E (MPS then H200 re-check). Sets the bar every LoRA run must clear.',
    },
    {
      phase: 'A1',
      body: '80k curriculum (MILPaC + Anuvaad sample, judiciary cap). decoder_attn LoRA r=16. Local MPS first; H200 DDP for speed.',
    },
    {
      phase: 'A2',
      body: 'Resume A1; 150k scale; LR 5e-5. Best joint dual-policy point. Production checkpoint.',
    },
    {
      phase: 'B',
      body: 'Assignment train 1136, resume A2, 800 steps. I BLEU rises; E collapses without replay mix. Ablation only.',
    },
  ],

  journeyC: [
    {
      phase: 'C0 freeze',
      body: 'joint_full_41000 domain SPM. Full joint on 16GB via dedupe. 48k/64k packing ablations only.',
    },
    {
      phase: 'C1a stop',
      body: 'Marian from-scratch scaffold on H200 stopped: 80k bitext cannot beat NLLB priors for quality.',
    },
    {
      phase: 'C1c v1',
      body: 'Bulk +8k SPM pieces + full emb train. Breaks good NLLB singles. Test catastrophic.',
    },
    {
      phase: 'C1c v2',
      body: 'Careful +1500, probe guards, emb grad mask, new_embed_rows.pt save fix. Still loses zero-shot after A1.',
    },
  ],

  opsH200: [
    {
      title: 'Co-residence',
      body: '2xH200 often full of root vLLM. Training only when free or util carved. Never kill production inference without owner OK.',
    },
    {
      title: 'DDP hang',
      body: 'torch.compile + DDP aborted NCCL after first loss_eval. Fix: compile off when world>1; unwrap for eval; broadcast_buffers=False.',
    },
    {
      title: 'Accuracy contract',
      body: 'Same data, LoRA surface, steps, dual-policy metric on MPS and CUDA. Dual GPU halves wall clock; does not invent a new experiment.',
    },
    {
      title: 'Emb checkpoint',
      body: 'C1c v2 grad-mask emb not in PEFT adapter. Save new_embed_rows.pt; retrain before scoring.',
    },
  ],

  decisions: [
    {
      id: 'prod',
      title: 'Production checkpoint',
      choice: 'Track D A2 best_primary',
      why: 'Best dual-policy I+E. Stage B fails E anti-forget. C1c v2 loses zero-shot; v1 fails hard.',
    },
    {
      id: 'ocr',
      title: 'Hindi extraction',
      choice: 'Tesseract OCR (pdftotext fallback)',
      why: 'Text layer mangles Devanagari; 5 backends compared.',
    },
    {
      id: 'align',
      title: 'Alignment',
      choice: 'LaBSE mutual-best',
      why: 'More pairs than BGE-M3 end-to-end (1458 vs 1347); filters hold quality.',
    },
    {
      id: 'split',
      title: 'Assignment split',
      choice: 'Document-level seed 42',
      why: 'Prevents judgment leakage across train/test.',
    },
    {
      id: 'spm',
      title: 'Track C SPM freeze',
      choice: 'joint_full 41k Unigram',
      why: '64k packs slightly better; 41k balances emb size and overfit risk.',
    },
    {
      id: 'lora',
      title: 'LoRA surface',
      choice: 'decoder_attn profile',
      why: 'Cross-attn is the MT hinge; ~0.5% params; avoid emb LoRA on Track D.',
    },
    {
      id: 'stageB',
      title: 'Stage B outcome',
      choice: 'Keep as ablation',
      why: 'I BLEU +1.2 vs A2 but MILPaC chrF drop 5.24 > limit 2.0.',
    },
    {
      id: 'c1',
      title: 'Track C1 path',
      choice: 'C1c over C1a',
      why: 'From-scratch on 80k is weak; extend NLLB carefully (then measured negative).',
    },
    {
      id: 'compile',
      title: 'H200 DDP',
      choice: 'torch.compile off under world>1',
      why: 'NCCL abort after first loss_eval with Dynamo.',
    },
    {
      id: 'emb',
      title: 'C1c emb save',
      choice: 'new_embed_rows.pt',
      why: 'PEFT LoRA-only adapters drop grad-mask emb rows.',
    },
  ],

  artifacts: [
    { path: 'data/processed/{train,dev,test}.jsonl', role: 'Assignment splits' },
    { path: 'data/external/parallel/stage_a_train.jsonl', role: 'Stage A train after E holdout' },
    { path: 'data/models/tokenizers/sentencepiece_legal_v2_joint_full_41000.model', role: 'Track C0 freeze' },
    { path: 'data/runs/...A2.../checkpoints/best_primary', role: 'Production adapters' },
    { path: 'data/analysis/final_dual_policy_report.json', role: 'Closed dual-policy dump' },
    { path: 'docs/EXPERIMENTS.md', role: 'Public tables and freezes' },
    { path: 'DESIGN_DECISIONS.md', role: 'Why each choice (26 sections)' },
    { path: 'docs/TRAINING_STRATEGY.md', role: 'Train contract T0-T5 + C1c' },
    { path: 'CHANGELOG.md', role: 'Execution history' },
    { path: '.local/dual_track_experiment_plan.md', role: 'Private adopted plan (not committed)' },
  ],

  chapters: [
    { id: 'open', label: 'Open' },
    { id: 'mode', label: 'Mode' },
    { id: 'pillars', label: 'Pillars' },
    { id: 'plan', label: 'Plan' },
    { id: 'corpus', label: 'Corpus' },
    { id: 'pipeline', label: 'Pipeline' },
    { id: 'skips', label: 'Skips' },
    { id: 'stage-a', label: 'Stage A' },
    { id: 'eval', label: 'Eval' },
    { id: 'tokens', label: 'Tokens' },
    { id: 'tracks', label: 'Tracks' },
    { id: 'track-d', label: 'Track D' },
    { id: 'track-c', label: 'Track C' },
    { id: 'ops', label: 'Ops' },
    { id: 'scoreboard', label: 'Scores' },
    { id: 'decision', label: 'Decision' },
    { id: 'atlas', label: 'Atlas' },
    { id: 'close', label: 'Close' },
  ],
};

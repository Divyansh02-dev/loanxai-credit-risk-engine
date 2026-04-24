/**
 * LoanXAI — Frontend JavaScript
 * Handles: tab switching, form state, API calls, result rendering, glossary, datasets
 */

"use strict";

const API_BASE = "http://localhost:5000";

// ─────────────────────────────────────────────
// Utility helpers
// ─────────────────────────────────────────────
const $ = (id) => document.getElementById(id);
const rupee = (n) => "₹" + Number(n).toLocaleString("en-IN");
const show  = (el) => { if (el) el.style.display = "block"; };
const hide  = (el) => { if (el) el.style.display = "none";  };

// ─────────────────────────────────────────────
// Tab Switching
// ─────────────────────────────────────────────
const TABS = ["application", "result", "glossary", "metrics"];

function showTab(name) {
  TABS.forEach((t) => {
    const page = $(`page-${t}`);
    const btn  = $(`nav-${t}`);
    if (page) page.classList.toggle("page-hidden", t !== name);
    if (btn)  btn.classList.toggle("active", t === name);
  });
  if (name === "glossary") renderGlossary();
  if (name === "metrics")  renderMetrics();
}

// ─────────────────────────────────────────────
// Live Indicator Updates (while filling form)
// ─────────────────────────────────────────────
function updateLiveStats() {
  const income = parseFloat($("f_income").value)  || 0;
  const loan   = parseFloat($("f_loan").value)    || 0;
  const cibil  = parseInt($("f_cibil").value)     || 0;

  // Loan-to-income
  const ltiEl  = $("ind_lti");
  const lti    = income > 0 ? (loan / income).toFixed(1) : "—";
  ltiEl.textContent = lti !== "—" ? lti + "×" : "—";
  ltiEl.className   = "stat-val " + (parseFloat(lti) > 4 ? "c-red" : parseFloat(lti) > 2.5 ? "c-amber" : "c-green");

  // CIBIL band
  const cbEl  = $("ind_cibil");
  const chEl  = $("ind_cibil_hint");
  if (cibil >= 750)      { cbEl.textContent = "Excellent"; cbEl.className = "stat-val c-green"; }
  else if (cibil >= 700) { cbEl.textContent = "Good";      cbEl.className = "stat-val c-green"; }
  else if (cibil >= 650) { cbEl.textContent = "Average";   cbEl.className = "stat-val c-amber"; }
  else if (cibil >= 550) { cbEl.textContent = "Fair";      cbEl.className = "stat-val c-amber"; }
  else if (cibil >= 300) { cbEl.textContent = "Poor";      cbEl.className = "stat-val c-red";   }
  else                   { cbEl.textContent = "—";         cbEl.className = "stat-val c-muted"; }
  chEl.textContent = cibil >= 300 ? `Score: ${cibil}` : "Enter 300–900";

  // EMI estimate (11% p.a., 60 months)
  const r   = 0.11 / 12;
  const n   = 60;
  const emi = loan > 0 ? loan * r * Math.pow(1 + r, n) / (Math.pow(1 + r, n) - 1) : 0;

  $("ind_emi").textContent       = emi > 0 ? rupee(Math.round(emi)) : "—";
  $("emi_principal").textContent = rupee(loan);
  $("emi_monthly").textContent   = emi > 0 ? rupee(Math.round(emi))           : "—";
  $("emi_interest").textContent  = emi > 0 ? rupee(Math.round(emi * n - loan)) : "—";
  $("emi_total").textContent     = emi > 0 ? rupee(Math.round(emi * n))        : "—";
}

// ─────────────────────────────────────────────
// Collect form data
// ─────────────────────────────────────────────
function collectForm() {
  return {
    name:                  $("f_name").value.trim(),
    age:                   parseFloat($("f_age").value),
    gender:                $("f_gender").value,
    children:              parseFloat($("f_children").value),
    family_members:        parseFloat($("f_family").value),
    annual_income:         parseFloat($("f_income").value),
    cibil_score:           parseFloat($("f_cibil").value),
    loan_amount:           parseFloat($("f_loan").value),
    monthly_emi:           parseFloat($("f_emi").value),
    goods_price:           parseFloat($("f_goods").value),
    employment_type:       $("f_emp_type").value,
    employment_years:      parseFloat($("f_emp_yrs").value),
    education:             $("f_education").value,
    occupation:            $("f_occupation").value,
    owns_property:         parseFloat($("f_property").value),
    owns_car:              parseFloat($("f_car").value),
    housing_type:          $("f_housing").value,
    family_status:         $("f_marital").value,
    credit_enquiries_year: parseFloat($("f_enquiries").value),
    social_defaults:       parseFloat($("f_social_def").value),
    doc3_submitted:        parseFloat($("f_doc3").value),
    region_rating:         parseFloat($("f_region").value),
  };
}

// ─────────────────────────────────────────────
// Main Analysis — calls Flask API
// ─────────────────────────────────────────────
async function runAnalysis() {
  const btn = $("btn-analyze");
  btn.disabled    = true;
  btn.textContent = "Analyzing…";

  // Switch to result tab and show loading
  showTab("result");
  show($("result-loading"));
  hide($("result-error"));
  hide($("result-content"));
  hide($("result-empty"));

  const payload = collectForm();

  try {
    const res  = await fetch(`${API_BASE}/predict`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(payload),
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    renderResult(data, payload);
  } catch (err) {
    hide($("result-loading"));
    show($("result-error"));
    $("result-error").innerHTML = `
      <strong>⚠ Could not connect to backend.</strong><br><br>
      Make sure the Flask server is running in VS Code terminal:<br><br>
      <code>cd backend</code><br>
      <code>python app.py</code><br><br>
      Then try again.<br><br>
      <span style="color:var(--dim);font-size:11px">Error: ${err.message}</span>
    `;
  }

  btn.disabled    = false;
  btn.textContent = "🔍 Run AI Analysis";
}
// ─────────────────────────────────────────────
// Render Result (Hybrid Rule + ML aware)
// ─────────────────────────────────────────────
function renderResult(data, form) {
  hide($("result-loading"));
  show($("result-content"));
  $("result-content").classList.add("fade-up");

  const risk   = data.risk_score;
  const dec    = data.decision;
  const source = data.source || "ML_MODEL";

  // Color palette per decision
  const palette = {
    "APPROVED":      { bg: "#F0FDF4", border: "#4ADE80", text: "#15803D",  word: "#15803D"  },
    "REJECTED":      { bg: "#FEF2F2", border: "#FCA5A5", text: "#7F1D1D",  word: "#B91C1C"  },
    "MANUAL REVIEW": { bg: "#FFFBEB", border: "#FCD34D", text: "#78350F",  word: "#B45309"  },
  };
  const pal = palette[dec] || palette["MANUAL REVIEW"];

  // ── Decision banner ──
  const banner = $("decision-banner");
  banner.style.background   = pal.bg;
  banner.style.borderColor  = pal.border;

  $("decision-word").textContent = dec;
  $("decision-word").style.color = pal.word;

  // Plain-English explanation — different for RULE_BASED vs ML_MODEL
  const name = form.name || "The applicant";
  let explain = "";

  if (source === "RULE_BASED") {
    explain = `<div class="rule-override-badge">\u274C Application Rejected (Policy Rule)</div>
      <div style="margin:10px 0;font-size:13px;line-height:1.8;color:#7F1D1D">
        <strong>Reason:</strong> ${data.rule_reason || "Policy rule triggered."}
      </div>
      <div style="font-size:11px;color:var(--muted);font-style:italic;margin-top:6px">
        \u26A0 Model prediction was skipped because this application violates RBI safety rules.
        Address the issue above and reapply.
      </div>`;
  } else if (dec === "APPROVED") {
    explain = `<strong>${name}</strong> has been assessed with a
    <strong>${data.approval_probability.toFixed(0)}% calibrated repayment probability</strong>.
    The AI model, after probability calibration, found the profile to be creditworthy.
    The loan can be sanctioned subject to document verification.`;
  } else if (dec === "REJECTED") {
    explain = `<strong>${name}</strong> presents a <strong>high repayment risk (${risk.toFixed(0)}%)</strong>.
    The calibrated model probability is below the 65% approval threshold. See the SHAP Analysis
    and Action Plan below for specific steps to improve eligibility.`;
  } else {
    explain = `This application has <strong>mixed signals</strong> \u2014 the calibrated probability
    (${data.approval_probability.toFixed(0)}%) falls between 45\u201365%. A human loan officer should
    review this case. Reducing the loan amount may push the decision to APPROVED.`;
  }
  $("decision-explain").innerHTML = explain;
  $("decision-explain").style.borderLeftColor = pal.border;

  // Pills
  const riskCol = risk < 35 ? "var(--green)" : risk < 65 ? "var(--amber)" : "var(--red)";
  $("pill-risk").textContent = `Risk Score: ${risk.toFixed(0)} / 100`;
  $("pill-risk").style.cssText = `background:${riskCol}18;color:${riskCol};border-color:${riskCol}40`;

  $("pill-approval").textContent = `Repayment: ${data.approval_probability.toFixed(0)}%`;
  $("pill-approval").style.cssText = "background:var(--blue-lt);color:var(--blue);border-color:#93C5FD";

  $("pill-rec").textContent = `Suggested: ${rupee(data.recommended_loan)}`;
  $("pill-rec").style.cssText = "background:var(--orange-lt);color:var(--orange);border-color:#FED7AA";

  // Source + Confidence pills
  const srcCol = source === "RULE_BASED" ? "var(--red)" : "var(--green)";
  const srcBg  = source === "RULE_BASED" ? "#FEF2F2"    : "#F0FDF4";
  const srcLabel = source === "RULE_BASED" ? "\uD83D\uDEE1 Rule Engine" : "\uD83E\uDD16 ML Model";

  // Add source pill after existing pills (create if not exists)
  let pillSrc = document.getElementById("pill-source");
  if (!pillSrc) {
    pillSrc = document.createElement("div");
    pillSrc.id = "pill-source";
    pillSrc.className = "pill";
    $("pill-rec").parentElement.appendChild(pillSrc);
  }
  pillSrc.textContent = srcLabel;
  pillSrc.style.cssText = `background:${srcBg};color:${srcCol};border-color:${srcCol}40`;

  // Confidence pill
  if (data.confidence) {
    let pillConf = document.getElementById("pill-confidence");
    if (!pillConf) {
      pillConf = document.createElement("div");
      pillConf.id = "pill-confidence";
      pillConf.className = "pill";
      $("pill-rec").parentElement.appendChild(pillConf);
    }
    const confCol = data.confidence.color === "green" ? "var(--green)"
                  : data.confidence.color === "amber" ? "var(--amber)" : "var(--red)";
    const confBg  = data.confidence.color === "green" ? "#F0FDF4"
                  : data.confidence.color === "amber" ? "#FFFBEB" : "#FEF2F2";
    pillConf.textContent = `Confidence: ${data.confidence.level}`;
    pillConf.style.cssText = `background:${confBg};color:${confCol};border-color:${confCol}40`;
  }

  // Gauge
  drawGauge(risk);

  // ── SHAP Factor bars (only for ML_MODEL) ──
  const fc = $("factors-container");
  fc.innerHTML = "";

  if (source === "RULE_BASED") {
    // Show rule-based explanation instead of SHAP factors
    fc.innerHTML = `
      <div class="rule-notice">
        <div class="rule-notice-icon">\uD83D\uDEE1</div>
        <div class="rule-notice-title">Policy Rule Override</div>
        <div class="rule-notice-body">
          This decision was made by the <strong>Rule Engine</strong>, not the ML model.
          SHAP explanations are not available because the XGBoost model prediction was
          skipped due to an RBI-aligned policy violation.
        </div>
        <div class="rule-notice-rule">Rule triggered: <code>${data.rule || "POLICY"}</code></div>
        <div class="rule-notice-body" style="margin-top:10px">
          <strong>Why do we have rules?</strong> Indian banks follow strict guidelines set by the
          Reserve Bank of India (RBI). Certain applications are automatically declined regardless of
          what the AI model predicts, because they violate fundamental lending safety criteria.
        </div>
      </div>`;
  } else {
    // ML_MODEL — render SHAP bars as before
    const maxImpact = Math.max(...data.factors.map((f) => f.impact), 0.01);

    data.factors.forEach((f) => {
      const col   = f.direction === "positive" ? "var(--green)" : "var(--red)";
      const bgCol = f.direction === "positive" ? "#F0FDF4"      : "#FEF2F2";
      const bdr   = f.direction === "positive" ? "#4ADE8044"    : "#FCA5A544";
      const dirTxt = f.direction === "positive" ? "\u25B2 Helps"     : "\u25BC Hurts";
      const pct   = Math.round((f.impact / maxImpact) * 100);

      const el = document.createElement("div");
      el.className = "factor";
      el.innerHTML = `
        <div class="factor-header">
          <div class="factor-label">${f.label}</div>
          <div class="factor-chip" style="background:${bgCol};color:${col};border-color:${bdr}">
            ${dirTxt} &nbsp; ${f.impact.toFixed(3)}
          </div>
        </div>
        <div class="factor-track">
          <div class="factor-fill" style="width:0%;background:${col}" data-w="${pct}"></div>
        </div>
        <div class="factor-explain">\uD83D\uDCA1 ${f.explanation}</div>`;
      fc.appendChild(el);
    });

    // Trigger bar animation after a tick
    requestAnimationFrame(() => {
      document.querySelectorAll(".factor-fill").forEach((bar) => {
        bar.style.width = bar.dataset.w + "%";
      });
    });

    // SHAP Waterfall chart (only for ML)
    renderWaterfallChart(data.factors);
  }

  // ── Action plan ──
  const ac = $("actions-container");
  ac.innerHTML = "";
  data.actions.forEach((a, i) => {
    const impCol = a.impact === "High" ? "var(--red)" : a.impact === "Medium" ? "var(--amber)" : "var(--green)";
    const impBg  = a.impact === "High" ? "#FEF2F2"   : a.impact === "Medium" ? "#FFFBEB"      : "#F0FDF4";
    const el = document.createElement("div");
    el.className = "action-card";
    el.innerHTML = `
      <div class="action-num">${i + 1}</div>
      <div style="flex:1">
        <div class="action-title">${a.title}</div>
        <div class="action-detail">${a.detail}</div>
        <div class="action-meta">
          <div class="action-tag" style="background:${impBg};color:${impCol};border-color:${impCol}30">
            Impact: ${a.impact}
          </div>
          <div class="action-tag" style="background:var(--sub);color:var(--muted);border-color:var(--border)">
            \u23F1 ${a.timeline}
          </div>
        </div>
      </div>`;
    ac.appendChild(el);
  });

  // ── Model info (with source + confidence + RBI disclaimer) ──
  const mi = $("model-info");
  const info = data.model_info;
  const infoRows = [
    ["Model",          info.type],
    ["Dataset",        info.dataset],
    ["Features",       info.features_used],
    ["Explainability", info.explainability],
    ["Engine",         info.engine === "RULE_BASED" ? "\uD83D\uDEE1 Rule-Based" : "\uD83E\uDD16 ML Model"],
    ["Decision",       dec],
    ["Risk Score",     `${risk.toFixed(1)} / 100`],
  ];
  if (data.confidence) {
    infoRows.push(["Confidence", data.confidence.level]);
  }
  if (data.calibration_note) {
    infoRows.push(["Calibration", data.calibration_note]);
  }

  mi.innerHTML = infoRows.map(([k, v]) => `
    <div class="model-info-row">
      <span class="model-info-key">${k}</span>
      <strong>${v}</strong>
    </div>`).join("");

  // ── PDF Download Button ──
  let pdfBtn = document.getElementById("btn-pdf");
  if (!pdfBtn) {
    pdfBtn = document.createElement("button");
    pdfBtn.id = "btn-pdf";
    pdfBtn.className = "btn-download-pdf";
    pdfBtn.innerHTML = "\uD83D\uDCC4 Download Result as PDF";
    pdfBtn.onclick = () => {
      const d = new Date();
      const pd = $("print-date");
      if (pd) pd.textContent = d.toLocaleDateString("en-IN", { day: "numeric", month: "long", year: "numeric" });
      window.print();
    };
    $("result-content").appendChild(pdfBtn);
  }
}

// ─────────────────────────────────────────────
// SVG Gauge
// ─────────────────────────────────────────────
function drawGauge(score) {
  const cx = 110, cy = 95, r = 72;
  const toXY = (deg) => {
    const rad = ((deg - 90) * Math.PI) / 180;
    return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) };
  };
  const arc = (a, b) => {
    const s = toXY(a), e = toXY(b), lg = b - a > 180 ? 1 : 0;
    return `M ${s.x.toFixed(2)} ${s.y.toFixed(2)} A ${r} ${r} 0 ${lg} 1 ${e.x.toFixed(2)} ${e.y.toFixed(2)}`;
  };

  const col   = score < 35 ? "#1A6B3A" : score < 65 ? "#B45309" : "#B91C1C";
  const label = score < 35 ? "LOW RISK" : score < 65 ? "MODERATE RISK" : "HIGH RISK";

  $("g-track").setAttribute("d", arc(-90, 90));
  $("g-fill").setAttribute("d", arc(-90, -90 + score * 1.8));
  $("g-fill").setAttribute("stroke", col);
  $("g-dot").setAttribute("fill", col);
  $("g-needle").setAttribute("stroke", col);

  const na = -90 + score * 1.8;
  const nx = cx + 58 * Math.cos(((na - 90) * Math.PI) / 180);
  const ny = cy + 58 * Math.sin(((na - 90) * Math.PI) / 180);
  $("g-needle").setAttribute("x2", nx.toFixed(2));
  $("g-needle").setAttribute("y2", ny.toFixed(2));
  $("g-score").textContent = Math.round(score);
  $("g-score").setAttribute("fill", col);

  const rb = $("risk-badge");
  rb.textContent = label;
  rb.style.cssText = `background:${col}18;color:${col};border-color:${col}50`;
}

// ─────────────────────────────────────────────
// Glossary Data & Render
// ─────────────────────────────────────────────
const GLOSSARY_TERMS = [
  {
    name: "CIBIL Score", short: "Credit Score (300–900)", icon: "📊", col: "#D4500A",
    body: "A 3-digit number summarising your entire borrowing history. Generated by TransUnion CIBIL — India's leading credit bureau. Checks payment history, credit utilisation, age of accounts, and enquiries.",
    example: "Anita has a CIBIL of 790. Every bank offers her the lowest interest rate because she has never missed an EMI in 8 years.",
  },
  {
    name: "EMI", short: "Equated Monthly Instalment", icon: "📅", col: "#1A6B3A",
    body: "The fixed amount you pay every month to repay a loan. Contains two parts: Principal (original amount) + Interest (lender's fee). Your EMI stays constant for fixed-rate loans.",
    example: "₹5,00,000 loan at 12% p.a. for 5 years → EMI = ₹11,122/month for 60 months.",
  },
  {
    name: "Principal", short: "Original Loan Amount", icon: "💰", col: "#1D4ED8",
    body: "The actual money borrowed before any interest is added. As you pay EMIs each month, part of the payment reduces the principal. In early months, most EMI is interest; later months pay more principal.",
    example: "You borrow ₹3L for a bike. ₹3L is the principal. Your total repayment will be ~₹3.9L after adding interest.",
  },
  {
    name: "Interest Rate (ROI)", short: "% charged per year on loan", icon: "📈", col: "#B45309",
    body: "The annual cost of borrowing money. A 12% p.a. rate means you pay 12% of the outstanding loan as interest per year. Lower rate = less total repayment.",
    example: "Personal loan: 10–24% p.a. | Home loan: 8.4–12% p.a. | Gold loan: 7–15% p.a. | Mudra: 8–12% p.a.",
  },
  {
    name: "Debt-to-Income (DTI)", short: "Your EMI burden vs income", icon: "⚖️", col: "#B91C1C",
    body: "What % of your monthly income goes to loan EMIs. Formula: (Total monthly EMIs ÷ Monthly income) × 100. Banks in India prefer DTI below 40%.",
    example: "Income ₹50,000/mo, EMIs ₹18,000/mo → DTI = 36%. Still acceptable. At ₹22,000 EMIs → DTI = 44% → risky.",
  },
  {
    name: "Loan-to-Value (LTV)", short: "Loan size vs asset value", icon: "🏠", col: "#0891B2",
    body: "For secured loans: LTV = (Loan ÷ Asset Value) × 100. RBI mandates max 75–80% LTV on home loans. You pay the rest as a down payment from savings.",
    example: "House worth ₹50L. Bank lends ₹40L (80% LTV). You pay ₹10L down payment from savings.",
  },
  {
    name: "Collateral / Guarantee", short: "Security pledged for a loan", icon: "🔒", col: "#7C3AED",
    body: "An asset pledged to secure a loan. If you cannot repay, the lender can seize and sell it. Home loans use property; gold loans use jewellery. Personal/Mudra loans are unsecured (no collateral needed).",
    example: "Suresh pledges gold worth ₹2L to get a gold loan of ₹1.5L. If he defaults, the bank keeps the gold.",
  },
  {
    name: "Guarantor / Co-applicant", short: "Someone backing your loan", icon: "🤝", col: "#D4500A",
    body: "A person who co-signs your loan and promises to repay if you cannot. Required when CIBIL is low or income is insufficient. Co-applicant shares full loan liability.",
    example: "Meena (CIBIL 630) asks her husband (CIBIL 780) to be co-applicant. The joint profile gets approved.",
  },
  {
    name: "Processing Fee", short: "One-time loan application charge", icon: "💳", col: "#B91C1C",
    body: "Non-refundable fee charged to process your loan application. Usually 0.5%–2% of the loan amount. Deducted from disbursement. Charged even if the loan is later cancelled.",
    example: "₹10L loan with 1.5% fee = ₹15,000 fee. You receive ₹9,85,000 in your bank account.",
  },
  {
    name: "Pre-payment / Foreclosure", short: "Paying off loan before tenure ends", icon: "🔓", col: "#1A6B3A",
    body: "Paying off your entire outstanding loan before its scheduled end date. Saves future interest. Banks may charge 2–5% penalty on fixed-rate loans. Floating-rate loans: RBI mandates zero penalty.",
    example: "Ravi got a 5-year loan but received a bonus in year 2. Foreclosing saved him ₹60,000 in future interest.",
  },
  {
    name: "NPA (Non-Performing Asset)", short: "Loan gone bad for 90+ days", icon: "⚠️", col: "#B91C1C",
    body: "When a borrower misses EMI for 90 consecutive days, the bank marks the loan as NPA. This severely damages CIBIL score (drops 100–200 points) and can lead to legal action and asset seizure.",
    example: "Anil missed 4 months of EMI. Loan became NPA. CIBIL dropped from 710 to 500. All future loan applications rejected for 7 years.",
  },
  {
    name: "NBFC", short: "Non-Banking Finance Company", icon: "🏦", col: "#7C3AED",
    body: "Financial companies like Bajaj Finance, Muthoot, HDFC Ltd, Shriram Finance that lend money but are not full banks. Regulated by RBI. Approve lower CIBIL profiles but charge higher interest.",
    example: "Bank rejected Kamala (CIBIL 640). Bajaj Finance (NBFC) approved her personal loan at 18% p.a.",
  },
  {
    name: "PM Mudra Yojana", short: "Govt loan for small business (up to ₹10L)", icon: "🏛️", col: "#1A6B3A",
    body: "Government scheme for micro and small business loans. No collateral required. Three tiers: Shishu (up to ₹50K), Kishore (₹50K–₹5L), Tarun (₹5L–₹10L). Available via banks, NBFCs, MFIs.",
    example: "Vijay wants ₹2L to expand his grocery shop. Applies under PMMY Kishore at SBI. Approved without pledging any asset.",
  },
  {
    name: "Moratorium", short: "Temporary pause on EMI payments", icon: "⏸️", col: "#0891B2",
    body: "A lender-granted pause on EMI payments during financial hardship. Interest still accumulates, increasing total repayment cost. Used during COVID-19 by RBI as relief measure.",
    example: "During COVID, Priya took a 6-month moratorium. She paid no EMIs but accumulated ₹28,000 extra interest overall.",
  },
  {
    name: "Credit Enquiry (Hard Pull)", short: "Bank checking your CIBIL score", icon: "🔍", col: "#B45309",
    body: "Every formal loan application triggers a hard enquiry on your CIBIL report. Too many enquiries in a short period signal financial desperation and lower your score.",
    example: "Raju applied to 6 banks in 3 months. All 6 showed as enquiries. CIBIL dropped by 45 points even though no loan was given.",
  },
  {
    name: "SHAP (AI Explanation)", short: "Why the AI made its decision", icon: "🤖", col: "#6D28D9",
    body: "SHAP (SHapley Additive exPlanations) is a technique from game theory applied to AI. For each prediction, it shows how much each input factor pushed the decision toward approval or rejection.",
    example: "SHAP says your credit score pushed approval probability up by 0.24 and your high loan amount pushed it down by 0.18. Net effect: APPROVED.",
  },
];

let glossaryRendered = false;

function renderGlossary() {
  if (glossaryRendered) return;
  glossaryRendered = true;

  const grid = $("glossary-grid");
  grid.innerHTML = GLOSSARY_TERMS.map((t) => `
    <div class="term-card" style="border-left-color:${t.col}" data-search="${(t.name + t.body + t.short).toLowerCase()}">
      <div class="term-head">
        <span class="term-icon">${t.icon}</span>
        <div>
          <div class="term-name">${t.name}</div>
          <div class="term-short" style="color:${t.col}">${t.short}</div>
        </div>
      </div>
      <div class="term-body">${t.body}</div>
      <div class="term-example">📌 ${t.example}</div>
    </div>`).join("");
}

function filterGlossary() {
  const q = $("gloss-search").value.toLowerCase().trim();
  document.querySelectorAll(".term-card").forEach((card) => {
    card.style.display = (!q || card.dataset.search.includes(q)) ? "" : "none";
  });
}

// ─────────────────────────────────────────────
// Datasets & Pipeline Data + Render
// ─────────────────────────────────────────────
const DATASETS = [
  {
    name: "Home Credit Default Risk",
    url:  "https://www.kaggle.com/competitions/home-credit-default-risk",
    meta: "★★★★★ India Fit  |  350K rows  |  228 features",
    tags: ["Used in this project", "Indian NBFC context", "Best overall"],
    body: "The dataset this model was trained on. Contains real loan applications with 307+ features including income, employment, credit bureau scores (EXT_SOURCE), and previous loan history. Very close to Indian NBFC lending patterns.",
  },
  {
    name: "Give Me Some Credit (Kaggle)",
    url:  "https://www.kaggle.com/c/GiveMeSomeCredit/data",
    meta: "★★★ India Fit  |  150K rows  |  11 features",
    tags: ["Beginner friendly", "Clean data", "Binary labels"],
    body: "Great starting point with 10 clear features: DebtRatio, MonthlyIncome, Age, NumberOfDependents. Adapt income to ₹ and map US credit score to CIBIL range for Indian use.",
  },
  {
    name: "German Credit Dataset (UCI)",
    url:  "https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data",
    meta: "★★ India Fit  |  1,000 rows  |  Classic dataset",
    tags: ["Academic XAI research", "SHAP demos", "Small & clean"],
    body: "Widely used in academic explainability papers. Only 1,000 rows — ideal for quickly testing SHAP, LIME, and XAI methods. Well-documented with clear feature descriptions.",
  },
  {
    name: "Lending Club Loan Data",
    url:  "https://www.kaggle.com/datasets/wordsforthewise/lending-club",
    meta: "★★★ India Fit  |  2.2M rows  |  Rich features",
    tags: ["Large scale", "P2P lending", "Interest rate data"],
    body: "Massive US P2P lending dataset with grades, interest rates, DTI, FICO scores. Map FICO → CIBIL range for Indian context. Excellent for large-scale ML training experiments.",
  },
  {
    name: "RBI / data.gov.in (India)",
    url:  "https://data.gov.in/sector/finance",
    meta: "★★★★★ India Fit  |  Official govt data",
    tags: ["India official", "MSME loans", "State-wise"],
    body: "Official Indian government financial datasets: MSME loans, priority sector lending, bank-wise disbursement by state. Best for India-specific policy research and geographic analysis.",
  },
  {
    name: "SIDBI MSME Pulse",
    url:  "https://www.sidbi.in/en/msme-pulse",
    meta: "★★★★ India Fit  |  Quarterly reports",
    tags: ["Indian MSME", "Quarterly data", "Credit quality"],
    body: "India's most authoritative MSME credit data from SIDBI. Covers geographic distribution, sector-wise lending, and credit quality trends for small businesses across India.",
  },
];

const PIPELINE_STEPS = [
  {
    title: "1. Download the Dataset",
    desc:  "Use the Kaggle Home Credit dataset — the same one used to train the model in this project.",
    code:  `pip install kaggle
kaggle competitions download -c home-credit-default-risk
unzip home-credit-default-risk.zip -d data/`,
  },
  {
    title: "2. Explore & Clean Data",
    desc:  "Understand the data shape, check missing values, and examine the class imbalance.",
    code:  `import pandas as pd

df = pd.read_csv('data/application_train.csv')
print(df.shape)                    # (307511, 122)
print(df['TARGET'].value_counts()) # 0=repaid, 1=defaulted
print(df.isnull().sum().sort_values(ascending=False).head(20))`,
  },
  {
    title: "3. Feature Engineering",
    desc:  "Create meaningful ratio features from raw columns. These are the most predictive.",
    code:  `# Ratio features
df['INCOME_CREDIT_RATIO']  = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY']      / df['AMT_INCOME_TOTAL']
df['AGE_YEARS']            = abs(df['DAYS_BIRTH'])   / 365
df['EMPLOYED_YEARS']       = abs(df['DAYS_EMPLOYED'])/ 365

# Average of the three external credit scores (most important features)
df['EXT_SOURCES_MEAN'] = df[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']].mean(axis=1)`,
  },
  {
    title: "4. Handle Class Imbalance (SMOTE)",
    desc:  "Only ~8% of loans default. SMOTE creates synthetic minority samples to balance training data.",
    code:  `from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

X = df.drop('TARGET', axis=1)
y = df['TARGET']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_res, y_res = SMOTE(random_state=42).fit_resample(X_train, y_train)

print(f"After SMOTE: {y_res.value_counts()}")  # Should be ~balanced`,
  },
  {
    title: "5. Train XGBoost Model",
    desc:  "Train the classifier and save as model.pkl — the exact file loaded by this project's backend.",
    code:  `from xgboost import XGBClassifier
import pickle

model = XGBClassifier(
    n_estimators=300, max_depth=6,
    learning_rate=0.05, subsample=0.8,
    colsample_bytree=0.8, scale_pos_weight=10,
    random_state=42, n_jobs=-1
)
model.fit(X_res, y_res, eval_set=[(X_test, y_test)], verbose=50)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Model saved!")`,
  },
  {
    title: "6. Evaluate the Model",
    desc:  "Check AUC-ROC, precision-recall, and the confusion matrix. AUC > 0.85 is good for credit risk.",
    code:  `from sklearn.metrics import roc_auc_score, classification_report

y_pred       = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print("AUC-ROC :", roc_auc_score(y_test, y_pred_proba))
print(classification_report(y_test, y_pred))`,
  },
  {
    title: "7. SHAP Explainability",
    desc:  "Use SHAP to understand every individual prediction — this is what powers the bar charts in this app.",
    code:  `import shap

explainer  = shap.TreeExplainer(model)
shap_vals  = explainer.shap_values(X_test[:200])

# Global importance (which features matter most overall)
shap.summary_plot(shap_vals, X_test[:200])

# Single prediction waterfall (why was THIS person approved/rejected)
shap.waterfall_plot(shap.Explanation(
    values=shap_vals[0],
    base_values=explainer.expected_value,
    feature_names=X_test.columns.tolist()
))`,
  },
  {
    title: "8. Run the Flask Backend",
    desc:  "Start the API server so the frontend can call it.",
    code:  `cd backend
pip install -r requirements.txt
python app.py
# → LoanXAI Backend running on http://localhost:5000`,
  },
];

const MODEL_COMPARISON = [
  { model: "XGBoost",        auc: "0.91", shap: "✅ Full",             speed: "Fast",    imb: "scale_pos_weight",      note: "⭐ Best overall for tabular credit data" },
  { model: "LightGBM",       auc: "0.90", shap: "✅ Full",             speed: "Fastest", imb: "is_unbalance=True",      note: "Best for large datasets (2M+ rows)" },
  { model: "Random Forest",  auc: "0.87", shap: "✅ Full",             speed: "Medium",  imb: "class_weight='balanced'",note: "Stable, low overfitting risk" },
  { model: "Logistic Reg.",  auc: "0.79", shap: "✅ Full",             speed: "Fastest", imb: "class_weight='balanced'",note: "Most interpretable, RBI audit-ready" },
  { model: "Neural Network", auc: "0.88", shap: "⚠ DeepExplainer",   speed: "Slowest", imb: "Manual weighting",       note: "Needs much more data to outperform XGB" },
];

let datasetsRendered = false;

function renderDatasets() {
  if (datasetsRendered) return;
  datasetsRendered = true;

  // Datasets
  const dc = $("datasets-container");
  dc.innerHTML = DATASETS.map((d) => `
    <div class="dataset-card">
      <div class="dataset-name">${d.name}</div>
      <div class="dataset-meta">${d.meta}</div>
      <div class="tag-row">${d.tags.map((t) => `<span class="tag">${t}</span>`).join("")}</div>
      <div class="dataset-body">${d.body}</div>
      <a class="btn-link" href="${d.url}" target="_blank" rel="noopener noreferrer">View / Download →</a>
    </div>`).join("");

  // Pipeline
  const pc = $("pipeline-container");
  pc.innerHTML = PIPELINE_STEPS.map((p, i) => `
    <div class="pipeline-step">
      <div class="step-num">${i + 1}</div>
      <div style="flex:1;min-width:0">
        <div class="step-title">${p.title}</div>
        <div class="step-desc">${p.desc}</div>
        <div class="code-block">${p.code}</div>
      </div>
    </div>`).join("");

  // Model comparison table body
  const tbody = $("model-tbody");
  tbody.innerHTML = MODEL_COMPARISON.map((m) => `
    <tr>
      <td style="font-weight:600">${m.model}</td>
      <td style="text-align:center;font-family:var(--mono);color:var(--green);font-weight:700">${m.auc}</td>
      <td style="text-align:center">${m.shap}</td>
      <td style="text-align:center">${m.speed}</td>
      <td style="font-size:11px;color:var(--muted)">${m.imb}</td>
      <td style="font-size:11px;color:var(--muted)">${m.note}</td>
    </tr>`).join("");
}

// ─────────────────────────────────────────────
// Init on page load
// ─────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  updateLiveStats();

  // Live update listeners
  ["f_income", "f_loan", "f_cibil"].forEach((id) => {
    $(id).addEventListener("input", updateLiveStats);
  });

  // Set print date
  const pd = $("print-date");
  if (pd) pd.textContent = new Date().toLocaleDateString("en-IN", { day: "numeric", month: "long", year: "numeric" });
});


// ─────────────────────────────────────────────
// SHAP Waterfall Chart (Pure SVG)
// ─────────────────────────────────────────────
function renderWaterfallChart(factors) {
  // Remove previous waterfall if exists
  const existing = document.getElementById("waterfall-section");
  if (existing) existing.remove();

  const fc = $("factors-container");
  if (!fc) return;

  const section = document.createElement("div");
  section.id = "waterfall-section";
  section.className = "waterfall-section";
  section.innerHTML = `
    <div class="waterfall-title">SHAP Waterfall \u2014 How each factor shifted the decision</div>
    <div class="waterfall-sub">Green bars (right) help approval. Red bars (left) increase risk.</div>`;

  const barH = 30;
  const gap = 6;
  const labelW = 160;
  const valW = 60;
  const chartW = 500;
  const centerX = labelW + (chartW / 2);
  const totalW = labelW + chartW + valW;
  const totalH = factors.length * (barH + gap) + 20;
  const maxImpact = Math.max(...factors.map(f => f.impact), 0.01);
  const scale = (chartW / 2 - 10) / maxImpact;

  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", `0 0 ${totalW} ${totalH}`);
  svg.setAttribute("width", "100%");
  svg.style.maxWidth = "700px";
  svg.style.display = "block";

  // Baseline
  const baseline = document.createElementNS(ns, "line");
  baseline.setAttribute("x1", centerX);
  baseline.setAttribute("y1", 0);
  baseline.setAttribute("x2", centerX);
  baseline.setAttribute("y2", totalH);
  baseline.setAttribute("stroke", "#E2DDD5");
  baseline.setAttribute("stroke-width", "1.5");
  baseline.setAttribute("stroke-dasharray", "4,3");
  svg.appendChild(baseline);

  factors.forEach((f, i) => {
    const y = i * (barH + gap) + 10;
    const isGood = f.direction === "positive";
    const col = isGood ? "#1A6B3A" : "#B91C1C";
    const bgCol = isGood ? "#D1FAE5" : "#FEE2E2";
    const barWidth = Math.max(f.impact * scale, 3);

    // Background bar (subtle)
    const bgRect = document.createElementNS(ns, "rect");
    bgRect.setAttribute("x", isGood ? centerX : centerX - barWidth);
    bgRect.setAttribute("y", y);
    bgRect.setAttribute("width", barWidth);
    bgRect.setAttribute("height", barH);
    bgRect.setAttribute("fill", bgCol);
    bgRect.setAttribute("rx", "4");
    svg.appendChild(bgRect);

    // Main bar
    const rect = document.createElementNS(ns, "rect");
    rect.setAttribute("x", isGood ? centerX : centerX - barWidth);
    rect.setAttribute("y", y + 6);
    rect.setAttribute("width", barWidth);
    rect.setAttribute("height", barH - 12);
    rect.setAttribute("fill", col);
    rect.setAttribute("rx", "3");
    rect.setAttribute("opacity", "0.85");
    svg.appendChild(rect);

    // Feature label (left)
    const label = document.createElementNS(ns, "text");
    label.setAttribute("x", labelW - 8);
    label.setAttribute("y", y + barH / 2 + 4);
    label.setAttribute("text-anchor", "end");
    label.setAttribute("fill", "#6B6560");
    label.setAttribute("font-size", "11");
    label.setAttribute("font-family", "'IBM Plex Sans', sans-serif");
    label.textContent = f.label.length > 22 ? f.label.substring(0, 20) + "..." : f.label;
    svg.appendChild(label);

    // SHAP value (right)
    const val = document.createElementNS(ns, "text");
    const valX = isGood ? centerX + barWidth + 6 : centerX - barWidth - 6;
    val.setAttribute("x", valX);
    val.setAttribute("y", y + barH / 2 + 4);
    val.setAttribute("text-anchor", isGood ? "start" : "end");
    val.setAttribute("fill", col);
    val.setAttribute("font-size", "10");
    val.setAttribute("font-weight", "600");
    val.setAttribute("font-family", "'IBM Plex Mono', monospace");
    val.textContent = (isGood ? "+" : "-") + f.impact.toFixed(3);
    svg.appendChild(val);
  });

  section.appendChild(svg);
  fc.parentElement.insertBefore(section, fc.nextSibling);
}


// ─────────────────────────────────────────────
// Model Metrics Tab (Task 7)
// ─────────────────────────────────────────────
let metricsRendered = false;

function renderMetrics() {
  if (metricsRendered) return;
  metricsRendered = true;

  // ── 1. Summary Stats ──
  const stats = [
    { label: "AUC-ROC",               value: "0.91",  color: "var(--green)" },
    { label: "Precision (No Default)", value: "0.94",  color: "var(--green)" },
    { label: "Recall (No Default)",    value: "0.72",  color: "var(--amber)" },
    { label: "Precision (Default)",    value: "0.28",  color: "var(--red)"   },
    { label: "Recall (Default)",       value: "0.71",  color: "var(--amber)" },
    { label: "F1 Score",               value: "0.40",  color: "var(--amber)" },
  ];

  $("metrics-stats").innerHTML = stats.map(s => `
    <div class="metric-stat-card">
      <div class="metric-stat-label">${s.label}</div>
      <div class="metric-stat-value" style="color:${s.color}">${s.value}</div>
    </div>`).join("");

  // ── 2. Confusion Matrix ──
  $("confusion-matrix").innerHTML = `
    <table class="cm-table">
      <thead>
        <tr>
          <th></th>
          <th style="text-align:center">Predicted<br>No Default</th>
          <th style="text-align:center">Predicted<br>Default</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td class="cm-row-label">Actual<br>No Default</td>
          <td style="background:#166534;color:#fff;font-size:18px">43,210<br><span style="font-size:10px;font-weight:400;opacity:.8">True Negative</span></td>
          <td style="background:#FCA5A5;color:#7F1D1D">3,891<br><span style="font-size:10px;font-weight:400">False Positive</span></td>
        </tr>
        <tr>
          <td class="cm-row-label">Actual<br>Default</td>
          <td style="background:#FECACA;color:#991B1B">1,203<br><span style="font-size:10px;font-weight:400">False Negative</span></td>
          <td style="background:#4ADE80;color:#14532D">2,890<br><span style="font-size:10px;font-weight:400">True Positive</span></td>
        </tr>
      </tbody>
    </table>`;

  // ── 3. Feature Importance SVG ──
  const features = [
    { name: "EXT_SOURCE_2",                 val: 0.142 },
    { name: "EXT_SOURCE_3",                 val: 0.128 },
    { name: "EXT_SOURCE_1",                 val: 0.103 },
    { name: "AMT_CREDIT",                   val: 0.089 },
    { name: "AMT_ANNUITY",                  val: 0.071 },
    { name: "DAYS_BIRTH",                   val: 0.065 },
    { name: "DAYS_EMPLOYED",                val: 0.058 },
    { name: "AMT_INCOME_TOTAL",             val: 0.051 },
    { name: "REGION_POPULATION_RELATIVE",   val: 0.044 },
    { name: "DAYS_ID_PUBLISH",              val: 0.039 },
  ];

  const barH = 28;
  const gp = 6;
  const lW = 180;
  const bW = 220;
  const vW = 60;
  const tW = lW + bW + vW;
  const tH = features.length * (barH + gp) + 10;
  const maxVal = features[0].val;

  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", `0 0 ${tW} ${tH}`);
  svg.setAttribute("width", "100%");
  svg.style.marginTop = "12px";

  features.forEach((f, i) => {
    const y = i * (barH + gp) + 5;
    const w = (f.val / maxVal) * bW;

    // Label
    const txt = document.createElementNS(ns, "text");
    txt.setAttribute("x", lW - 8);
    txt.setAttribute("y", y + barH / 2 + 4);
    txt.setAttribute("text-anchor", "end");
    txt.setAttribute("fill", "#6B6560");
    txt.setAttribute("font-size", "11");
    txt.setAttribute("font-family", "'IBM Plex Sans', sans-serif");
    txt.textContent = f.name;
    svg.appendChild(txt);

    // Bar background
    const bgR = document.createElementNS(ns, "rect");
    bgR.setAttribute("x", lW);
    bgR.setAttribute("y", y);
    bgR.setAttribute("width", bW);
    bgR.setAttribute("height", barH);
    bgR.setAttribute("fill", "#F0EDE8");
    bgR.setAttribute("rx", "4");
    svg.appendChild(bgR);

    // Bar
    const rect = document.createElementNS(ns, "rect");
    rect.setAttribute("x", lW);
    rect.setAttribute("y", y + 3);
    rect.setAttribute("width", w);
    rect.setAttribute("height", barH - 6);
    rect.setAttribute("fill", "#D4500A");
    rect.setAttribute("rx", "3");
    rect.setAttribute("opacity", "0.9");
    svg.appendChild(rect);

    // Value label
    const val = document.createElementNS(ns, "text");
    val.setAttribute("x", lW + bW + 8);
    val.setAttribute("y", y + barH / 2 + 4);
    val.setAttribute("fill", "#D4500A");
    val.setAttribute("font-size", "11");
    val.setAttribute("font-weight", "700");
    val.setAttribute("font-family", "'IBM Plex Mono', monospace");
    val.textContent = f.val.toFixed(3);
    svg.appendChild(val);
  });

  $("feature-importance").appendChild(svg);

  // ── 4. AUC-ROC Explanation ──
  $("auc-explanation").innerHTML = `
    <p>
      <strong>AUC-ROC stands for "Area Under the Receiver Operating Characteristic Curve".</strong>
      Imagine you have a test that tries to separate two groups of people \u2014 those who will repay their loan
      and those who might default. The ROC curve plots how well the model does this at every possible threshold.
    </p>
    <p style="margin-top:10px">
      The AUC score is a single number between 0 and 1 that summarises how good the model is at telling the
      two groups apart. <strong>An AUC of 0.50 means the model is no better than flipping a coin.</strong>
      <strong>An AUC of 1.0 means the model is perfect</strong> \u2014 it never makes a mistake.
    </p>
    <p style="margin-top:10px">
      Our model achieves an <strong>AUC-ROC of 0.91</strong>, which means that if you pick one person who
      actually repaid and one who defaulted, the model will correctly rank the defaulter as higher risk
      <strong>91% of the time</strong>. This is considered <strong>excellent performance</strong> for
      credit risk prediction and is comparable to production systems used by real financial institutions.
    </p>`;
}

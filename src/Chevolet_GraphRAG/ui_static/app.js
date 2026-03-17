/* ==============================================
   Chevy AI — Frontend Logic
   Uses ONLY existing backend variable contracts:
     ChatRequest:  session_id, user_query, model_hint, feedback, resolved, top_k
     ChatResponse: session_id, answer, confidence, resolved, top_image_path,
                   top_sources, top_manual_sources, top_faq_sources, graph_paths, debug
     ImageSearchRequest: session_id, query, model_hint, top_k
   ============================================== */

const sessionId = `sess-${Math.random().toString(36).slice(2, 10)}`;

// ─── DOM refs (matching index.html IDs) ───
const chatLog = document.getElementById("chat-log");
const chatForm = document.getElementById("chat-form");
const queryInput = document.getElementById("query");
const modelHintInput = document.getElementById("model-hint");
const sendBtn = document.getElementById("send-btn");
const welcome = document.getElementById("welcome");
const typingEl = document.getElementById("typing");

const topImageEmpty = document.getElementById("top-image-empty");
const imageGallery = document.getElementById("image-gallery");
const topSources = document.getElementById("top-sources");
const topSourcesEmpty = document.getElementById("top-sources-empty");
const topFaqSources = document.getElementById("top-faq-sources");
const topFaqSourcesEmpty = document.getElementById("top-faq-sources-empty");

const toggleSidebar = document.getElementById("toggle-sidebar");
const closeSidebar = document.getElementById("close-sidebar");
const sidebar = document.getElementById("sidebar");

// ─── State ───
let lastBotMsgId = null;

// ─── Sidebar toggle ───
toggleSidebar.addEventListener("click", () => sidebar.classList.toggle("collapsed"));
closeSidebar.addEventListener("click", () => sidebar.classList.add("collapsed"));

// ─── Quick chips ───
document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    queryInput.value = chip.dataset.query;
    chatForm.dispatchEvent(new Event("submit", { cancelable: true }));
  });
});

// ─── Message rendering ───
function activateChat() {
  welcome.style.display = "none";
  chatLog.classList.add("active");
}

function showTyping() { typingEl.style.display = "flex"; scrollToBottom(); }
function hideTyping() { typingEl.style.display = "none"; }
function scrollToBottom() { chatLog.scrollTop = chatLog.scrollHeight; }

// ─── Image popup & adjacent pages ───
function openImagePopup(src) {
  window.open(src, "image_popup", "width=720,height=920,scrollbars=yes,resizable=yes");
}

function getFileName(path) {
  if (!path) return "";
  const parts = path.split("/");
  return parts[parts.length - 1] || path;
}

function formatPageLabel(source) {
  if (!source) return "";
  if (source.display_page_label) return `p.${source.display_page_label}`;
  return `PDF p.${source.page_no}`;
}

function formatSourceRelevance(source) {
  if (!source) return "";
  return source.relevance_label ? `관련도 ${source.relevance_label}` : "";
}

function formatSourceScoreDetails(source) {
  if (!source) return "";
  const parts = [];
  if (Number.isFinite(Number(source.retrieval_score))) {
    parts.push(`retrieval ${Number(source.retrieval_score).toFixed(3)}`);
  }
  if (Number.isFinite(Number(source.rerank_score))) {
    parts.push(`rerank ${Number(source.rerank_score).toFixed(3)}`);
  }
  return parts.join(" | ");
}

function createClickableImage(src, label) {
  const wrap = document.createElement("div");
  wrap.className = "gallery-item";
  const img = document.createElement("img");
  img.src = src;
  img.alt = label;
  img.loading = "lazy";
  img.style.cursor = "pointer";
  img.onclick = () => openImagePopup(src);
  img.onerror = () => { wrap.style.display = "none"; };
  const caption = document.createElement("span");
  caption.className = "gallery-caption";
  caption.textContent = label;
  wrap.appendChild(img);
  wrap.appendChild(caption);
  return wrap;
}

function createEvidenceCard(data) {
  const primarySource =
    (data.top_manual_sources || data.top_sources || [])[0] ||
    (data.top_faq_sources || [])[0] ||
    null;
  if (!data.top_image_path && !primarySource) return null;

  const card = document.createElement("div");
  card.className = "evidence-card";

  if (data.top_image_path) {
    const imageWrap = document.createElement("div");
    imageWrap.className = "evidence-card-image";
    const img = document.createElement("img");
    img.src = data.top_image_path;
    img.alt = "대표 근거 페이지";
    img.loading = "lazy";
    img.onclick = () => openImagePopup(data.top_image_path);
    imageWrap.appendChild(img);
    card.appendChild(imageWrap);
  }

  const meta = document.createElement("div");
  meta.className = "evidence-card-meta";

  const title = document.createElement("div");
  title.className = "evidence-card-title";
  title.textContent = "대표 근거";
  meta.appendChild(title);

  if (primarySource) {
    const sourceFile = document.createElement("div");
    sourceFile.className = "evidence-card-source";
    sourceFile.textContent = getFileName(primarySource.source_file);
    meta.appendChild(sourceFile);

    const sourcePage = document.createElement("div");
    sourcePage.className = "evidence-card-page";
    sourcePage.textContent = [formatPageLabel(primarySource), formatSourceRelevance(primarySource)]
      .filter(Boolean)
      .join(" | ");
    meta.appendChild(sourcePage);

    const scoreDetails = formatSourceScoreDetails(primarySource);
    if (scoreDetails) {
      const sourceScore = document.createElement("div");
      sourceScore.className = "evidence-card-detail";
      sourceScore.textContent = scoreDetails;
      meta.appendChild(sourceScore);
    }

    if (primarySource.manual_type) {
      const sourceType = document.createElement("div");
      sourceType.className = "evidence-card-detail";
      sourceType.textContent = `manual_type: ${primarySource.manual_type}`;
      meta.appendChild(sourceType);
    }
  }

  card.appendChild(meta);
  return card;
}

function appendUserMsg(text) {
  activateChat();
  const div = document.createElement("div");
  div.className = "msg user";
  div.textContent = text;
  chatLog.appendChild(div);
  scrollToBottom();
}

function appendBotMsg(data) {
  /* data comes straight from ChatResponse fields:
     answer, confidence, top_image_path, top_sources/top_manual_sources, top_faq_sources, graph_paths */
  const id = `bot-${Date.now()}`;
  lastBotMsgId = id;
  const primarySource =
    (data.top_manual_sources || data.top_sources || [])[0] ||
    (data.top_faq_sources || [])[0] ||
    null;

  const div = document.createElement("div");
  div.className = "msg bot";
  div.id = id;

  // answer text
  const answerEl = document.createElement("div");
  answerEl.className = "answer-text";
  answerEl.textContent = data.answer;

  // source relevance badge
  if (primarySource && primarySource.relevance_label) {
    const badge = document.createElement("span");
    badge.className = "confidence-badge";
    badge.textContent = `관련도 ${primarySource.relevance_label}`;
    answerEl.appendChild(document.createTextNode("\n"));
    answerEl.appendChild(badge);
  }

  const evidenceCard = createEvidenceCard(data);
  if (evidenceCard) {
    div.appendChild(evidenceCard);
  }

  div.appendChild(answerEl);

  // inline sources preview (top_sources from ChatResponse)
  const sources = data.top_manual_sources || data.top_sources || [];
  if (sources.length > 0) {
    const srcDiv = document.createElement("div");
    srcDiv.className = "inline-sources";
    srcDiv.textContent = "📌 ";
    sources.slice(0, 3).forEach((s) => {
      const tag = document.createElement("span");
      tag.textContent = `${getFileName(s.source_file)} ${formatPageLabel(s)}`;
      srcDiv.appendChild(tag);
    });
    if (sources.length > 3) {
      const more = document.createElement("span");
      more.textContent = `+${sources.length - 3}`;
      srcDiv.appendChild(more);
    }
    div.appendChild(srcDiv);
  }

  // feedback buttons (uses existing resolved & feedback fields)
  const fbRow = document.createElement("div");
  fbRow.className = "feedback-row";

  const goodBtn = document.createElement("button");
  goodBtn.className = "feedback-btn";
  goodBtn.textContent = "👍 도움됐어요";
  goodBtn.onclick = () => sendFeedback("해결됨", true, goodBtn, badBtn);

  const badBtn = document.createElement("button");
  badBtn.className = "feedback-btn";
  badBtn.textContent = "👎 부족해요";
  badBtn.onclick = () => sendFeedback("해결되지 않음, 추가 정보 필요", false, badBtn, goodBtn);

  fbRow.appendChild(goodBtn);
  fbRow.appendChild(badBtn);
  div.appendChild(fbRow);

  chatLog.appendChild(div);
  scrollToBottom();
}

function appendErrorMsg(message) {
  const div = document.createElement("div");
  div.className = "msg bot";
  div.textContent = `⚠️ ${message}`;
  chatLog.appendChild(div);
  scrollToBottom();
}

// ─── Sidebar rendering ───
function renderSidebar(data) {
  const gallery = imageGallery || document.getElementById("image-gallery");
  const oldTopImage = document.getElementById("top-image");
  const primarySource = (data.top_manual_sources || data.top_sources || [])[0] || null;

  if (gallery) {
    gallery.innerHTML = "";
    if (data.top_image_path) {
      const label = primarySource
        ? `${getFileName(primarySource.source_file)} | ${formatPageLabel(primarySource)}`
        : "대표 근거 페이지";
      gallery.appendChild(createClickableImage(data.top_image_path, label));
      gallery.style.display = "flex";
      if (topImageEmpty) topImageEmpty.style.display = "none";
    } else {
      gallery.style.display = "none";
      if (topImageEmpty) topImageEmpty.style.display = "block";
    }
  } else if (oldTopImage) {
    if (data.top_image_path) {
      oldTopImage.src = data.top_image_path;
      oldTopImage.style.display = "block";
      oldTopImage.style.cursor = "pointer";
      oldTopImage.onclick = () => openImagePopup(data.top_image_path);
      if (topImageEmpty) topImageEmpty.style.display = "none";
    } else {
      oldTopImage.style.display = "none";
      oldTopImage.removeAttribute("src");
      if (topImageEmpty) topImageEmpty.style.display = "block";
    }
  }

  // top sources (uses top_sources from ChatResponse)
  const manualSources = data.top_manual_sources || data.top_sources || [];
  const faqSources = data.top_faq_sources || [];
  if (topSources) {
    topSources.innerHTML = "";
    if (manualSources.length > 0) {
      if (topSourcesEmpty) topSourcesEmpty.style.display = "none";
      manualSources.slice(0, 5).forEach((s) => {
        const li = document.createElement("li");
        const parts = [
          getFileName(s.source_file),
          formatPageLabel(s),
          formatSourceRelevance(s),
          formatSourceScoreDetails(s),
          s.path_summary || "",
        ].filter(Boolean);
        li.textContent = parts.join(" | ");
        topSources.appendChild(li);
      });
    } else {
      if (topSourcesEmpty) topSourcesEmpty.style.display = "block";
    }
  }

  // faq sources (separated from manual sources)
  if (topFaqSources) {
    topFaqSources.innerHTML = "";
    if (faqSources.length > 0) {
      if (topFaqSourcesEmpty) topFaqSourcesEmpty.style.display = "none";
      faqSources.slice(0, 5).forEach((s) => {
        const li = document.createElement("li");
        const parts = [
          s.source_file,
          formatSourceRelevance(s),
          formatSourceScoreDetails(s),
          s.path_summary || "FAQ",
        ].filter(Boolean);
        li.textContent = parts.join(" | ");
        topFaqSources.appendChild(li);
      });
    } else {
      if (topFaqSourcesEmpty) topFaqSourcesEmpty.style.display = "block";
    }
  }
}

// ─── Chat submit (ChatRequest fields ONLY) ───
chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const query = queryInput.value.trim();
  if (!query) return;

  appendUserMsg(query);
  queryInput.value = "";
  showTyping();
  sendBtn.disabled = true;

  /* Build payload using EXACTLY ChatRequest fields:
     session_id, user_query, model_hint, feedback, resolved, top_k */
  const payload = {
    session_id: sessionId,
    user_query: query,
    model_hint: modelHintInput.value || null,
    feedback: null,
    resolved: null,
    top_k: 5,
  };

  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const txt = await res.text();
      throw new Error(txt || `HTTP ${res.status}`);
    }

    const data = await res.json();
    hideTyping();
    appendBotMsg(data);
    renderSidebar(data);
  } catch (err) {
    hideTyping();
    appendErrorMsg(`오류: ${err.message}`);
  } finally {
    sendBtn.disabled = false;
  }
});

// ─── Feedback (uses existing /feedback endpoint + FeedbackRequest fields) ───
async function sendFeedback(feedbackText, resolved, activeBtn, otherBtn) {
  activeBtn.classList.add(resolved ? "active-good" : "active-bad");
  otherBtn.classList.remove("active-good", "active-bad");

  /* FeedbackRequest fields: session_id, feedback, resolved */
  const payload = {
    session_id: sessionId,
    feedback: feedbackText,
    resolved: resolved,
  };

  try {
    const res = await fetch("/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) return;

    const data = await res.json();
    if (!resolved && data.answer) {
      appendBotMsg(data);
      renderSidebar(data);
    }
  } catch (_) {
    /* silent fail for feedback */
  }
}

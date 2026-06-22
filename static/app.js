// --- App State Management ---
let activeUserId = "alice";
let clustersData = [];
let profileData = { weights: {}, history: [] };
let systemStats = {};

// Track card open times for dwell feedback logging
const expandedCards = {};

// --- Element References ---
const elBackendStatus = document.getElementById("backend-status");
const elOllamaStatus = document.getElementById("ollama-status");
const elBtnRefresh = document.getElementById("btn-refresh");

const elTotalArticles = document.getElementById("val-total-articles");
const elUnprocessed = document.getElementById("val-unprocessed");
const elTotalClusters = document.getElementById("val-total-clusters");

const elBtnIngest = document.getElementById("btn-ingest");
const elBtnProcess = document.getElementById("btn-process");
const elIngestSpinner = document.getElementById("ingest-spinner");
const elProcessSpinner = document.getElementById("process-spinner");
const elControlLogs = document.getElementById("control-logs");

const elInputUserId = document.getElementById("input-user-id");
const elBtnLoadProfile = document.getElementById("btn-load-profile");
const elWeightsContainer = document.getElementById("weights-container");
const elFeedbackLogBody = document.getElementById("feedback-log-body");

const elChatViewport = document.getElementById("chat-viewport");
const elChatInput = document.getElementById("chat-input");
const elBtnChatSend = document.getElementById("btn-chat-send");
const elTogglePersonalization = document.getElementById("toggle-personalization");

const elClustersGrid = document.getElementById("clusters-grid");

// --- API Helper Methods ---
async function apiFetch(endpoint, options = {}) {
    try {
        const response = await fetch(endpoint, options);
        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || "API request failed");
        }
        return await response.json();
    } catch (error) {
        console.error(`API Error (${endpoint}):`, error);
        throw error;
    }
}

// --- Main App Logic ---

// Set visual health dots
function setHealthIndicator(element, status, text) {
    const dot = element.querySelector(".dot");
    const label = element.querySelector("span:not(.dot)");
    
    dot.className = "dot " + (status ? "green" : "red");
    label.textContent = text;
}

// Write message to the control console log box
function logConsole(message) {
    const time = new Date().toLocaleTimeString();
    elControlLogs.innerText += `\n[${time}] ${message}`;
    elControlLogs.scrollTop = elControlLogs.scrollHeight;
}

// Load DB & System Statistics
async function loadStats() {
    try {
        const stats = await apiFetch("/api/stats");
        systemStats = stats;
        
        // Update connection status
        setHealthIndicator(elBackendStatus, true, "API Server: Connected");
        setHealthIndicator(elOllamaStatus, stats.ollama_online, `Ollama LLM: ${stats.ollama_online ? "Active (" + stats.ollama_model + ")" : "Offline"}`);
        
        // Update stats widgets
        elTotalArticles.textContent = stats.articles_total;
        elUnprocessed.textContent = stats.articles_unprocessed;
        elTotalClusters.textContent = stats.clusters_total;
        
        // Update background job spinners
        if (stats.jobs.ingest.status === "running") {
            elIngestSpinner.classList.remove("hidden");
            elBtnIngest.disabled = true;
        } else {
            elIngestSpinner.classList.add("hidden");
            elBtnIngest.disabled = false;
        }
        
        if (stats.jobs.process.status === "running") {
            elProcessSpinner.classList.remove("hidden");
            elBtnProcess.disabled = true;
        } else {
            elProcessSpinner.classList.add("hidden");
            elBtnProcess.disabled = false;
        }
    } catch (err) {
        setHealthIndicator(elBackendStatus, false, "API Server: Disconnected");
        setHealthIndicator(elOllamaStatus, false, "Ollama LLM: Unknown");
    }
}

// Load Active User Profile (Weights & feedback logs)
async function loadUserProfile() {
    activeUserId = elInputUserId.value.trim() || "alice";
    try {
        const profile = await apiFetch(`/api/profile/${activeUserId}`);
        profileData = profile;
        renderWeightsGraph();
        renderFeedbackHistory();
    } catch (err) {
        logConsole(`Error loading profile: ${err.message}`);
    }
}

// Render the cluster weights visualizer graph
function renderWeightsGraph() {
    elWeightsContainer.innerHTML = "";
    const weights = profileData.weights;
    
    const entries = Object.entries(weights);
    if (entries.length === 0) {
        elWeightsContainer.innerHTML = `<div class="empty-state">No interactions recorded. Interact with news clusters to train this user profile.</div>`;
        return;
    }
    
    // Sort weights by absolute value descending
    entries.sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
    
    entries.forEach(([clusterId, weight]) => {
        // Find cluster label
        const cluster = clustersData.find(c => c.id === parseInt(clusterId));
        const label = cluster ? cluster.label : `Cluster ${clusterId}`;
        const absWeight = Math.abs(weight);
        const sign = weight >= 0 ? "+" : "-";
        
        // Normalize width percentage (caps at 1.0 weight)
        const widthPct = Math.min(100, Math.round(absWeight * 100));
        const barClass = weight >= 0 ? "positive" : "negative";
        
        const row = document.createElement("div");
        row.className = "weight-row";
        row.innerHTML = `
            <div class="weight-info">
                <span class="weight-label" title="${label}">${label}</span>
                <span class="weight-value">${sign}${absWeight.toFixed(2)}</span>
            </div>
            <div class="weight-bar-bg">
                <div class="weight-bar-fill ${barClass}" style="width: ${widthPct}%"></div>
            </div>
        `;
        elWeightsContainer.appendChild(row);
    });
}

// Render the user's feedback history table logs
function renderFeedbackHistory() {
    elFeedbackLogBody.innerHTML = "";
    const history = profileData.history;
    
    if (history.length === 0) {
        elFeedbackLogBody.innerHTML = `<tr><td colspan="4" class="no-logs">No feedback logged.</td></tr>`;
        return;
    }
    
    history.forEach(log => {
        const cluster = clustersData.find(c => c.id === log.cluster_id);
        const label = cluster ? cluster.label : `Cluster ${log.cluster_id}`;
        
        // Pretty format signals
        let signalLabel = log.signal;
        if (log.signal === "thumbs_up") signalLabel = "👍 Thumbs Up";
        else if (log.signal === "thumbs_down") signalLabel = "👎 Thumbs Down";
        else if (log.signal === "dwell") signalLabel = `📖 Dwell (${Math.round(log.value)}s)`;
        
        const row = document.createElement("tr");
        row.innerHTML = `
            <td title="${label}">${label}</td>
            <td>${signalLabel}</td>
            <td>${log.value >= 0 ? "+" : ""}${log.value.toFixed(1)}</td>
            <td>${log.created_at.substring(11, 16)}</td>
        `;
        elFeedbackLogBody.appendChild(row);
    });
}

// Fetch and render the full Cluster directory list
async function loadClusters() {
    try {
        const clusters = await apiFetch("/api/clusters");
        clustersData = clusters;
        
        elClustersGrid.innerHTML = "";
        if (clusters.length === 0) {
            elClustersGrid.innerHTML = `<div class="grid-loading">No topic clusters in database. Trigger crawler and processing loops first.</div>`;
            return;
        }
        
        clusters.forEach(c => {
            const card = renderClusterCard(c);
            elClustersGrid.appendChild(card);
        });
    } catch (err) {
        elClustersGrid.innerHTML = `<div class="grid-loading" style="color: var(--danger)">Failed to fetch clusters: ${err.message}</div>`;
    }
}

// Format the HTML/DOM elements of a cluster card
function renderClusterCard(c) {
    const card = document.createElement("div");
    card.className = "cluster-card";
    card.dataset.id = c.id;
    
    // Check if user has feedback for this cluster
    const userWeight = profileData.weights[c.id] || 0.0;
    const isUp = userWeight > 0.01;
    const isDown = userWeight < -0.01;
    
    const formattedDate = c.created_at ? c.created_at.substring(0, 10) : "N/A";
    
    // Build articles HTML
    let articlesHtml = "";
    if (c.articles && c.articles.length > 0) {
        c.articles.forEach(art => {
            const articleLabel = art.source ? `[${art.source}]` : "[Web]";
            articlesHtml += `
                <li class="cluster-article-item">
                    <a href="${art.url}" target="_blank" class="cluster-article-link">${art.title}</a>
                    <span class="cluster-article-meta">${articleLabel} ${art.published_at.substring(0, 10)}</span>
                </li>
            `;
        });
    } else {
        articlesHtml = "<li class='cluster-article-item' style='color: var(--text-dim)'>No articles linked.</li>";
    }
    
    card.innerHTML = `
        <div class="cluster-card-header">
            <h4 class="cluster-label">${c.label}</h4>
            <span class="cluster-size">${c.article_count} article${c.article_count === 1 ? "" : "s"}</span>
        </div>
        <p class="cluster-summary">${c.summary || "No summary available."}</p>
        
        <!-- Expandable details -->
        <div class="cluster-expand-area">
            <div class="expand-headline">Linked Articles</div>
            <ul class="cluster-article-list">
                ${articlesHtml}
            </ul>
        </div>
        
        <div class="cluster-footer">
            <span class="cluster-date">Created: ${formattedDate}</span>
            <div class="feedback-actions">
                <button class="btn-feedback btn-thumbs-up ${isUp ? 'active-up' : ''}" title="Thumbs Up">
                    <i class="fa-solid fa-thumbs-up"></i>
                </button>
                <button class="btn-feedback btn-thumbs-down ${isDown ? 'active-down' : ''}" title="Thumbs Down">
                    <i class="fa-solid fa-thumbs-down"></i>
                </button>
                <button class="btn-feedback btn-expand" title="Expand Articles List">
                    <i class="fa-solid fa-book-open"></i>
                </button>
            </div>
        </div>
    `;
    
    // Bind click events for thumbs up / down
    card.querySelector(".btn-thumbs-up").addEventListener("click", (e) => {
        e.stopPropagation();
        submitFeedback(c.id, isUp ? "neutral" : "thumbs_up");
    });
    
    card.querySelector(".btn-thumbs-down").addEventListener("click", (e) => {
        e.stopPropagation();
        submitFeedback(c.id, isDown ? "neutral" : "thumbs_down");
    });
    
    // Bind click event for expand/collapse (tracks dwell timer feedback)
    card.querySelector(".btn-expand").addEventListener("click", (e) => {
        e.stopPropagation();
        toggleCardExpansion(card, c.id);
    });
    
    return card;
}

// Track dwell time of expanded card
function toggleCardExpansion(card, clusterId) {
    const isExpanded = card.classList.toggle("expanded");
    const icon = card.querySelector(".btn-expand i");
    
    if (isExpanded) {
        icon.className = "fa-solid fa-book";
        expandedCards[clusterId] = Date.now();
        logConsole(`Reading cluster [${clusterId}] ... timer started.`);
    } else {
        icon.className = "fa-solid fa-book-open";
        const startTime = expandedCards[clusterId];
        if (startTime) {
            const durationSec = (Date.now() - startTime) / 1000;
            delete expandedCards[clusterId];
            
            // Only log if duration is significant (e.g. > 1 second)
            if (durationSec > 1.0) {
                logConsole(`Finished reading cluster [${clusterId}] after ${durationSec.toFixed(1)} seconds.`);
                submitDwellFeedback(clusterId, durationSec);
            }
        }
    }
}

// Submit thumbs feedback to FastAPI
async function submitFeedback(clusterId, signal) {
    try {
        const result = await apiFetch("/api/feedback", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                user_id: activeUserId,
                cluster_id: clusterId,
                signal: signal
            })
        });
        
        logConsole(`Feedback recorded: User '${activeUserId}' flagged cluster ${clusterId} as ${signal}.`);
        
        // Reload directories to redraw active flags and refresh weight visualizers
        await loadUserProfile();
        await loadClusters();
    } catch (err) {
        logConsole(`Feedback failed: ${err.message}`);
    }
}

// Submit reading dwell time to FastAPI
async function submitDwellFeedback(clusterId, seconds) {
    try {
        await apiFetch("/api/feedback", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                user_id: activeUserId,
                cluster_id: clusterId,
                signal: "dwell",
                dwell_seconds: seconds
            })
        });
        
        // Reload visual weights
        await loadUserProfile();
    } catch (err) {
        console.error("Dwell logging failed", err);
    }
}

// --- RAG Chat System ---

// Push a new bubble to the chat panel
function appendChatBubble(sender, text, sources = []) {
    const bubble = document.createElement("div");
    bubble.className = `chat-bubble ${sender}`;
    bubble.innerText = text;
    
    elChatViewport.appendChild(bubble);
    
    // Append similarity source pills under assistant answers
    if (sender === "assistant" && sources.length > 0) {
        const srcContainer = document.createElement("div");
        srcContainer.className = "chat-sources";
        srcContainer.innerHTML = "<h5>Relevant Sources Used:</h5>";
        
        const pills = document.createElement("div");
        pills.className = "source-pills";
        
        sources.forEach(src => {
            const pill = document.createElement("div");
            pill.className = "source-pill";
            pill.innerHTML = `<i class="fa-solid fa-cube"></i> [${Math.round(src.similarity * 100)}%] ${src.label}`;
            
            // Clicking a source highlights the cluster card in the grid
            pill.addEventListener("click", () => {
                const card = document.querySelector(`.cluster-card[data-id="${src.id}"]`);
                if (card) {
                    card.scrollIntoView({ behavior: "smooth", block: "center" });
                    card.classList.add("glowing-accent");
                    setTimeout(() => card.classList.remove("glowing-accent"), 2500);
                }
            });
            pills.appendChild(pill);
        });
        srcContainer.appendChild(pills);
        elChatViewport.appendChild(srcContainer);
    }
    
    // Auto-scroll viewport
    elChatViewport.scrollTop = elChatViewport.scrollHeight;
    return bubble;
}

// Perform RAG query
async function submitChatQuery() {
    const queryText = elChatInput.value.trim();
    if (!queryText) return;
    
    elChatInput.value = "";
    appendChatBubble("user", queryText);
    
    // Insert typing indicator
    const typingBubble = document.createElement("div");
    typingBubble.className = "chat-bubble assistant typing-bubble";
    typingBubble.innerHTML = `
        <div class="typing-indicator">
            <span class="typing-dot"></span>
            <span class="typing-dot"></span>
            <span class="typing-dot"></span>
        </div>
    `;
    elChatViewport.appendChild(typingBubble);
    elChatViewport.scrollTop = elChatViewport.scrollHeight;
    
    const usePersonalization = elTogglePersonalization.checked;
    
    try {
        const response = await apiFetch("/api/ask", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                query: queryText,
                user_id: usePersonalization ? activeUserId : null
            })
        });
        
        // Remove typing indicator
        typingBubble.remove();
        
        // Append response
        appendChatBubble("assistant", response.answer, response.sources);
    } catch (err) {
        typingBubble.remove();
        appendChatBubble("assistant", `Error: ${err.message}. Please make sure your local Ollama server is running and the database contains indexed articles.`);
    }
}

// --- Control Room Background Job Triggers ---

async function triggerIngestion() {
    try {
        const res = await apiFetch("/api/ingest", { method: "POST" });
        logConsole(res.message);
        elIngestSpinner.classList.remove("hidden");
        elBtnIngest.disabled = true;
        
        // Start polling stats to track job progression
        startStatsPolling();
    } catch (err) {
        logConsole(`Ingestion launch failed: ${err.message}`);
    }
}

async function triggerProcessing() {
    try {
        const res = await apiFetch("/api/process", { method: "POST" });
        logConsole(res.message);
        elProcessSpinner.classList.remove("hidden");
        elBtnProcess.disabled = true;
        
        // Start polling stats to track job progression
        startStatsPolling();
    } catch (err) {
        logConsole(`Processing launch failed: ${err.message}`);
    }
}

// --- Event Listeners and Init ---

let pollingInterval = null;

function startStatsPolling() {
    if (pollingInterval) clearInterval(pollingInterval);
    
    // Poll stats every 3 seconds to catch status transitions during background processing
    pollingInterval = setInterval(async () => {
        await loadStats();
        
        // If all jobs complete, slow down the polling to every 10 seconds
        const ingestRunning = systemStats.jobs.ingest.status === "running";
        const processRunning = systemStats.jobs.process.status === "running";
        
        if (!ingestRunning && !processRunning) {
            clearInterval(pollingInterval);
            pollingInterval = setInterval(loadStats, 10000);
            
            // Reload news items since directory may have changed
            await loadClusters();
            await loadUserProfile();
        }
    }, 3000);
}

// Refresh overall dashboard
async function refreshDashboard() {
    logConsole("Refreshing data dashboard...");
    await loadStats();
    await loadClusters();
    await loadUserProfile();
}

// Initialize Everything
document.addEventListener("DOMContentLoaded", () => {
    // Bind triggers
    elBtnRefresh.addEventListener("click", refreshDashboard);
    elBtnLoadProfile.addEventListener("click", loadUserProfile);
    
    elBtnChatSend.addEventListener("click", submitChatQuery);
    elChatInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") submitChatQuery();
    });
    
    elBtnIngest.addEventListener("click", triggerIngestion);
    elBtnProcess.addEventListener("click", triggerProcessing);
    
    // Load data
    refreshDashboard();
    
    // Start standard low-frequency polling for health checks (10s)
    pollingInterval = setInterval(loadStats, 10000);
});

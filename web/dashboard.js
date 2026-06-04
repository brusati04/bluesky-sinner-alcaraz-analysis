/* --- dashboard.js --- */

// State variables
let globalData = null;
let network = null;
let nodesDataset = null;
let edgesDataset = null;
let activeCategory = 'all';
let activeAlgorithm = 'louvain';
let selectedCommunityId = null;
let selectedNodeId = null;

// Accent colors corresponding to CSS variables
const CATEGORY_COLORS = {
    'hype/meme': { background: '#ff375f', border: 'rgba(255, 55, 95, 0.4)', highlight: '#ff375f' },
    'utility/bot': { background: '#bf5af2', border: 'rgba(191, 90, 242, 0.4)', highlight: '#bf5af2' },
    'analytical/journalism': { background: '#64d2ff', border: 'rgba(100, 210, 255, 0.4)', highlight: '#64d2ff' },
    'fan_chat': { background: '#ff9f0a', border: 'rgba(255, 159, 10, 0.4)', highlight: '#ff9f0a' },
    'fallback': { background: '#8e8e93', border: 'rgba(142, 142, 147, 0.4)', highlight: '#8e8e93' }
};

const EMOTION_COLORS = {
    joy: '#30d158',
    trust: '#0a84ff',
    anticipation: '#ffd60a',
    surprise: '#64d2ff',
    sadness: '#5e5ce6',
    anger: '#ff453a',
    fear: '#ff9f0a',
    disgust: '#ff375f'
};

// Initialize application on load
window.addEventListener('DOMContentLoaded', () => {
    loadDashboardData();
    setupEventListeners();
});

// Load the exported JSON data
async function loadDashboardData() {
    try {
        const response = await fetch('data/dashboard_data.json');
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        globalData = await response.json();
        
        // Populate header stats
        document.getElementById('stat-nodes').innerText = globalData.nodes.length;
        document.getElementById('stat-edges').innerText = globalData.edges.length;
        
        const activeAlgData = globalData.algorithms[activeAlgorithm];
        document.getElementById('stat-modularity').innerText = activeAlgData.modularity.toFixed(4);
        
        const commIds = Object.keys(activeAlgData.communities);
        document.getElementById('stat-comms').innerText = commIds.length;
        
        // Render community directory & graph
        renderCommunityDirectory();
        initializeNetworkGraph();
        
        // Initialize Lucide Icons
        if (window.lucide) {
            window.lucide.createIcons();
        }
    } catch (error) {
        console.error("Error loading dashboard data:", error);
        document.getElementById('community-list').innerHTML = `
            <div class="loading-spinner" style="color: #ef4444;">
                <i data-lucide="alert-triangle"></i>
                <p>Failed to load data payload. Run the python script first!</p>
                <p style="font-size: 0.7rem; color: var(--text-muted);">${error.message}</p>
            </div>
        `;
        if (window.lucide) {
            window.lucide.createIcons();
        }
    }
}

// Set up UI Event listeners
function setupEventListeners() {
    // Category filter button click events
    const filterBtns = document.querySelectorAll('.filter-btn');
    filterBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
            const btnEl = e.currentTarget;
            filterBtns.forEach(b => b.classList.remove('active'));
            btnEl.classList.add('active');
            
            activeCategory = btnEl.getAttribute('data-category');
            filterDashboard();
        });
    });
    
    // Directory Search bar
    const searchInput = document.getElementById('comm-search');
    searchInput.addEventListener('input', (e) => {
        const query = e.target.value.toLowerCase().trim();
        filterDirectoryList(query);
    });
    
    // Graph control buttons
    document.getElementById('btn-fit').addEventListener('click', () => {
        if (network) network.fit({ animation: { duration: 1000, easingFunction: 'easeInOutQuad' } });
    });
    
    document.getElementById('btn-clear-selection').addEventListener('click', () => {
        clearGraphSelection();
    });

    // Algorithm Selector Dropdown listener
    const algSelect = document.getElementById('algorithm-select');
    if (algSelect) {
        algSelect.addEventListener('change', (e) => {
            activeAlgorithm = e.target.value;
            
            const activeAlgData = globalData.algorithms[activeAlgorithm];
            
            // Update stats
            document.getElementById('stat-modularity').innerText = activeAlgData.modularity.toFixed(4);
            document.getElementById('stat-comms').innerText = Object.keys(activeAlgData.communities).length;
            
            // Clear selection states
            selectedCommunityId = null;
            selectedNodeId = null;
            
            // Re-render community directory
            renderCommunityDirectory();
            
            // Re-color network nodes and filter hidden nodes
            updateNetworkGraphLayout();
            
            // Reset detail panel
            showDefaultDetailState();
            
            // Apply taxonomy filters
            filterDashboard();
        });
    }
}

// Populate Left Sidebar community directory
function renderCommunityDirectory() {
    const listContainer = document.getElementById('community-list');
    listContainer.innerHTML = '';
    
    const activeAlgData = globalData.algorithms[activeAlgorithm];
    
    // Convert communities object to sorted array (sort by post count descending)
    const commsArray = Object.values(activeAlgData.communities)
        .sort((a, b) => b.post_count - a.post_count);
        
    commsArray.forEach(comm => {
        // Skip communities with 0 posts in directory to keep it clean
        if (comm.post_count === 0) return;
        
        const card = document.createElement('div');
        card.className = `directory-card comm-category-${comm.category.replace('/', '-')}`;
        card.setAttribute('data-id', comm.id);
        card.setAttribute('data-category', comm.category);
        
        // Sentiment dot class
        let sentClass = 'sent-neu';
        if (comm.avg_sentiment >= 0.05) sentClass = 'sent-pos';
        else if (comm.avg_sentiment <= -0.05) sentClass = 'sent-neg';
        
        card.innerHTML = `
            <div class="dir-card-header">
                <span class="dir-card-title">Community #${comm.id}</span>
                <span class="dir-card-badge dir-badge-${comm.category === 'utility/bot' ? 'bot' : comm.category.split('/')[0]}">${comm.category}</span>
            </div>
            <div class="dir-card-stats">
                <span><i data-lucide="users" style="width: 10px; height:10px; display:inline-block; vertical-align:middle; margin-right:2px;"></i> ${comm.size} Users</span>
                <span>${comm.post_count} Posts</span>
                <span class="dir-sentiment">
                    <span class="sentiment-dot ${sentClass}"></span>
                    ${comm.avg_sentiment.toFixed(2)}
                </span>
            </div>
        `;
        
        card.addEventListener('click', () => {
            selectCommunity(comm.id, true);
        });
        
        listContainer.appendChild(card);
    });
    
    if (window.lucide) {
        window.lucide.createIcons();
    }
}

// Filter directory cards based on selected category & search query
function filterDashboard() {
    // 1. Filter the Directory Cards in Left Sidebar
    const cards = document.querySelectorAll('.directory-card');
    cards.forEach(card => {
        const commCategory = card.getAttribute('data-category');
        if (activeCategory === 'all' || commCategory === activeCategory) {
            card.classList.remove('hidden');
        } else {
            card.classList.add('hidden');
        }
    });
    
    // 2. Dim nodes in the graph not belonging to active category
    if (network && nodesDataset) {
        const allNodes = nodesDataset.get();
        const updatedNodes = allNodes.map(node => {
            const nodeCommId = String(node.communities[activeAlgorithm]);
            const commData = globalData.algorithms[activeAlgorithm].communities[nodeCommId];
            const nodeCategory = commData ? commData.category : 'fallback';
            
            let colorObj = CATEGORY_COLORS[nodeCategory] || CATEGORY_COLORS['fallback'];
            
            // If category is filtered, check if node matches
            if (activeCategory !== 'all' && nodeCategory !== activeCategory) {
                // Dim the node
                return {
                    id: node.id,
                    color: {
                        background: 'rgba(40, 40, 40, 0.15)',
                        border: 'rgba(255, 255, 255, 0.02)',
                        highlight: 'rgba(10, 132, 255, 0.3)'
                    },
                    font: { color: 'rgba(255, 255, 255, 0.15)' }
                };
            } else {
                // Restore node color
                return {
                    id: node.id,
                    color: {
                        background: colorObj.background,
                        border: colorObj.border,
                        highlight: colorObj.highlight
                    },
                    font: { color: '#f1f5f9' }
                };
            }
        });
        
        nodesDataset.update(updatedNodes);
    }
}

// Filter sidebar directory cards by text search
function filterDirectoryList(query) {
    const cards = document.querySelectorAll('.directory-card');
    cards.forEach(card => {
        const commId = card.getAttribute('data-id');
        const commData = globalData.algorithms[activeAlgorithm].communities[commId];
        const commCategory = card.getAttribute('data-category');
        
        if (!commData) return;

        // Check if query matches community id, category, keywords, description, or top users
        const matchId = commId.includes(query);
        const matchCat = commCategory.toLowerCase().includes(query);
        const matchDesc = commData.description.toLowerCase().includes(query);
        const matchWord = commData.top_words.some(w => w.toLowerCase().includes(query));
        const matchUser = commData.top_users.some(u => u.toLowerCase().includes(query));
        
        const isVisibleInCategory = activeCategory === 'all' || commCategory === activeCategory;
        
        if (isVisibleInCategory && (matchId || matchCat || matchDesc || matchWord || matchUser || query === '')) {
            card.classList.remove('hidden');
        } else {
            card.classList.add('hidden');
        }
    });
}

// Initialize Vis Network Canvas
function initializeNetworkGraph() {
    const container = document.getElementById('mynetwork');
    const loadingScreen = document.getElementById('network-loading-screen');
    const progressText = document.getElementById('physics-progress');
    
    // Build Nodes Dataset
    const nodes = globalData.nodes.map(node => {
        const nodeCommId = String(node.communities[activeAlgorithm]);
        const commData = globalData.algorithms[activeAlgorithm].communities[nodeCommId];
        const category = commData ? commData.category : 'fallback';
        const colorStyle = CATEGORY_COLORS[category] || CATEGORY_COLORS['fallback'];
        
        // Scale size by pagerank
        const minSize = 8;
        const maxSize = 35;
        const sizeVal = minSize + (node.pagerank * 500); // Scale multiplier
        
        // Clean label if DID (show shortened name)
        let labelText = node.label;
        if (labelText.startsWith('did:')) {
            labelText = labelText.substring(0, 15) + '...';
        }
        
        return {
            id: node.id,
            label: labelText,
            title: `User: ${node.id}<br/>Community: #${nodeCommId} (${category})<br/>PageRank: ${node.pagerank.toFixed(5)}<br/>Avg Sentiment: ${node.sentiment_avg.toFixed(2)}`,
            value: sizeVal,
            size: sizeVal,
            communities: node.communities,
            pagerank: node.pagerank,
            sentiment_avg: node.sentiment_avg,
            color: {
                background: colorStyle.background,
                border: colorStyle.border,
                highlight: colorStyle.highlight
            },
            font: {
                size: sizeVal > 15 ? 12 : 9,
                face: 'Inter',
                color: '#f1f5f9'
            }
        };
    });
    
    // Build Edges Dataset
    const edges = globalData.edges.map(edge => {
        return {
            from: edge.from,
            to: edge.to,
            width: Math.min(6, 1 + edge.weight * 0.75),
            title: `${edge.relationship} Interaction (Weight: ${edge.weight})`
        };
    });
    
    nodesDataset = new vis.DataSet(nodes);
    edgesDataset = new vis.DataSet(edges);
    
    const data = {
        nodes: nodesDataset,
        edges: edgesDataset
    };
    
    const options = {
        nodes: {
            shape: 'dot',
            borderWidth: 1.5,
            shadow: {
                enabled: true,
                color: 'rgba(0,0,0,0.5)',
                size: 4,
                x: 0,
                y: 2
            }
        },
        edges: {
            color: {
                color: 'rgba(255, 255, 255, 0.1)',
                highlight: 'rgba(10, 132, 255, 0.6)',
                hover: 'rgba(10, 132, 255, 0.3)'
            },
            arrows: {
                to: { enabled: true, scaleFactor: 0.35 }
            },
            smooth: {
                type: 'continuous',
                roundness: 0.2
            }
        },
        physics: {
            forceAtlas2Based: {
                gravitationalConstant: -18,
                centralGravity: 0.015,
                springLength: 75,
                springConstant: 0.08
            },
            solver: 'forceAtlas2Based',
            stabilization: {
                enabled: true,
                iterations: 200,
                updateInterval: 25
            }
        },
        interaction: {
            hover: true,
            tooltipDelay: 100,
            hideEdgesOnDrag: true
        }
    };
    
    // Create Network Graph instance
    network = new vis.Network(container, data, options);
    
    // Physics stabilization progress bar
    network.on("stabilizationProgress", (params) => {
        const progress = Math.round((params.iterations / params.total) * 100);
        progressText.innerText = `${progress}%`;
    });
    
    // Hide loading screen when stabilized
    network.on("stabilizationIterationsDone", () => {
        loadingScreen.style.opacity = '0';
        setTimeout(() => {
            loadingScreen.classList.add('hidden');
        }, 500);
        
        // Turn off physics after initial layout to prevent sliding lag
        network.setOptions({ physics: false });
    });
    
    // Node selection / click handler
    network.on("selectNode", (params) => {
        const clickedNodeId = params.nodes[0];
        const clickedNode = nodesDataset.get(clickedNodeId);
        
        if (clickedNode) {
            selectedNodeId = clickedNodeId;
            const nodeCommId = clickedNode.communities[activeAlgorithm];
            selectCommunity(nodeCommId, false); // select community, don't refit graph (keep zoom on clicked node)
            highlightConnectedNodes(clickedNodeId);
        }
    });
    
    // Graph background click handler
    network.on("deselectNode", () => {
        selectedNodeId = null;
        restoreGraphColors();
        showDefaultDetailState();
    });
}

// Filter nodes/edges based on activeAlgorithm (hiding unassigned nodes) and re-run layout
function updateNetworkGraphLayout() {
    if (!network || !nodesDataset || !edgesDataset) return;
    
    // Show the loading overlay
    const loadingScreen = document.getElementById('network-loading-screen');
    const progressText = document.getElementById('physics-progress');
    if (loadingScreen) {
        loadingScreen.classList.remove('hidden');
        loadingScreen.style.opacity = '1';
        progressText.innerText = '0%';
    }
    
    // Filter nodes: keep only nodes assigned to a community in activeAlgorithm (community != -1)
    const activeNodes = globalData.nodes.filter(node => {
        const commId = node.communities[activeAlgorithm];
        return commId !== -1;
    });
    
    const visibleNodeIds = new Set(activeNodes.map(n => n.id));
    
    const newNodes = activeNodes.map(node => {
        const nodeCommId = String(node.communities[activeAlgorithm]);
        const commData = globalData.algorithms[activeAlgorithm].communities[nodeCommId];
        const category = commData ? commData.category : 'fallback';
        const colorStyle = CATEGORY_COLORS[category] || CATEGORY_COLORS['fallback'];
        
        const minSize = 8;
        const maxSize = 35;
        const sizeVal = minSize + (node.pagerank * 500);
        
        let labelText = node.label;
        if (labelText.startsWith('did:')) {
            labelText = labelText.substring(0, 15) + '...';
        }
        
        return {
            id: node.id,
            label: labelText,
            title: `User: ${node.id}<br/>Community: #${nodeCommId} (${category})<br/>PageRank: ${node.pagerank.toFixed(5)}<br/>Avg Sentiment: ${node.sentiment_avg.toFixed(2)}`,
            value: sizeVal,
            size: sizeVal,
            communities: node.communities,
            pagerank: node.pagerank,
            sentiment_avg: node.sentiment_avg,
            color: {
                background: colorStyle.background,
                border: colorStyle.border,
                highlight: colorStyle.highlight
            },
            font: {
                size: sizeVal > 15 ? 12 : 9,
                face: 'Inter',
                color: '#f1f5f9'
            }
        };
    });
    
    // Filter edges: keep only edges where both ends are in the visible nodes set
    const newEdges = globalData.edges.filter(edge => {
        return visibleNodeIds.has(edge.from) && visibleNodeIds.has(edge.to);
    }).map(edge => {
        return {
            from: edge.from,
            to: edge.to,
            width: Math.min(6, 1 + edge.weight * 0.75),
            title: `${edge.relationship} Interaction (Weight: ${edge.weight})`
        };
    });
    
    // Clear datasets and load new filtered data
    nodesDataset.clear();
    nodesDataset.add(newNodes);
    
    edgesDataset.clear();
    edgesDataset.add(newEdges);
    
    // Re-enable physics stabilization to layout the new graph nicely
    network.setOptions({ physics: { enabled: true } });
    network.stabilize(200);
}

// Select a community and load its profile details
function selectCommunity(commId, fitToNodes = false) {
    selectedCommunityId = commId;
    const activeAlgData = globalData.algorithms[activeAlgorithm];
    const comm = activeAlgData.communities[String(commId)];
    if (!comm) return;
    
    // Highlight directory card on sidebar
    const directoryCards = document.querySelectorAll('.directory-card');
    directoryCards.forEach(card => {
        card.classList.remove('selected');
        if (card.getAttribute('data-id') === String(commId)) {
            card.classList.add('selected');
            card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
    });
    
    // Fit network view to nodes in this community if requested
    if (fitToNodes && network && nodesDataset) {
        const nodesInComm = nodesDataset.get().filter(n => String(n.communities[activeAlgorithm]) === String(commId)).map(n => n.id);
        if (nodesInComm.length > 0) {
            network.selectNodes(nodesInComm);
            network.fit({
                nodes: nodesInComm,
                animation: { duration: 1000, easingFunction: 'easeInOutQuad' }
            });
            highlightCommunityNodes(commId);
        }
    }
    
    // Populate Detail Sidebar (Right)
    const stateDefault = document.getElementById('state-default');
    const stateSelected = document.getElementById('state-selected');
    
    stateDefault.classList.add('hidden');
    stateSelected.classList.remove('hidden');
    
    // Headers & Badges
    const badge = document.getElementById('detail-category-badge');
    badge.innerText = comm.category;
    badge.className = 'category-badge';
    
    const catClass = comm.category === 'utility/bot' ? 'bot' : comm.category.split('/')[0];
    badge.classList.add(`cat-badge-${catClass}`);
    
    document.getElementById('detail-id-badge').innerText = `Community #${comm.id}`;
    
    // Dynamic naming based on category
    let titleText = `Community #${comm.id}`;
    if (comm.category === 'hype/meme') titleText = `Hype Chat #${comm.id}`;
    else if (comm.category === 'utility/bot') titleText = `Utility / Bot Feed #${comm.id}`;
    else if (comm.category === 'analytical/journalism') titleText = `News & Stats Group #${comm.id}`;
    
    document.getElementById('detail-title').innerText = titleText;
    document.getElementById('detail-description').innerText = comm.description;
    
    // Stats
    document.getElementById('detail-size').innerText = comm.size;
    document.getElementById('detail-post-count').innerText = comm.post_count;
    
    // Sentiment
    const sentimentDisplay = document.getElementById('detail-avg-sentiment');
    const sign = comm.avg_sentiment > 0 ? '+' : '';
    sentimentDisplay.innerText = `${sign}${comm.avg_sentiment.toFixed(2)}`;
    
    if (comm.avg_sentiment >= 0.05) {
        sentimentDisplay.style.color = 'var(--accent-green)';
    } else if (comm.avg_sentiment <= -0.05) {
        sentimentDisplay.style.color = '#ef4444';
    } else {
        sentimentDisplay.style.color = 'var(--text-secondary)';
    }
    
    // Sentiment meter fill (-1 to +1 slider mapping to 0% to 100% width)
    const sliderPercent = ((comm.avg_sentiment + 1) / 2) * 100;
    document.getElementById('detail-sentiment-fill').style.left = `${sliderPercent}%`;
    
    // Keywords pills
    const wordsContainer = document.getElementById('detail-top-words');
    wordsContainer.innerHTML = '';
    if (comm.top_words.length === 0) {
        wordsContainer.innerHTML = `<span class="tag-pill">None</span>`;
    } else {
        comm.top_words.forEach(word => {
            const pill = document.createElement('span');
            pill.className = 'tag-pill';
            // Highlight player names
            if (['sinner', 'jannik', 'alcaraz', 'carlos', 'carlitos'].includes(word.toLowerCase())) {
                pill.className = 'tag-pill tag-pill-highlight';
            }
            pill.innerText = word;
            wordsContainer.appendChild(pill);
        });
    }
    
    // Entities pills
    const entsContainer = document.getElementById('detail-top-entities');
    entsContainer.innerHTML = '';
    if (comm.top_entities.length === 0) {
        entsContainer.innerHTML = `<span class="tag-pill">None</span>`;
    } else {
        comm.top_entities.forEach(ent => {
            const pill = document.createElement('span');
            pill.className = 'tag-pill tag-pill-highlight';
            pill.innerText = ent;
            entsContainer.appendChild(pill);
        });
    }
    
    // Emotion progress bars
    const emotionsList = document.getElementById('detail-emotions-list');
    emotionsList.innerHTML = '';
    
    const sortedEmotions = Object.entries(comm.emotions)
        .sort((a, b) => b[1] - a[1]);
        
    if (sortedEmotions.length === 0) {
        emotionsList.innerHTML = '<p style="font-size:0.75rem; color:var(--text-muted);">No emotional markers found in this sub-community.</p>';
    } else {
        sortedEmotions.forEach(([emo, val]) => {
            const color = EMOTION_COLORS[emo] || '#3b82f6';
            const percent = (val * 100).toFixed(1);
            
            const row = document.createElement('div');
            row.className = 'emotion-row';
            row.innerHTML = `
                <span class="emotion-name">${emo}</span>
                <div class="emotion-bar-track">
                    <div class="emotion-bar-fill" style="width: ${percent}%; background-color: ${color};"></div>
                </div>
                <span class="emotion-val">${percent}%</span>
            `;
            emotionsList.appendChild(row);
        });
    }
    
    // Top users chips
    const usersContainer = document.getElementById('detail-top-users');
    usersContainer.innerHTML = '';
    comm.top_users.forEach(username => {
        const chip = document.createElement('div');
        chip.className = 'member-chip';
        chip.innerHTML = `
            <i data-lucide="user" class="member-avatar"></i>
            <span>${username.split('.')[0]}</span>
        `;
        chip.addEventListener('click', () => {
            if (network) {
                network.selectNodes([username]);
                network.focus(username, { scale: 1.1, animation: { duration: 800 } });
                highlightConnectedNodes(username);
            }
        });
        usersContainer.appendChild(chip);
    });
    
    // Posts Feed
    const feedDisplayCount = document.getElementById('feed-count-display');
    const postsFeed = document.getElementById('detail-posts-feed');
    postsFeed.innerHTML = '';
    
    feedDisplayCount.innerText = `Showing ${comm.posts.length} posts`;
    
    if (comm.posts.length === 0) {
        postsFeed.innerHTML = '<p style="font-size:0.8rem; color:var(--text-muted); text-align:center; padding: 2rem 0;">No raw posts found in this community partition.</p>';
    } else {
        comm.posts.forEach(post => {
            const card = document.createElement('div');
            card.className = 'post-card';
            
            const dateStr = new Date(post.created_at).toLocaleDateString(undefined, {month:'short', day:'numeric', hour:'2-digit', minute:'2-digit'});
            
            let sentEmoji = '😐';
            let sentColor = 'var(--text-secondary)';
            if (post.sentiment >= 0.05) { sentEmoji = '😊'; sentColor = 'var(--accent-green)'; }
            else if (post.sentiment <= -0.05) { sentEmoji = '😢'; sentColor = '#ef4444'; }
            
            card.innerHTML = `
                <div class="post-meta">
                    <span class="post-author">@${post.author.split('.')[0]}</span>
                    <span class="post-date">${dateStr}</span>
                </div>
                <div class="post-text">${escapeHtml(post.text)}</div>
                <div class="post-footer">
                    <span class="post-sentiment" style="color: ${sentColor}">
                        <span>${sentEmoji}</span> VADER: ${post.sentiment.toFixed(2)}
                    </span>
                    <span class="post-sentiment" style="opacity: 0.8; text-transform: capitalize;">
                        Emotion: ${post.dominant_emotion}
                    </span>
                    <div class="post-engagement">
                        <span class="engage-item"><i data-lucide="heart" class="engage-icon"></i> ${post.likes}</span>
                        <span class="engage-item"><i data-lucide="repeat" class="engage-icon"></i> ${post.reposts}</span>
                    </div>
                </div>
            `;
            postsFeed.appendChild(card);
        });
    }
    
    if (window.lucide) {
        window.lucide.createIcons();
    }
}

// Clear selected state and reset detail panel
function showDefaultDetailState() {
    selectedCommunityId = null;
    selectedNodeId = null;
    
    document.getElementById('state-selected').classList.add('hidden');
    document.getElementById('state-default').classList.remove('hidden');
    
    const directoryCards = document.querySelectorAll('.directory-card');
    directoryCards.forEach(c => c.classList.remove('selected'));
}

// Clear selected node and restore graph styles
function clearGraphSelection() {
    if (network) {
        network.unselectNodes();
        restoreGraphColors();
        showDefaultDetailState();
        network.fit({ animation: { duration: 800 } });
    }
}

// Highlight nodes inside a community, dim all other communities
function highlightCommunityNodes(commId) {
    if (!nodesDataset) return;
    
    const allNodes = nodesDataset.get();
    const updatedNodes = allNodes.map(node => {
        if (String(node.communities[activeAlgorithm]) !== String(commId)) {
            // Dim non-community node
            return {
                id: node.id,
                color: {
                    background: 'rgba(40, 40, 40, 0.15)',
                    border: 'rgba(255, 255, 255, 0.02)',
                    highlight: 'rgba(10, 132, 255, 0.3)'
                },
                font: { color: 'rgba(255, 255, 255, 0.15)' }
            };
        } else {
            // Restore colors
            const activeAlgData = globalData.algorithms[activeAlgorithm];
            const commData = activeAlgData.communities[String(commId)];
            const colorStyle = CATEGORY_COLORS[commData.category] || CATEGORY_COLORS['fallback'];
            return {
                id: node.id,
                color: {
                    background: colorStyle.background,
                    border: colorStyle.border,
                    highlight: colorStyle.highlight
                },
                font: { color: '#f1f5f9' }
            };
        }
    });
    
    nodesDataset.update(updatedNodes);
}

// Highlight neighbors of a selected node, dim the rest of the graph
function highlightConnectedNodes(selectedNodeId) {
    if (!network || !nodesDataset || !edgesDataset) return;
    
    const connectedNodes = network.getConnectedNodes(selectedNodeId);
    
    const allNodes = nodesDataset.get();
    const updatedNodes = allNodes.map(node => {
        const isSelf = node.id === selectedNodeId;
        const isNeighbor = connectedNodes.includes(node.id);
        
        if (isSelf || isNeighbor) {
            // Highlight / Keep normal color
            const nodeCommId = String(node.communities[activeAlgorithm]);
            const commData = globalData.algorithms[activeAlgorithm].communities[nodeCommId];
            const category = commData ? commData.category : 'fallback';
            const colorStyle = CATEGORY_COLORS[category] || CATEGORY_COLORS['fallback'];
            
            return {
                id: node.id,
                color: {
                    background: colorStyle.background,
                    border: '#f1f5f9', // white border
                    highlight: colorStyle.highlight
                },
                font: { color: '#f1f5f9' }
            };
        } else {
            // Dim
            return {
                id: node.id,
                color: {
                    background: 'rgba(40, 40, 40, 0.12)',
                    border: 'rgba(255, 255, 255, 0.02)',
                    highlight: 'rgba(10, 132, 255, 0.3)'
                },
                font: { color: 'rgba(255, 255, 255, 0.12)' }
            };
        }
    });
    
    nodesDataset.update(updatedNodes);
}

// Restore default colors to all nodes
function restoreGraphColors() {
    if (!nodesDataset) return;
    
    const allNodes = nodesDataset.get();
    const updatedNodes = allNodes.map(node => {
        const nodeCommId = String(node.communities[activeAlgorithm]);
        const commData = globalData.algorithms[activeAlgorithm].communities[nodeCommId];
        const category = commData ? commData.category : 'fallback';
        const colorStyle = CATEGORY_COLORS[category] || CATEGORY_COLORS['fallback'];
        
        return {
            id: node.id,
            color: {
                background: colorStyle.background,
                border: colorStyle.border,
                highlight: colorStyle.highlight
            },
            font: { color: '#f1f5f9' }
        };
    });
    
    nodesDataset.update(updatedNodes);
    
    // Apply taxonomy filter in case it was active
    filterDashboard();
}

// Helper to escape HTML tags to prevent XSS
function escapeHtml(text) {
    if (!text) return "";
    return text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

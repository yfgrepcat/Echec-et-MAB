let board = null;
let game = new Chess();
let isPlaying = false;
let isAgentThinking = false;
let currentMode = "human_vs_mab";

const $status = $('#status');
const $dot = $('.dot');
const $history = $('#history-container');
const $currentArm = $('#current-arm');
const $currentTime = $('#current-time');

let rewardChart, armChart, armTimeChart, benchChart, levelChart, phaseChart;

let whiteTime = 300;
let blackTime = 300;
let activeTurn = 'w';
let lastTick = Date.now();
let clockInterval = null;

function formatTime(seconds) {
    if (isNaN(seconds) || seconds < 0) return "00:00";
    const m = Math.floor(seconds / 60).toString().padStart(2, '0');
    const s = Math.floor(seconds % 60).toString().padStart(2, '0');
    return `${m}:${s}`;
}

function handleGameOver(resultText) {
    if (!isPlaying) return;
    isPlaying = false;
    isAgentThinking = false;
    setStatus("Partie terminée : " + resultText, "dot");
    $('#gameOverMessage').text(resultText);
    $('#gameOverModal').fadeIn();
}

$('#closeModalBtn').click(() => {
    $('#gameOverModal').fadeOut();
});

function startLiveClock() {
    if (clockInterval) clearInterval(clockInterval);
    lastTick = Date.now();
    clockInterval = setInterval(() => {
        if (!isPlaying) return;
        const now = Date.now();
        const delta = (now - lastTick) / 1000.0;
        lastTick = now;
        
        if (activeTurn === 'w') {
            whiteTime = Math.max(0, whiteTime - delta);
            $('#clock-white').text(formatTime(whiteTime));
            if (whiteTime <= 0) handleGameOver("Temps écoulé (Les Noirs gagnent)");
        } else {
            blackTime = Math.max(0, blackTime - delta);
            $('#clock-black').text(formatTime(blackTime));
            if (blackTime <= 0) handleGameOver("Temps écoulé (Les Blancs gagnent)");
        }
    }, 100);
}

function updateClocks(res) {
    if (res.white_clock !== undefined) {
        whiteTime = res.white_clock;
        $('#clock-white').text(formatTime(whiteTime));
    }
    if (res.black_clock !== undefined) {
        blackTime = res.black_clock;
        $('#clock-black').text(formatTime(blackTime));
    }
    if (res.turn) {
        activeTurn = res.turn;
        lastTick = Date.now(); // reset delta for the new turn
    }
}

$('#gameMode').change(function() {
    if ($(this).val() === 'mab_vs_sf') {
        $('#sfLevel').fadeIn();
    } else {
        $('#sfLevel').hide();
    }
});

// Worker Selection Logic

/**
 * Fetch available workers for a given bandit type and populate a <select>.
 * @param {string} banditType  - "basic_linucb" or "neural_linucb"
 * @param {string} selectId    - jQuery selector for the <select> element
 * @param {string} defaultLabel - Label for the default/empty option
 */
function loadWorkers(banditType, selectId, defaultLabel, selectedValue = null) {
    const $select = $(selectId);
    $select.empty();

    if (defaultLabel) {
        $select.append(`<option value="">${defaultLabel}</option>`);
    }

    $.get('/api/workers', { bandit_type: banditType }, (res) => {
        if (!res.workers || res.workers.length === 0) {
            $select.append('<option value="" disabled>Aucun worker disponible</option>');
            return;
        }
        res.workers.forEach((w) => {
            $select.append(`<option value="${w.id}">${w.filename}</option>`);
        });
        if (selectedValue) {
            $select.val(selectedValue);
            // Hide the text input if we successfully selected an existing/new worker
            if (selectId === '#trainWorkerSelect' && $select.val() !== "") {
                $('#trainNewWorkerName').hide();
            }
        } else if (!defaultLabel && res.workers.length > 0) {
            $select.val(res.workers[0].id);
        }
    });
}

// Initialize worker lists on page load
$(document).ready(function() {
    loadWorkers('basic_linucb', '#workerSelect', null);
    loadWorkers('basic_linucb', '#analysisWorkerSelect', 'Tous les workers');
    loadWorkers('basic_linucb', '#benchWorkerSelect', null);
    loadWorkers('basic_linucb', '#trainWorkerSelect', 'Nouveau worker...');
});

// When Play bandit type changes → reload play worker list
$('#banditType').change(function() {
    loadWorkers($(this).val(), '#workerSelect', null);
});

// When Analysis bandit type changes → reload analysis worker list and refresh data
$('#analysisBanditType').change(function() {
    loadWorkers($(this).val(), '#analysisWorkerSelect', 'Tous les workers');
    // Auto-refresh analysis after a short delay to let the select populate
    setTimeout(loadAnalysis, 300);
});

// When Analysis worker select changes → refresh data
$('#analysisWorkerSelect').change(function() {
    loadAnalysis();
});

// When Benchmark bandit type changes → reload bench worker list and refresh
$('#benchBanditType').change(function() {
    loadWorkers($(this).val(), '#benchWorkerSelect', null);
    setTimeout(loadBenchmarks, 300);
});

// When Benchmark worker select changes → refresh
$('#benchWorkerSelect').change(function() {
    loadBenchmarks();
});

// When Training bandit type changes → reload train worker list
$('#trainBanditType').change(function() {
    loadWorkers($(this).val(), '#trainWorkerSelect', 'Nouveau worker...');
    $('#trainNewWorkerName').show();
});

// When Training worker select changes → show/hide new worker name input
$('#trainWorkerSelect').change(function() {
    if ($(this).val() === "") {
        $('#trainNewWorkerName').show();
    } else {
        $('#trainNewWorkerName').hide();
    }
});

// Chess Board

function onDragStart (source, piece, position, orientation) {
  if (!isPlaying || isAgentThinking || game.game_over()) return false;
  if (currentMode !== "human_vs_mab") return false; 
  if (piece.search(/^b/) !== -1) return false; 
}

function onDrop (source, target) {
  let move = game.move({ from: source, to: target, promotion: 'q' });
  if (move === null) return 'snapback';
  sendHumanMove(move.from + move.to + (move.promotion ? move.promotion : ''));
}

function onSnapEnd () {
  board.position(game.fen());
}

const config = {
  draggable: true, position: 'start',
  onDragStart: onDragStart, onDrop: onDrop, onSnapEnd: onSnapEnd,
  pieceTheme: 'https://chessboardjs.com/img/chesspieces/wikipedia/{piece}.png'
};

board = Chessboard('board', config);

// Game Start

$('#startBtn').click(() => {
    currentMode = $('#gameMode').val();
    const sfLevel = $('#sfLevel').val();
    const timeControl = parseInt($('#timeControl').val());
    const banditType = $('#banditType').val();
    const workerId = $('#workerSelect').val();
    
    $.ajax({
        url: '/api/start', method: 'POST', contentType: 'application/json',
        data: JSON.stringify({
            mode: currentMode,
            sf_level: sfLevel,
            time_control: timeControl,
            bandit_type: banditType,
            worker_id: workerId || null
        }),
        success: (res) => {
            game.load(res.fen);
            board.position(res.fen);
            isPlaying = true;
            $history.empty();
            $currentArm.text("-"); $currentTime.text("-");
            updateClocks(res);
            startLiveClock();
            
            if (currentMode === "human_vs_mab") {
                setStatus("À vous de jouer", "playing");
            } else {
                setStatus("Simulation auto en cours...", "thinking");
                triggerAutoMove();
            }
        },
        error: (err) => {
            const message = err.responseJSON?.error || "Impossible de démarrer la partie";
            setStatus(message, "dot");
            isPlaying = false;
            isAgentThinking = false;
        }
    });
});

// Moves

function sendHumanMove(uci) {
    if (currentMode !== "human_vs_mab") return;
    isAgentThinking = true;
    setStatus("L'agent réfléchit...", "thinking");
    addHistoryItem("Vous", uci);
    
    $.ajax({
        url: '/api/move', method: 'POST', contentType: 'application/json',
        data: JSON.stringify({move: uci}),
        success: (res) => {
            game.load(res.fen);
            board.position(res.fen);
            updateClocks(res);
            
            if (res.game_over) {
                handleGameOver(res.result);
            } else {
                triggerAutoMove();
            }
        },
        error: (err) => {
            console.error(err); isAgentThinking = false;
            setStatus("Erreur", "dot"); game.undo(); board.position(game.fen());
        }
    });
}

function triggerAutoMove() {
    if (!isPlaying) return;
    isAgentThinking = true;
    
    $.ajax({
        url: '/api/auto_move', method: 'POST',
        success: (res) => {
            game.load(res.fen);
            board.position(res.fen);
            updateClocks(res);
            
            if (res.info) {
                addHistoryItem(res.info.turn, res.info.move, res.info.elapsed + "s", "Bras " + res.info.arm);
                if (res.info.turn.includes("MAB")) {
                    $currentArm.text(res.info.arm);
                    $currentTime.text(res.info.elapsed + " s");
                }
            }
            
            if (res.game_over) {
                handleGameOver(res.result);
            } else {
                if (currentMode === "human_vs_mab") {
                    isAgentThinking = false;
                    setStatus("À vous de jouer", "playing");
                } else {
                    setTimeout(triggerAutoMove, 400); 
                }
            }
        },
        error: (err) => {
            console.error(err); isPlaying = false; isAgentThinking = false;
            setStatus("Erreur auto_move", "dot");
        }
    });
}

function setStatus(text, dotClass) {
    $status.text(text);
    $dot.removeClass('playing thinking dot').addClass(dotClass);
}

function addHistoryItem(player, move, time="", arm="") {
    const timeStr = time ? `<span style="color:#94a3b8">${time} ${arm !== "Bras -" ? arm : ""}</span>` : "";
    $history.prepend(`<div class="history-item"><span><strong>${player}</strong>: ${move}</span>${timeStr}</div>`);
}

// Tabs

$('.tab-btn').click(function() {
    $('.tab-btn').removeClass('active'); $(this).addClass('active');
    $('.tab-content').hide(); $('#' + $(this).data('tab') + '-tab').fadeIn();
    if ($(this).data('tab') === 'analysis' || $(this).data('tab') === 'training') loadAnalysis();
    if ($(this).data('tab') === 'benchmarks') loadBenchmarks();
});

$('#refreshAnalysis').click(loadAnalysis);
$('#refreshBench').click(loadBenchmarks);

// Training

let trainPollInterval = null;

function pollTrainStatus() {
    $.get('/api/train_status', (res) => {
        if (res.is_training) {
            $('#trainProgressContainer').show();
            $('#trainStatusIndicator').show();
            $('#trainStatus').text('Modèle en cours de mise à jour...');
            loadTrainingChart();
        } else {
            if (trainPollInterval) {
                clearInterval(trainPollInterval);
                trainPollInterval = null;
                $('#trainStatus').text('Entraînement terminé !');
                $('#trainProgressContainer').hide();
                loadAnalysis();
            }
        }
    });
}

let currentTrainingWorkerId = null;

$('#startTrainBtn').click(() => {
    const games = $('#trainGames').val();
    const sf = $('#trainLevel').val();
    const tc = parseInt($('#trainTime').val());
    const banditType = $('#trainBanditType').val();
    
    let workerId = $('#trainWorkerSelect').val();
    if (!workerId) {
        workerId = $('#trainNewWorkerName').val().trim();
        if (!workerId) {
            workerId = "new_run_" + Math.floor(Date.now() / 1000);
        }
    }
    
    if (banditType === "neural_linucb" && !workerId.endsWith("_neural")) {
        workerId += "_neural";
    }
    currentTrainingWorkerId = workerId;

    $('#trainStatus').text('Entraînement lancé en arrière-plan...');
    $('#trainProgressContainer').show();
    $.ajax({
        url: '/api/train', method: 'POST', contentType: 'application/json',
        data: JSON.stringify({games, sf_level: sf, time_control: tc, bandit_type: banditType, worker_id: currentTrainingWorkerId}),
        success: () => {
            if (trainPollInterval) clearInterval(trainPollInterval);
            trainPollInterval = setInterval(pollTrainStatus, 1000);
            
            // Reload the worker lists in other tabs so the new worker appears
            setTimeout(() => {
                loadWorkers($('#trainBanditType').val(), '#trainWorkerSelect', 'Nouveau worker...', currentTrainingWorkerId);
                loadWorkers($('#banditType').val(), '#workerSelect', null);
                loadWorkers($('#analysisBanditType').val(), '#analysisWorkerSelect', 'Tous les workers');
                loadWorkers($('#benchBanditType').val(), '#benchWorkerSelect', null);
            }, 3000); // Give it some time to save the initial model
        }
    });
});

// Benchmark

$('#runBenchBtn').click(() => {
    const banditType = $('#benchBanditType').val();
    const workerId = $('#benchWorkerSelect').val();
    $('#benchStatus').text('Benchmark lancé... (peut prendre du temps)');
    $.ajax({
        url: '/api/run_benchmark', method: 'POST', contentType: 'application/json',
        data: JSON.stringify({
            bandit_type: banditType,
            worker_id: workerId || null
        }),
        success: () => {
            if (!benchPollInterval) benchPollInterval = setInterval(loadBenchmarks, 3000);
        }
    });
});

// Analysis

function loadAnalysis() {
    const workerId = $('#analysisWorkerSelect').val();
    const params = { bandit_type: $('#analysisBanditType').val() };
    if (workerId) {
        params.worker_id = workerId;
    }

    $.get('/api/analysis', params, (res) => {
        if (!res.stats && !res.recent_rewards) return;
        const ctxReward = document.getElementById('rewardChart').getContext('2d');
        if (rewardChart) rewardChart.destroy();
        rewardChart = new Chart(ctxReward, {
            type: 'line',
            data: {
                labels: Array.from({length: res.recent_rewards.length}, (_, i) => i + 1),
                datasets: [{ label: 'Récompense', data: res.recent_rewards, borderColor: '#38bdf8', tension: 0.3, fill: false }]
            },
            options: { responsive: true, plugins: { legend: { display: false } } }
        });
        
        const ctxArm = document.getElementById('armChart').getContext('2d');
        if (armChart) armChart.destroy();
        armChart = new Chart(ctxArm, {
            type: 'bar',
            data: {
                labels: Object.keys(res.arm_counts).map(a => "Bras " + a),
                datasets: [{ label: 'Utilisation', data: Object.values(res.arm_counts), backgroundColor: ['#818cf8', '#34d399', '#fbbf24', '#f87171'] }]
            },
            options: { responsive: true }
        });
        
        if (res.arm_time_stats) {
            const ctxArmTime = document.getElementById('armTimeChart').getContext('2d');
            if (armTimeChart) armTimeChart.destroy();
            armTimeChart = new Chart(ctxArmTime, {
                type: 'bar',
                data: {
                    labels: Object.keys(res.arm_time_stats).map(a => "Bras " + a),
                    datasets: [{ label: 'Temps Moyen (s)', data: Object.values(res.arm_time_stats), backgroundColor: ['#818cf8', '#34d399', '#fbbf24', '#f87171'] }]
                },
                options: { responsive: true }
            });
        }
        
        if (res.phase_stats) {
            const ctxPhase = document.getElementById('phaseChart').getContext('2d');
            if (phaseChart) phaseChart.destroy();
            
            // Sort the keys to ensure chronological order
            const keys = Object.keys(res.phase_stats).map(Number).sort((a,b) => a-b);
            const labels = keys.map(k => `Coups ${k}-${k+4}`);
            const data = keys.map(k => res.phase_stats[k]);
            
            phaseChart = new Chart(ctxPhase, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Temps moyen (secondes)',
                        data: data,
                        backgroundColor: 'rgba(16, 185, 129, 0.2)',
                        borderColor: '#10b981',
                        borderWidth: 2,
                        tension: 0.3,
                        fill: true
                    }]
                },
                options: { 
                    responsive: true,
                    scales: { y: { beginAtZero: true, title: { display: true, text: 'Secondes', color: '#94a3b8' } } }
                }
            });
        }
    });
}

function loadTrainingChart() {
    let workerId = currentTrainingWorkerId; // Use the locked ID
    if (!workerId) return; // Do nothing if training hasn't started or we don't know the ID

    const params = { worker_id: workerId, bandit_type: $('#trainBanditType').val() };

    $.get('/api/analysis', params, (res) => {
        if (!res.level_stats) return;
        
        const ctxLevel = document.getElementById('levelChart').getContext('2d');
        if (levelChart) levelChart.destroy();
        const datasets = [];
        const colors = ['#38bdf8', '#fbbf24', '#f87171', '#34d399', '#818cf8'];
        let i = 0;
        for (const [level, data] of Object.entries(res.level_stats)) {
            datasets.push({
                label: 'Niveau ' + level + ' (Moy: ' + data.avg_reward + ')',
                data: data.rewards,
                borderColor: colors[i % colors.length],
                tension: 0.3, fill: false
            });
            i++;
        }
        levelChart = new Chart(ctxLevel, {
            type: 'line',
            data: {
                labels: Array.from({length: datasets[0] ? datasets[0].data.length : 0}, (_, j) => j + 1),
                datasets: datasets
            },
            options: { 
                responsive: true,
                animation: false // Disable animation so it updates smoothly during polling
            }
        });
    });
}

// Benchmarks Loading

let benchPollInterval = null;

function loadBenchmarks() {
    const banditType = $('#benchBanditType').val();
    $.get('/api/benchmarks', { bandit_type: banditType }, (res) => {
        if (res.is_running) {
            $('#benchStatus').text("Benchmark en cours d'exécution...");
            if (!benchPollInterval) benchPollInterval = setInterval(loadBenchmarks, 3000);
        } else {
            $('#benchStatus').text(res.results ? 'Benchmark terminé.' : 'Aucun benchmark pour le moment.');
            if (benchPollInterval) {
                clearInterval(benchPollInterval);
                benchPollInterval = null;
            }
        }
        
        if (!res.results) {
            $('#benchmarkTable tbody').html('<tr><td colspan="5">Aucune donnée trouvée. (Lancez un benchmark)</td></tr>');
            return;
        }
        
        let tbody = '';
        const levels = []; const winrates = [];
        res.results.forEach(row => {
            tbody += `<tr><td>SF Lv ${row.level}</td><td>${row.wins}</td><td>${row.losses}</td><td>${row.draws}</td><td>${(row.winrate * 100).toFixed(1)}%</td></tr>`;
            levels.push('Niveau ' + row.level);
            winrates.push(row.winrate * 100);
        });
        $('#benchmarkTable tbody').html(tbody);
        
        const ctxBench = document.getElementById('benchmarkChart').getContext('2d');
        if (benchChart) benchChart.destroy();
        benchChart = new Chart(ctxBench, {
            type: 'line',
            data: {
                labels: levels,
                datasets: [{ label: 'Winrate (%)', data: winrates, borderColor: '#34d399', backgroundColor: 'rgba(52, 211, 153, 0.2)', tension: 0.3, fill: true }]
            },
            options: { responsive: true, scales: { y: { beginAtZero: true, max: 100 } } }
        });
    });
}

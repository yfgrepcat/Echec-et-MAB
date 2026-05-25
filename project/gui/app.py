import os
import sys
import json
import time
import glob
import threading
import subprocess
from pathlib import Path
from flask import Flask, request, jsonify, render_template
import chess
import chess.engine
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from mab_agent import ChessMAB, sanitize_bandit_config
from utils.time_manager import Clock
from experiments.training import run_training_session, DummyEngine

app = Flask(__name__)

game_state = {
    "board": None,
    "mab": None,
    "white_clock": None,
    "black_clock": None,
    "engine": None,
    "history": [],
    "mode": "human_vs_mab",
    "sf_level": 10,
    "mab_color": chess.BLACK,
    "last_turn_time": 0
}

training_state = {
    "is_training": False,
    "current_game": 0,
    "total_games": 0,
    "wins": 0,
    "losses": 0,
    "draws": 0
}

def update_training_progress(current, total, result):
    training_state["current_game"] = current
    training_state["total_games"] = total
    if result == "1-0":
        training_state["wins"] += 1
    elif result == "0-1":
        training_state["losses"] += 1
    else:
        training_state["draws"] += 1

import shutil
ENGINE_PATH = shutil.which("stockfish") or str(ROOT_DIR / "bin" / "stockfish")


def create_game_engine(skill_level: int):
    """Create the GUI chess engine.

    The GUI should not silently fall back to a random engine; if Stockfish is
    unavailable, we raise an explicit error so the UI can surface the problem.
    """
    if not ENGINE_PATH or not os.path.exists(ENGINE_PATH):
        raise RuntimeError(
            "Stockfish introuvable. Installez-le ou placez le binaire dans le PATH."
        )
    try:
        engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
        engine.configure({"Skill Level": skill_level})
        return engine
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Impossible de lancer Stockfish depuis {ENGINE_PATH}."
        ) from exc

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/start", methods=["POST"])
def start_game():
    data = request.json or {}
    mode = data.get("mode", "human_vs_mab")
    sf_level = int(data.get("sf_level", 10))
    time_control = data.get("time_control", 300)
    bandit_type = data.get("bandit_type", "basic_linucb")
    bandit_config = data.get("bandit_config", None)
    
    game_state["mode"] = mode
    game_state["sf_level"] = sf_level
    game_state["board"] = chess.Board()
    game_state["white_clock"] = Clock(time_control)
    game_state["black_clock"] = Clock(time_control)
    game_state["last_turn_time"] = time.time()
    game_state["history"] = []
    
    if game_state["engine"] is not None:
        try:
            game_state["engine"].quit()
        except:
            pass
            
    try:
        game_state["engine"] = create_game_engine(sf_level)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    
    try:
        bandit_config = sanitize_bandit_config(bandit_config)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    model_path = os.path.join(ROOT_DIR, "models", "final_model.npz")
    game_state["mab"] = ChessMAB(
        game_state["engine"],
        model_path=model_path,
        bandit_type=bandit_type,
        bandit_config=bandit_config,
    )
    game_state["mab"].load()
    
    if mode == "human_vs_mab":
        game_state["mab_color"] = chess.BLACK
    else:
        game_state["mab_color"] = chess.WHITE
        
    return jsonify({
        "fen": game_state["board"].fen(),
        "status": "started",
        "white_clock": time_control,
        "black_clock": time_control,
        "turn": "w"
    })

@app.route("/api/move", methods=["POST"])
def make_move():
    data = request.json
    user_move_uci = data.get("move")
    
    board = game_state["board"]
    if not board or board.is_game_over():
        return jsonify({"error": "Game over"}), 400
        
    try:
        user_move = chess.Move.from_uci(user_move_uci)
        if user_move not in board.legal_moves:
            return jsonify({"error": "Illegal move"}), 400
    except:
         return jsonify({"error": "Invalid move format"}), 400
         
    turn_color = board.turn
    active_clock = game_state["white_clock"] if turn_color == chess.WHITE else game_state["black_clock"]
    
    elapsed = time.time() - game_state["last_turn_time"]
    active_clock.spend(elapsed)

    board.push(user_move)
    game_state["history"].append({
        "turn": "Humain",
        "move": user_move_uci,
        "elapsed": round(elapsed, 2), "arm": "-", "budget": "-", "clock_left": round(active_clock.time_left, 2)
    })
    
    return jsonify({
        "fen": board.fen(),
        "game_over": board.is_game_over(),
        "result": board.result() if board.is_game_over() else None,
        "white_clock": round(game_state["white_clock"].time_left, 2),
        "black_clock": round(game_state["black_clock"].time_left, 2),
        "turn": "w" if board.turn == chess.WHITE else "b"
    })

@app.route("/api/auto_move", methods=["POST"])
def auto_move():
    board = game_state["board"]
    mode = game_state["mode"]
    mab = game_state["mab"]
    engine = game_state["engine"]
    
    if not board or board.is_game_over():
        return jsonify({"error": "Game over or not started"}), 400
        
    start_time = time.time()
    turn_color = board.turn
    is_mab_turn = (turn_color == game_state["mab_color"]) or (mode == "mab_vs_mab")
    active_clock = game_state["white_clock"] if turn_color == chess.WHITE else game_state["black_clock"]
    
    if is_mab_turn:
        move, arm, reward, elapsed, budget, _ = mab.play(board, active_clock, training=False)
        board.push(move)
        
        info = {
            "turn": "MAB (Blancs)" if turn_color == chess.WHITE else "MAB (Noirs)",
            "move": move.uci(),
            "arm": arm,
            "elapsed": round(elapsed, 2),
            "budget": round(budget, 2),
            "clock_left": round(active_clock.time_left, 2)
        }
    else:
        limit = chess.engine.Limit(white_clock=game_state["white_clock"].time_left, black_clock=game_state["black_clock"].time_left)
        result = engine.play(board, limit)
        move = result.move
        elapsed = time.time() - start_time
        active_clock.spend(elapsed)
        board.push(move)
        info = {
            "turn": f"Stockfish (Lv {game_state['sf_level']})",
            "move": move.uci(),
            "arm": "-",
            "elapsed": round(elapsed, 2),
            "budget": "-",
            "clock_left": round(active_clock.time_left, 2)
        }
        
    game_state["history"].append(info)
    
    if mode == "human_vs_mab":
        game_state["last_turn_time"] = time.time()
    
    return jsonify({
        "fen": board.fen(),
        "move": move.uci(),
        "info": info,
        "game_over": board.is_game_over(),
        "result": board.result() if board.is_game_over() else None,
        "white_clock": round(game_state["white_clock"].time_left, 2),
        "black_clock": round(game_state["black_clock"].time_left, 2),
        "turn": "w" if board.turn == chess.WHITE else "b"
    })
    
@app.route("/api/analysis", methods=["GET"])
def get_analysis():
    log_files = glob.glob(os.path.join(ROOT_DIR, "logs", "games_worker_*.jsonl"))
    rows = []
    for log_file in log_files:
        with open(log_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
                    
    if not rows:
        return jsonify({"stats": None})
        
    df = pd.DataFrame(rows)
    arm_counts = df["arm"].value_counts().to_dict()
    recent_rewards_series = df["reward"].clip(-15, 15).rolling(window=20).mean().dropna()
    recent_rewards = recent_rewards_series.tail(200).tolist()
    
    level_stats = {}
    if "stockfish_level" in df.columns:
        for level, group in df.groupby("stockfish_level"):
            smoothed = group["reward"].clip(-15, 15).rolling(window=20).mean().dropna()
            level_stats[str(level)] = {
                "avg_reward": round(group["reward"].mean(), 3),
                "rewards": smoothed.tail(200).tolist()
            }
            
    phase_stats = {}
    if "ply" in df.columns and "elapsed" in df.columns:
        df_phase = df.copy()
        df_phase['move_number'] = df_phase['ply'] // 2
        df_phase['phase_jeu'] = (df_phase['move_number'] // 5) * 5
        phase_data = df_phase.groupby('phase_jeu')['elapsed'].mean()
        phase_stats = {int(k): round(float(v), 2) for k, v in phase_data.items()}
        
    arm_time_stats = {}
    if "arm" in df.columns and "elapsed" in df.columns:
        arm_time_df = df[df["arm"] != "-"]
        if not arm_time_df.empty:
            arm_time_data = arm_time_df.groupby("arm")["elapsed"].mean()
            arm_time_stats = {str(k): round(float(v), 2) for k, v in arm_time_data.items()}
    
    return jsonify({
        "arm_counts": arm_counts,
        "recent_rewards": recent_rewards,
        "level_stats": level_stats,
        "phase_stats": phase_stats,
        "arm_time_stats": arm_time_stats
    })

@app.route("/api/train", methods=["POST"])
def run_train():
    data = request.json or {}
    games = int(data.get("games", 10))
    sf_level = int(data.get("sf_level", 10))
    time_control = int(data.get("time_control", 60))
    bandit_type = data.get("bandit_type", "basic_linucb")
    bandit_config = data.get("bandit_config", None)
    
    training_state.update({
        "is_training": True, "current_game": 0, "total_games": games,
        "wins": 0, "losses": 0, "draws": 0
    })
    
    try:
        bandit_config = sanitize_bandit_config(bandit_config)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    def train_task():
        run_training_session(
            worker_id=0,
            total_games=games,
            use_openings=True,
            use_random_positions=False,
            stockfish_level=sf_level,
            time_control=time_control,
            bandit_type=bandit_type,
            bandit_config=bandit_config,
            progress_callback=update_training_progress,
        )
        training_state["is_training"] = False
    threading.Thread(target=train_task).start()
    return jsonify({"status": "started"})

@app.route("/api/train_status", methods=["GET"])
def get_train_status():
    return jsonify(training_state)

benchmark_is_running = False

@app.route("/api/run_benchmark", methods=["POST"])
def run_benchmark():
    global benchmark_is_running
    if benchmark_is_running:
        return jsonify({"status": "already_running"})
        
    data = request.json or {}
    bandit_type = data.get("bandit_type", "basic_linucb")
        
    benchmark_is_running = True
    def bench_task():
        global benchmark_is_running
        subprocess.run([
            sys.executable,
            os.path.join(ROOT_DIR, "experiments", "benchmark_simulate.py"),
            "--simulate",
            "--runs", "1",
            "--games-per-run", "2",
            "--bandits", bandit_type
        ])
        benchmark_is_running = False
    threading.Thread(target=bench_task).start()
    return jsonify({"status": "started"})

@app.route("/api/benchmarks", methods=["GET"])
def get_benchmarks():
    global benchmark_is_running
    try:
        df = pd.read_csv(os.path.join(ROOT_DIR, "logs", "benchmark_results.csv"))
        return jsonify({
            "results": df.to_dict(orient="records"),
            "is_running": benchmark_is_running
        })
    except:
        return jsonify({"results": None, "is_running": benchmark_is_running})

if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)

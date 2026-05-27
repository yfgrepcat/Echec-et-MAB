import os
import re
import sys
import time
import threading
import subprocess
from pathlib import Path
from flask import Flask, request, jsonify, render_template
import chess
import chess.engine

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from mab_agent import ChessMAB, sanitize_bandit_config
from utils.time_manager import Clock
from experiments.training import run_training_session
from experiments.reporting import (
    load_benchmark_results,
    load_training_logs,
    summarize_training_logs,
)

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

@app.route("/api/workers", methods=["GET"])
def list_workers():
    """Return the list of trained workers available for a given bandit type."""
    bandit_type = request.args.get("bandit_type", "basic_linucb")
    models_dir = os.path.join(ROOT_DIR, "models")
    workers = []

    if not os.path.isdir(models_dir):
        return jsonify({"workers": []})

    for fname in sorted(os.listdir(models_dir)):
        if not fname.endswith(".npz"):
            continue
        if bandit_type == "neural_linucb":
            # Match worker_*_neural.npz
            m = re.match(r"^worker_(.+)_neural\.npz$", fname)
            if m:
                workers.append({"id": m.group(1) + "_neural", "filename": fname})
        else:
            # Match worker_*.npz but NOT worker_*_neural.npz
            if "_neural.npz" in fname:
                continue
            m = re.match(r"^worker_(.+)\.npz$", fname)
            if m:
                workers.append({"id": m.group(1), "filename": fname})

    return jsonify({"workers": workers})

@app.route("/api/start", methods=["POST"])
def start_game():
    data = request.json or {}
    mode = data.get("mode", "human_vs_mab")
    sf_level = int(data.get("sf_level", 10))
    time_control = data.get("time_control", 300)
    bandit_type = data.get("bandit_type", "basic_linucb")
    bandit_config = data.get("bandit_config", None)
    worker_id = data.get("worker_id", None)
    
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

    # Build model path from worker_id if provided
    if worker_id:
        model_path = os.path.join(ROOT_DIR, "models", f"worker_{worker_id}.npz")
    else:
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
    worker_id = request.args.get("worker_id", None)
    bandit_type = request.args.get("bandit_type", "basic_linucb")

    df = load_training_logs(worker_id=worker_id, bandit_type=bandit_type)
    return jsonify(summarize_training_logs(df))

@app.route("/api/train", methods=["POST"])
def run_train():
    data = request.json or {}
    games = int(data.get("games", 10))
    sf_level = int(data.get("sf_level", 10))
    time_control = int(data.get("time_control", 60))
    bandit_type = data.get("bandit_type", "basic_linucb")
    bandit_config = data.get("bandit_config", None)
    
    worker_id = data.get("worker_id", "")
    if not worker_id:
        worker_id = f"new_run_{int(time.time())}"
        
    if bandit_type == "neural_linucb" and not worker_id.endswith("_neural"):
        worker_id += "_neural"

    
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
            worker_id=worker_id,
            total_games=games,
            use_openings=True,
            use_random_positions=False,
            stockfish_level=sf_level,
            agent_stockfish_level=10,
            opponent_stockfish_level=sf_level,
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
    worker_id = data.get("worker_id", None)

    # Build model path from worker_id
    if worker_id:
        model_path = os.path.join(ROOT_DIR, "models", f"worker_{worker_id}.npz")
    else:
        model_path = os.path.join(ROOT_DIR, "models", "final_model.npz")

    benchmark_is_running = True
    def bench_task():
        global benchmark_is_running
        try:
            cmd = [
                sys.executable,
                os.path.join(ROOT_DIR, "experiments", "benchmark.py"),
                "--model-path", model_path,
                "--bandit-type", bandit_type,
                "--games-per-level", "2",
                "--levels", "5", "8", "10", "12",
                "--agent-stockfish-level", "10",
            ]
            subprocess.run(cmd)
        finally:
            benchmark_is_running = False
    threading.Thread(target=bench_task).start()
    return jsonify({"status": "started"})

@app.route("/api/benchmarks", methods=["GET"])
def get_benchmarks():
    global benchmark_is_running
    bandit_type = request.args.get("bandit_type", "basic_linucb")
    try:
        df = load_benchmark_results(bandit_type)
        if df.empty:
            return jsonify({"results": None, "is_running": benchmark_is_running})
        return jsonify({
            "results": df.to_dict(orient="records"),
            "is_running": benchmark_is_running
        })
    except:
        return jsonify({"results": None, "is_running": benchmark_is_running})

if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)

import multiprocessing as mp

from experiments.training import run_training_session


NUM_WORKERS = 4

GAMES_PER_WORKER = 50

STOCKFISH_LEVELS = [1, 5, 10, 15]

BANDIT_TYPE = "basic_linucb"
BANDIT_CONFIG = {}


def worker(worker_id):
    run_training_session(
        worker_id=worker_id,
        total_games=GAMES_PER_WORKER,
        use_openings=False,
        use_random_positions=True,
        stockfish_level=STOCKFISH_LEVELS[worker_id % len(STOCKFISH_LEVELS)],
        bandit_type=BANDIT_TYPE,
        bandit_config=BANDIT_CONFIG,
    )


if __name__ == "__main__":
    processes: list[mp.Process] = []
    for i in range(NUM_WORKERS):
        p = mp.Process(target=worker, args=(i,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

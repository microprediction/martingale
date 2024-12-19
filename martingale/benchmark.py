# benchmark.py
import numpy as np
import multiprocessing as mp
from evaluation import evaluator

def gen_factory():
    def simple_gen(n=20000):
        x = 0
        for i in range(n):
            x += np.random.randn()
            yield {'x': x}
    return simple_gen

class SimpleModel:
    def __init__(self):
        self.mean = 0.0
        self.count = 0
    def update(self, x):
        self.count += 1
        self.mean += (x - self.mean) / self.count
    def get_mean(self):
        return self.mean

def worker_evaluator(gen_factory, model_cls_list, worker_runs, n, burn_in, result_queue):
    epoch_data = {}
    for _ in range(worker_runs):
        gen = gen_factory()
        run_results = evaluator(gen, model_cls_list, n=n, burn_in=burn_in)
        for res in run_results:
            epoch = res['epoch']
            model = res['model']
            w = res['epoch_importance']
            mean_err = res['mean_error']

            key = (epoch, model)
            if key not in epoch_data:
                epoch_data[key] = {'weighted_sum': 0.0, 'weight': 0.0}
            epoch_data[key]['weighted_sum'] += mean_err * w
            epoch_data[key]['weight'] += w

    aggregated_results = []
    for (epoch, model_name), vals in epoch_data.items():
        total_weight = vals['weight'] if vals['weight'] != 0 else 1.0
        mean_error = vals['weighted_sum'] / total_weight
        aggregated_results.append({
            'model': model_name,
            'epoch': epoch,
            'mean_error': mean_error,
            'epoch_importance': total_weight
        })

    result_queue.put(aggregated_results)


def benchmark(gen_factory, model_cls_list, runs=2, worker_runs=50, n=10000, burn_in=10000):
    result_queue = mp.Queue()
    processes = []

    for _ in range(runs):
        p = mp.Process(target=worker_evaluator, args=(gen_factory, model_cls_list, worker_runs, n, burn_in, result_queue))
        p.start()
        processes.append(p)

    all_worker_results = []
    for _ in range(runs):
        worker_res = result_queue.get()
        all_worker_results.append(worker_res)

    for p in processes:
        p.join()

    final_epoch_data = {}
    for worker_res in all_worker_results:
        for res in worker_res:
            epoch = res['epoch']
            model = res['model']
            w = res['epoch_importance']
            mean_err = res['mean_error']

            key = (epoch, model)
            if key not in final_epoch_data:
                final_epoch_data[key] = {'weighted_sum': 0.0, 'weight': 0.0}
            final_epoch_data[key]['weighted_sum'] += mean_err * w
            final_epoch_data[key]['weight'] += w

    final_results = []
    for (epoch, model_name), vals in final_epoch_data.items():
        total_weight = vals['weight'] if vals['weight'] != 0 else 1.0
        mean_error = vals['weighted_sum'] / total_weight
        final_results.append({
            'epoch': epoch,
            'model': model_name,
            'mean_error': mean_error,
            'std_error': np.sqrt(mean_error)
        })

    final_results.sort(key=lambda x: (x['epoch'], x['model']))
    return final_results


if __name__ == "__main__":
    model_cls_list = [SimpleModel]
    results = benchmark(gen_factory, model_cls_list, runs=2, worker_runs=5, n=200, burn_in=9800)
    print("Final Aggregated Benchmark Results:")
    for r in results:
        print(r)

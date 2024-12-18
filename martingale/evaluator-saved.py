import numpy as np
import multiprocessing as mp
from martingale.stats.fewvar import FEWVar
from martingale.strhash import str_hash


def evaluator(gen, model_cls_list, n=2000, epoch_len=200, epoch_fading_factor=0.1, burn_in=10000 ):
    """
    Evaluate multiple models on a process generated by `gen`.
    We:
    - Burn in the generator for `burn_in` steps.
    - Compute ergodic std of dx during burn-in for normalization.
    - Run the next `n` steps, updating models and recording normalized errors.
    Returns a dictionary with summary statistics of model performance.
    """
    error_fading_factor = 1/epoch_len

    # Burn-in: Collect data to compute ergodic std
    burn_in_x = []
    count = 0
    epoch = 0
    for obs in gen:
        burn_in_x.append(obs['x'])
        count += 1
        if count >= burn_in:
            break

    burn_in_x = np.array(burn_in_x)
    # Compute dx = difference in consecutive x
    dx = np.diff(burn_in_x)
    ergodic_std = np.std(dx)
    # Avoid division by zero; if ergodic_std is zero, just set it to 1.0
    if ergodic_std == 0:
        ergodic_std = 1.0

    # Initialize models
    models = [cls() for cls in model_cls_list]

    # FEWVar instances to track performance: We'll track (x - model_mean)/ergodic_std
    # for each model separately.
    model_performance = [FEWVar(fading_factor=error_fading_factor) for _ in models]

    # Now run the next n steps
    steps = 0
    epoch_importance = 1
    all_epoch_results = list()
    stream = str_hash() # random hash string
    for obs in gen:
        if steps >= n:
            break
        x_true = obs['x']

        for m_idx, model in enumerate(models):
            model.update(x=x_true)
            model_mean = model.get_mean()
            error = (x_true - model_mean) / ergodic_std
            model_performance[m_idx].update(error)

        steps += 1

        if steps % epoch_len==0:
            epoch += 1
            epoch_results = {}
            for m_idx, cls in enumerate(model_cls_list):
                mean_err = model_performance[m_idx].get_mean()
                var_err = model_performance[m_idx].get_var()
                epoch_results[cls.__name__] = {
                    'stream':stream,
                    'epoch':epoch,
                    'mean_error': mean_err,
                    'var_error': var_err,
                    'std_error': np.sqrt(var_err),
                    'epoch_importance':epoch_importance
                }

            epoch_importance *= epoch_fading_factor

            all_epoch_results.append(epoch_results)
    return all_epoch_results

def worker_evaluator(gen_factory, model_cls_list, n, burn_in, result_queue):
    """
    Worker process function:
    - Creates a new generator from gen_factory
    - Calls evaluator
    - Puts results in result_queue
    """
    gen = gen_factory()
    res = evaluator(gen, model_cls_list, n=n, burn_in=burn_in)
    result_queue.put(res)

def benchmark(gen_factory, model_cls_list, runs=4, n=200, burn_in=9800):
    """
    Benchmark multiple models by running multiple independent realizations
    of the process in parallel using multiprocessing.
    gen_factory: a callable that returns a new generator each time it's called.
    model_cls_list: list of model classes
    runs: number of independent runs
    """

    result_queue = mp.Queue()
    processes = []

    for _ in range(runs):
        p = mp.Process(target=worker_evaluator, args=(gen_factory, model_cls_list, n, burn_in, result_queue))
        p.start()
        processes.append(p)

    # Collect results
    all_results = []
    for _ in range(runs):
        res = result_queue.get()
        all_results.append(res)

    # Wait for all processes to finish
    for p in processes:
        p.join()

    # Aggregate results across all runs
    # We'll average the mean_error and var_error across runs for each model
    aggregated = {}
    for model_cls in model_cls_list:
        name = model_cls.__name__
        mean_errors = [r[name]['mean_error'] for r in all_results]
        var_errors = [r[name]['var_error'] for r in all_results]

        aggregated[name] = {
            'mean_error': np.mean(mean_errors),
            'var_error': np.mean(var_errors),
            'std_error': np.sqrt(np.mean(var_errors))
        }

    return aggregated

# Example usage (adjust to your actual environment):
# Suppose we have a factory function that returns a new generator:
# def gen_factory():
#     params = brown_parameters()
#     return brown_gen(n=20000, params=params)

# Suppose we have a list of model classes:
# model_cls_list = [MyModelClass1, MyModelClass2, MyModelClass3]

# Now we can run:
# results = benchmark(gen_factory, model_cls_list, runs=4, n=200, burn_in=9800)
# print("Benchmark Results:", results)
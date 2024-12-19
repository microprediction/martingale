from martingale.evaluation import evaluator
from martingale.benchmarks.brownbenchmark import BrownBenchmark
from martingale.processes.browngen import brown_gen
from martingale.stats.fewmean import FEWMean


def brown_challenge(challengers):
    return evaluator(gen=brown_gen,model_cls_list=challengers+[BrownBenchmark])


if __name__=='__main__':
    challengers = [FEWMean]
    report = brown_challenge(challengers=challengers)
    print(report)
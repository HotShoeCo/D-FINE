class BestStat:
    def __init__(self, metric_index: int = 0):
        """
        metric_index: which element to pull from each coco_eval list for comparison.
        Metrics are now scalar values extracted at the configured index.
        """
        self.metric_index = metric_index
        self._pareto_front = []

    def reset(self):
        self._pareto_front = []

    def update(self, epoch, stats):
        current_stats = {k: stats[k] for k in stats if k.startswith("coco_eval")}
        current_stats["epoch"] = epoch
        to_keep = []
        is_current_pareto = True

        for saved_stats in self._pareto_front:
            if self.dominates(current_stats, saved_stats):
                pass
            elif self.dominates(saved_stats, current_stats):
                is_current_pareto = False
                to_keep.append(saved_stats)
            else:
                to_keep.append(saved_stats)

        if is_current_pareto:
            to_keep.append(current_stats)
        
        self._pareto_front = to_keep

    def get_best(self):
        return self._pareto_front[-1] if self._pareto_front else {"epoch": -1}

    def is_current_best(self, epoch):
        return any(stat["epoch"] == epoch for stat in self._pareto_front)

    def dominates(self, model_a_stats, model_b_stats):
        keys = [k for k in model_a_stats if k.startswith("coco_eval")]
        idx = self.metric_index
        a_better_or_equal = all(
            model_a_stats[k][idx] >= model_b_stats[k][idx]
            for k in keys
        )
        a_strictly_better = any(
            model_a_stats[k][idx] > model_b_stats[k][idx]
            for k in keys
        )
        return a_better_or_equal and a_strictly_better

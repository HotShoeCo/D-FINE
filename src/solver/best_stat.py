class BestStatTracker:
    def __init__(self, mode="single"):
        self.mode = mode
        self._best_stat = {"epoch": -1}
        self._pareto_front = []

    def reset(self):
        self._best_stat = {"epoch": -1}
        self._pareto_front = []

    def update(self, current_stats):
        if self.mode == "single":
            self._update_single(current_stats)
        elif self.mode == "pareto":
            self._update_pareto(current_stats)
        else:
            raise ValueError(f"Unknown BestStatTracker mode: {self.mode}")

    def _update_single(self, current_stats):
        is_better = all(
            (k not in self._best_stat or current_stats[k] >= self._best_stat[k])
            for k in current_stats if k.startswith("coco_eval")
        )
        if is_better:
            self._best_stat["epoch"] = current_stats["epoch"]
            for k in current_stats:
                if k.startswith("coco_eval"):
                    self._best_stat[k] = current_stats[k]

    def _update_pareto(self, current_stats):
        self._pareto_front = self._update_pareto_front(current_stats, self._pareto_front)

    def get_best_model_stats(self):
        """Returns the stats of the 'best' model based on the tracker's mode."""
        if self.mode == "single":
            return self._best_stat
        elif self.mode == "pareto":
            return self._pareto_front[-1] if self._pareto_front else {"epoch": -1}
        else:
            raise ValueError(f"Unknown BestStatTracker mode: {self.mode}")

    def is_current_best(self, current_epoch):
        """Checks if the current epoch's stats represent the 'best' based on the tracker's mode."""
        if self.mode == "single":
            return current_epoch == self._best_stat.get("epoch", -1)
        elif self.mode == "pareto":
            return any(stat["epoch"] == current_epoch for stat in self._pareto_front)
        else:
            raise ValueError(f"Unknown BestStatTracker mode: {self.mode}")

    def dominates(self, model_a_stats, model_b_stats):
        a_better_or_equal = all(model_a_stats[key] >= model_b_stats[key]
                               for key in model_a_stats if key.startswith("coco_eval"))
        a_strictly_better = any(model_a_stats[key] > model_b_stats[key]
                             for key in model_a_stats if key.startswith("coco_eval"))
        return a_better_or_equal and a_strictly_better

    def _update_pareto_front(self, current_stats, pareto_front):
        to_keep = []
        is_current_pareto = True

        for saved_stats in pareto_front:
            if self.dominates(current_stats, saved_stats):
                pass
            elif self.dominates(saved_stats, current_stats):
                is_current_pareto = False
                to_keep.append(saved_stats)
            else:
                to_keep.append(saved_stats)

        if is_current_pareto:
            to_keep.append(current_stats)
        return to_keep
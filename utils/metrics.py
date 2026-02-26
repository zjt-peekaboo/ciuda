# utils/metrics.py
import numpy as np


class AverageAccuracy:
    """Track average accuracy across tasks"""

    def __init__(self):
        self.accuracies = []

    def update(self, task_id, accuracy):
        """Update accuracy for a task"""
        if task_id >= len(self.accuracies):
            self.accuracies.append(accuracy)
        else:
            self.accuracies[task_id] = accuracy

    def get_average(self):
        """Get average accuracy"""
        if len(self.accuracies) == 0:
            return 0.0
        return np.mean(self.accuracies)

    def get_all(self):
        """Get all accuracies"""
        return self.accuracies


class ForgettingMeasure:
    """Track forgetting measure across tasks"""

    def __init__(self, num_tasks):
        self.num_tasks = num_tasks
        self.max_accuracies = [0.0] * num_tasks
        self.current_accuracies = [0.0] * num_tasks

    def update(self, task_id, accuracy):
        """Update accuracy for a task"""
        self.current_accuracies[task_id] = accuracy

        # Update max accuracy
        if accuracy > self.max_accuracies[task_id]:
            self.max_accuracies[task_id] = accuracy

    def get_forgetting(self):
        """Calculate average forgetting"""
        forgetting = []
        for i in range(len(self.max_accuracies)):
            if self.max_accuracies[i] > 0:
                forget = self.max_accuracies[i] - self.current_accuracies[i]
                forgetting.append(forget)

        if len(forgetting) == 0:
            return 0.0
        return np.mean(forgetting)

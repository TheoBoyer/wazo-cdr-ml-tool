"""

    Task

"""

class Task:
    """
        Task
    """
    def __init__(self, data, metric_type):
        self.data = data
        self.metric_type = metric_type

    def make_dataset(self):
        return self.data.make_dataset()

    def read_raw_data(self):
        return self.data.read_raw_data()

    def get_requirements(self):
        return self.data.requirements

    def save_pipeine(self, path):
        self.data.save(path)

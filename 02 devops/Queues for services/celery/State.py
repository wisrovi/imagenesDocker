"""
This program have a class heredeting luigi.Task
With the luigi.Task classes, you can define dependencies
between tasks and specify how they should be executed in
parallel or serially. Each task can have requirements, such
as the need for another task to complete before it can be executed.
This allows you to create complex workflows and control the
execution of tasks efficiently.
"""

from abc import abstractmethod
import os
import uuid

import luigi


class Luigi_base(luigi.Task):
    """
    Class to define dependencies between tasks and specify how
    they should be executed.
    """

    priority = 3
    task_family = "family"
    NAME_VAR = "text"

    kwargs = luigi.DictParameter(default={})
    name_file = str(uuid.uuid4()).replace("-", "")

    def output(self):
        folder = self.kwargs["folder"]
        return luigi.LocalTarget(os.path.join(folder, f"{self.name_file}.txt"))

    def requires(self):
        return self.depends_of()

    @abstractmethod
    def depends_of(self):
        return None

    def run(self):
        self.logic()

        folder = self.kwargs["folder"]
        with open(os.path.join(folder, f"{self.name_file}.txt"), "w") as file:
            file.write("File")

    def logic(self):
        """
        This method is for to apply the logic for task.
        create a database on redis and save the data.
        """

        pass

    @abstractmethod
    def execute(self, data: dict) -> dict:
        return {"text": "text"}

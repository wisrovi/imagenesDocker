import luigi
import os
import uuid
from abc import abstractmethod


class LuigiBase(luigi.Task):
    """
    Base class for defining task dependencies and execution logic.
    """

    kwargs = luigi.DictParameter(default={})
    name_file = str(uuid.uuid4()).replace("-", "")

    def output(self):
        # Define the output path for the task
        return luigi.LocalTarget(
            os.path.join(self.kwargs["folder"], f"{self.name_file}.txt")
        )

    def requires(self):
        # Define task dependencies based on `depends_of` method
        return self.depends_of()

    @abstractmethod
    def depends_of(self):
        # Abstract method to specify dependencies for the task
        return None

    def run(self):
        # Logic execution and writing output file
        self.logic()
        with open(self.output().path, "w") as file:
            file.write("Task completed")

    def logic(self):
        # Placeholder for task-specific logic
        pass

    @abstractmethod
    def execute(self, data: dict) -> dict:
        # Abstract method to execute task-specific operations
        return {}


class TaskA(LuigiBase):
    """
    First task in the sequence.
    """

    def depends_of(self):
        # TaskA has no dependencies
        return None

    def logic(self):
        # Specific logic for TaskA
        print("Executing TaskA")

    def execute(self, data: dict) -> dict:
        return {"result": "Result from TaskA"}


class TaskB(LuigiBase):
    """
    Second task that depends on TaskA.
    """

    def depends_of(self):
        # TaskB depends on TaskA
        return TaskA(kwargs=self.kwargs)

    def logic(self):
        # Specific logic for TaskB
        print("Executing TaskB")

    def execute(self, data: dict) -> dict:
        return {"result": "Result from TaskB"}


class TaskC(LuigiBase):
    """
    Third task that depends on TaskB.
    """

    def depends_of(self):
        # TaskC depends on TaskB
        return TaskB(kwargs=self.kwargs)

    def logic(self):
        # Specific logic for TaskC
        print("Executing TaskC")

    def execute(self, data: dict) -> dict:
        return {"result": "Result from TaskC"}


if __name__ == "__main__":
    # Define the output directory in kwargs
    kwargs = {"folder": "output_folder"}
    # Create the folder if it doesn't exist
    os.makedirs(kwargs["folder"], exist_ok=True)

    # Execute TaskC, which triggers TaskB and TaskA in sequence
    luigi.build(
        [TaskC(kwargs=kwargs)],  # Start with the last task in the sequence
        local_scheduler=False,  # Use the central scheduler (luigid)
        scheduler_host="localhost",
        workers=4,
        detailed_summary=True,
    )

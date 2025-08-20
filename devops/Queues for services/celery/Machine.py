"""
This class is used to create a state machine for the microservice.
"""

import datetime
from abc import abstractmethod
import uuid
from icecream import ic
import tempfile
import luigi

from eyesDcar_tools.redis.Data_shared import Data_shared
from eyesDcar_tools.kafka.Kafka import KAFKA
from eyesDcar_tools.redis.Queue import Queue
from eyesDcar_tools.config import Config

queue = Queue()


class StateMachine:
    """
    This class is used to create a state machine for the microservice.
    """

    result = None
    __id = None

    def __init__(self, idd: str = None):
        """
        This method is the constructor of the class.

        @type id: str
        @param id: unique identifier for the state machine
        """

        if idd is None:
            self.__id = str(uuid.uuid4()).replace("-", "")

        else:
            self.__id = idd

    @property
    def id(self):
        """
        This method is used to get the unique identifier of the state machine.

        @rtype: str
        @return: unique identifier of the state machine
        """

        return self.__id

    def __enter__(self):
        """
        This method is used to enter the context manager.

        @rtype: StateMachine
        @return: the state machine
        """

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        This method is used to exit the context manager.

        @type exc_type: type
        @param exc_type: type of exception

        @type exc_value: Exception
        @param exc_value: exception

        @type traceback: traceback
        @param traceback: traceback
        """

        ic(self.response_to, self.key)

        if (
            self.response_to is not None
            and self.key is not None
            and self.kwargs is not None
        ):
            kafka_util = KAFKA(self.response_to, producer=True)
            kafka_util.prepare_message_for_send(
                key=self.key,
                value=self.kwargs,
            )
            kafka_util.send_messages()

    def run(self, **kwargs: dict) -> dict:
        """
        This method is used to run the state machine.

        @type kwargs: dict
        @param kwargs: input data

        @rtype: dict
        @return: output data
        """

        self.response_to = kwargs.get("response_to", None)
        self.key = kwargs.get("key", None)

        start_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with Data_shared(self.__id) as db:
            db.save(kwargs)

        # This is the code that is executed by the state machine
        with tempfile.TemporaryDirectory() as temp_dir:
            data_ = {
                "id": self.__id,
                "folder": str(temp_dir),
                "launched": datetime.datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S",
                ),
            }

            self.process(data_)

        with Data_shared(self.__id) as db:
            result = db.read()

        # *************** summary ***************
        summary = {}
        summary["start_time"] = start_time_str
        end_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        summary["end_time"] = end_time_str
        elapsed_time = datetime.datetime.strptime(
            summary["end_time"], "%Y-%m-%d %H:%M:%S"
        ) - datetime.datetime.strptime(
            summary["start_time"],
            "%Y-%m-%d %H:%M:%S",
        )
        summary["elapsed_time"] = str(elapsed_time)
        # *************** end summary ***************

        output = {}
        output["output"] = result
        output["input"] = kwargs
        output["summary"] = summary

        with Data_shared(self.__id) as db:
            db.save(output)

        return result

    @abstractmethod
    def process(self, data_: dict) -> dict:
        """
        This method is used to process the data.

        Is presented as an abstract method, so it must be implemented
        in the child class.

        @type data_: dict
        @param data_: input data

        @rtype: dict
        @return: output data
        """


class StateMachine_luigi(StateMachine):
    STATE_START: callable = None

    def process(self, data_: dict = {}):
        """
        This override method is used to process the data.
        and run the state machine reporting the status of the task on luigi.

        @type data_: dict
        @param data_: input data

        @rtype: dict
        @return: output data
        """

        self.kwargs = data_  # data_ es un diccionario

        print("StateMachine", data_)

        if self.STATE_START is None:
            return {}

        if Config.LUIGI_LOCALHOST:
            luigi.build(
                [
                    self.STATE_START(kwargs=data_),
                ],
                workers=Config.LUIGI_WORKERS,
                detailed_summary=Config.LUIGI_SUMMARY,
            )
        else:
            luigi.build(
                [
                    self.STATE_START(kwargs=data_),
                ],
                scheduler_host=Config.LUIGI_SERVER,
                workers=Config.LUIGI_WORKERS,
                detailed_summary=Config.LUIGI_SUMMARY,
            )

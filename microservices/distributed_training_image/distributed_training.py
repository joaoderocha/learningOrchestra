import importlib
from concurrent.futures import ThreadPoolExecutor
from utils import Database, Data, Metadata, ObjectStorage
from constants import Constants
import traceback
from horovod.ray import RayExecutor


class Parameters:
    __DATASET_KEY_CHARACTER = "$"
    __CLASS_INSTANCE_CHARACTER = "#"
    __DATASET_WITH_OBJECT_KEY_CHARACTER = "."
    __REMOVE_KEY_CHARACTER = ""

    def __init__(self, database: Database, data: Data):
        self.__database_connector = database
        self.__data = data

    def treat(self, method_parameters: dict) -> dict:
        parameters = method_parameters.copy()

        for name, value in parameters.items():
            if type(value) is list:
                new_value = []
                for item in value:
                    new_value.append(self.__treat_value(item))
                parameters[name] = new_value
            else:
                parameters[name] = self.__treat_value(value)

        return parameters

    def __treat_value(self, value: object) -> object:
        if self.__is_dataset(value):
            dataset_name = self.__get_dataset_name_from_value(
                value)

            if self.__has_dot_in_dataset_name(value):
                object_name = self.__get_name_after_dot_from_value(value)
                return self.__data.get_object_from_dataset(
                    dataset_name, object_name)

            else:
                return self.__data.get_dataset_content(
                    dataset_name)

        elif self.__is_a_class_instance(value):
            return self.__get_a_class_instance(value)

        else:
            return value

    def __get_a_class_instance(self, class_code: str) -> object:
        class_instance_name = "class_instance"
        class_instance = None
        context_variables = {}

        class_code = class_code.replace(
            self.__CLASS_INSTANCE_CHARACTER,
            f'{class_instance_name}=')

        import tensorflow
        exec(class_code, locals(), context_variables)

        return context_variables[class_instance_name]

    def __is_a_class_instance(self, value: object) -> bool:
        if type(value) != str:
            return False
        else:
            return self.__CLASS_INSTANCE_CHARACTER in value

    def __is_dataset(self, value: object) -> bool:
        if type(value) != str:
            return False
        else:
            return self.__DATASET_KEY_CHARACTER in value

    def __get_dataset_name_from_value(self, value: str) -> str:
        dataset_name = value.replace(self.__DATASET_KEY_CHARACTER,
                                     self.__REMOVE_KEY_CHARACTER)
        return dataset_name.split(self.__DATASET_WITH_OBJECT_KEY_CHARACTER)[
            Constants.FIRST_ARGUMENT]

    def __has_dot_in_dataset_name(self, dataset_name: str) -> bool:
        return self.__DATASET_WITH_OBJECT_KEY_CHARACTER in dataset_name

    def __get_name_after_dot_from_value(self, value: str) -> str:
        return value.split(
            self.__DATASET_WITH_OBJECT_KEY_CHARACTER)[Constants.SECOND_ARGUMENT]


class Execution:
    __DATASET_KEY_CHARACTER = "$"
    __REMOVE_KEY_CHARACTER = ""

    def __init__(self,
                 database_connector: Database,
                 executor_name: str,
                 executor_service_type: str,
                 parent_name: str,
                 parent_name_service_type: str,
                 metadata_creator: Metadata,
                 class_method: str,
                 parameters_handler: Parameters,
                 storage: ObjectStorage,
                 executor: RayExecutor,
                 compile_code: str = '',
                 ):
        self.__metadata_creator = metadata_creator
        self.__thread_pool = ThreadPoolExecutor()
        self.__database_connector = database_connector
        self.__storage = storage
        self.__parameters_handler = parameters_handler
        self.executor_name = executor_name
        self.parent_name = parent_name
        self.class_method = class_method
        self.executor_service_type = executor_service_type
        self.parent_name_service_type = parent_name_service_type
        self.distributed_executor = executor
        self.compile_code = compile_code

    def create(self,
               module_path: str,
               class_name: str,
               method_parameters: dict,
               description: str) -> None:

        self.__metadata_creator.create_file(self.parent_name,
                                            self.executor_name,
                                            module_path,
                                            class_name,
                                            self.class_method,
                                            self.executor_service_type)

        self.__thread_pool.submit(self.__pipeline,
                                  module_path,
                                  method_parameters,
                                  description)

    def __pipeline(self,
                   module_path: str,
                   method_parameters: dict,
                   description: str) -> None:
        try:
            importlib.import_module(module_path)
            model_instance = self.__storage.read(self.parent_name,
                                                 self.parent_name_service_type)

            model_definition = model_instance.to_json()
            treated_parameters = self.__parameters_handler.treat(method_parameters)

            kwargs = self._build_parameters(
                model=model_definition,
                model_name=self.parent_name,
                training_parameters=treated_parameters,
                compile_code=self.compile_code,
            )

            self.distributed_executor.start(executable_cls=ExecutionBackground, executable_kwargs=kwargs)
            self.distributed_executor.execute(lambda worker: worker.compile)

            method_result = self.distributed_executor.execute(lambda worker: worker.train)

            model_instance.set_weights(method_result)

            self.__storage.save(method_result, self.executor_name,
                                self.executor_service_type)

            self.__metadata_creator.update_finished_flag(self.executor_name,
                                                         flag=True)
            self.distributed_executor.shutdown()


        except Exception as exception:
            traceback.print_exc()
            self.__metadata_creator.create_execution_document(
                self.executor_name,
                description,
                method_parameters,
                repr(exception))
            return None

        self.__metadata_creator.create_execution_document(self.executor_name,
                                                          description,
                                                          method_parameters,
                                                          )

    def __execute_a_object_method(self, class_instance: object, method: str,
                                  parameters: dict) -> object:
        class_method = getattr(class_instance, method)
        print('class_method', class_method, flush=True)
        treated_parameters = self.__parameters_handler.treat(parameters)
        print('treated_parameters', treated_parameters, flush=True)
        method_result = class_method(**treated_parameters)
        print('method_result', method_result, flush=True)
        if self.executor_service_type == Constants.TRAIN_TENSORFLOW_TYPE or \
                self.executor_service_type == Constants.TRAIN_SCIKITLEARN_TYPE or \
                method_result is None:
            return class_instance
        return method_result

    def _build_parameters(self, model: str, model_name: str, training_parameters: dict, compile_code: str):
        return {
            'model': model,
            'model_name': model_name,
            'training_parameters': training_parameters,
            'compile_code': compile_code
        }


class ExecutionBackground:
    def __init__(self, **kwargs):
        import tensorflow
        import horovod
        self.model = tensorflow.keras.models.model_from_json(kwargs['model'])
        self.model_name = kwargs['model_name']
        self.training_parameters = kwargs['training_parameters']
        self.compile_code = kwargs['compile_code']

    def compile(self):
        context = {self.model_name: self.model}
        if self.compile_code is not None:
            exec(self.compile_code, locals(), context)
        return self.model.optimizer is not None

    def train(self):
        self.model.fit(**self.training_parameters)
        return self.model.get_weights()

# compile_code = """
# import tensorflow as tf
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(0.001),
#     loss=tf.keras.losses.SparseCategoricalCrossentropy(),
#     metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
# )
# response = model
# """
#
# d = dict({
#     'module_path': 'tensorflow.keras.models',
#     'class_name': 'model',
#     'class': 'Sequential',
#     'class_parameters': {'layers': [tensorflow.keras.layers.Flatten(input_shape=(28, 28)),
#                                     tensorflow.keras.layers.Dense(128, activation='relu'),
#                                     tensorflow.keras.layers.Dense(10, activation='softmax')]},
#     'compile_code': compile_code
# })

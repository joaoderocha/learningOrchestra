import ast
import importlib
import types
from concurrent.futures import ThreadPoolExecutor
from utils import Database, Data, Metadata, ObjectStorage
from constants import Constants
from horovod.ray import RayExecutor
import tensorflow
import traceback
from training_function.train_function import train, get_rank


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
                print('parameter:', name, 'type: ', type(parameters[name]), flush=True)
            else:
                parameters[name] = self.__treat_value(value)
                print('parameter:', name, 'type: ', type(parameters[name]), flush=True)

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
        import horovod.tensorflow.keras as hvd
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

    def update(self,
               module_path: str,
               method_parameters: dict,
               description: str) -> None:
        self.__metadata_creator.update_finished_flag(self.executor_name, False)

        self.__thread_pool.submit(self.__pipeline,
                                  module_path,
                                  method_parameters,
                                  description)

    def __pipeline(self,
                   module_path: str,
                   method_parameters: dict,
                   description: str) -> None:
        try:
            print('rodando single')
            importlib.import_module(module_path)
            model_instance = self.__storage.read(self.parent_name,
                                                 self.parent_name_service_type)

            method_result = self.__execute_a_object_method(model_instance,
                                                           self.class_method,
                                                           method_parameters)

            self.__storage.save(method_result, self.executor_name,
                                self.executor_service_type)
            self.__metadata_creator.update_finished_flag(self.executor_name,
                                                         flag=True)

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

        treated_parameters = self.__parameters_handler.treat(parameters)
        method_result = class_method(**treated_parameters)

        if self.executor_service_type == Constants.TRAIN_TENSORFLOW_TYPE or \
                self.executor_service_type == Constants.TRAIN_SCIKITLEARN_TYPE or \
                method_result is None:
            return class_instance

        return method_result


class DistributedExecution(Execution):
    def __init__(self, database_connector: Database, executor_name: str, executor_service_type: str, parent_name: str,
                 parent_name_service_type: str, metadata_creator: Metadata, class_method: str,
                 parameters_handler: Parameters, storage: ObjectStorage, ray_executor: RayExecutor,
                 compile_code: str = '',
                 monitoring_path: str = ''):
        super().__init__(database_connector, executor_name, executor_service_type, parent_name,
                         parent_name_service_type, metadata_creator, class_method, parameters_handler, storage)
        self.__metadata_creator = metadata_creator
        self.__thread_pool = ThreadPoolExecutor()
        self.__database_connector = database_connector
        self.__storage = storage
        self.__parameters_handler = parameters_handler
        self.distributed_executor = ray_executor
        self.compile_code = compile_code
        self.monitoring_path = monitoring_path

    def start(self,
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
            print('rodando distribuido')
            importlib.import_module(module_path)
            rank0callbacks = method_parameters['rank0callbacks']
            del method_parameters['rank0callbacks']
            model_instance = self.__storage.read(self.parent_name,
                                                 self.parent_name_service_type)

            model_definition = model_instance.to_json()
            treated_parameters = self.__parameters_handler.treat(method_parameters)

            callbacks = method_parameters['callbacks']
            del treated_parameters['callbacks']

            a = self.distributed_executor.run(get_rank)
            print('ranks', a, flush=True)

            kwargs = dict({
                'model': model_definition,
                'model_name': self.parent_name,
                'training_parameters': treated_parameters,
                'compile_code': self.compile_code,
                'callbacks': callbacks,
                'rank0callbacks': rank0callbacks,
                'monitoring_path': self.monitoring_path
            })

            method_result = self.distributed_executor.run(train, kwargs=kwargs)

            self.__execute_a_object_method(model_instance, 'set_weights', dict({'weights': method_result[0]}))

            self.__storage.save(method_result, self.executor_name,
                                self.executor_service_type)
            self.__metadata_creator.update_finished_flag(self.executor_name,
                                                         flag=True)

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


class DistributedBuilderExecution(Execution):
    def __init__(self, database_connector: Database, executor_name: str, executor_service_type: str,
                 metadata_creator: Metadata,
                 parameters_handler: Parameters, storage: ObjectStorage, ray_executor: RayExecutor, code: str,
                 monitoring_path: str = ''):
        super().__init__(database_connector, executor_name, executor_service_type, '',
                         '', metadata_creator, '', parameters_handler, storage)
        self.__metadata_creator = metadata_creator
        self.__thread_pool = ThreadPoolExecutor()
        self.__database_connector = database_connector
        self.__storage = storage
        self.__parameters_handler = parameters_handler
        self.distributed_executor = ray_executor
        self.code = code
        self.monitoring_path = monitoring_path

    def build(self,
              name: str,
              code: str,
              monitoring_path: str,
              method_parameters: dict,
              description: str) -> None:
        print('tentei acessa banco')
        self.__database_connector.insert_one_in_file(
            name,
            dict({'code': code, 'monitoring_path': monitoring_path, 'description': description})
        )
        print('startando thread')
        self.__thread_pool.submit(self.__pipeline,
                                  code,
                                  method_parameters,
                                  description)

    def __pipeline(self,
                   code: str,
                   method_parameters: dict,
                   description: str) -> None:
        tree = ast.parse(code)

        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
            raise ValueError("provided code fragment is not a single function")

        comp = compile(code, filename="file.py", mode="single")
        func = types.FunctionType(comp.co_consts[0], {})

        kwargs = self.__parameters_handler.treat(method_parameters)

        method_result = self.distributed_executor.run(func, kwargs=kwargs)

        self.__storage.save(method_result, self.executor_name,
                            self.executor_service_type)
        self.__metadata_creator.update_finished_flag(self.executor_name,
                                                     flag=True)

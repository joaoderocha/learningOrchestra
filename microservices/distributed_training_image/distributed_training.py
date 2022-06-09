import importlib
from concurrent.futures import ThreadPoolExecutor
from numpy import array2string, ndarray
from utils import Database, Data, Metadata, ObjectStorage
from constants import Constants
import traceback
from training_function.train_function import train
from horovod.ray import RayExecutor
from ray.util import inspect_serializability


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
        print('\nparameters: ', parameters)
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
            print('\nis_dataset ', flush=True)
            dataset_name = self.__get_dataset_name_from_value(
                value)

            if self.__has_dot_in_dataset_name(value):
                object_name = self.__get_name_after_dot_from_value(value)

                a = self.__data.get_object_from_dataset(
                    dataset_name, object_name)
                print('\ntipo', type(a), flush=True)

                return array2string(a) if isinstance(a, ndarray) else a
            else:
                a = self.__data.get_dataset_content(
                    dataset_name)
                print('\ntipo', type(a), flush=True)

                return array2string(a) if isinstance(a, ndarray) else a
        elif self.__is_a_class_instance(value):
            print('\nis_a_class', flush=True)
            return self.__get_a_class_instance(value)

        else:
            return value

    def __get_a_class_instance(self, class_code: str) -> object:
        class_instance_name = "class_instance"
        class_instance = None
        context_variables = {}
        print('class_code: ', class_code, flush=True)
        class_code = class_code.replace(
            self.__CLASS_INSTANCE_CHARACTER,
            f'{class_instance_name}=')

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


def _build_parameters(model: str, model_name: str, training_parameters: dict, callbacks: str = '',
                      compile_code: str = ''):
    return {
        'model': model,
        'model_name': model_name,
        'training_parameters': training_parameters,
        'compile_code': compile_code,
        'callbacks': callbacks,
    }


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

            # tree = ast.parse(function_code)
            #
            # if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
            #     raise ValueError("provided code fragment is not a single function")
            #
            # comp = compile(function_code, filename="file.py", mode="single")
            # func = types.FunctionType(comp.co_consts[0], {})

            importlib.import_module(module_path)
            print('Starting executor...', flush=True)
            self.distributed_executor.start()
            print('executor ready...', flush=True)
            model_instance = self.__storage.read(self.parent_name,
                                                 self.parent_name_service_type)
            model_definition = model_instance.to_json()
            treated_parameters = self.__parameters_handler.treat(method_parameters)

            callbacks = method_parameters['callbacks']
            del treated_parameters['callbacks']
            compile_code = self.compile_code
            model_name = self.parent_name

            print('\nmodel definition', model_definition, type(model_definition), flush=True)
            inspect_serializability(model_definition, name="model_definition")
            print('\nmodel_name', model_name, type(model_name), flush=True)
            inspect_serializability(model_name, name="model_name")
            print('\ncompile_code', compile_code, type(compile_code), flush=True)
            inspect_serializability(compile_code, name="compile_code")
            print('\ncallbacks', callbacks, type(callbacks), flush=True)
            inspect_serializability(callbacks, name="callbacks")
            print('\ntraining_parameters', treated_parameters, type(treated_parameters), flush=True)
            inspect_serializability(treated_parameters, name="treated_parameters")

            inspect_serializability(train, name="func")

            kwargs = dict({
                'model': model_definition,
                'model_name': self.parent_name,
                'training_parameters': treated_parameters,
                'compile_code': self.compile_code,
                'callbacks': callbacks,
            })

            inspect_serializability(kwargs, name='kwargs')

            method_result = self.distributed_executor.run(train, kwargs=kwargs)

            print('method_results', method_result, f'\n len: {len(method_result)}', flush=True)

            self.__execute_a_object_method(model_instance, 'set_weights', dict({'weights': method_result[0]}))
            print('saving results to model...', flush=True)
            self.__storage.save(method_result, self.executor_name,
                                self.executor_service_type)
            print('updating flag...', flush=True)
            self.__metadata_creator.update_finished_flag(self.executor_name,
                                                         flag=True)

        except Exception as exception:
            print('error', exception, flush=True)
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

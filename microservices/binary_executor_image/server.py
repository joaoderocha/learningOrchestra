from flask import jsonify, request, Flask
import os
from binary_execution import Execution, Parameters, DistributedExecution, DistributedBuilderExecution
from utils import UserRequest, Database, ObjectStorage, Data, Metadata, ProcessController, find_port
from typing import Union, Tuple
from constants import Constants

import ray
from horovod.ray import RayExecutor
import horovod.tensorflow.keras as hvd
import training_function

address = f'{os.environ["NODE_IP_ADDRESS"]}:{os.environ["HOST_PORT"]}'
runtime_env = {"py_modules": [training_function], "pip": "./requirements.txt"}
ray.init(address=address, runtime_env=runtime_env)
settings = RayExecutor.create_settings(timeout_s=60, placement_group_timeout_s=60)
executor = RayExecutor(settings, use_gpu=False, cpus_per_worker=2, num_workers=2)

app = Flask(__name__)

database = Database(
    os.environ[Constants.DATABASE_URL],
    os.environ[Constants.DATABASE_REPLICA_SET],
    int(os.environ[Constants.DATABASE_PORT]),
    os.environ[Constants.DATABASE_NAME],
)
request_validator = UserRequest(database)
storage = ObjectStorage(database)
data = Data(database, storage)
metadata_creator = Metadata(database)
parameters_handler = Parameters(database, data)
process_controller = ProcessController()


@app.route(Constants.MICROSERVICE_URI_PATH, methods=["POST"])
def create_execution() -> jsonify:
    service_type = request.args.get(Constants.TYPE_FIELD_NAME)
    print('rodando single')
    model_name = request.json[Constants.MODEL_NAME_FIELD_NAME]
    parent_name = request.json[Constants.PARENT_NAME_FIELD_NAME]
    filename = request.json[Constants.NAME_FIELD_NAME]
    description = request.json[Constants.DESCRIPTION_FIELD_NAME]
    class_method = request.json[Constants.METHOD_FIELD_NAME]
    method_parameters = request.json[Constants.METHOD_PARAMETERS_FIELD_NAME]
    monitoring_path = ''
    try:
        monitoring_path = request.json[Constants.MONITORING_PATH_FIELD_NAME]
    except Exception:
        pass

    print(f'{model_name}, {parent_name}, {filename}, {description}, {class_method}, {method_parameters},'
          f' {monitoring_path}', flush=True)

    request_errors = analyse_post_request_errors(
        request_validator,
        data,
        filename,
        model_name,
        parent_name,
        class_method,
        method_parameters)

    print(f'{request_errors}', flush=True)

    if request_errors is not None:
        return request_errors

    monitoring_response = None
    if monitoring_path is not '':
        process_nickname, url = init_monitoring(filename, monitoring_path)
        monitoring_response = {
            'process_nickname': process_nickname,
            'url': url,
        }

    parent_name_service_type = data.get_type(parent_name)

    train_model = Execution(
        database,
        filename,
        service_type,
        parent_name,
        parent_name_service_type,
        metadata_creator,
        class_method,
        parameters_handler,
        storage
    )

    print(f'{train_model}', flush=True)

    module_path, class_name = data.get_module_and_class_from_a_instance(
        model_name)
    train_model.create(
        module_path, class_name, method_parameters, description)

    return (
        jsonify({
            Constants.MESSAGE_RESULT:
                f'{Constants.MICROSERVICE_URI_SWITCHER[service_type]}'
                f'{filename}{Constants.MICROSERVICE_URI_GET_PARAMS}',
            Constants.EXTRA_RESULTS: monitoring_response if monitoring_response is not None else {}
        }),
        Constants.HTTP_STATUS_CODE_SUCCESS_CREATED,
    )


@app.route(f'{Constants.MICROSERVICE_URI_PATH}/<filename>', methods=["PATCH"])
def update_execution(filename: str) -> jsonify:
    service_type = request.args.get(Constants.TYPE_FIELD_NAME)
    description = request.json[Constants.DESCRIPTION_FIELD_NAME]
    method_parameters = request.json[Constants.METHOD_PARAMETERS_FIELD_NAME]
    model_name = request.json[Constants.MODEL_NAME_FIELD_NAME]

    request_errors = analyse_patch_request_errors(
        request_validator,
        data,
        filename,
        model_name,
        method_parameters)

    if request_errors is not None:
        return request_errors

    module_path, function = data.get_module_and_class_from_a_instance(
        model_name)
    model_name = data.get_model_name_from_a_child(filename)
    method_name = data.get_class_method_from_a_executor_name(filename)

    parent_name_service_type = data.get_type(model_name)

    default_model = Execution(
        database,
        filename,
        service_type,
        model_name,
        parent_name_service_type,
        metadata_creator,
        method_name,
        parameters_handler,
        storage)

    default_model.update(
        module_path, method_parameters, description)

    return (
        jsonify({
            Constants.MESSAGE_RESULT:
                f'{Constants.MICROSERVICE_URI_SWITCHER[service_type]}'
                f'{filename}{Constants.MICROSERVICE_URI_GET_PARAMS}',
            Constants.EXTRA_RESULTS: {}
        }),
        Constants.HTTP_STATUS_CODE_SUCCESS_CREATED,
    )


@app.route(f'{Constants.MICROSERVICE_URI_PATH}/<filename>', methods=["DELETE"])
def delete_default_model(filename: str) -> jsonify:
    service_type = request.args.get(Constants.TYPE_FIELD_NAME)

    try:
        request_validator.existent_filename_validator(
            filename
        )
    except Exception as nonexistent_model_filename:
        return (
            jsonify(
                {Constants.MESSAGE_RESULT: str(nonexistent_model_filename)}),
            Constants.HTTP_STATUS_CODE_NOT_FOUND,
        )

    storage.delete(filename, service_type)

    return (
        jsonify({
            Constants.MESSAGE_RESULT: Constants.DELETED_MESSAGE,
            Constants.EXTRA_RESULTS: {}
        }),
        Constants.HTTP_STATUS_CODE_SUCCESS,
    )


@app.route(Constants.MICROSERVICE_URI_PATH, methods=['GET'])
def get_monitoring() -> jsonify:
    monitoring_nickname = request.json[Constants.MONITORING_NICKNAME_FIELD_NAME]
    url = process_controller.get_url(monitoring_nickname)
    if url is None:
        return (
            jsonify({Constants.MESSAGE_RESULT: {}}),
            Constants.HTTP_STATUS_CODE_NOT_FOUND
        )

    return (
        jsonify({
            Constants.MESSAGE_RESULT: url,
        }),
        Constants.HTTP_STATUS_CODE_SUCCESS
    )


@app.route(Constants.MICROSERVICE_DISTRIBUTED_TRAINING_URI_PATH, methods=['POST'])
def create_distributed_execution() -> jsonify:
    print('dist execution', flush=True)
    service_type = request.args.get(Constants.TYPE_FIELD_NAME)
    model_name = request.json[Constants.MODEL_NAME_FIELD_NAME]
    parent_name = request.json[Constants.PARENT_NAME_FIELD_NAME]
    filename = request.json[Constants.NAME_FIELD_NAME]
    description = request.json[Constants.DESCRIPTION_FIELD_NAME]
    class_method = request.json[Constants.METHOD_FIELD_NAME]
    method_parameters = request.json[Constants.METHOD_PARAMETERS_FIELD_NAME]
    compilation_code = request.json[Constants.COMPILATION_FIELD_NAME]
    monitoring_path = ''
    try:
        monitoring_path = request.json[Constants.MONITORING_PATH_FIELD_NAME]
    except Exception:
        pass

    parent_name_service_type = data.get_type(parent_name)

    train_model = DistributedExecution(
        database,
        filename,
        service_type,
        parent_name,
        parent_name_service_type,
        metadata_creator,
        class_method,
        parameters_handler,
        storage,
        executor,
        compilation_code,
        monitoring_path
    )

    module_path, class_name = data.get_module_and_class_from_a_instance(
        model_name)
    train_model.start(
        module_path, class_name, method_parameters, description)

    return (
        jsonify({
            Constants.MESSAGE_RESULT:
                f'{Constants.MICROSERVICE_URI_SWITCHER[service_type]}'
                f'{filename}{Constants.MICROSERVICE_URI_GET_PARAMS}',
        }),
        Constants.HTTP_STATUS_CODE_SUCCESS_CREATED,
    )


@app.route(Constants.MICROSERVICE_DISTRIBUTED_BUILDER_URI_PATH, methods=['POST'])
def create_builder_horovod() -> jsonify:
    print('builder execution', flush=True)
    service_type = request.args.get(Constants.TYPE_FIELD_NAME)
    filename = request.json[Constants.NAME_FIELD_NAME]
    description = request.json[Constants.DESCRIPTION_FIELD_NAME]
    method_parameters = request.json[Constants.METHOD_PARAMETERS_FIELD_NAME]
    code = request.json[Constants.CODE_FIELD_NAME]
    monitoring_path = ''
    try:
        monitoring_path = request.json[Constants.MONITORING_PATH_FIELD_NAME]
    except Exception:
        pass
    print(service_type, filename, description, method_parameters, code, monitoring_path, flush=True)
    train_model = DistributedBuilderExecution(
        database,
        filename,
        service_type,
        metadata_creator,
        parameters_handler,
        storage,
        executor,
        code,
        monitoring_path
    )
    print('criei executor', flush=True)
    train_model.build(filename,
                      code, monitoring_path, method_parameters, description)

    return (
        jsonify({
            Constants.MESSAGE_RESULT:
                f'{Constants.MICROSERVICE_URI_SWITCHER[service_type]}'
                f'{filename}{Constants.MICROSERVICE_URI_GET_PARAMS}',
        }),
        Constants.HTTP_STATUS_CODE_SUCCESS_CREATED,
    )


@app.route(Constants.MICROSERVICE_DISTRIBUTED_BUILDER_URI_PATH, methods=['PATCH'])
def update_builder_horovod() -> jsonify:
    pass


@app.route(f'{Constants.MICROSERVICE_DISTRIBUTED_BUILDER_URI_PATH}/<filename>', methods=['DELETE'])
def delete_builder_horovod(filename: str) -> jsonify:
    service_type = request.args.get(Constants.TYPE_FIELD_NAME)

    try:
        request_validator.existent_filename_validator(
            filename
        )
    except Exception as nonexistent_model_filename:
        return (
            jsonify(
                {Constants.MESSAGE_RESULT: str(nonexistent_model_filename)}),
            Constants.HTTP_STATUS_CODE_NOT_FOUND,
        )

    storage.delete(filename, service_type)

    return (
        jsonify({
            Constants.MESSAGE_RESULT: Constants.DELETED_MESSAGE,
            Constants.EXTRA_RESULTS: {}
        }),
        Constants.HTTP_STATUS_CODE_SUCCESS,
    )


def init_monitoring(filename, monitoring_path) -> Tuple[str, str]:
    process, process_nickname = process_controller.create_process(
        ['tensorboard', '--logdir', f'{monitoring_path}', '--bind_all'],
        process_nickname=f'{filename}_monitoring', monitoring_path=monitoring_path)
    port = find_port(process)
    url = process_controller.add_port(process_nickname, port)
    return process_nickname, url


def analyse_post_request_errors(request_validator: UserRequest,
                                data: Data,
                                filename: str,
                                model_name: str,
                                parent_name: str,
                                class_method: str,
                                method_parameters: dict) \
        -> Union[tuple, None]:
    try:
        request_validator.not_duplicated_filename_validator(
            filename
        )
    except Exception as duplicated_train_filename:
        return (
            jsonify({Constants.MESSAGE_RESULT: str(duplicated_train_filename)}),
            Constants.HTTP_STATUS_CODE_CONFLICT,
        )

    try:
        request_validator.existent_filename_validator(
            parent_name
        )
    except Exception as invalid_parent_name:
        return (
            jsonify({Constants.MESSAGE_RESULT: str(invalid_parent_name)}),
            Constants.HTTP_STATUS_CODE_NOT_ACCEPTABLE,
        )

    try:
        request_validator.existent_filename_validator(
            model_name
        )
    except Exception as invalid_parent_name:
        return (
            jsonify({Constants.MESSAGE_RESULT: str(invalid_parent_name)}),
            Constants.HTTP_STATUS_CODE_NOT_ACCEPTABLE,
        )

    module_path, class_name = data.get_module_and_class_from_a_instance(
        model_name)

    try:
        request_validator.valid_method_class_validator(
            module_path,
            class_name,
            class_method
        )
    except Exception as invalid_method_name:
        return (
            jsonify({Constants.MESSAGE_RESULT: str(invalid_method_name)}),
            Constants.HTTP_STATUS_CODE_NOT_ACCEPTABLE,
        )

    try:
        request_validator.valid_method_parameters_validator(
            module_path,
            class_name,
            class_method,
            method_parameters
        )
    except Exception as invalid_method_parameters:
        return (
            jsonify({Constants.MESSAGE_RESULT: str(invalid_method_parameters)}),
            Constants.HTTP_STATUS_CODE_NOT_ACCEPTABLE,
        )

    return None


def analyse_patch_request_errors(request_validator: UserRequest,
                                 data: Data,
                                 filename: str,
                                 model_name: str,
                                 method_parameters: dict) \
        -> Union[tuple, None]:
    try:
        request_validator.existent_filename_validator(
            filename
        )
    except Exception as nonexistent_train_filename:
        return (
            jsonify(
                {Constants.MESSAGE_RESULT: str(nonexistent_train_filename)}),
            Constants.HTTP_STATUS_CODE_NOT_FOUND,
        )

    module_path, class_name = data.get_module_and_class_from_a_executor_name(
        model_name)

    class_method = data.get_class_method_from_a_executor_name(filename)
    try:
        request_validator.valid_method_parameters_validator(
            module_path,
            class_name,
            class_method,
            method_parameters
        )
    except Exception as invalid_function_parameters:
        return (
            jsonify(
                {Constants.MESSAGE_RESULT: str(invalid_function_parameters)}),
            Constants.HTTP_STATUS_CODE_NOT_ACCEPTABLE,
        )

    return None


if __name__ == "__main__":
    print('flask', flush=True)
    print('Starting ray cluster...', flush=True)
    executor.start()
    app.run(
        host=os.environ["MICROSERVICE_IP"],
        port=int(os.environ["MICROSERVICE_PORT"])
    )


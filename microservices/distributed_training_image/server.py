from flask import jsonify, request, Flask
import os
from distributed_training import Execution, Parameters
from utils import UserRequest, Database, ObjectStorage, Data, Metadata
from typing import Union
from constants import Constants
import ray
from horovod.ray import RayExecutor
import horovod.tensorflow.keras as hvd

ray.init(address=f'{os.environ["NODE_IP_ADDRESS"]}:{os.environ["HOST_PORT"]}')

hvd.init()

settings = RayExecutor.create_settings(180)
executor = RayExecutor(settings, num_workers=2, use_gpu=False, )


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


@app.route(Constants.MICROSERVICE_URI_PATH, methods=["POST"])
def create_execution() -> jsonify:
    service_type = request.args.get(Constants.TYPE_FIELD_NAME)
    model_name = request.json[Constants.MODEL_NAME_FIELD_NAME]
    parent_name = request.json[Constants.PARENT_NAME_FIELD_NAME]
    filename = request.json[Constants.NAME_FIELD_NAME]
    description = request.json[Constants.DESCRIPTION_FIELD_NAME]
    class_method = request.json[Constants.METHOD_FIELD_NAME]
    method_parameters = request.json[Constants.METHOD_PARAMETERS_FIELD_NAME]
    compilation_code = request.json[Constants.COMPILATION_FIELD_NAME]
    print(f'{model_name}, {parent_name}, {filename}, {description}, {class_method}, {method_parameters},', flush=True)

    # request_errors = analyse_post_request_errors(
    #     request_validator,
    #     data,
    #     filename,
    #     model_name,
    #     parent_name,
    #     class_method,
    #     method_parameters)

    # print(f'{request_errors}', flush=True)
    #
    # if request_errors is not None:
    #     return request_errors

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
        storage,
        executor,
        compilation_code,
    )

    print(f'{train_model}', flush=True)

    module_path, class_name = data.get_module_and_class_from_a_instance(
        model_name)
    print('module_path: ', module_path, 'class_name: ', class_name)
    train_model.create(
        module_path, class_name, method_parameters, description)

    return (
        jsonify({
            Constants.MESSAGE_RESULT:
                f'{Constants.MICROSERVICE_URI_SWITCHER[service_type]}'
                f'{filename}{Constants.MICROSERVICE_URI_GET_PARAMS}',
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


# @app.route(Constants.MICROSERVICE_URI_PATH, methods=['GET'])
# def get_monitoring() -> jsonify:
#     monitoring_nickname = request.json[Constants.MONITORING_NICKNAME_FIELD_NAME]
#     url = process_controller.get_url(monitoring_nickname)
#     if url is None:
#         return (
#             jsonify({Constants.MESSAGE_RESULT: {}}),
#             Constants.HTTP_STATUS_CODE_NOT_FOUND
#         )
#
#     return (
#         jsonify({
#             Constants.MESSAGE_RESULT: url,
#         }),
#         Constants.HTTP_STATUS_CODE_SUCCESS
#     )
#
#
# def init_monitoring(filename, monitoring_path) -> Tuple[str, str]:
#     process, process_nickname = process_controller.create_process(
#         ['tensorboard', '--logdir', f'{monitoring_path}', '--bind_all'],
#         process_nickname=f'{filename}_monitoring', monitoring_path=monitoring_path)
#     port = find_port(process)
#     url = process_controller.add_port(process_nickname, port)
#     return process_nickname, url


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
    app.run(
        host=os.environ["MICROSERVICE_IP"],
        port=int(os.environ["MICROSERVICE_PORT"])
    )

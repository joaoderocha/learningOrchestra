import ast
import os
import types
from typing import Union

import ray
from flask import jsonify, request, Flask
from horovod.ray import RayExecutor

from code_execution import Parameters, Function, Execution, DistributedExecution
from constants import Constants
from utils import Data, Database, UserRequest, Metadata, ObjectStorage

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
function_treat = Function()


@app.route(Constants.MICROSERVICE_URI_PATH, methods=["POST"])
def create_execution() -> jsonify:
    print(f'REQUEST{request.json} \n \n', flush=True)
    filename = request.json[Constants.NAME_FIELD_NAME]
    description = request.json[Constants.DESCRIPTION_FIELD_NAME]
    service_type = request.args.get(Constants.TYPE_PARAM_NAME)
    function_parameters = request.json[Constants.FUNCTION_PARAMETERS_FIELD_NAME]
    function = request.json[Constants.FUNCTION_FIELD_NAME]
    distributed = request.json[Constants.DISTRIBUTED_FIELD_NAME]

    print(f'ATRIBUTO {Constants.DISTRIBUTED_FIELD_NAME} \n \n', flush=True)

    print(filename, description, service_type, function_parameters, function, distributed, flush=True)

    request_errors = analyse_post_request_errors(
        request_validator,
        filename
    )

    if request_errors is not None:
        return request_errors

    if distributed:
        execution = DistributedExecution(
            database,
            filename,
            service_type,
            storage,
            metadata_creator,
            parameters_handler,
            function_treat
        )
    else:
        execution = Execution(
            database,
            filename,
            service_type,
            storage,
            metadata_creator,
            parameters_handler,
            function_treat)

    execution.create(function, function_parameters, description)

    return (
        jsonify({
            Constants.MESSAGE_RESULT:
                f'{Constants.MICROSERVICE_URI_SWITCHER[service_type]}'
                f'{filename}{Constants.MICROSERVICE_URI_GET_PARAMS}'}),
        Constants.HTTP_STATUS_CODE_SUCCESS_CREATED,
    )


@app.route(f'{Constants.MICROSERVICE_URI_PATH}/<filename>', methods=["PATCH"])
def update_execution(filename: str) -> jsonify:
    service_type = request.args.get(Constants.TYPE_PARAM_NAME)
    description = request.json[Constants.DESCRIPTION_FIELD_NAME]
    function = request.json[Constants.FUNCTION_FIELD_NAME]
    function_parameters = request.json[Constants.FUNCTION_PARAMETERS_FIELD_NAME]

    request_errors = analyse_patch_request_errors(
        request_validator,
        filename)

    if request_errors is not None:
        return request_errors

    execution = Execution(
        database,
        filename,
        service_type,
        storage,
        metadata_creator,
        parameters_handler,
        function_treat)

    execution.update(function, function_parameters, description)

    return (
        jsonify({
            Constants.MESSAGE_RESULT:
                f'{Constants.MICROSERVICE_URI_SWITCHER[service_type]}'
                f'{filename}{Constants.MICROSERVICE_URI_GET_PARAMS}'}),
        Constants.HTTP_STATUS_CODE_SUCCESS_CREATED,
    )


@app.route(f'{Constants.MICROSERVICE_URI_PATH}/<filename>', methods=["DELETE"])
def delete_default_model(filename: str) -> jsonify:
    try:
        request_validator.existent_filename_validator(
            filename
        )
    except Exception as nonexistent_model_filename:
        return (
            jsonify(
                {Constants.MESSAGE_RESULT: str(nonexistent_model_filename)}),
            Constants.HTTP_STATUS_CODE_NOT_FOUND
        )

    storage.delete(filename)

    return (
        jsonify({
            Constants.MESSAGE_RESULT: Constants.DELETED_MESSAGE}),
        Constants.HTTP_STATUS_CODE_SUCCESS,
    )


def analyse_post_request_errors(request_validator: UserRequest,
                                filename: str) \
        -> Union[tuple, None]:
    try:
        request_validator.not_duplicated_filename_validator(
            filename
        )
    except Exception as duplicated_filename:
        return (
            jsonify({Constants.MESSAGE_RESULT: str(duplicated_filename)}),
            Constants.HTTP_STATUS_CODE_CONFLICT,
        )

    return None


def analyse_patch_request_errors(request_validator: UserRequest,
                                 filename: str) \
        -> Union[tuple, None]:
    try:
        request_validator.existent_filename_validator(
            filename
        )
    except Exception as nonexistent_filename:
        return (
            jsonify(
                {Constants.MESSAGE_RESULT: str(nonexistent_filename)}),
            Constants.HTTP_STATUS_CODE_NOT_FOUND,
        )

    return None


def runDistributed(function_code, args):
    tree = ast.parse(function_code)

    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError("provided code fragment is not a single function")

    comp = compile(function_code, filename="file.py", mode="single")
    func = types.FunctionType(comp.co_consts[0], {})
    submit(func, args)


def submit(func, args):
    ray.init(address="auto", _redis_password="5241590000000000")
    settings = RayExecutor.create_settings(timeout_s=30)
    executor = RayExecutor(
        settings,
        use_gpu=False,
        num_workers_per_host=1,
        num_hosts=3,
    )
    executor.start()
    executor.run(func, args, locals())
    executor.shutdown()


if __name__ == "__main__":
    app.run(
        host=os.environ["MICROSERVICE_IP"],
        port=int(os.environ["MICROSERVICE_PORT"])
    )

class Constants:
    MODULE_PATH_FIELD_NAME = "modulePath"
    CLASS_FIELD_NAME = "class"
    MODEL_NAME_FIELD_NAME = "modelName"
    PARENT_NAME_FIELD_NAME = "parentName"
    NAME_FIELD_NAME = "name"
    FINISHED_FIELD_NAME = "finished"
    DESCRIPTION_FIELD_NAME = "description"
    METHOD_FIELD_NAME = "method"
    METHOD_PARAMETERS_FIELD_NAME = "methodParameters"
    TYPE_FIELD_NAME = "type"
    EXCEPTION_FIELD_NAME = "exception"
    MONITORING_PATH_FIELD_NAME = "monitoringPath"
    EXTRA_RESULTS = "extra_results"
    MONITORING_NICKNAME_FIELD_NAME = "nickname"

    MODELS_VOLUME_PATH = "MODELS_VOLUME_PATH"
    BINARY_VOLUME_PATH = "BINARY_VOLUME_PATH"
    TRANSFORM_VOLUME_PATH = "TRANSFORM_VOLUME_PATH"
    CODE_EXECUTOR_VOLUME_PATH = "CODE_EXECUTOR_VOLUME_PATH"

    DELETED_MESSAGE = "deleted file"

    HTTP_STATUS_CODE_SUCCESS = 200
    HTTP_STATUS_CODE_SUCCESS_CREATED = 201
    HTTP_STATUS_CODE_CONFLICT = 409
    HTTP_STATUS_CODE_NOT_ACCEPTABLE = 406
    HTTP_STATUS_CODE_NOT_FOUND = 404
    GET_METHOD_NAME = "GET"

    DATABASE_URL = "DATABASE_URL"
    DATABASE_PORT = "DATABASE_PORT"
    DATABASE_NAME = "DATABASE_NAME"
    DATABASE_REPLICA_SET = "DATABASE_REPLICA_SET"

    ID_FIELD_NAME = "_id"
    METADATA_DOCUMENT_ID = 0

    MESSAGE_RESULT = "result"

    MODEL_SCIKITLEARN_TYPE = "model/scikitlearn"
    MODEL_TENSORFLOW_TYPE = "model/tensorflow"

    TUNE_SCIKITLEARN_TYPE = "tune/scikitlearn"
    TUNE_TENSORFLOW_TYPE = "tune/tensorflow"

    TRAIN_SCIKITLEARN_TYPE = "train/scikitlearn"
    TRAIN_TENSORFLOW_TYPE = "train/tensorflow"

    EVALUATE_SCIKITLEARN_TYPE = "evaluate/scikitlearn"
    EVALUATE_TENSORFLOW_TYPE = "evaluate/tensorflow"

    PREDICT_SCIKITLEARN_TYPE = "predict/scikitlearn"
    PREDICT_TENSORFLOW_TYPE = "predict/tensorflow"

    PYTHON_FUNCTION_TYPE = "function/python"
    DATASET_TENSORFLOW_TYPE = "dataset/tensorflow"

    TRANSFORM_SCIKITLEARN_TYPE = "transform/scikitlearn"
    TRANSFORM_TENSORFLOW_TYPE = "transform/tensorflow"

    EXPLORE_SCIKITLEARN_TYPE = "explore/scikitlearn"
    EXPLORE_TENSORFLOW_TYPE = "explore/tensorflow"

    MONITORING_TENSORFLOW_TYPE = "monitoring/tensorflow"
    COMPILATION_FIELD_NAME = "compile_code"

    API_PATH = "/api/learningOrchestra/v1/"

    MICROSERVICE_URI_SWITCHER = {
        TUNE_SCIKITLEARN_TYPE: f'{API_PATH}{TUNE_SCIKITLEARN_TYPE}',
        TUNE_TENSORFLOW_TYPE: f'{API_PATH}{TUNE_TENSORFLOW_TYPE}',
        TRAIN_SCIKITLEARN_TYPE: f'{API_PATH}{TRAIN_SCIKITLEARN_TYPE}',
        TRAIN_TENSORFLOW_TYPE: f'{API_PATH}{TRAIN_TENSORFLOW_TYPE}',
        EVALUATE_SCIKITLEARN_TYPE: f'{API_PATH}{EVALUATE_SCIKITLEARN_TYPE}',
        EVALUATE_TENSORFLOW_TYPE: f'{API_PATH}{EVALUATE_TENSORFLOW_TYPE}',
        PREDICT_SCIKITLEARN_TYPE: f'{API_PATH}{PREDICT_SCIKITLEARN_TYPE}',
        PREDICT_TENSORFLOW_TYPE: f'{API_PATH}{PREDICT_TENSORFLOW_TYPE}',
        MONITORING_TENSORFLOW_TYPE: f'{API_PATH}{MONITORING_TENSORFLOW_TYPE}',
    }

    MICROSERVICE_URI_PATH = "/binaryExecutor"
    MICROSERVICE_DISTRIBUTED_TRAINING_URI_PATH = "/distributedTraining"
    MICROSERVICE_DISTRIBUTED_BUILDER_URI_PATH = "/builderHorovod"
    MICROSERVICE_URI_GET_PARAMS = "?query={}&limit=20&skip=0"

    FIRST_ARGUMENT = 0
    SECOND_ARGUMENT = 1

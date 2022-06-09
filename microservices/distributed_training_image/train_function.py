import tensorflow
import numpy


class InstanceTreatment:
    __CLASS_INSTANCE_CHARACTER = "#"
    __REMOVE_KEY_CHARACTER = ""

    def __init__(self):
        pass

    def treat(self, method_parameters: []) -> []:
        parameters = method_parameters.copy()
        print('parameters: ', parameters)
        iterable = parameters if isinstance(parameters, list) else parameters.items()
        new_value = []
        for item in iterable:
            new_value.append(self.__treat_value(item))
        return new_value

    def __treat_value(self, value: object) -> object:
        if self.__is_a_class_instance(value):
            print('is_a_class', flush=True)
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

        import tensorflow
        import horovod.tensorflow.keras as hvd
        exec(class_code, locals(), context_variables)

        return context_variables[class_instance_name]

    def __is_a_class_instance(self, value: object) -> bool:
        if type(value) != str:
            return False
        else:
            return self.__CLASS_INSTANCE_CHARACTER in value


class ExecutionBackground:
    def __init__(self, **kwargs):
        print('model iniciado', flush=True)
        self.instanceTreatment = InstanceTreatment()
        self.model = tensorflow.keras.models.model_from_json(kwargs['model'])
        self.model_name = kwargs['model_name']
        self.training_parameters = dict({
            **kwargs['training_parameters'],
            'x': numpy.fromstring(kwargs['x']),
            'y': numpy.fromstring(kwargs['y']),
            'callbacks': self.instanceTreatment.treat(kwargs['callbacks'])
        })
        self.compile_code = kwargs['compile_code']
        print('modelo iniciado...', self.model, flush=True)

    def compile(self):
        print('buildando...', flush=True)
        context = {self.model_name: self.model}

        if self.compile_code is not None or '':
            exec(self.compile_code, locals(), context)
            print('Model compiled', flush=True)
        return self.model.optimizer is not None

    def train(self):
        print('treiando...', flush=True)
        self.model.fit(**self.training_parameters)
        return self.model.get_weights()


def train(*args, **kwargs):
    import horovod.tensorflow.keras as hvd
    hvd.init()
    exe = ExecutionBackground(**kwargs)
    exe.compile()
    return exe.train()

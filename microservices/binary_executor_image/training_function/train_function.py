from subprocess import Popen, PIPE, STDOUT
from typing import Tuple
from datetime import datetime
import tensorflow
import requests


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
        import horovod.tensorflow.keras as hvd
        hvd.init()
        print('model iniciado', flush=True)
        self.instanceTreatment = InstanceTreatment()
        self.model = tensorflow.keras.models.model_from_json(kwargs['model'])
        self.model_name = kwargs['model_name']
        self.callbacks = kwargs['callbacks'] + kwargs['rank0callbacks'] if hvd.rank() == 0 else kwargs['callbacks']
        self.monitoring_path = kwargs['monitoring_path']
        if self.monitoring_path is not '':
            self.monitoring_process = Popen(
                ['nohup', 'tensorboard', '--logdir', f'{self.monitoring_path}', '--port', '9500', '--bind_all'],
                stdout=PIPE, stderr=STDOUT)
        self.training_parameters = dict({
            **kwargs['training_parameters'],
            'callbacks': self.instanceTreatment.treat(self.callbacks)
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


class ProcessController:
    def __init__(self) -> None:
        self.__processDict = dict()
        self.__localhost = requests.get('https://api.ipify.org').text

    def create_process(self, arg_list: list, process_nickname: str, monitoring_path: str) -> Tuple[Popen, str]:
        nickname = process_nickname
        if nickname in self.__processDict:
            nickname = process_nickname + datetime.strftime("%Y%m%d-%H%M%S")

        process = Popen(arg_list, stdout=PIPE, stderr=STDOUT)
        self.__processDict[nickname] = {'process': process, 'path': monitoring_path}

        return process, nickname

    def kill_process(self, process_nickname: str) -> None:
        if process_nickname in self.__processDict:
            process = self.__processDict.pop(process_nickname)
            process.kill()

    def get_process(self, process_nickname: str) -> Popen:
        if process_nickname in self.__processDict:
            return self.__processDict.get(process_nickname)['process']

    def get_url(self, process_nickname: str) -> str:
        if process_nickname in self.__processDict:
            return self.__processDict.get(process_nickname)['url']

    def add_port(self, process_nickname: str, port: str) -> str:
        if process_nickname in self.__processDict:
            self.__processDict.get(process_nickname)['url'] = f'http://{self.__localhost}:{port}'
            return self.__processDict.get(process_nickname)['url']


def find_port(proc) -> str:
    for line in iter(proc.stdout.readline, b''):
        decoded_line = line.decode('utf-8')
        left_index = decoded_line.find('http://')
        if left_index > 0:
            right_index = decoded_line.rfind('/')
            url = decoded_line[left_index:right_index + 1]
            return url[url.rfind(':') + 1:url.rfind('/')]


def train(*args, **kwargs):
    import horovod.tensorflow.keras as hvd
    hvd.init()
    exe = ExecutionBackground(**kwargs)
    exe.compile()
    return exe.train()

import importlib
import json
import pickle

import tensorflow


def create_model(class_definition: dict):
    module = importlib.import_module(class_definition['module_path'])
    class_method = getattr(module, class_definition['class'])
    class_instance = class_method(**class_definition['class_parameters'])
    context = {class_definition['class_name']: class_instance}
    exec(class_definition['compile_code'], None, context)

    return context[class_definition['class_name']]


compile_code = """
import tensorflow as tf
import horovod.tensorflow.keras as hvd

hvd.init()

print(hvd.rank())

model.compile(
    optimizer=hvd.DistributedOptimizer(tf.optimizers.SGD(lr=0.0125 * hvd.size(), momentum=0.9)),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

response = model
"""

d = dict({
    'module_path': 'tensorflow.keras.models',
    'class_name': 'model',
    'class': 'Sequential',
    'class_parameters': {'layers': [tensorflow.keras.layers.Flatten(input_shape=(28, 28)),
                                    tensorflow.keras.layers.Dense(128, activation='relu'),
                                    tensorflow.keras.layers.Dense(10, activation='softmax')]},
    'compile_code': compile_code
})

a = create_model(d)

model_config = a.to_json()
print(model_config)


class ExecutionBackground:
    def __init__(self, *args, **kwargs):
        import tensorflow
        import horovod
        self.model = tensorflow.keras.models.model_from_json(kwargs['model'])
        self.model_name = kwargs['model_name']
        self.compile_code = kwargs['compile_code']
        self.training_parameters = kwargs['training_parameters']

    def compile(self):
        context = {self.model_name: self.model}
        if self.compile_code is not None:
            exec(self.compile_code, locals(), context)
        return self.model.optimizer is not None

    def train(self):
        self.model.fit(**self.training_parameters)
        return self.model.get_weights()

#
# (x_train, y_train), (x_test, y_test) = tensorflow.keras.datasets.mnist.load_data()
#
# import ray
# from horovod.ray import RayExecutor
# import horovod.tensorflow.keras as hvd
#
# ray.init(address='auto')
#
# hvd.init()
#
# kwargs = dict({
#     'model': model_config,
#     'model_name': 'model',
#     'training_parameters': {
#         'x': x_train,  # json.dumps(pickle.dumps(x_train).decode('latin-1')),
#         'y': y_train,  # json.dumps(pickle.dumps(x_train).decode('latin-1')),
#         "validation_split": 0.1,
#         "epochs": 5,
#         "callbacks": [
#             hvd.callbacks.BroadcastGlobalVariablesCallback(0),
#             hvd.callbacks.MetricAverageCallback(),
#             hvd.callbacks.LearningRateWarmupCallback(initial_lr=0.0125 * hvd.size(), warmup_epochs=3, verbose=1),
#             tensorflow.keras.callbacks.TensorBoard(histogram_freq=1)
#         ]
#     },
#     'compile_code': compile_code,
# })
#
# settings = RayExecutor.create_settings(30)
# executor = RayExecutor(settings, num_workers=1, use_gpu=False)
# executor.start(executable_cls=ExecutionBackground, executable_kwargs=kwargs)
# executor.execute(lambda worker: worker.compile())
# executor.execute(lambda worker: worker.train())
#

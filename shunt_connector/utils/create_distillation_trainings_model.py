# -*- coding: utf-8 -*-
"""
Calculates the FLOPs (actually MACCs) for a given model or layer.
FLOPs are only counted for convolutional layers.

Copyright 2021 Christian Doppler Laboratory for Embedded Machine Learning

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Libs
import numpy as np

import tensorflow.keras.backend as K
from tensorflow.keras.layers import Add, Multiply, Subtract, Flatten, Lambda, Activation, Concatenate
from tensorflow.keras.models import Model

__author__ = 'Bernhard Haas'
__copyright__ = 'Copyright 2021, Christian Doppler Laboratory for ' \
                'Embedded Machine Learning'
__credits__ = ['']
__license__ = 'Apache 2.0'
__version__ = '1.0.0'
__maintainer__ = 'Bernhard Haas'
__email__ = 'bernhardhaas55@gmail.com'
__status__ = 'Release'

def create_classification_distillation_model(model_student,
                                             model_teacher,
                                             add_dark_knowledge=False,
                                             temperature=3,
                                             add_attention_transfer=False,
                                             shunt_locations=None,
                                             index_offset=None,
                                             max_number_transfers=None,
                                             train_on_shunt_features=False):

    if not add_attention_transfer and not add_dark_knowledge and not train_on_shunt_features:
        return model_student

    outputs_teacher = []
    outputs_student = []

    if add_attention_transfer:
        # count how many COV2D layers there are
        add_index_list = []
        for i, layer in enumerate(model_teacher.layers[shunt_locations[1]:-3]):
            if isinstance(layer, Add):
                add_index_list.append(shunt_locations[1]+i)

        number_adds = len(add_index_list)

        if max_number_transfers == 0: # auto mode
            max_number_transfers = int(np.amax([2, np.ceil(number_adds / 3)]))

        if max_number_transfers >= number_adds:
            transfer_indices_teacher = add_index_list
        else: # too many conv found
            indices = list(map(int, list(np.linspace(0, number_adds-1, max_number_transfers))))
            transfer_indices_teacher = list(np.asarray(add_index_list)[indices])

        for index in transfer_indices_teacher:
            print('Chose {}, {} layers for attention transfer!'.format(model_teacher.layers[index].name, model_student.layers[index+index_offset].name))
            output_teacher = model_teacher.layers[index].output
            output_teacher = Lambda(lambda x: K.sum(x*x, axis=-1))(output_teacher)
            output_teacher = Flatten()(output_teacher)
            output_student = model_student.layers[index+index_offset].output
            output_student = Lambda(lambda x: K.sum(x*x, axis=-1))(output_student)
            output_student = Flatten()(output_student)
            outputs_teacher.append(output_teacher)
            outputs_student.append(output_student)

    if train_on_shunt_features:
        print('{}, {} layers added for feature matching!'.format(model_teacher.layers[shunt_locations[1]].name, model_student.layers[shunt_locations[1]+index_offset].name))
        outputs_teacher.append(model_teacher.layers[shunt_locations[1]].output)
        outputs_student.append(model_student.layers[shunt_locations[1]+index_offset].output)

    if add_dark_knowledge:
        # teacher network
        softmax_input = model_teacher.layers[-2].output
        softmax_input = Lambda(lambda x: x / temperature, name='Temperature_teacher')(softmax_input)
        prediction_with_temperature = Activation('softmax', name='Softened_softmax_teacher')(softmax_input)
        outputs_teacher.append(prediction_with_temperature)

        # student network
        softmax_input = model_student.layers[-2].output
        softend_softmax_input = Lambda(lambda x: x / temperature, name='Temperature_student')(softmax_input)
        prediction_with_temperature = Activation('softmax', name='Softened_softmax_student')(softend_softmax_input)
        outputs_student.append(prediction_with_temperature)

    model_student_with_outputs = Model(model_student.input, [model_student.output] + outputs_student, name='Student')
    model_teacher_with_outputs = Model(model_teacher.input, outputs_teacher, name='Teacher')
    model_teacher_with_outputs.trainable = False

    input_net = model_student.input

    outputs_teacher = model_teacher_with_outputs(input_net)
    outputs_student = model_student_with_outputs(input_net)

    if not isinstance(outputs_teacher, list):   # outputs has to be a list
        outputs_teacher = [outputs_teacher]

    losses = []

    if add_attention_transfer:  # add layers for attention transfer loss
        for i in range(len(transfer_indices_teacher)):
            teacher_out = Lambda(lambda x: K.l2_normalize(x, axis=1))(outputs_teacher[i])
            student_out = Lambda(lambda x: K.l2_normalize(x, axis=1))(outputs_student[i+1])
            loss = Subtract(name='a_t_{}'.format(i))([teacher_out, student_out])
            losses.append(loss)

    if train_on_shunt_features:
        if add_dark_knowledge:
            teacher_out = outputs_teacher[-2]
            student_out = outputs_student[-2]
        else:
            teacher_out = outputs_teacher[-1]
            student_out = outputs_student[-1]

        teacher_out = Flatten()(teacher_out)
        student_out = Flatten()(student_out)

        loss = Subtract(name='f_m')([teacher_out, student_out])
        losses.append(loss)

    if add_dark_knowledge:   # add dark knowledge loss
        dark_knowledge_loss = Lambda(lambda x: K.log(x), name='log_student')(outputs_student[-1])
        dark_knowledge_loss = Multiply(name='dark_knowledge_without_temperature')([outputs_teacher[-1], dark_knowledge_loss])
        dark_knowledge_loss = Lambda(lambda x: x * temperature * temperature, name='d_k')(dark_knowledge_loss)
        losses.append(dark_knowledge_loss)

    model_distillation = Model(input_net, [outputs_student[0]] + losses, name='knowledge_distillation')

    print(model_distillation.summary(line_length=150))

    return model_distillation


def create_semantic_distillation_model(model_teacher, model_student):

    input_net = model_teacher.input

    model_student._name = 'Student'
    model_teacher._name = 'Teacher'

    output_teacher = model_teacher(input_net)
    output_student = model_student(input_net)

    model_teacher.trainable = False
    model_student.trainable = True

    x = Concatenate(name='Concat')([output_student, output_teacher])

    model_distillation = Model(input_net, x, name='knowledge_distillation')

    return model_distillation
    
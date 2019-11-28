from os import path

import math

import decisiontree as dt
from dataset import Dataset

print('loading dataset ...')
file_location = path.join(path.abspath(
    path.dirname(__file__)), '../data', 'cleveland.data')
dataset = Dataset(file_location)

# todo: attribute_age
attribute_sex = dt.AttributeFactory.create('sex', [0, 1], lambda x: x['sex'])
attribute_cp = dt.AttributeFactory.create('cp', [1, 2, 3, 4], lambda x: x['cp'])
# todo: attribute_trestbps
# todo: attribute_chol
attribute_fbs = dt.AttributeFactory.create('fbs', [0, 1], lambda x: x['fbs'])
attribute_restecg = dt.AttributeFactory.create('restecg', [0, 2], lambda x: x['restecg'])
# todo: attribute_thalach
attribute_exang = dt.AttributeFactory.create('exang', [0, 1], lambda x: x['exang'])
# todo: attribute_oldpeak
attribute_slope = dt.AttributeFactory.create('slope', [1, 2, 3, 4], lambda x: x['slope'])
attribute_ca = dt.AttributeFactory.create('ca', [0, 1, 2, 3], lambda x: x['ca'])
attribute_thai = dt.AttributeFactory.create('thai', [3, 4, 5, 6, 7], lambda x: x['thai'])

attribute_num = dt.AttributeFactory.create('num', [0, 1], lambda x: x['num'])

print('generating decision tree and pruning (using kfold) ...')
attribute_selector = dt.InformationGainAttributeSelector()
min_error_tree = (math.inf, None)
for training_indices, validation_indices in dataset.get_cross_validation_indexes():
    dtree = dt.DecisionTreeFactory.create(dataset.get_training_dataset().iloc[training_indices, :],
                                          [attribute_sex, attribute_cp, attribute_fbs, attribute_restecg,
                                           attribute_exang, attribute_slope, attribute_ca, attribute_thai],
                                          attribute_num, attribute_selector)
    dtree.prune(dataset.get_training_dataset().iloc[validation_indices, :])
    error_rate = dtree.evaluate(dataset.get_training_dataset().iloc[validation_indices, :])
    if error_rate <= min_error_tree[0]:
        min_error_tree = (error_rate, dtree)

error_rate = dtree.evaluate(dataset.test_dataset)
print('error rate is {0}'.format(error_rate))

min_error_tree[1].draw('Cleveland Heart Diseases')

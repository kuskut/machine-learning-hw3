from os import path

import pandas as pd

import decisiontree as dt
from decisiontree.decisiontree import DTreeDecisionNode

file_location = path.join(path.abspath(
    path.dirname(__file__)), '../data', 'test.data')

samples = pd.read_csv(file_location, header=None, index_col=False,
                      names=['outlook', 'temperature', 'humidity', 'wind', 'played'])

attribute_outlook = dt.AttributeFactory.create('outlook', ['sunny', 'overcast', 'rain'], lambda x: x['outlook'])
attribute_temperature = dt.AttributeFactory.create('temperature', ['cool', 'mild', 'hot'], lambda x: x['temperature'])
attribute_humidity = dt.AttributeFactory.create('humidity', ['normal', 'high'], lambda x: x['humidity'])
attribute_wind = dt.AttributeFactory.create('wind', ['weak', 'strong'], lambda x: x['wind'])
attribute_played = dt.AttributeFactory.create('played', ['yes', 'no'], lambda x: x['played'])

attribute_selector = dt.GiniIndexAttributeSelector()

dtree = dt.DecisionTreeFactory.create(samples,
                                      [attribute_outlook, attribute_temperature, attribute_humidity, attribute_wind],
                                      attribute_played, attribute_selector)


def printtree(prefix, node):
    if isinstance(node, DTreeDecisionNode):
        print(prefix + node.attribute.name)
        for link in node.links:
            printtree(prefix + '<' + link.attribute_class + '>' + '-', link.node)
    else:
        print(prefix + '> ' + node.target_value)


printtree('', dtree.root_node)

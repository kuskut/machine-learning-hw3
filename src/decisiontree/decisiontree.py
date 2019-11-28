from abc import ABC, abstractmethod
import graphviz
from os import path
import random


class DTreeNode(ABC):
    @abstractmethod
    def classify(self, sample):
        pass


class DTreeLink:
    def __init__(self, attribute_class, node):
        self.attribute_class = attribute_class
        self.node = node


class DTreeDecisionNode(DTreeNode):
    def __init__(self, attribute, links, predominant_target_value):
        self.attribute = attribute
        self.links = links
        self.predominant_target_value = predominant_target_value
        self.is_active = True

    def classify(self, sample):
        """
        checks all the link values with the sample input and then
        :param sample:
        :return:
        """
        if not self.is_active:
            return self.predominant_target_value

        sample_attribute_class = self.attribute.get_class(sample)

        for link in self.links:
            if link.attribute_class == sample_attribute_class:
                return link.node.evaluate(sample)

        raise ValueError('the sample {0} does not belong to any class of {1]'.format(str(sample), self.attribute.name))

    def deactivate(self):
        self.is_active = False

    def activate(self):
        self.is_active = True


class DTreeTerminalNode(DTreeNode):
    def __init__(self, target_value):
        self.target_value = target_value

    def classify(self, sample):
        """
        simply returns the target value for all samples as this is a leaf node
        :param sample:
        :return: target value
        """
        return self.target_value


class DecisionTree:
    def __init__(self, root_node, target_attribute):
        self.root_node = root_node
        self.target_attribute = target_attribute

    def classify(self, sample):
        """
        classifies input sample with this decision tree
        :param sample:
        :return: predicted target attribute value
        """
        return self.root_node.evaluate(sample)

    def evaluate(self, test_samples):
        errors_count = 0
        for sample in test_samples:
            predicted_value = self.classify(sample)
            if predicted_value != self.target_attribute.get_class(sample):
                errors_count += 1

        return errors_count / len(test_samples)

    def _draw_node(self, graph, node):
        nname = str(random.randrange(1, 1000000))
        if isinstance(node, DTreeDecisionNode):
            graph.attr('node', shape='box')
            graph.node(nname, label=node.attribute.name)
            for link in node.links:
                child_lbl = self._draw_node(graph, link.node)
                graph.edge(nname, child_lbl, link.attribute_class)
        else:
            graph.attr('node', shape='ellipse')
            graph.node(name=nname, label=node.target_value)

        return nname

    def draw(self, name):
        file_path = path.join(path.abspath(path.dirname(__file__)), '../../docs', 'dtree.gv')
        graph = graphviz.Graph(name, filename=file_path)

        self._draw_node(graph, self.root_node)

        graph.view()


class DecisionTreeFactory:
    @staticmethod
    def _create_node(samples, attributes, target_attribute, attribute_selector):
        if len(samples) == 0:
            return DTreeTerminalNode('XX UNKNOWN XX')

        tattr = samples.loc[:, target_attribute.name].mode().iloc[0]

        # if there is no more attributes -> return terminal with most abundant target value
        if len(attributes) == 0:
            return DTreeTerminalNode(tattr)

        # if all the samples have same target_attribute -> return terminal node with target value
        if samples.loc[:, target_attribute.name].nunique() == 1:
            return DTreeTerminalNode(tattr)

        # select best attribute
        selected_attribute = attribute_selector.select(samples, attributes, target_attribute)
        attributes.remove(selected_attribute)

        # create sub nodes
        links = []
        for attribute_class in selected_attribute.get_all_classes():
            child_node = DecisionTreeFactory._create_node(selected_attribute.filter(samples, attribute_class),
                                                          attributes[:], target_attribute, attribute_selector)
            link = DTreeLink(attribute_class, child_node)
            links.append(link)

        return DTreeDecisionNode(selected_attribute, links, tattr)

    @staticmethod
    def create(samples, attributes, target_attribute, attribute_selector):
        root_node = DecisionTreeFactory._create_node(samples, attributes, target_attribute, attribute_selector)
        return DecisionTree(root_node, target_attribute)

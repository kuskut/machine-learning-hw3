from abc import ABC, abstractmethod

import math


class Attribute(ABC):
    def __init__(self, name, all_classes):
        self.name = name
        self.all_classes = all_classes

    def get_all_classes(self):
        return self.all_classes

    @abstractmethod
    def get_class(self, sample):
        """
        get class of the input sample for the attribute
        :param sample:
        :return: class (int)
        """
        pass

    def filter(self, samples, cls):
        mask = [(self.get_class(x) == cls) for i, x in samples.iterrows()]
        return samples.loc[mask, :]


class AttributeFactory:
    @staticmethod
    def create(name, all_classes, get_class_fn):
        class AttributeImpl(Attribute):
            def __init__(self):
                super().__init__(name, all_classes)

            def get_class(self, sample):
                return get_class_fn(sample)

        return AttributeImpl()


class AttributeSelector(ABC):
    @abstractmethod
    def select(self, samples, attributes, target_attribute):
        pass


class InformationGainAttributeSelector(AttributeSelector):
    def _compute_entropy(self, samples, target_attribute):
        entropy = 0

        if len(samples) == 0:
            return entropy

        for cls in target_attribute.get_all_classes():
            p = len(target_attribute.filter(samples, cls)) / len(samples)
            if p != 0:
                entropy += -1 * p * math.log(p, 2)

        return entropy

    def select(self, samples, attributes, target_attribute):
        current_entropy = self._compute_entropy(samples, target_attribute)
        max_gain = (None, -math.inf)
        for attribute in attributes:
            total_entropy = 0
            for cls in attribute.get_all_classes():
                filtered_samples = attribute.filter(samples, cls)
                factor = len(filtered_samples) / len(samples)
                total_entropy += factor * self._compute_entropy(filtered_samples, target_attribute)

            total_entropy = current_entropy - total_entropy
            if total_entropy > max_gain[1]:
                max_gain = (attribute, total_entropy)

        return max_gain[0]


class GiniIndexAttributeSelector(AttributeSelector):
    def _compute_gindex(self, samples, target_attribute):
        gindex = 0

        if len(samples) == 0:
            return gindex

        for cls in target_attribute.get_all_classes():
            p = len(target_attribute.filter(samples, cls)) / len(samples)
            gindex += p ** 2

        return 1 - gindex

    def select(self, samples, attributes, target_attribute):
        current_gindex = self._compute_gindex(samples, target_attribute)
        max_gain = (None, -math.inf)
        for attribute in attributes:
            total_gindex = 0
            for cls in attribute.get_all_classes():
                filtered_samples = attribute.filter(samples, cls)
                factor = len(filtered_samples) / len(samples)
                total_gindex += factor * self._compute_gindex(filtered_samples, target_attribute)

            total_gindex = current_gindex - total_gindex
            if total_gindex > max_gain[1]:
                max_gain = (attribute, total_gindex)
                
        return max_gain[0]


class DTreeNode(ABC):
    @abstractmethod
    def evaluate(self, sample):
        pass


class DTreeLink:
    def __init__(self, attribute_class, node):
        self.attribute_class = attribute_class
        self.node = node


class DTreeDecisionNode(DTreeNode):
    def __init__(self, attribute, links):
        self.attribute = attribute
        self.links = links

    def evaluate(self, sample):
        """
        checks all the link values with the sample input and then
        :param sample:
        :return:
        """
        sample_attribute_class = self.attribute.get_class(sample)

        for link in self.links:
            if link.attribute_class == sample_attribute_class:
                return link.node.evaluate(sample)

        raise ValueError('the sample {0} does not belong to any class of {1]'.format(str(sample), self.attribute.name))


class DTreeTerminalNode(DTreeNode):
    def __init__(self, target_value):
        self.target_value = target_value

    def evaluate(self, sample):
        """
        simply returns the target value for all samples as this is a leaf node
        :param sample:
        :return: target value
        """
        return self.target_value


class DecisionTree:
    def __init__(self, root_node):
        self.root_node = root_node

    def evaluate(self, sample):
        """
        classifies input sample with this decision tree
        :param sample:
        :return: predicted target attribute value
        """
        return self.root_node.evaluate(sample)


class DecisionTreeFactory:
    @staticmethod
    def _create_node(samples, attributes, target_attribute, attribute_selector):
        if len(samples) == 0:
            return DTreeTerminalNode('XX UNKNOWN XX')

        # if there is no more attributes -> return terminal with most abundant target value
        if len(attributes) == 0:
            tattr = samples.loc[:, target_attribute.name].mode().iloc[0]
            return DTreeTerminalNode(tattr)

        # if all the samples have same target_attribute -> return terminal node with target value
        if samples.loc[:, target_attribute.name].nunique() == 1:
            tattr = samples.loc[:, target_attribute.name].mode().iloc[0]
            return DTreeTerminalNode(tattr)

        # select best attribute
        selected_attribute = attribute_selector.select(samples, attributes, target_attribute)
        attributes.remove(selected_attribute)

        # create sub nodes
        links = []
        for attribute_class in selected_attribute.get_all_classes():
            child_node = DecisionTreeFactory._create_node(selected_attribute.filter(samples, attribute_class),
                                                          attributes,
                                                          target_attribute, attribute_selector)
            link = DTreeLink(attribute_class, child_node)
            links.append(link)

        return DTreeDecisionNode(selected_attribute, links)

    @staticmethod
    def create(samples, attributes, target_attribute, attribute_selector):
        root_node = DecisionTreeFactory._create_node(samples, attributes, target_attribute, attribute_selector)
        return DecisionTree(root_node)

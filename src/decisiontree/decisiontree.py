from abc import ABC, abstractmethod


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
                                                          attributes[:],
                                                          target_attribute, attribute_selector)
            link = DTreeLink(attribute_class, child_node)
            links.append(link)

        return DTreeDecisionNode(selected_attribute, links)

    @staticmethod
    def create(samples, attributes, target_attribute, attribute_selector):
        root_node = DecisionTreeFactory._create_node(samples, attributes, target_attribute, attribute_selector)
        return DecisionTree(root_node)

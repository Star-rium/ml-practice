import numpy as np

class ID3:
  def __init__(self):
    self.tree = None
    self.default_class = None

  def entropy(self, feature, target_col):
    elements, count = np.unique(target_col, return_counts = True)
    entropy = np.sum(
        [(-count[i]/np.sum(count))*np.log2(count[i]/np.sum(count))
        for i in range(len(elements))])
    # print(f"feature: {feature}, entropy: {entropy}")
    return entropy

  def information_gain(self, data, feature, target):
    total_entropy = self.entropy(feature, data[target])
    values, counts = np.unique(data[feature], return_counts=True)

    weighted_entropy = np.sum([
        (counts[i] / np.sum(counts)) *
        self.entropy(data[data[feature] == values[i]] ,data[data[feature] == values[i]][target])
        for i in range(len(values))
    ])
    # print(f"feature: {feature}, total entropy: {total_entropy}, weighted entropy: {weighted_entropy}")
    # print(f"Information feature: {feature}, information gain: {total_entropy - weighted_entropy}")
    return total_entropy - weighted_entropy

  def id3(self, data, original_data, features, target, parent_node_class=None):
    # base case 1: if all class in the set are the same
    if len(np.unique(data[target])) == 1:
    #   print(f"returning with pure data")
      return np.unique(data[target])[0]

    # base case 2: if subset is empty -> return most frequent class in original dataset
    if len(data) == 0:
    #   print("returning with empty set")
      return np.unique(original_data[target])[
          np.argmax(np.unique(original_data[target], return_counts=True)[1])
      ]
    # base case 3: if feature set is empty -> return parent node class
    if len(features) == 0:
    #   print("returning with empty feature set")
      return parent_node_class

    # get most frequent class for fallback
    parent_node_class = np.unique(data[target])[np.argmax(np.unique(data[target], return_counts=True)[1])]
    gains = [self.information_gain(data, feature, target) for feature in features]
    best_feature = features[np.argmax(gains)]
    # print(f"Best feature: {best_feature}")

    # initialize tree, create subset
    tree = {best_feature: {}}
    remaining_features = [f for f in features if f != best_feature]

    # build tree for each value in the best feature
    for value in np.unique(data[best_feature]):
    #   print(f"selected value: {value}")
      subset = data[data[best_feature] == value]
      subtree = self.id3(subset, original_data, remaining_features, target, parent_node_class)
      tree[best_feature][value] = subtree

    return tree

  def build_tree(self, data, features, target):
    vals, count = np.unique(data[target], return_counts=True)
    self.default_class = vals[np.argmax(count)]
    self.tree = self.id3(data, data, features, target)
    return self.tree

  def predict(self, sample, tree=None):
    if tree is None:
      tree = self.tree
    if not isinstance(tree, dict): # base case: encountered leaf node
      return tree
    root_feature = next(iter(tree))
    feature_value = sample.get(root_feature)
    if feature_value in tree[root_feature]:
      # recursion for each subtree
      subtree = tree[root_feature][feature_value]
      return self.predict(sample, subtree)
    else:
      # if unseen category -> return default category
      return self.default_class

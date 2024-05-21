import numpy as np
import utils
from graphviz import Digraph

GLOBAL_COUNT=0
class Node:
    '''
        Class defines the nodes of the decision tree
        self.attr: [String] attribute of the node
        self.val: [Float] value of the attribute
        self.avg_attr: [Float] average of the attribute

        self.left: [Node] left child of the node
        self.right: [Node] right child of the node

    '''

    def __init__(self, attribute, value, type_arr):
        '''
            Initializes the node
        '''
        global GLOBAL_COUNT
        GLOBAL_COUNT+=1
        self.node_id=GLOBAL_COUNT
        
        self.attr_idx = attribute
        self.val = value
        self.attr_type = type_arr[self.attr_idx]
        self.left = None
        self.right = None
        self.leaf = False
        self.classification = None

    def make_leaf(self, classification):
        '''
            Makes the node a leaf node
        '''
        # print(f'Making leaf node {classification}')
        self.leaf = True
        self.left = None
        self.right = None
        self.classification = classification

    def get_classification(self):
        '''
            Returns the classification of the node
        '''
        return self.classification

    def predict_node(self, X):
        '''
            Predicts the class of the instance X
        '''
        if self.leaf:
            return self.classification
        else:
            if self.attr_type == 'cont':
                if X[self.attr_idx] <= self.val:
                    return self.left.predict_node(X)
                else:
                    return self.right.predict_node(X)
            else:
                if X[self.attr_idx] == self.val:
                    return self.left.predict_node(X)
                else:
                    return self.right.predict_node(X)

    def __eq__(self, other) -> bool:
        # print(self,other)
        if other is None:
            return False
        if self.leaf and other.leaf:
            return self.classification == other.classification
        if self.attr_idx != other.attr_idx:
            return False
        if self.val != other.val:
            return False
        if self.attr_type != other.attr_type:
            return False
        if self.left != other.left:
            return False
        if self.right != other.right:
            return False
        return True

    def dfs_count(self):
        ans = 1
        if not self.left is None:
            ans += self.left.dfs_count()
        if not self.right is None:
            ans += self.right.dfs_count()
        return ans

    def calc_prune_error(self, X_val, y_val):
        preds = np.array([self.predict_node(x) for x in X_val])
        return np.sum(preds != y_val)

    def prune_base(self, y_train, X_val, y_val, n_node):

        leaf = utils.classify_array(y_train)
        errors_leaf = np.sum(y_val != leaf)
        errors_node = np.sum(y_val != np.array(
            [self.predict_node(x) for x in X_val]))

        if errors_leaf <= errors_node:
            n_node.make_leaf(leaf)
        

    def prune_rec(self, X_train, y_train, X_val, y_val,type_arr):

        n_node = Node(self.attr_idx, self.val, type_arr)
        n_node.leaf = self.leaf
        n_node.classification = self.classification
        n_node.left = self.left
        n_node.right = self.right
        # print(f"start {type(n_node)}")
        if self.leaf:
            n_node = self.prune_base(y_train, X_val, y_val, n_node)

        else:
            X_train_yes, Y_train_yes, X_train_no, Y_train_no = utils.filter(
                X_train, y_train, self.attr_idx, self.val,type_arr)
            X_val_yes, Y_val_yes, X_val_no, Y_val_no = utils.filter(
                X_val, y_val, self.attr_idx, self.val,type_arr)

            if not (self.left is None or self.left.leaf==True):
                n_node.left = self.left.prune_rec(X_train_yes,Y_train_yes, X_val_yes, Y_val_yes,type_arr)
            if not (self.right is None or self.right.leaf==True):
                n_node.right = self.right.prune_rec(X_train_no,Y_train_no, X_val_no, Y_val_no,type_arr)

            self.prune_base(y_train, X_val, y_val, n_node)

        # print(f"end {type(n_node)}")
        return n_node


class DecisionTree:
    '''
        Class defines the decision tree
        self.root: [Node] root of the tree
        self.X:  [Nest List] training features
        self.y:  [List] training labels
    '''

    def __init__(self, X, y, column_names, min_leaf_size, max_depth,impurity_measure):
        '''
            Initializes the tree
        '''
        self.measure=impurity_measure
        self.root = None
        self.root_pruned = None
        self.X = X
        self.y = y
        self.min_leaf_size = min_leaf_size
        self.max_depth = max_depth
        self.column_names = column_names
        self.type_arr = utils.assign_feature_type(X, 2)

    def fit(self):
        '''
                Builds the decision tree
        '''
        global GLOBAL_COUNT
        GLOBAL_COUNT=0
        self.root = self.build_tree(self.X, self.y)

    def is_leaf(self, X_lo, y_lo):
        '''
            Checks if the node is a leaf
            node: [Node] node to be checked
        '''
        if X_lo.shape[0] <= self.min_leaf_size:
            return True
        if utils.check_purity(y_lo):
            return True
        return False

    def build_tree(self, X, y, depth=0):
        '''
            Recursively builds the decision tree
            node: [Node] node to be built
            depth: [Int] depth of the node
        '''

        
        if self.is_leaf(X, y) or depth == self.max_depth:
            node = Node(0, 0, self.type_arr)
            node.make_leaf(utils.classify_array(y))
            return node
        else:
            depth += 1
            best_attr, best_val = utils.get_best_split(
                X, y, self.type_arr,self.measure)
            node = Node(best_attr, best_val, self.type_arr)
            
            X_left, y_left, X_right, y_right = utils.create_children_np(
                X, y, best_attr, best_val, self.type_arr)
            
            if X_left.shape[0] == 0 or X_right.shape[0] == 0:
                node.make_leaf(utils.classify_array(y))    
                return node

            left_tree = self.build_tree(X_left, y_left, depth)
            right_tree = self.build_tree(X_right, y_right, depth)
            
            # print("hii",left_tree,right_tree)
            if left_tree == right_tree:    
                node.make_leaf(utils.classify_array(y_left))
            else:
                node.leaf = False
                node.left = left_tree
                node.right = right_tree

            return node

    def predict(self, X):
        '''
            Predicts the labels of the test data
            X: [Nest List] test features
        '''
        if self.root is None:
            return None
        else:
            return np.array([self.root.predict_node(x) for x in X])

    def pruned_predict(self, X):
        if self.root_pruned is None:
            return None
        else:
            return np.array([self.root_pruned.predict_node(x) for x in X])

    def calc_accuracy(self, X, y,print_report=True):
        '''
            Calculates the accuracy of the decision tree
            X: [Nest List] test features
            y: [List] test labels
        '''
        y_pred = self.predict(X)
        from sklearn.metrics import classification_report
        if print_report:
            print(classification_report(y, y_pred))
        return utils.calc_accuracy(y, y_pred)

    def calc_pruned_accuracy(self, X, y,print_report=True):
        y_pred = self.pruned_predict(X)
        from sklearn.metrics import classification_report
        if print_report:
            print(classification_report(y, y_pred))
        return utils.calc_accuracy(y, y_pred)

    def print_tree(self, node=None, depth=0):
        '''
            Prints the tree in a readable format
            node: [Node] node to be printed
            depth: [Int] depth of the node
        '''

        if node.leaf:
            print('|'*depth+'Leaf: '+str(node.classification))
        else:
            compoperator = '<='
            if node.attr_type == 'discrete':
                compoperator = '=='
            print('|'*depth+'Attribute: ' +
                  self.column_names[node.attr_idx]+'  ' + compoperator + '  Value: '+str(node.val))
            self.print_tree(node.left, depth+1)
            self.print_tree(node.right, depth+1)

    def count_nodes(self):
        return self.root.dfs_count()

    def post_prune(self, X_train, y_train, X_val, y_val):
        '''
            Recursively prunes the tree
        '''
        global GLOBAL_COUNT
        GLOBAL_COUNT=0
        self.root_pruned = self.root.prune_rec(X_train, y_train, X_val, y_val,self.type_arr)


def render_node(vertex, feature_names, count):
    if vertex.leaf:
        return f'ID {vertex.node_id},\nClassification -> {vertex.classification}\n'
    return f'ID {vertex.node_id}\n{feature_names[vertex.attr_idx]} <= {vertex.val}\n'


def tree_to_gv(node_root, feature_names,file_name="decision_tree.gv"):
    f = Digraph('Decision Tree', filename=file_name)
    # f.attr(rankdir='LR', size='1000,500')

    f.attr('node', shape='rectangle')
    q = [node_root]
    idx = 0
    while len(q) > 0:
        node = q.pop(0)
        if node is None:
            continue
        if not node.left is None:
            f.edge(render_node(node, feature_names, idx), render_node(
                node.left, feature_names, idx), label='True')
            idx += 1
            q.append(node.left)
        if not node.right is None:
            f.edge(render_node(node, feature_names, idx), render_node(
                node.right, feature_names, idx), label='False')
            idx += 1
            q.append(node.right)
    f.render(f'./{file_name}', view=True)

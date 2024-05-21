import argparse
import utils
import tree
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
import time
import copy
MAX_HEIGHT = None
DATA_PATH = None
MIN_LEAF_SIZE = None
MEASURE = None
RANDOM_SEED = None


def select_best_tree(X_full, y_full, measure):
    X_b_train = None
    X_b_test = None
    X_b_val = None
    y_b_train = None
    y_b_test = None
    y_b_val = None
    best_acc = -1
    acc_list=[]
    acc_train_list=[]
    for i in range(1, 11):
        X_train, X_val, X_test, y_train, y_val, y_test = utils.train_val_test_split(
            X_full, y_full, 0.6, 0.2, seed=i+RANDOM_SEED)
        Tree = tree.DecisionTree(
            X_train, y_train, feature_names, MIN_LEAF_SIZE, MAX_HEIGHT, measure)
        Tree.fit()
        train_acc = Tree.calc_accuracy(X_train, y_train, print_report=False)
        test_acc = Tree.calc_accuracy(X_test, y_test, print_report=False)
        print(f"Training on split {i} complete")
        print(f"Training accuracy: {train_acc}")
        print(f"Testing accuracy: {test_acc}")
        acc_list.append(test_acc)
        acc_train_list.append(train_acc)
        if test_acc > best_acc:
            best_acc = test_acc
            X_b_train = X_train
            X_b_test = X_test
            y_b_train = y_train
            y_b_test = y_test
            X_b_val = X_val
            y_b_val = y_val
            best_tree=copy.deepcopy(Tree)
    # best_tree=tree.DecisionTree(X_b_train, y_b_train, feature_names, MIN_LEAF_SIZE, MAX_HEIGHT, measure)
    # best_tree.fit()
    print("\n\n")
    print(f"Average train accuracy over 10 test train splits is {np.mean(acc_train_list)}")
    print(f"Average test acuracy over 10 test train splits is {np.mean(acc_list)}")
    return best_tree, X_b_train, X_b_test, X_b_val, y_b_train, y_b_test, y_b_val


def height_ablation(X_train, y_train,X_test,y_test ,X_val, y_val, measure):
    test_acc_depth_list = []
    train_acc_depth_list=[]
    val_acc_depth_list=[]
    num_node_list = []

    for i in range(1, 25):
        print(f"Checking Height = {i}")
        Tree_3 = tree.DecisionTree(
            X_train, y_train, feature_names, MIN_LEAF_SIZE, i, measure)
        Tree_3.fit()
        test_acc_depth_list.append(Tree_3.calc_accuracy(X_test, y_test, print_report=False))
        train_acc_depth_list.append(Tree_3.calc_accuracy(X_train, y_train, print_report=False))
        val_acc_depth_list.append(Tree_3.calc_accuracy(X_val, y_val, print_report=False))
        num_node_list.append(Tree_3.count_nodes())

    figure, axis = plt.subplots(1, 2)
    axis[0].plot(range(1, 25), test_acc_depth_list, label="Test")
    axis[0].plot(range(1, 25), train_acc_depth_list, label="Train")
    axis[0].plot(range(1, 25), val_acc_depth_list, label="Validation")
    axis[0].set_xlabel("Max Depth")
    axis[0].set_ylabel("Accuracy")
    axis[0].set_title("Accuracy vs Max Depth")
    axis[0].legend()

    axis[1].plot(num_node_list, train_acc_depth_list, label="Train")
    axis[1].plot(num_node_list, val_acc_depth_list, label="Validation")
    axis[1].plot(num_node_list, test_acc_depth_list, label="Test")
    axis[1].set_xlabel("Number of Nodes")
    axis[1].set_ylabel("Accuracy")
    axis[1].set_title("Accuracy vs Number of Nodes")
    axis[1].legend()
    plt.show()

    Optimal_depth = 1+np.argmax(np.array(val_acc_depth_list))
    Best_tree = tree.DecisionTree(
        X_train, y_train, feature_names, MIN_LEAF_SIZE, Optimal_depth, measure)
    Best_tree.fit()
    print(f"Optimal depth: {Optimal_depth}")
    print(f"Number of nodes: {Best_tree.count_nodes()}")
    return Best_tree


def run_entropy_exp(X_full,y_full,X_train_,y_train_,X_test_,y_test_):
    print("\n"+"-"*50 + "TREE with Entropy" + "-"*50)
    MEASURE = "entropy"
    start = time.time()
    Tree_1_entropy = tree.DecisionTree(
        X_train_, y_train_, feature_names, MIN_LEAF_SIZE, MAX_HEIGHT, MEASURE)
    Tree_1_entropy.fit()
    end = time.time()
    print("Training complete")
    print("Training accuracy:", Tree_1_entropy.calc_accuracy(X_train_, y_train_))
    print("Testing accuracy:", Tree_1_entropy.calc_accuracy(X_test_, y_test_))
    print("Time taken:", end - start)

    print("\n"+"-"*50 + "TREE over 10 random splits ENTROPY" + "-"*50)
    BEST_TREE, X_train, X_test, X_val, y_train, y_test,y_val = select_best_tree(X_full, y_full, "entropy")
    print("\n"+"-"*50 + "BEST TREE OVER 10 RANDOM SPLITS ENTROPY" + "-"*50)
    print("Training accuracy:", BEST_TREE.calc_accuracy(X_train, y_train))
    print("Testing accuracy:", BEST_TREE.calc_accuracy(X_test, y_test))
    tree.tree_to_gv(BEST_TREE.root, feature_names, "unprunedDT.gv")

    print("\n"+"-"*50 + "DEPTH Vs Accuracy ENTROPY" + "-"*50)
    BEST_TREE_HEIGHT = height_ablation(
        X_train, y_train, X_test, y_test,X_val,y_val, "entropy")

    print("\n"+"-"*50+"PRUNING OPERATIONS"+"-"*50)
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("X_val:", X_val.shape)
    print("y_train:", y_train.shape)
    print("y_test:", y_test.shape)
    print("y_val:", y_val.shape)
    print("Unpruned best tree accuracies:")
    print("Training accuracy:", BEST_TREE.calc_accuracy(X_train, y_train))
    print("Testing accuracy:", BEST_TREE.calc_accuracy(X_test, y_test))
    print("Validation accuracy:", BEST_TREE.calc_accuracy(X_val, y_val))

    BEST_TREE.post_prune(X_train, y_train, X_val, y_val)
    print("\n"+"-"*50+"Post Pruning complete"+"-"*50)
    print("Training accuracy:", BEST_TREE.calc_pruned_accuracy(X_train, y_train))
    print("Testing accuracy:", BEST_TREE.calc_pruned_accuracy(X_test, y_test))
    print("Validation accuracy:", BEST_TREE.calc_pruned_accuracy(X_val, y_val))
    tree.tree_to_gv(BEST_TREE.root_pruned, feature_names, "prunedDT.gv")
    tree.tree_to_gv(BEST_TREE_HEIGHT.root, feature_names,"unpruned_best_tree_optimal_height_entropy.gv")



def run_gini_exp(X_full,y_full,X_train_,y_train_,X_test_,y_test_):
    MAX_HEIGHT=15
    print("\n"+"-"*50 + "TREE with Gini" + "-"*50)
    MEASURE = "gini"

    start = time.time()
    Tree_1_gini = tree.DecisionTree(
        X_train_, y_train_, feature_names, MIN_LEAF_SIZE, MAX_HEIGHT, MEASURE)
    Tree_1_gini.fit()
    end = time.time()
    print("Training complete")
    print("Training accuracy:", Tree_1_gini.calc_accuracy(X_train_, y_train_))
    print("Testing accuracy:", Tree_1_gini.calc_accuracy(X_test_, y_test_))
    print("Time taken:", end - start)


    print("\n"+"-"*50 + "TREE over 10 random splits gini" + "-"*50)
    BEST_TREE, X_train, X_test, X_val, y_train, y_test,y_val = select_best_tree(X_full, y_full, "gini")
    print("\n"+"-"*50 + "BEST TREE OVER 10 RANDOM SPLITS gini" + "-"*50)
    print("Training accuracy:", BEST_TREE.calc_accuracy(X_train, y_train))
    print("Testing accuracy:", BEST_TREE.calc_accuracy(X_test, y_test))
    tree.tree_to_gv(BEST_TREE.root, feature_names, "unpruned_gini_DT.gv")

    print("\n"+"-"*50 + "DEPTH Vs Accuracy gini" + "-"*50)
    BEST_TREE_HEIGHT = height_ablation(
        X_train, y_train, X_test, y_test,X_val,y_val, "gini")

    print("\n"+"-"*50+"PRUNING OPERATIONS"+"-"*50)
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("X_val:", X_val.shape)
    print("y_train:", y_train.shape)
    print("y_test:", y_test.shape)
    print("y_val:", y_val.shape)
    print("Unpruned best tree accuracies:")
    print("Training accuracy:", BEST_TREE.calc_accuracy(X_train, y_train))
    print("Testing accuracy:", BEST_TREE.calc_accuracy(X_test, y_test))
    print("Validation accuracy:", BEST_TREE.calc_accuracy(X_val, y_val))

    BEST_TREE.post_prune(X_train, y_train, X_val, y_val)
    print("\n"+"-"*50+"Post Pruning complete"+"-"*50)
    print("Training accuracy:", BEST_TREE.calc_pruned_accuracy(X_train, y_train))
    print("Testing accuracy:", BEST_TREE.calc_pruned_accuracy(X_test, y_test))
    print("Validation accuracy:", BEST_TREE.calc_pruned_accuracy(X_val, y_val))
    tree.tree_to_gv(BEST_TREE.root_pruned, feature_names, "pruned_gini_DT.gv")
    tree.tree_to_gv(BEST_TREE_HEIGHT.root, feature_names,"unpruned_best_tree_optimal_height_gini.gv")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--min_leaf_size", type=int, default=1)
    parser.add_argument("--impurity_measure", type=str, default="entropy")
    parser.add_argument("--random_seed", type=int, default="42")
    parser.add_argument("--data_path", type=str, default="diabetes.csv")

    args = parser.parse_args()

    MAX_HEIGHT = args.max_depth
    DATA_PATH = args.data_path
    MIN_LEAF_SIZE = args.min_leaf_size
    MEASURE = args.impurity_measure
    RANDOM_SEED = args.random_seed

    print(f"MAX DEPTH = {MAX_HEIGHT}")
    print(f"DATA PATH = {DATA_PATH}")

    if (MAX_HEIGHT < 0):
        MAX_HEIGHT = 10

    print("-"*50 + "PREPROCESSING" + "-"*50)
    feature_names = utils.get_column_names(DATA_PATH)
    X_full, y_full = utils.get_X_y(DATA_PATH)
    X_train, X_test, y_train, y_test = utils.train_test_split(
        X_full, y_full, 0.8)
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test:", y_test.shape)

    run_entropy_exp(X_full=X_full, y_full=y_full,X_train_=X_train,y_train_=y_train,X_test_=X_test,y_test_=y_test)
    run_gini_exp(X_full=X_full, y_full=y_full,X_train_=X_train,y_train_=y_train,X_test_=X_test,y_test_=y_test)

    

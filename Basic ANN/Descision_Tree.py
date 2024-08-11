import numpy as np
import matplotlib.pyplot as plt

X_train = np.array([[1,1,1],[1,0,1],[1,0,0],[1,0,0],[1,1,1],[0,1,1],[0,0,0],[1,0,1],[0,1,0],[1,0,0]])
y_train = np.array([1,1,0,0,1,0,0,1,1,0])

def entropy(x):
    z=(-1*x*np.log2(x))
    z-= (1-x)*(np.log2(1-x))
    return z

def split_dataset(X, node_indices, feature):
    L = []
    R = []
    for i in range(node_indices):   
        if X[i][feature] == 1:
            L.append(i)
        else:
            R.append(i)
    return L,R

def information_gain(X, y, node_indices, feature):
    left_indices, right_indices = split_dataset(X, node_indices, feature)
    
    X_node, y_node = X[node_indices], y[node_indices]
    X_left, y_left = X[left_indices], y[left_indices]
    X_right, y_right = X[right_indices], y[right_indices]
    
    InfoGain = 0

    node_entropy = entropy(y_node)
    left_entropy = entropy(y_left)
    right_entropy = entropy(y_right)
    
    w_left = len(X_left) / len(X_node)
    w_right = len(X_right) / len(X_node)

    weighted_entropy = w_left * left_entropy + w_right * right_entropy 
    
    InfoGain = node_entropy - weighted_entropy
    
    return InfoGain

def best_split(X,y,node_indices):
    total_features = X.shape[1]
    best_feature = -1
    max_info_gain=0
    for feature in range(total_features):
        info_gain = information_gain(X, y, node_indices, feature)
        if info_gain > max_info_gain:
            max_info_gain = info_gain
            best_feature = feature
    return best_feature

tree = []

def build_tree_recursive(X, y, node_indices, branch_name, max_depth, current_depth):
  
    if current_depth == max_depth:
        formatting = " "*current_depth + "-"*current_depth
        print(formatting, "%s leaf node with indices" % branch_name, node_indices)
        return
   
    # Otherwise, get best split and split the data
    # Get the best feature and threshold at this node
    best_feature = best_split(X, y, node_indices) 
    tree.append((current_depth, branch_name, best_feature, node_indices))
    
    formatting = "-"*current_depth
    print("%s Depth %d, %s: Split on feature: %d" % (formatting, current_depth, branch_name, best_feature))
    
    # Split the dataset at the best feature
    left_indices, right_indices = split_dataset(X, node_indices, best_feature)
    
    # continue splitting the left and the right child. Increment current depth
    build_tree_recursive(X, y, left_indices, "Left", max_depth, current_depth+1)
    build_tree_recursive(X, y, right_indices, "Right", max_depth, current_depth+1)


build_tree_recursive(X_train,y_train,1,0,5,0)
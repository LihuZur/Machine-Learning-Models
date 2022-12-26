import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt


def kNN(curr_data, curr_labels, image, k):
    dist_from_image = find_dist(curr_data, image) # Finding the distance between image and every item in the dataset
    indexes = np.argsort(dist_from_image)[:k]  # Only taking the indexes of the k closest images

    req_labels = np.take(curr_labels, indexes).astype(int)  # Using a numpy function to extract the required labels
    res_arr = np.bincount(req_labels)
    max_elements = [i for i in range(len(res_arr)) if res_arr[i] == max(res_arr)]
    return np.random.choice(max_elements)

# Finding the distances between image and all data images
def find_dist(curr_data, image):
    res = np.zeros(len(curr_data))
    for i in range(len(curr_data)):
        res[i] = np.linalg.norm(curr_data[i] - image)
    return res


# Checking with k=10 and n=1000
def check_10():
    curr_data = train[:1000]
    curr_labels = train_labels[:1000]

    test_res = np.zeros(len(test))
    for i in range(len(test)):
        test_res[i] = kNN(curr_data, curr_labels, test[i], 10)
    a = (test_res.astype(int) == test_labels.astype(int))
    print(np.count_nonzero(a)/len(test))


# running with input n,k
def run_test(n, k):
    # Taking n points and labels from the data
    curr_data = train[:n]
    curr_labels = train_labels[:n]

    # preparing numpy arrays for the calculation
    test_res = np.zeros(len(test))
    k_res = np.zeros(k)
    k_arr = np.arange(1, k+1)

    # calculating
    for i in range(k):
        for j in range(len(test)):
            test_res[j] = kNN(curr_data, curr_labels, test[j], k_arr[i])

        a = (test_res.astype(int) == test_labels.astype(int))
        k_res[i] = np.count_nonzero(a)/len(test)

    plt.plot(k_arr, k_res)
    plt.show()


# Checking with k=1 and increasing n values
def final_test():
    n_arr = np.arange(100, 5100, 100)
    n_res = np.zeros(len(n_arr))
    test_res = np.zeros(len(test))

    for i in range(len(n_arr)):
        curr_data = train[:n_arr[i]]
        curr_labels = train_labels[:n_arr[i]]
        for j in range(len(test)):
            test_res[j] = kNN(curr_data, curr_labels, test[j], 1)
        a = (test_res.astype(int) == test_labels.astype(int))
        n_res[i] = np.count_nonzero(a)/len(test)

    plt.plot(n_arr, n_res)
    plt.show()


if __name__ == '__main__':
    # Set-up lines
    mnist = fetch_openml('mnist_784', as_frame=False)
    data = mnist['data']
    labels = mnist['target']

    idx = np.random.RandomState(0).choice(70000, 11000)
    train = data[idx[:10000], :].astype(int)
    train_labels = labels[idx[:10000]]
    test = data[idx[10000:], :].astype(int)
    test_labels = labels[idx[10000:]]

    # Running the functions
    check_10()
    # run_test(1000, 100)
    # final_test()

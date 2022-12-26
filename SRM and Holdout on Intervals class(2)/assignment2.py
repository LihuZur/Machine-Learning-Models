#################################
# Your name: Lihu Zur
#################################

import numpy as np
import matplotlib.pyplot as plt
import intervals


class Assignment2(object):
    """Assignment 2 skeleton.

    Please use these function signatures for this assignment and submit this file, together with the intervals.py.
    """

    def sample_from_D(self, m):
        """Sample m data samples from D.
        Input: m - an integer, the size of the data sample.

        Returns: np.ndarray of shape (m,2) :
                A two dimensional array of size m that contains the pairs where drawn from the distribution P.
        """
        x = np.sort(np.random.uniform(size=m))
        interval_1 = (x > 0.2) & (x < 0.4)
        interval_2 = (x > 0.6) & (x < 0.8)
        intersection = ((~interval_1) & (~interval_2))  # The union of the compliments is what we actually need
        y = np.array([self.choose_label(intersection[i]) for i in range(m)])  # Choosing labels according to the distribution
        y.reshape(m,)
        return np.column_stack((x,y)) # ordering the x and y as pairs of an x value and a label for it, to be sent for testing


    def experiment_m_range_erm(self, m_first, m_last, step, k, T):
        """Runs the ERM algorithm.
        Calculates the empirical error and the true error.
        Plots the average empirical and true errors.
        Input: m_first - an integer, the smallest size of the data sample in the range.
               m_last - an integer, the largest size of the data sample in the range.
               step - an integer, the difference between the size of m in each loop.
               k - an integer, the maximum number of intervals.
               T - an integer, the number of times the experiment is performed.

        Returns: np.ndarray of shape (n_steps,2).
            A two dimensional array that contains the average empirical error
            and the average true error for each m in the range accordingly.
        """
        n_arr = np.arange(m_first, m_last + 1, step)
        emp_err = np.zeros(n_arr.shape)
        true_err = np.zeros(n_arr.shape)

        for i in range(n_arr.shape[0]):  # For each n in n_arr
            emp_i = 0
            true_i = 0

            for j in range(T):  # Run 100 times on the current n
                sample = self.sample_from_D(n_arr[i]) # Getting a sample of the required size
                ERM_intervals, ERM_emp_err = intervals.find_best_interval(sample[:, 0], sample[:, 1], k)
                emp_i += ERM_emp_err
                true_i += self.calc_true_error(ERM_intervals)

            emp_err[i] = emp_i / (T * n_arr[i])
            true_err[i] = true_i / T

        plt.plot(n_arr, emp_err, label="Empirical error")
        plt.plot(n_arr, true_err, label="True error")
        plt.legend()
        plt.xlabel("n")
        plt.ylabel("error")
        plt.show()

        return np.stack((emp_err, true_err))

    def experiment_k_range_erm(self, m, k_first, k_last, step):
        """Finds the best hypothesis for k= 1,2,...,10.
        Plots the empirical and true errors as a function of k.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the ERM algorithm.
        """
        k_arr = np.arange(k_first, k_last + 1, step)
        emp_err = np.zeros(k_arr.shape)
        true_err = np.zeros(k_arr.shape)
        sample = self.sample_from_D(m) # Getting a sample of the required size
        xs = sample[:, 0]
        ys = sample[:, 0]

        for i in range(k_arr.shape[0]):
            ERM_intervals, ERM_emp_err = intervals.find_best_interval(sample[:, 0], sample[:, 1], k_arr[i])
            emp_err[i] = ERM_emp_err / m
            true_err[i] = self.calc_true_error(ERM_intervals)

        plt.plot(k_arr, emp_err, label="Empirical error")
        plt.plot(k_arr, true_err, label="True error")
        plt.legend()
        plt.xlabel("k")
        plt.ylabel("error")
        plt.show()

        return np.argmin(emp_err)


    def experiment_k_range_srm(self, m, k_first, k_last, step):
        """Run the experiment in (c).
        Plots additionally the penalty for the best ERM hypothesis.
        and the sum of penalty and empirical error.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the SRM algorithm.
        """
        k_arr = np.arange(k_first, k_last + 1, step)
        emp_err = np.zeros(k_arr.shape)
        true_err = np.zeros(k_arr.shape)
        penalty = self.calc_penalty(k_arr, m)
        sample = self.sample_from_D(m)  # Getting a sample of the required size
        xs = sample[:, 0]
        ys = sample[:, 0]

        for i in range(k_arr.shape[0]):
            ERM_intervals, ERM_emp_err = intervals.find_best_interval(sample[:, 0], sample[:, 1], k_arr[i])
            emp_err[i] = ERM_emp_err / m
            true_err[i] = self.calc_true_error(ERM_intervals)

        penalty_plus_emp = penalty + emp_err

        plt.plot(k_arr, emp_err, label="Empirical error")
        plt.plot(k_arr, true_err, label="True error")
        plt.plot(k_arr, penalty, label="Penalty")
        plt.plot(k_arr, penalty_plus_emp, label="Penalty + empirical error")
        plt.legend()
        plt.xlabel("k")
        plt.show()

        return np.argmin(penalty_plus_emp)*step + k_first


    def cross_validation(self, m):
        """Finds a k that gives a good test error.
        Input: m - an integer, the size of the data sample.

        Returns: The best k value (an integer) found by the cross validation algorithm.
        """
        sample = self.sample_from_D(m)
        np.random.shuffle(sample)
        validation = sample[:int(m / 5)]  # 20% validation
        train = np.array(sorted(sample[int(m/5):], key=lambda x: x[0]))  # Rest 80% training

        x_train = train[:, 0]
        y_train = train[:, 1]
        x_validation = validation[:, 0]
        y_validation = validation[:, 1]
        ERM_validation_errors = np.zeros(10)  # Storing the errors on the validation data for each k

        for k in range(1, 11):
            k_intervals , emp_err_k = intervals.find_best_interval(x_train, y_train, k)
            ERM_validation_errors[k-1] = self.calc_validation_error(k_intervals, x_validation, y_validation)

        return np.argmin(ERM_validation_errors) + 1



    #################################
    # Place for additional methods

    # Choosing a label for a given input
    def choose_label(self, num):
        if num:
            return np.random.choice([0.0,1.0], size=1, p=[0.2, 0.8])
        return np.random.choice([0.0,1.0], size=1, p=[0.9, 0.1])

    # Finding the rest of the intervals
    def get_other_intervals(self, intervals):
        rest = [(0.0, intervals[0][0])]
        n = len(intervals)
        for i in range(n - 1):
            rest.append((intervals[i][1], intervals[i + 1][0])) #Taking what's in between the initial intervals

        rest.append((intervals[len(intervals) - 1][1], 1.0)) # Taking what's left from the last initial interval till the end
        return rest

    # Finding the "weight" of the intersection of the given 2 intervals in the [0,1] borders
    def find_intersection_weight(self, i1, i2):
        len1 = i1[1] - i1[0]
        len2 = i2[1] - i2[0]

        # No intersection
        if(i1[0] >= i2[1] or i2[0] >= i1[1]):
            return 0

        # i1 is inside i2
        if(i1[0] >= i2[0] and i1[1] <= i2[1]):
            return len1

        #i2 is inside i1
        if(i2[0] >= i1[0] and i2[1] <= i1[1]):
            return len2

        # i1 and i2 intersect, i1 continues beyond i2
        if(i1[0] > i2[0] and i2[1] < i1[1]):
            return i2[1] - i1[0]

        # i1 and i2 intersect, i1 continues beyond i2
        return i1[1] - i2[0]

    # Calculating the true error, iterating through all the intervals and for their compliments in respect to [0,1]
    def calc_true_error(self, intervals):
        one_classified = [(0, 0.2), (0.4, 0.6), (0.8, 1)]
        zero_classified = self.get_other_intervals(one_classified)
        other_intervals = self.get_other_intervals(intervals)  # Hypothesis returns 0 here

        res = 0
        p1 = 0.8
        p0 = 0.1

        # Calculating for intervals (where the hypothesis returns 1)
        for i in intervals:
            for i1 in one_classified:
                res += (1 - p1) * self.find_intersection_weight(i, i1)
            for i0 in zero_classified:
                res += (1 - p0) * self.find_intersection_weight(i, i0)

        # Calculating for other_intervals (where the hypothesis returns 0)
        for i in other_intervals:
            for i1 in one_classified:
                res += p1 * self.find_intersection_weight(i, i1)
            for i0 in zero_classified:
                res += p0 * self.find_intersection_weight(i, i0)

        return res  # Res now has the total some of weighted mistakes (The true error)

    # Calculating the penalty for given k_arr and given m size dataset, using delta=0.1 and VC_dim = 2k
    def calc_penalty(self, k_arr, n):
        calc = (2 * k_arr + np.log(20))/ n
        return 2 * np.sqrt(calc)

    # Calculating the validation error
    def calc_validation_error(self, k_intervals, x_v, y_v):
        error = 0

        for i in range(len(x_v)):
            x_pred = 0
            for interval in k_intervals:
                if(x_v[i] >= interval[0] and x_v[i] <= interval[1]):  # x is inside an interval
                    x_pred = 1
                    break  # No need to check other intervals

                if(x_v[i] < interval[0]):  # Intervals are sorted, so if this condition happens it means x is not inside any of them
                    break
            error += (x_pred != y_v[i])  # Adding to the error if the prediction was wrong

        return error / len(x_v)


    #################################


if __name__ == '__main__':
    ass = Assignment2()
    ass.experiment_m_range_erm(10, 100, 5, 3, 100)
    ass.experiment_k_range_erm(1500, 1, 10, 1)
    ass.experiment_k_range_srm(1500, 1, 10, 1)
    print(ass.cross_validation(1500))

    #Finding the best 3 intervals with ERM
    sample = ass.sample_from_D(1500)
    xs = sample[:, 0]
    ys = sample[:, 1]
    print(intervals.find_best_interval(xs, ys, 3)[0])


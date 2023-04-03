# import pandas as pd
import numpy as np
import math


def gradient_descent2(p, q):
    m_curr = b_curr = 0
    n = len(p)
    iterations = 10
    prev_cost = 0
    # iterations=415533
    learning_rate = 0.0002
    i = 0
    # for i in range(iterations):
    while 1:
        i = i + 1
        q_predicted = m_curr * p + b_curr
        cost = (1 / n) * sum([val ** 2 for val in (q - q_predicted)])
        if math.isclose(prev_cost, cost, rel_tol=1e-20, abs_tol=0):
            break;
        m_der = -(2 / n) * sum(p * (q - q_predicted))
        b_der = -(2 / n) * sum((q - q_predicted))
        m_curr = m_curr - (learning_rate * m_der)
        b_curr = b_curr - (learning_rate * b_der)
        # print("m={}, b={}, cost={}, iterations={}"
        #       .format(m_curr,b_curr,cost,i))
        prev_cost = cost
        print(i)
    print(m_curr)
    print(b_curr)


df = pd.read_csv("test_scores.CSV")
x = np.array(df.math)
y = np.array(df.cs)
gradient_descent2(x, y)

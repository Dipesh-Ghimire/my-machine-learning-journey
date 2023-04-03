import numpy as np

def gradient_descent(x,y):
    m_curr = b_curr = 0
    iterations = 100000

    n = len(x)
    learning_rate = 0.00001

    for i in range(iterations):
        y_predicted = m_curr*x+b_curr
        cost = (1/n)*sum([val**2 for val in (y-y_predicted)])
        m_der = -(2/n) * sum(x*(y-y_predicted))
        b_der = -(2/n) * sum(y-y_predicted)
        m_curr = m_curr -(learning_rate*m_der)
        b_curr = b_curr -(learning_rate*b_der)
        print("m {}, b {}, cost{}, iterations {}".format(m_curr,b_curr,cost,i))


# sum() function takes a list & returns sum of elements of that list
# Matrix Multiplication in numpy array is faster instead of list
x = np.array([1,2,3,4,5])
y = np.array([5,7,9,11,13])
gradient_descent(x,y)

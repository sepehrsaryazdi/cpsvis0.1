import numpy as np

def compute_m_inverse(r0, r2, c0, c2, e03, e23):
    A = np.array([[1 / e03, 0, 0], [0, 1 / e23, 0], [0, 0, 1 / (e03 * e23)]])
    B = np.array([c2, c0, -np.cross(r2, r0)]).T
    m_inverse = np.matmul(A, B)
    return m_inverse


def compute_c3(m_inverse, e03, e23, A023):
    c3 = np.matmul(m_inverse, np.array([[e03], [e23], [A023]]))
    c3 = c3.T.flatten()

    return c3


def compute_r3(c0, c2, c3, e30, e32):
    A = np.array([c0, c2, c3])
    r3 = np.matmul(np.linalg.inv(A), np.array([[e30], [e32], [0]]))
    r3 = r3.T.flatten()
    return r3
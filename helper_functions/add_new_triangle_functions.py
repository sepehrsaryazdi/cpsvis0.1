import numpy as np

def outitude_edge_params(A,B,a_minus,a_plus, b_minus, b_plus, e_minus, e_plus):
    return A*(e_plus*a_plus+e_minus*b_minus-e_minus*e_plus) + B*(e_plus*b_plus+e_minus*a_minus - e_minus*e_plus)

def compute_m_inverse(r0, r2, c0, c2, e03, e23):
    C = np.array([r0, r2, np.cross(c0, c2)])
    A = np.array([[1 / e03, 0, 0], [0, 1 / e23, 0], [0, 0, 1 / (e03 * e23)]])
    B = np.array([c2, c0, np.cross(r2, r0)]).T
    #m_inverse = np.matmul(A, B)
    m_inverse = np.linalg.inv(C)
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

def compute_outitude_sign(c0,c1,c2,c3):
    D = [c1,c2,c3]
    D_prime = [c0,c1,c3]
    C = [c0,c1,c2]
    C_prime = [c0,c2,c3]
    return np.linalg.det(D) + np.linalg.det(D_prime) - np.linalg.det(C) - np.linalg.det(C_prime)


def compute_t(e01, e12, e20, e10, e21, e02):
    return e01*e12*e20/(e10*e21*e02)

def compute_all_until_r3c3(r0, r2, c0, c2, e03, e23, e30, e32, A023):
    m_inverse = compute_m_inverse(r0, r2, c0, c2, e03, e23)
    c3 = compute_c3(m_inverse, e03, e23, A023)
    r3 = compute_r3(c0, c2, c3, e30, e32)
    return (r3, c3)
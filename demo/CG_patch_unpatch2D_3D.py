import numpy as np
def cg_patch(A, l1, l2, o1, o2):
    n1, n2 = np.shape(A)
    tmp = np.mod(n1 - l1, o1)
    if tmp != 0:
        # print(np.shape(A), o1-tmp, n2)
        # A = np.concatenate([A, np.zeros((o1 - tmp, n2))], axis=0)
        A = np.pad(A, ((0, o1 - tmp), (0, 0)), 'reflect')

    tmp = np.mod(n2 - l2, o2)
    if tmp != 0:
        # A = np.concatenate([A, np.zeros((A.shape[0], o2 - tmp))], axis=-1);
        A = np.pad(A, ((0, 0), (0, o2 - tmp)), 'reflect')

    N1, N2 = np.shape(A)
    X = []
    for i1 in range(0, N1 - l1 + 1, o1):
        for i2 in range(0, N2 - l2 + 1, o2):
            tmp = np.reshape(A[i1:i1 + l1, i2:i2 + l2], (l1 * l2, 1))
            X.append(tmp)
    X = np.array(X)
    return X[:, :, 0].T


def cg_patch_inv(X1, n1, n2, l1, l2, o1, o2):
    tmp1 = np.mod(n1 - l1, o1)
    tmp2 = np.mod(n2 - l2, o2)
    if (tmp1 != 0) and (tmp2 != 0):
        A = np.zeros((n1 + o1 - tmp1, n2 + o2 - tmp2))
        mask = np.zeros((n1 + o1 - tmp1, n2 + o2 - tmp2))

    if (tmp1 != 0) and (tmp2 == 0):
        A = np.zeros((n1 + o1 - tmp1, n2))
        mask = np.zeros((n1 + o1 - tmp1, n2))

    if (tmp1 == 0) and (tmp2 != 0):
        A = np.zeros((n1, n2 + o2 - tmp2))
        mask = np.zeros((n1, n2 + o2 - tmp2))

    if (tmp1 == 0) and (tmp2 == 0):
        A = np.zeros((n1, n2))
        mask = np.zeros((n1, n2))

    N1, N2 = np.shape(A)
    ids = 0
    for i1 in range(0, N1 - l1 + 1, o1):
        for i2 in range(0, N2 - l2 + 1, o2):
            # print(i1,i2)
            #       [i1,i2,ids]
            A[i1:i1 + l1, i2:i2 + l2] = A[i1:i1 + l1, i2:i2 + l2] + np.reshape(X1[:, ids], (l1, l2))
            mask[i1:i1 + l1, i2:i2 + l2] = mask[i1:i1 + l1, i2:i2 + l2] + np.ones((l1, l2))
            ids = ids + 1

    A = A / mask
    # A = A[0:n1, 0:n2]

    return A[0:n1, 0:n2]


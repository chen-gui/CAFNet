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


def CG_patch3d(A, l1=4, l2=4, l3=4, s1=2, s2=2, s3=2):
    """
    patch3d: decompose 3D data into patches

    Input
      D: input image
      mode: patching mode
      l1: first patch size
      l2: second patch size
      l3: third patch size
      s1: first shifting size
      s2: second shifting size
      s3: third shifting size

    Output
      X: patches
    """

    [n1, n2, n3] = A.shape
    tmp = np.mod(n1 - l1, s1)
    if tmp != 0:
        # A = np.concatenate((A, np.zeros([s1 - tmp, n2, n3])), axis=0)
        A = np.pad(A, ((0, s1 - tmp), (0, 0), (0, 0)), 'reflect')

    tmp = np.mod(n2 - l2, s2)
    if tmp != 0:
        # A = np.concatenate((A, np.zeros([A.shape[0], s2 - tmp, n3])), axis=1)
        A = np.pad(A, ((0, 0), (0, s2 - tmp), (0, 0)), 'reflect')

    tmp = np.mod(n3 - l3, s3)
    if tmp != 0:
        # A = np.concatenate((A, np.zeros([A.shape[0], A.shape[1], s3 - tmp])),
        #                    axis=2)  # concatenate along the third dimension
        A = np.pad(A, ((0, 0), (0, 0), (0, s3 - tmp)), 'reflect')


    [N1, N2, N3] = A.shape
    for i1 in range(0, N1 - l1 + 1, s1):
        for i2 in range(0, N2 - l2 + 1, s2):
            for i3 in range(0, N3 - l3 + 1, s3):
                if i1 == 0 and i2 == 0 and i3 == 0:
                    X = np.reshape(A[i1:i1 + l1, i2:i2 + l2, i3:i3 + l3], (l1 * l2 * l3, 1))
                else:
                    tmp = np.reshape(A[i1:i1 + l1, i2:i2 + l2, i3:i3 + l3], (l1 * l2 * l3, 1))
                    X = np.concatenate((X, tmp), axis=1)

    return X

def CG_patch3d_coords(coords, l1=4, l2=4, l3=4, s1=2, s2=2, s3=2):
    """
    输入：coords.shape = (n1, n2, n3, 3)，即每个点的 (x, y, z) 坐标
    输出：X_coords.shape = (l1*l2*l3*3, num_patches)，每列为一个 patch 的坐标展开
    """
    n1, n2, n3, _ = coords.shape
    tmp = np.mod(n1 - l1, s1)
    if tmp != 0:
        coords = np.pad(coords, ((0, s1 - tmp), (0, 0), (0, 0), (0, 0)), 'reflect')

    tmp = np.mod(n2 - l2, s2)
    if tmp != 0:
        coords = np.pad(coords, ((0, 0), (0, s2 - tmp), (0, 0), (0, 0)), 'reflect')

    tmp = np.mod(n3 - l3, s3)
    if tmp != 0:
        coords = np.pad(coords, ((0, 0), (0, 0), (0, s3 - tmp), (0, 0)), 'reflect')

    N1, N2, N3, _ = coords.shape
    for i1 in range(0, N1 - l1 + 1, s1):
        for i2 in range(0, N2 - l2 + 1, s2):
            for i3 in range(0, N3 - l3 + 1, s3):
                patch = coords[i1:i1 + l1, i2:i2 + l2, i3:i3 + l3, :]  # shape (l1, l2, l3, 3)
                patch = np.reshape(patch, (l1 * l2 * l3, 3, 1))
                if i1 == 0 and i2 == 0 and i3 == 0:
                    X_coords = patch
                else:
                    X_coords = np.concatenate((X_coords, patch), axis=2)

    return X_coords

def CG_patch3d_inv(X, n1, n2, n3, l1=4, l2=4, l3=4, s1=2, s2=2, s3=2):
    """
    patch3d_inv: insert patches into the 3D data

    Input
      D: input image
      mode: patching mode
      n1: first dimension size
      n1: second dimension size
      n3: third dimension size
      l1: first patch size
      l2: second patch size
      l3: third patch size
      s1: first shifting size
      s2: second shifting size
      s3: third shifting size

    Output
      X: patches

    """
    # global A
    # tmp1 = np.mod(n1 - l1, s1)
    # tmp2 = np.mod(n2 - l2, s2)
    # tmp3 = np.mod(n3 - l3, s3)
    # if tmp1 != 0 and tmp2 != 0 and tmp3 != 0:
    #     A = np.zeros([n1 + s1 - tmp1, n2 + s2 - tmp2, n3 + s3 - tmp3])
    #     mask = np.zeros([n1 + s1 - tmp1, n2 + s2 - tmp2, n3 + s3 - tmp3])
    #
    # if tmp1 != 0 and tmp2 != 0 and tmp3 == 0:
    #     A = np.zeros([n1 + s1 - tmp1, n2 + s2 - tmp2, n3])
    #     mask = np.zeros([n1 + s1 - tmp1, n2 + s2 - tmp2, n3])
    #
    # if tmp1 != 0 and tmp2 == 0 and tmp3 == 0:
    #     A = np.zeros([n1 + s1 - tmp1, n2, n3])
    #     mask = np.zeros([n1 + s1 - tmp1, n2, n3])
    #
    # if tmp1 != 0 and tmp2 == 0 and tmp3 != 0: #陈桂添加的情况
    #     A = np.zeros([n1 + s1 - tmp1, n2, n3 + s3 - tmp3])
    #     mask = np.zeros([n1 + s1 - tmp1, n2, n3 + s3 - tmp3])
    #
    # if tmp1 == 0 and tmp2 != 0 and tmp3 == 0:
    #     A = np.zeros([n1, n2 + s2 - tmp2, n3])
    #     mask = np.zeros([n1, n2 + s2 - tmp2, n3])
    #
    # if tmp1 == 0 and tmp2 == 0 and tmp3 != 0:
    #     A = np.zeros([n1, n2, n3 + s3 - tmp3])
    #     mask = np.zeros([n1, n2, n3 + s3 - tmp3])
    #
    # if tmp1 == 0 and tmp2 == 0 and tmp3 == 0:
    #     A = np.zeros([n1, n2, n3])
    #     mask = np.zeros([n1, n2, n3])
    # else:
    #     raise ValueError("Unhandled case for tmp1, tmp2, tmp3 values.")

    A = np.zeros((n1, n2, n3), dtype=np.float32)
    mask = np.zeros((n1, n2, n3), dtype=np.float32)
    tmp = np.mod(n1 - l1, s1)
    if tmp != 0:
        A = np.pad(A, ((0, s1 - tmp), (0, 0), (0, 0)), 'reflect')
        mask = np.pad(mask, ((0, s1 - tmp), (0, 0), (0, 0)), 'reflect')

    tmp = np.mod(n2 - l2, s2)
    if tmp != 0:
        A = np.pad(A, ((0, 0), (0, s2 - tmp), (0, 0)), 'reflect')
        mask = np.pad(mask, ((0, 0), (0, s2 - tmp), (0, 0)), 'reflect')

    tmp = np.mod(n3 - l3, s3)
    if tmp != 0:
        A = np.pad(A, ((0, 0), (0, 0), (0, s3 - tmp)), 'reflect')
        mask = np.pad(mask, ((0, 0), (0, 0), (0, s3 - tmp)), 'reflect')

    N1, N2, N3 = np.shape(A)

    id = -1
    for i1 in range(0, N1 - l1 + 1, s1):
        for i2 in range(0, N2 - l2 + 1, s2):
            for i3 in range(0, N3 - l3 + 1, s3):
                id = id + 1
                A[i1:i1 + l1, i2:i2 + l2, i3:i3 + l3] = A[i1:i1 + l1, i2:i2 + l2, i3:i3 + l3] + np.reshape(X[:, id],
                                                                                                           (l1, l2,
                                                                                                            l3))
                mask[i1:i1 + l1, i2:i2 + l2, i3:i3 + l3] = mask[i1:i1 + l1, i2:i2 + l2, i3:i3 + l3] + np.ones(
                    (l1, l2, l3))
    A = A / mask

    A = A[0:n1, 0:n2, 0:n3]

    return A

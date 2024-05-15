from sympy import Integer, Rational, Float

import numpy as np
from tqdm import tqdm, trange


def affine_legendre_recursion_accurate(lmax: int, shift: Float, scale: Float):
    coeffs = [[Integer(1)], [shift, scale]]
    for n in range(2, lmax + 1):
        coeffs_n = []
        A_nm1_0 = coeffs[n - 1][0]
        A_nm1_1 = coeffs[n - 1][1]
        A_nm2_0 = coeffs[n - 2][0]

        A_n_0 = Rational(2*n - 1, n)*A_nm1_0*shift - Rational(n - 1, n)*A_nm2_0 + Rational(2*n - 1, 3*n)*A_nm1_1*scale

        coeffs_n.append(A_n_0)

        for l in range(1, n - 1):
            A_nm1_l = coeffs[n - 1][l]
            A_nm2_l = coeffs[n - 2][l]
            A_nm1_lp1 = coeffs[n - 1][l + 1]
            A_nm1_lm1 = coeffs[n - 1][l - 1]

            A_n_l = Rational(2*n - 1, n)*A_nm1_l*shift - Rational(n - 1, n)*A_nm2_l + Rational(2*n - 1, n)*Rational(l + 1, 2*l + 3)*A_nm1_lp1*scale + Rational(2*n - 1, n)*Rational(l, 2*l - 1)*A_nm1_lm1*scale

            coeffs_n.append(A_n_l)
        
        A_nm1_nm1 = coeffs[n - 1][n - 1]
        A_nm1_nm2 = coeffs[n - 1][n - 2]

        A_n_nm1 = Rational(2*n - 1, n)*A_nm1_nm1*shift + Rational(2*n - 1, n)*Rational(n - 1, 2*n - 3)*A_nm1_nm2*scale

        coeffs_n.append(A_n_nm1)

        A_n_n = A_nm1_nm1*scale

        coeffs_n.append(A_n_n)

        coeffs.append(coeffs_n)
    
    return coeffs


def affine_legendre_recursion(lmax: int, shift: np.ndarray, scale: np.ndarray):
    m_a = np.zeros(lmax + 1)
    m_b = np.zeros(lmax + 1)
    m_c = np.zeros(lmax + 1)
    m_d = np.zeros(lmax + 1)

    m_c[1] = 2.0/5.0
    m_d[1] = 1.0
    for n in range(2, lmax + 1):
        dn = float(n)
        inv_dn = 1.0/dn
        m_a[n] = (2.0*dn - 1.0)*inv_dn
        m_b[n] = (dn - 1.0)*inv_dn
        m_c[n] = (dn + 1.0)/(2.0*dn + 3.0)
        m_d[n] = dn/(2.0*dn - 1.0)
    
    expansion = []
    for n in range(0, lmax + 1):
        exp_n = []
        for l in range(0, n + 1):
            exp_n.append(np.zeros(shift.shape))
        expansion.append(exp_n)

    expansion[0][0] = np.ones(shift.shape)
    expansion[1][0] = shift
    expansion[1][1] = scale
    for n in range(2, lmax + 1):
        expansion[n][0] = (m_a[n]*shift*expansion[n - 1][0]
                - m_b[n]*expansion[n - 2][0]
                + m_a[n]*(1.0/3.0)*scale*expansion[n - 1][1])
        for l in range(0, n - 1):
            expansion[n][l] = (m_a[n]*shift*expansion[n - 1][l]
                    - m_b[n]*expansion[n - 2][l]
                    + m_a[n]*m_c[l]*scale*expansion[n - 1][l + 1]
                    + m_a[n]*m_d[l]*scale*expansion[n - 1][l - 1])

        expansion[n][n - 1] = (m_a[n]*shift*expansion[n - 1][n - 1]
                + m_a[n]*m_d[n - 1]*scale*expansion[n - 1][n - 2])
        expansion[n][n] = scale*expansion[n - 1][n - 1]

    return expansion


def shifted_legendre_recursion_accurate(lmax: int, shift: Float):
    coeffs = [[Integer(1)], [shift, Integer(1)]]
    for n in range(2, lmax + 1):
        coeffs_n = []
        A_nm1_0 = coeffs[n - 1][0]
        A_nm1_1 = coeffs[n - 1][1]
        A_nm2_0 = coeffs[n - 2][0]

        A_n_0 = Rational(2*n - 1, n)*A_nm1_0*shift - Rational(n - 1, n)*A_nm2_0 + Rational(2*n - 1, 3*n)*A_nm1_1

        coeffs_n.append(A_n_0)

        for l in range(1, n - 1):
            A_nm1_l = coeffs[n - 1][l]
            A_nm2_l = coeffs[n - 2][l]
            A_nm1_lp1 = coeffs[n - 1][l + 1]
            A_nm1_lm1 = coeffs[n - 1][l - 1]

            A_n_l = Rational(2*n - 1, n)*A_nm1_l*shift - Rational(n - 1, n)*A_nm2_l + Rational(2*n - 1, n)*Rational(l + 1, 2*l + 3)*A_nm1_lp1 + Rational(2*n - 1, n)*Rational(l, 2*l - 1)*A_nm1_lm1

            coeffs_n.append(A_n_l)
        
        A_nm1_nm1 = coeffs[n - 1][n - 1]
        A_nm1_nm2 = coeffs[n - 1][n - 2]

        A_n_nm1 = Rational(2*n - 1, n)*A_nm1_nm1*shift + Rational(2*n - 1, n)*Rational(n - 1, 2*n - 3)*A_nm1_nm2

        coeffs_n.append(A_n_nm1)

        A_n_n = A_nm1_nm1

        coeffs_n.append(A_n_n)

        coeffs.append(coeffs_n)
    
    return coeffs


def shifted_legendre_recursion(lmax: int, shift: np.ndarray):
    m_a = np.zeros(lmax + 1)
    m_b = np.zeros(lmax + 1)
    m_c = np.zeros(lmax + 1)
    m_d = np.zeros(lmax + 1)

    m_c[1] = 2.0/5.0
    m_d[1] = 1.0
    for n in range(2, lmax + 1):
        dn = float(n)
        inv_dn = 1.0/dn
        m_a[n] = (2.0*dn - 1.0)*inv_dn
        m_b[n] = (dn - 1.0)*inv_dn
        m_c[n] = (dn + 1.0)/(2.0*dn + 3.0)
        m_d[n] = dn/(2.0*dn - 1.0)
        
    expansion = []
    for n in range(0, lmax + 1):
        exp_n = []
        for l in range(0, n + 1):
            exp_n.append(np.zeros(shift.shape))
        expansion.append(exp_n)

    expansion[0][0] = np.ones(shift.shape)
    expansion[1][0] = shift
    expansion[1][1] = np.ones(shift.shape)
    for n in range(2, lmax + 1):
        expansion[n][0] = (m_a[n]*shift*expansion[n - 1][0]
                - m_b[n]*expansion[n - 2][0]
                + m_a[n]*(1.0/3.0)*expansion[n - 1][1])
        for l in range(1, n - 1):
            expansion[n][l] = (m_a[n]*shift*expansion[n - 1][l]
                    - m_b[n]*expansion[n - 2][l]
                    + m_a[n]*m_c[l]*expansion[n - 1][l + 1]
                    + m_a[n]*m_d[l]*expansion[n - 1][l - 1])

        expansion[n][n - 1] = (m_a[n]*shift*expansion[n - 1][n - 1]
                + m_a[n]*m_d[n - 1]*expansion[n - 1][n - 2])
        expansion[n][n] = expansion[n - 1][n - 1]

    return expansion


def scaled_legendre_recursion_accurate(lmax: int, scale: Float):
    coeffs = [[Integer(1)], [Integer(0), scale]]
    for n in range(2, lmax + 1):
        coeffs_n = []
        A_nm1_0 = coeffs[n - 1][0]
        A_nm1_1 = coeffs[n - 1][1]
        A_nm2_0 = coeffs[n - 2][0]

        A_n_0 = Rational(2*n - 1, 3*n)*A_nm1_1*scale - Rational(n - 1, n)*A_nm2_0

        coeffs_n.append(A_n_0)

        for l in range(1, n - 1):
            A_nm1_l = coeffs[n - 1][l]
            A_nm2_l = coeffs[n - 2][l]
            A_nm1_lp1 = coeffs[n - 1][l + 1]
            A_nm1_lm1 = coeffs[n - 1][l - 1]

            A_n_l = + Rational(2*n - 1, n)*Rational(l + 1, 2*l + 3)*A_nm1_lp1*scale + Rational(2*n - 1, n)*Rational(l, 2*l - 1)*A_nm1_lm1*scale - Rational(n - 1, n)*A_nm2_l

            coeffs_n.append(A_n_l)
        
        A_nm1_nm1 = coeffs[n - 1][n - 1]
        A_nm1_nm2 = coeffs[n - 1][n - 2]

        A_n_nm1 = Rational(2*n - 1, n)*Rational(n - 1, 2*n - 3)*A_nm1_nm2*scale

        coeffs_n.append(A_n_nm1)

        A_n_n = A_nm1_nm1*scale

        coeffs_n.append(A_n_n)

        coeffs.append(coeffs_n)
    
    return coeffs


def scaled_legendre_recursion(lmax: int, scale: np.ndarray):
    m_a = np.zeros(lmax + 1)
    m_b = np.zeros(lmax + 1)
    m_c = np.zeros(lmax + 1)
    m_d = np.zeros(lmax + 1)

    m_c[1] = 2.0/5.0
    m_d[1] = 1.0
    for n in range(2, lmax + 1):
        dn = float(n)
        inv_dn = 1.0/dn
        m_a[n] = (2.0*dn - 1.0)*inv_dn
        m_b[n] = (dn - 1.0)*inv_dn
        m_c[n] = (dn + 1.0)/(2.0*dn + 3.0)
        m_d[n] = dn/(2.0*dn - 1.0)
        
    expansion = []
    for n in range(0, lmax + 1):
        exp_n = []
        for l in range(0, n + 1):
            exp_n.append(np.zeros(scale.shape))
        expansion.append(exp_n)

    expansion[0][0] = np.ones(scale.shape)
    expansion[1][0] = np.zeros(scale.shape)
    expansion[1][1] = scale
    for n in range(2, lmax + 1):
        expansion[n][0] = (m_a[n]*(1.0/3.0)*scale*expansion[n - 1][1]
                - m_b[n]*expansion[n - 2][0])

        lmin = 1 + ((n & 1) ^ 1)
        for l in range(lmin, n - 1, 2):
            expansion[n][l] = (m_a[n]*m_c[l]*scale*expansion[n - 1][l + 1]
                    + m_a[n]*m_d[l]*scale*expansion[n - 1][l - 1]
                    - m_b[n]*expansion[n - 2][l])

        expansion[n][n] = scale*expansion[n - 1][n - 1]

    return expansion


if __name__ == "__main__":
    a_arr = np.concatenate((np.linspace(-1.5,0.0,40), np.linspace(0.0,1.5,40)[1:]))
    b_arr = np.linspace(0,1.0,81)[1:]

    lmax = 200

    print("Calculating accurate shifted recursion...")
    accurate_shifted_list = []
    for a in tqdm(a_arr):
        accurate_shifted_list.append(shifted_legendre_recursion_accurate(lmax, Float(a, 64)))

    print("Calculating accurate scaled recursion...")
    accurate_scaled_list = []
    for b in tqdm(b_arr):
        accurate_scaled_list.append(scaled_legendre_recursion_accurate(lmax, Float(b, 64)))

    print("Calculating shifted arrays...")
    shifted_list = shifted_legendre_recursion(lmax, a_arr)
    print("Calculating scaled arrays...")
    scaled_list = scaled_legendre_recursion(lmax, b_arr)
            
    indices = np.zeros((int((lmax + 1)*(lmax + 2)/2), 2), dtype=np.int32)
    accurate_shifted = np.zeros((int((lmax + 1)*(lmax + 2)/2), a_arr.size))
    accurate_scaled = np.zeros((int((lmax + 1)*(lmax + 2)/2), b_arr.size))
    shifted = np.zeros((int((lmax + 1)*(lmax + 2)/2), a_arr.size))
    scaled = np.zeros((int((lmax + 1)*(lmax + 2)/2), b_arr.size))

    print("Compressing arrays...")
    ind = 0
    for n in trange(lmax + 1):
        for l in range(n + 1):
            accurate_shifted_arr = [float(accurate_shifted_list[i][n][l]) for i, _ in enumerate(a_arr)]
            accurate_scaled_arr = [float(accurate_scaled_list[i][n][l]) for i, _ in enumerate(b_arr)]
            accurate_shifted[ind] = np.array(accurate_shifted_arr)
            accurate_scaled[ind] = np.array(accurate_scaled_arr)
            shifted[ind] = shifted_list[n][l]
            scaled[ind] = scaled_list[n][l]
            indices[ind][0] = n
            indices[ind][1] = l
            ind += 1

    
    print("Saving...")
    np.savez_compressed(
        f"affine_legendre_lmax_accuracy_{lmax}.npz", indices=indices, accurate_shifted=accurate_shifted, accurate_scaled=accurate_scaled, shifted=shifted, scaled=scaled)
    print("Finished!")
    

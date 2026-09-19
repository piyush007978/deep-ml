import numpy as np

def polynomial_derivative(coeffs):
    """
    Return coefficients of the derivative of a polynomial.
    coeffs are in descending order of powers, e.g. [1, 2, 3] -> x^2 + 2x + 3
    """
    n = len(coeffs) - 1  # highest power
    deriv = [coeffs[i] * (n - i) for i in range(n)]  # drop last term (constant), it vanishes
    return deriv if deriv else [0]


def evaluate_polynomial(coeffs, x):
    """
    Evaluate a polynomial (descending-order coefficients) at point x using Horner's method.
    """
    result = 0
    for c in coeffs:
        result = result * x + c
    return result

def quotient_rule_derivative(g_coeffs: list, h_coeffs: list, x: float) -> float:
    """
    Compute the derivative of f(x) = g(x)/h(x) at point x using the quotient rule.
    
    Args:
        g_coeffs: Coefficients of numerator polynomial in descending order
        h_coeffs: Coefficients of denominator polynomial in descending order
        x: Point at which to evaluate the derivative
        
    Returns:
        The derivative value f'(x)
    """
    # Your code here
    g_prime_coeffs = polynomial_derivative(g_coeffs)
    h_prime_coeffs = polynomial_derivative(h_coeffs)

    g_val = evaluate_polynomial(g_coeffs, x)
    h_val = evaluate_polynomial(h_coeffs, x)
    g_prime_val = evaluate_polynomial(g_prime_coeffs, x)
    h_prime_val = evaluate_polynomial(h_prime_coeffs, x)

    if h_val == 0:
        raise ValueError("Denominator h(x) is zero at the given point x.")

    numerator = g_prime_val * h_val - g_val * h_prime_val
    denominator = h_val ** 2

    return numerator / denominator
#Utilized ChatGPT to build on code base and modify callback to compare to table.
#import region
"""imports call functions to use to calculate the probabilities and arguments"""
import scipy.integrate as spi
import scipy.special as sp
import numpy as np
#endregion

def gamma_function(alpha):
    """Computes the Gamma function using numerical integration."""
    integral, _ = spi.quad(lambda t: np.exp(-t) * t ** (alpha - 1), 0, np.inf)
    return integral


def compute_F(z, m):
    """Computes F(z) using numerical integration."""
    Km = sp.gamma(0.5 * (m + 1)) / (np.sqrt(m * np.pi) * sp.gamma(0.5 * m))
    integral, _ = spi.quad(lambda u: (1 + u ** 2 / m) ** (-(m + 1) / 2), -np.inf, z)
    return Km * integral


# Table A9 stored as a dictionary with (m, z) as keys
t_table = {
    1: {0.00: 0.5, 0.26: 0.6, 0.54: 0.7, 0.88: 0.8, 1.36: 0.9, 1.96: 0.95, 3.08: 0.975, 6.31: 0.99},
    2: {0.00: 0.5, 0.32: 0.6, 0.62: 0.7, 1.06: 0.8, 1.89: 0.9, 2.92: 0.95, 4.30: 0.975, 6.96: 0.99},
    3: {0.00: 0.5, 0.28: 0.6, 0.58: 0.7, 0.98: 0.8, 1.64: 0.9, 2.35: 0.95, 3.18: 0.975, 5.84: 0.99},
    4: {0.00: 0.5, 0.27: 0.6, 0.57: 0.7, 0.94: 0.8, 1.53: 0.9, 2.13: 0.95, 2.78: 0.975, 4.60: 0.99},
    5: {0.00: 0.5, 0.27: 0.6, 0.56: 0.7, 0.92: 0.8, 1.48: 0.9, 2.02: 0.95, 2.57: 0.975, 4.03: 0.99},
    6: {0.00: 0.5, 0.26: 0.6, 0.55: 0.7, 0.91: 0.8, 1.44: 0.9, 1.94: 0.95, 2.45: 0.975, 3.71: 0.99},
    7: {0.00: 0.5, 0.26: 0.6, 0.55: 0.7, 0.90: 0.8, 1.41: 0.9, 1.89: 0.95, 2.36: 0.975, 3.50: 0.99},
    8: {0.00: 0.5, 0.26: 0.6, 0.55: 0.7, 0.89: 0.8, 1.40: 0.9, 1.86: 0.95, 2.31: 0.975, 3.36: 0.99},
    9: {0.00: 0.5, 0.26: 0.6, 0.54: 0.7, 0.88: 0.8, 1.38: 0.9, 1.83: 0.95, 2.26: 0.975, 3.25: 0.99},
    10: {0.00: 0.5, 0.26: 0.6, 0.54: 0.7, 0.88: 0.8, 1.37: 0.9, 1.81: 0.95, 2.23: 0.975, 3.17: 0.99},
    11: {0.00: 0.5, 0.26: 0.6, 0.54: 0.7, 0.88: 0.8, 1.36: 0.9, 1.80: 0.95, 2.20: 0.975, 2.72: 0.99},
    12: {0.00: 0.5, 0.26: 0.6, 0.54: 0.7, 0.87: 0.8, 1.35: 0.9, 1.78: 0.95, 2.18: 0.975, 3.05: 0.99},
    13: {0.00: 0.5, 0.26: 0.6, 0.54: 0.7, 0.87: 0.8, 1.34: 0.9, 1.77: 0.95, 2.16: 0.975, 3.01: 0.99},
    14: {0.00: 0.5, 0.26: 0.6, 0.54: 0.7, 0.87: 0.8, 1.34: 0.9, 1.76: 0.95, 2.14: 0.975, 2.98: 0.99},
    15: {0.00: 0.5, 0.26: 0.6, 0.54: 0.7, 0.87: 0.8, 1.34: 0.9, 1.75: 0.95, 2.13: 0.975, 2.58: 0.99},
    16: {0.00: 0.5, 0.26: 0.6, 0.54: 0.7, 0.86: 0.8, 1.33: 0.9, 1.75: 0.95, 2.12: 0.975, 2.45: 0.99},
    17: {0.00: 0.5, 0.26: 0.6, 0.54: 0.7, 0.86: 0.8, 1.33: 0.9, 1.74: 0.95, 2.11: 0.975, 2.36: 0.99},
    18: {0.00: 0.5, 0.26: 0.6, 0.54: 0.7, 0.86: 0.8, 1.32: 0.9, 1.73: 0.95, 2.10: 0.975, 2.31: 0.99},
    19: {0.00: 0.5, 0.26: 0.6, 0.54: 0.7, 0.86: 0.8, 1.32: 0.9, 1.72: 0.95, 2.09: 0.975, 2.26: 0.99},
    20: {0.00: 0.5, 0.26: 0.6, 0.54: 0.7, 0.86: 0.8, 1.31: 0.9, 1.71: 0.95, 2.08: 0.975, 2.23: 0.99},
}


def find_closest_table_value(m, computed_F):
    """Finds the closest F(z) value from the table for a given m and computed_F."""
    table_values = t_table[m]  # Getting the F(z) values for the specific m
    closest_z = min(table_values, key=lambda z: abs(table_values[z] - computed_F))  # Find closest z
    closest_F = table_values[closest_z]  # Retrieve the corresponding F(z) value
    return closest_z, closest_F


def main():
    while True:
        try:
            m = int(input("Enter degrees of freedom (m): "))
            z = float(input("Enter z value: "))

            computed_F = compute_F(z, m)
            closest_Fz, table_z = find_closest_table_value(m, computed_F)

            print(f"Computed F(z) for m={m}, z={z}: {computed_F:.5f}")
            if closest_Fz is not None:
                print(f"Closest Table Value: F({closest_Fz}) = {table_z}")
            else:
                print("No matching table value found.")

            repeat = input("Do you want to repeat? (yes/no): ").strip().lower()
            if repeat != 'yes':
                break
        except ValueError:
            print("Invalid input. Please enter numerical values.")


if __name__ == "__main__":
    main()

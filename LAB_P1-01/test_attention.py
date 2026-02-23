import numpy as np
import attention as a


def main():

    # Exemplo numérico simples
    Q = np.array([[1, 0, 1],
                  [0, 1, 0]])

    K = np.array([[1, 0, 1],
                  [0, 1, 0]])

    V = np.array([[1, 2],
                  [3, 4]])

    output = a.scaled_dot_product_attention(Q, K, V)

    print("Saída da Attention:")
    print(output)


if __name__ == "__main__":
    main()
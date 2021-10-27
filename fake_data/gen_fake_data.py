import numpy as np


def gen_fake_data():
    seed = 15
    np.random.seed(seed)
    
    x = np.random.randint(0, 20000, (1, 4096))
    mask = np.ones([1, 4096])

    np.save("x.npy", x)
    np.save("mask.npy", mask)

if __name__ == "__main__":
    gen_fake_data()

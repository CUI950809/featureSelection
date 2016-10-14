from utility.gen_data import gen_data
from utility.gen_data import plot_data

def test_gen_data():
    x1, x2, y1, y2 = gen_data(n_samples=1000)
    plot_data(x1, x2, y1, y2)


if __name__ == "__main__":
    test_gen_data()
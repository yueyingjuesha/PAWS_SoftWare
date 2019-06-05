from load_helper import *
import numpy as np
import os


def count(data):
    return data ** 2


def calculate_sum(data):
    return np.sqrt(data)


class toy_runner():
    def __init__(self, mode, path):
        self.mode = mode
        self.data = load_csv(path)
        self.result = None
        self.run_flag = False

    def run(self):
        if self.mode == 1:
            print("run code 1")
            self.result = count(self.data)
            self.run_flag = True
        elif self.mode == 2:
            print("run code 2")
            self.result = calculate_sum(self.data)
            self.run_flag = True
        else:
            print("illegal mode")

    def check_result(self):
        return self.run_flag

    def get_result(self):
        return self.result

    def save_result(self, dir):
        path = os.path.join(dir, "toy_output.csv")
        print(path)
        save_csv(self.result, path)


if __name__ == "__main__":
    input_path = "../../Data/toy_input.csv"
    output_path = "../../Data/toy_result.csv"

    data = load_csv(input_path)

    runner = toy_runner(3, data)
    print(runner.check_result())
    runner.run()
    print(runner.check_result())
    runner.save_result(output_path)

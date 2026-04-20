import torch
import task3
import task4
import task5

def test(verbose=True):
    task3.test(verbose=verbose)
    task4.test(verbose=verbose)
    task5.test(verbose=verbose)

def main():
    torch.manual_seed(42)
    test(verbose=False)

if __name__ == "__main__":
    main()
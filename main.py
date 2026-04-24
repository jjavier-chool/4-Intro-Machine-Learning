import torch

from task1 import STOCKS, START_DATE, END_DATE
from task3 import BATCH_SIZE, EPOCHS, WEIGHT_DECAY, BAD_RUNS
import task3
import task4
import task5

def main(todo: str='all'):
    torch.manual_seed(42)
    if todo == 'all':
        todo = '3,4,5'
    
    if todo:
        print(STOCKS, ":", START_DATE, "-", END_DATE)
        print(f'B = {BATCH_SIZE} : E = {EPOCHS} (early {BAD_RUNS}) : λ = {WEIGHT_DECAY}')

    try:
        for task in todo.split(','):
            {
                '3': task3,
                '4': task4,
                '5': task5
            }[task].test(verbose=False)
    except KeyError as e:
        task = e.args[0]
        if task in {'1', '2'}:
            print("Task", task, "is not executable")
        else:
            print("No task", task)

if __name__ == "__main__":
    import sys
    main(*sys.argv[1:])
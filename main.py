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
    
    tasks = todo.split(',')
    if '1' in tasks:
        print("Task 1 is not executable")
        return
    if '2' in tasks:
        print("Task 2 is not executable")

    if todo:
        print(STOCKS, ":", START_DATE, "-", END_DATE)
        print(f'B = {BATCH_SIZE} : E = {EPOCHS} (early {BAD_RUNS}) : λ = {WEIGHT_DECAY}')

    try:
        for task in todo.split(','):
            {
                '3': task3, 'rnn': task3,
                '4': task4, 'gru': task4,
                '5': task5, 'lstm': task5
            }[task].test(verbose=False)
    except KeyError as e:
        print("No such task", e.args[0])

if __name__ == "__main__":
    import sys
    main(*sys.argv[1:])
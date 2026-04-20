import torch
import task3
import task4
import task5

def main(todo: str='all'):
    torch.manual_seed(42)
    if todo == 'all':
        todo = '3,4,5'
    
    for task in todo.split(','):
        {
            '3': task3,
            '4': task4,
            '5': task5
        }[task].test(verbose=False)

if __name__ == "__main__":
    import sys
    main(*sys.argv[1:])
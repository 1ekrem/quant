from quant.lib.main_utils import *
from quant.data import alpha


def calculate_alpha():
    alpha.calculate_uk_alpha(latest=True)
    
    
def main():
    calculate_alpha()


if __name__ == '__main__':
    main()

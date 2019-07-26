import os, pickle, shutil, IPython
import numpy as np
import matplotlib.pyplot as plt

def Weighted_Loss(Input, Target):
    loss = Input - Target
    return (3 - (-1)*np.sign(loss)) * 1.0 / 2 * (7 / (1 + 0.01 * Target)) * np.abs(loss)

def main():
    Target = np.arange(2000)
    plt.plot(range(2000), Weighted_Loss(Target + 20, Target), '-', label='predicted - real = 20', color='red')
    plt.plot(range(2000), Weighted_Loss(Target + 10, Target), '-', label='predicted - real = 10', color='pink')

    plt.plot(range(2000), Weighted_Loss(Target - 10, Target), '-', label='predicted - real = -10', color='lightskyblue')
    plt.plot(range(2000), Weighted_Loss(Target - 20, Target), '-', label='predicted - real = -20', color='blue')
    
    
    plt.title('Weighted Loss')
    plt.xlabel('real value')
    plt.ylabel('weighted loss')
    plt.legend()
    plt.savefig('custom loss.png')




if __name__ == "__main__":
    main()

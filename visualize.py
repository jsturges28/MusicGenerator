import os
import pickle
import matplotlib.pyplot as plt
import re

def display_learning_curve_set(dir, base):
    '''
    Plot the learning curves for a set of results
    
    :param base: Directory containing a set of results files
    '''
    # Find the list of files in the local directory that match base_[\d]+.pkl
    files = [f for f in os.listdir(dir) if re.match(r'%s.+.pkl'%(base), f)]
    files.sort()
    
    for f in files:
        with open("%s/%s"%(dir,f), "rb") as fp:
            history = pickle.load(fp)
            plt.plot(history['loss'])
    plt.ylabel('RMSE + KL divergence loss')
    plt.xlabel('epochs')
    plt.legend(files)
    plt.legend(files, fontsize='small')
    plt.savefig('data.png')

if __name__ == "__main__":
    display_learning_curve_set('model', 'history')

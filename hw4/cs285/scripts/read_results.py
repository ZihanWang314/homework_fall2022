import glob
import tensorflow as tf
from matplotlib import pyplot as plt

def get_section_results(file):
    """
        requires tensorflow==1.12.0
    """
    X = []
    Y = []
    for e in tf.compat.v1.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == 'Train_EnvstepsSoFar':
                X.append(v.simple_value)
            elif v.tag == 'Eval_AverageReturn':
                Y.append(v.simple_value)
    return X, Y



def plot(logdir):
    eventfile = glob.glob('data/' + logdir + '/events*')[0]
    X, Y = get_section_results(eventfile)
    return X, Y

if __name__ == '__main__':
    import glob

    X, Y = plot('hw4_q4_reacher_ensemble1_reacher-cs285-v0_02-11-2022_20-20-19')
    plt.plot(X, Y)
    X, Y = plot('hw4_q4_reacher_ensemble3_reacher-cs285-v0_02-11-2022_20-20-40')
    plt.plot(X, Y)
    X, Y = plot('hw4_q4_reacher_ensemble5_reacher-cs285-v0_02-11-2022_20-21-00')
    plt.plot(X, Y)
    plt.legend(['ensemble1', 'ensemble3', 'ensemble5'])
    plt.savefig('q4_ensemble')
    plt.clf()

    X, Y = plot('hw4_q4_reacher_horizon5_reacher-cs285-v0_02-11-2022_20-17-40')
    plt.plot(X, Y)
    X, Y = plot('hw4_q4_reacher_horizon15_reacher-cs285-v0_02-11-2022_20-17-59')
    plt.plot(X, Y)
    X, Y = plot('hw4_q4_reacher_horizon30_reacher-cs285-v0_02-11-2022_20-18-15')
    plt.plot(X, Y)
    plt.legend(['horizon5', 'horizon15', 'horizon30'])
    plt.savefig('q4_horizon')
    plt.clf()
    
    X, Y = plot('hw4_q4_reacher_numseq100_reacher-cs285-v0_02-11-2022_20-18-48')
    plt.plot(X, Y)
    X, Y = plot('hw4_q4_reacher_numseq1000_reacher-cs285-v0_02-11-2022_20-19-11')
    plt.plot(X, Y)
    plt.legend(['numseq100', 'numseq1000'])
    plt.savefig('q4_numseq')
    plt.clf()
    
    X, Y = plot('hw4_q5_cheetah_random_cheetah-cs285-v0_02-11-2022_20-36-41')
    plt.plot(X, Y)
    X, Y = plot('hw4_q5_cheetah_cem_2_cheetah-cs285-v0_03-11-2022_03-45-13')
    plt.plot(X, Y)
    X, Y = plot('hw4_q5_cheetah_cem_4_cheetah-cs285-v0_03-11-2022_03-45-16')
    plt.plot(X, Y)
    plt.legend(['random', 'cem_2', 'cem_4'])
    plt.savefig('q5')
    plt.clf()

    X, Y = plot('hw4_q6_cheetah_rlenl0_cheetah-cs285-v0_02-11-2022_21-52-53')
    plt.plot(X, Y)
    X, Y = plot('hw4_q6_cheetah_rlen1_cheetah-cs285-v0_03-11-2022_05-05-13')
    plt.plot(X, Y)
    X, Y = plot('hw4_q6_cheetah_rlen10_cheetah-cs285-v0_03-11-2022_05-05-15')
    plt.plot(X, Y)
    plt.legend(['SAC', 'Dyna', 'MBPO'])
    plt.savefig('q6')
    plt.clf()
    

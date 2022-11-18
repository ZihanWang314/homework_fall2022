import glob
import os
from matplotlib import pyplot as plt
from tensorflow.python.summary.summary_iterator import summary_iterator
import numpy as np

def get_section_results(file, Xname='Train_EnvstepsSoFar', Yname='Train_AverageReturn'):
    X = []
    Y = []
    for e in summary_iterator(file):
        for v in e.summary.value:
            if v.tag == Xname:
                X.append(v.simple_value)
            elif v.tag == Yname:
                Y.append(v.simple_value)
    return X, Y

def plot_figure(logdirs, Xname='Train_EnvstepsSoFar', Yname='Train_AverageReturn'):
    Xs = []
    Ys = []
    for logdir in logdirs:
        eventfile = [i for i in os.listdir(logdir) if 'events' in i][0]
        eventfile = logdir + '/' + eventfile
        X, Y = get_section_results(eventfile, Xname, Yname)
        Xs.append(X)
        Ys.append(Y)
    Xs = np.array(Xs).mean(0)
    Ys = np.array(Ys).mean(0)
    return Xs, Ys    

if __name__ == '__main__':
    #q1
    plt.xlabel('timestep')
    plt.ylabel('return')
    logdirs_q1 = ['q1_MsPacman-v0_18-10-2022_00-56-52']
    Xs, Ys = plot_figure(logdirs_q1, Yname='Train_AverageReturn')
    Xs = Xs[1:]
    plt.plot(Xs, Ys, c='blue')
    Xs, Ys = plot_figure(logdirs_q1, Yname='Train_BestReturn')
    Xs = Xs[2:]
    plt.plot(Xs, Ys, c='red')
    plt.legend(['AverageReturn', 'BestReturn'])
    plt.savefig('q1')
    plt.clf()


    #q2
    logdirs_q2_dqn = ['q2_dqn_1_LunarLander-v3_17-10-2022_22-12-31', 'q2_dqn_2_LunarLander-v3_17-10-2022_22-12-49', 'q2_dqn_3_LunarLander-v3_17-10-2022_22-13-21']
    logdirs_q2_doubledqn = ['q2_doubledqn_1_LunarLander-v3_17-10-2022_22-16-10', 'q2_doubledqn_2_LunarLander-v3_17-10-2022_22-16-43', 'q2_doubledqn_2_LunarLander-v3_17-10-2022_22-16-58']
    Xs, Ys = plot_figure(logdirs_q2_dqn)
    Xs = Xs[1:]
    plt.plot(Xs, Ys, c='blue', alpha=1)

    Xs, Ys = plot_figure(logdirs_q2_doubledqn)
    Xs = Xs[1:]
    plt.plot(Xs, Ys, c='orange', alpha=1)
    plt.legend(['dqn', 'double_dqn'])

    for file in logdirs_q2_dqn:
        Xs, Ys = plot_figure([file])
        Xs = Xs[1:]
        plt.plot(Xs, Ys, c='blue', alpha=0.3)

    for file in logdirs_q2_doubledqn:
        Xs, Ys = plot_figure([file])
        Xs = Xs[1:]
        plt.plot(Xs, Ys, c='orange', alpha=0.3)

    plt.savefig('q2')
    plt.clf()

    #q3
    logdirs_q3 = ['q3_hparam1_LunarLander-v3_18-10-2022_01-24-55', 'q2_dqn_1_LunarLander-v3_17-10-2022_22-12-31', 'q3_hparam2_LunarLander-v3_18-10-2022_01-25-32', 'q3_hparam3_LunarLander-v3_18-10-2022_01-25-46']
    colors = ['black', 'yellow', 'green', 'blue']
    for idx, file in enumerate(logdirs_q3):
        Xs, Ys = plot_figure([file])
        Xs = Xs[1:]
        plt.plot(Xs, Ys, c=colors[idx])
    plt.legend(['param1:bs=16', 'baseline:bs=32', 'param2:bs=64', 'params3:bs=128'])
    plt.savefig('q3')
    plt.clf()

    #q4
    logdirs_q4 = ['q4_ac_1_1_CartPole-v0_18-10-2022_01-34-24', 'q4_100_1_CartPole-v0_18-10-2022_01-34-30', 'q4_1_100_CartPole-v0_18-10-2022_01-34-50', 'q4_10_10_CartPole-v0_18-10-2022_01-35-05']
    colors = ['black', 'yellow', 'green', 'blue']
    for idx, file in enumerate(logdirs_q4):
        Xs, Ys = plot_figure([file], Yname='Eval_AverageReturn')
        plt.plot(Xs, Ys, c=colors[idx])
    plt.legend(['ntu_1_ngsptu_1', 'ntu_100_ngsptu_1', 'ntu_1_ngsptu_100', 'ntu_10_ngsptu_10'])
    plt.savefig('q4')
    plt.clf()

    #q5
    logdirs_q5_1 = ['q5_10_10_InvertedPendulum-v4_18-10-2022_01-49-22']
    Xs, Ys = plot_figure(logdirs_q5_1)
    plt.plot(Xs, Ys)
    plt.legend(['InvertedPendulum result'])
    plt.savefig('q5-1')
    plt.clf()
    logdirs_q5_2 = ['q5_10_10_HalfCheetah-v4_18-10-2022_01-52-18']
    Xs, Ys = plot_figure(logdirs_q5_2)
    plt.plot(Xs, Ys)
    plt.legend(['HalfCheetah result'])
    plt.savefig('q5-2')
    plt.clf()
    
    #q6
    logdirs_q6_1 = ['q6a_sac_InvertedPendulum_InvertedPendulum-v4_18-10-2022_20-34-16']
    logdirs_q6_2 = ['q6b_sac_HalfCheetah_lr3e-5_HalfCheetah-v4_18-10-2022_21-04-56']
    Xs, Ys = plot_figure(logdirs_q6_1)
    plt.plot(Xs, Ys)
    plt.legend(['InvertedPendulum SAC result'])
    plt.savefig('q6-1')
    plt.clf()
    Xs, Ys = plot_figure(logdirs_q6_2)
    plt.plot(Xs, Ys)
    plt.legend(['HalfCheetah SAC result'])
    plt.savefig('q6-2')
    plt.clf()


    
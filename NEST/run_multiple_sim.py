import pandas as pd
import numpy as np
import os
from simulate_network import simulate_network
from datetime import datetime
now = datetime.now()
dt_string = now.strftime("%Y-%m-%d_%H%M%S")

def run_multiple_sim(results_dir=None, n_trial = 200, mult_coherence = [0.0,0.128,0.512], 
                        sim_parameters = None, sim_col='standard'):
    winner = np.zeros((len(mult_coherence),2))
    for i,coherence in enumerate(mult_coherence):
        print(coherence)
        trials = []
        win = []
        delta_s_A_winner = []
        delta_s_B_winner = []
        for j in range(n_trial): 
            print(j)
            ret_vals, events_A, events_B, events_inh, activity, inputs = simulate_network(coherence,sim_parameters,sim_col)     

            simtime = sim_parameters[sim_col]['simtime']
            int_stimulus_A = np.zeros((int(simtime)))
            int_stimulus_B = np.zeros((int(simtime)))

            for n in range(1,int(simtime)):
                int_stimulus_A[n] = int_stimulus_A[n-1]+inputs['stimulus pop A'][n]
                int_stimulus_B[n] = int_stimulus_B[n-1]+inputs['stimulus pop A'][n]
            
            inputs['integral stim pop A']=int_stimulus_A
            inputs['integral stim pop B']=int_stimulus_B

            if np.mean(activity['activity (Hz) pop_A'].to_numpy()[-100:-1])>np.mean(activity['activity (Hz) pop_B'].to_numpy()[-100:-1]):
                trials.append(j)
                win.append('A')
                delta_s_A_winner.append(int_stimulus_A[-1] - int_stimulus_B[-1])

            if np.mean(activity['activity (Hz) pop_A'].to_numpy()[-100:-1])<=np.mean(activity['activity (Hz) pop_B'].to_numpy()[-100:-1]):
                trials.append(j)
                win.append('B')
                delta_s_B_winner.append(int_stimulus_A[-1] - int_stimulus_B[-1])
            
            saving_dir = results_dir+'/'+sim_col+'/c'+str(coherence)+'/trial_'+ str(j)

            if not os.path.exists(saving_dir):
                os.makedirs(saving_dir)
            

            events_A.to_csv(saving_dir+'/events_pop_A.csv')
            events_B.to_csv(saving_dir+'/events_pop_B.csv')
            events_inh.to_csv(saving_dir+'/events_pop_inh.csv')
            activity.to_csv(saving_dir+'/frequency.csv')
            inputs.to_csv(saving_dir+'/stimuli.csv')

        trial_winner = {'trial':trials, 'winner':win}
        trial_winner=pd.DataFrame(trial_winner, index=trials)
        trial_winner = trial_winner.sort_values(by=['trial'])
        trial_winner.to_csv(results_dir+'/'+sim_col+'/c'+str(coherence)+'/trial_winner.csv')

        delta_s_A_winner = {'delta_s_A_winner':delta_s_A_winner}
        delta_s_B_winner = {'delta_s_B_winner':delta_s_B_winner}
        delta_s_A_winner = pd.DataFrame(delta_s_A_winner)
        delta_s_B_winner = pd.DataFrame(delta_s_B_winner)
        delta_s_A_winner.to_csv(results_dir+'/'+sim_col+'/c'+str(coherence)+'/delta_s_A_winner.csv')
        delta_s_B_winner.to_csv(results_dir+'/'+sim_col+'/c'+str(coherence)+'/delta_s_B_winner.csv')

    return

def main():

    current_path = os.getcwd()+'/results'
    sim_parameters = pd.read_csv('simulation_parameters.csv', index_col=0)

    run_multiple_sim(results_dir=current_path, n_trial = 2, mult_coherence = [0.512], 
                        sim_parameters = sim_parameters, simulation_name=['prova'])

if __name__ == "__main__":
	main()


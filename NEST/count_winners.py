
import os
import pandas as pd

root_path ="C:\\Users\\Alessandra Trapani\\OneDrive - Politecnico di Milano\\"

results_main=['standard\\','stim_end_500.0\\','stim_end_700.0\\','stim_end_900.0\\','stim_reverse\\','t_reverse\\']

for dt_string in results_main:
    dirs_results = os.listdir(root_path+'results\\'+dt_string)
    print(dirs_results)
    for n in dirs_results:
        if os.path.exists(root_path+'results\\'+dt_string+n+'\\'):
            win_pop='A_win'
            path = root_path+'results\\'+dt_string+n+'\\'+win_pop+ '\\'
            trials = []
            win = []
            if os.path.exists(path):
                dirs = os.listdir(path)
                for i in range(len(dirs)):
                    trials.append(int(dirs[i][6:]))
                    win.append('A')

            win_pop='B_win'
            path =  root_path+'results\\'+dt_string+n+'\\'+win_pop+ '\\'
            if os.path.exists(path):
                dirs= os.listdir(path)
                for i in range(len(dirs)):
                    trials.append(int(dirs[i][6:]))
                    win.append('B')
            df = {'trial':trials, 'winner':win}
            tryout=pd.DataFrame(df, index=trials)
            tryout = tryout.sort_values(by=['trial'])
            tryout.to_csv(root_path+'results\\'+dt_string+n +'\\'+'trial_winner.csv')

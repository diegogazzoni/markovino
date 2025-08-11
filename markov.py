import scipy as sp
import numpy as np
      
class MarkovModel(object):
    def __init__(self, tmatrix, cmatrix=None, fmatrix=None, states=None, lag_step=1):
        self.transition_matrix = tmatrix
        self.count_matrix = cmatrix
        self.flux_matrix = fmatrix
        self.ordered_states_list = states
        self.lag_step = lag_step
        if not np.allclose(tmatrix.sum(axis=1), 1):
            raise ValueError('The given transition matrix is not row-stochastic!')
    
    def stationary_distribution(self):   
        _, evecs = self.spectrum()
        p_inf = np.abs(evecs[:, 0])
        p_inf /= np.sum(p_inf)
        return p_inf        

    def spectrum(self):    
        evals, evecs = np.linalg.eig(self.transition_matrix.T) # calcolare right vectors

        i = np.argsort(evals)[::-1]
        evals = evals[i]
        evecs = evecs[:, i]
        return evals, evecs
    
    def timescales(self, n, dt):
        l, v = self.spectrum()
        return -(self.lag_step*dt)/np.log(np.abs(l[1:n]))

def create_from_trajectories(trajectories, states, lag_step) :
    n_skipped = 0
    num_states = len(states)
    count_matrix = np.zeros((num_states, num_states))
    for i, traj in enumerate(trajectories):
        if len(traj) <= lag_step: # skip: too short to contain lag-time transitions
            n_skipped += 1
            print(f"Skipping, too short trajectory ({len(traj)} samples)")
            continue
        if n_skipped == len(trajectories):
            raise ValueError(f"Impossible to produce Markov model with the requested lagtime")
        state_i = traj[:-lag_step]
        state_j = traj[lag_step:]
        np.add.at(count_matrix, (state_i, state_j), 1);
    num_reductions = 0
    disconnected_states = []
    while num_reductions < num_states:
        cols_sum    = np.sum( count_matrix, axis=0 ) # to check if state is 'source'
        rows_sum    = np.sum( count_matrix, axis=1 ) # to check if state is 'sink'
        c_diag      = np.diagonal(count_matrix)
        i_zero = np.where( (rows_sum == 0) | (cols_sum == 0) | (rows_sum == c_diag) | (cols_sum == c_diag))[0] 
        if set(disconnected_states) == set(i_zero):
            break
        disconnected_states = i_zero
        count_matrix[disconnected_states, :] = 0
        count_matrix[:, disconnected_states] = 0
        num_reductions = len(disconnected_states)
    count_matrix   = np.delete(count_matrix, disconnected_states, axis=0)
    count_matrix   = np.delete(count_matrix, disconnected_states, axis=1)
    updated_states = np.delete(states, disconnected_states, axis=0).tolist()
    flux_matrix    = np.delete(flux_matrix, disconnected_states, axis=0)
    flux_matrix    = np.delete(flux_matrix, disconnected_states, axis=1)
    
    trans_matrix = (count_matrix / np.sum(count_matrix, axis=1).reshape((-1,1)))
    if not np.allclose(trans_matrix.sum(axis=1), 1):
        raise ValueError(f'Failed to build a stochastic transition matrix!')
    model = MarkovModel(tmatrix=trans_matrix, cmatrix=count_matrix, fmatrix=flux_matrix, states=updated_states, lag_step=lag_step)
    l, v = model.spectrum();
    
    adj_matrix = sp.sparse.csr_matrix( (count_matrix > 0).astype(int) ) 
    n_components, labels = sp.sparse.csgraph.connected_components(adj_matrix, directed=True, connection='weak')
    if n_components > 1:
        print(f'\033 MSM has {n_components} isolated components!\033[0m')  
    if np.isclose(l[0], l[1]):
        print(f'lambda1={l[0]} ~ lambda2={l[1]}!')

    trajectories_output = trajectories
    if num_reductions > 0: 
        i_new = 0
        new_inds = []
        for j in range(num_states):
            if j not in disconnected_states:
                new_inds.append(i_new)
                i_new+=1
            else:
                new_inds.append(-1)
        updated_traj_list = []
        for old_traj in trajectories:
            updated_traj = []
            for n in range(len(old_traj)):
                t = old_traj[n] 
                sample = new_inds[t];
                if sample >= 0:
                    updated_traj.append(sample) 
            updated_traj_list.append( np.array(updated_traj) )
        trajectories_output = updated_traj_list

    return model, trajectories_output, fluxes_output 

import configparser, os, json
import numpy as np
import pandas as pd
import rpy2.robjects as ro
import gurobipy as gp
from gurobipy import GRB
from tqdm import tqdm

def check(arr):
    formatted_array = [
        f"\033[91m{x}\033[0m" if x > 0 else str(x)  # 91m = red
        for x in arr
    ]
    print("  ".join(formatted_array))

config = configparser.ConfigParser()
config.read('config.txt')
np.random.seed(int(config['data']['seed']))
T = int(config['data']['time_horizon'])
marg_csv = pd.read_csv("final_data/marginal_per_county.csv")

def solve(problem):
    S = problem['number_of_scenarios']
    state_file = f"final_data/{problem['state']}_{S}.csv"
    if os.path.exists(state_file):
        scen_csv = pd.read_csv(state_file)
        counties = sorted(scen_csv["county"].unique())
        N = len(counties)
        L = np.zeros((N, T, S))
        for index, row in scen_csv.iterrows():
            i = counties.index(row['county'])
            L[i, row['T']] = eval(row['loss'])
    else:
        from rpy2.robjects.packages import importr
        from rpy2.robjects import numpy2ri
        from scipy.stats import norm
        numpy2ri.activate()
        tweedie = importr("tweedie")
        corr_csv = pd.read_csv("final_data/correlation_per_state.csv")
        state_corr = corr_csv[corr_csv["state"] == problem['state']]
        counties = eval(state_corr["counties"].values[0])
        N = len(counties)
        correlation = np.array(eval(state_corr["correlation"].values[0]))
        z = np.random.multivariate_normal(np.zeros(N), correlation, size=(S, T))
        z = np.transpose(z, axes=(2, 1, 0))
        u = norm.cdf(z)
        L = np.empty_like(u)
        scenarios = []
        for i in tqdm(range(N), desc="Processing counties"):
            county_marg = marg_csv[marg_csv["County"] == counties[i]]
            p = float(county_marg["p"].values[0])
            mu = float(county_marg["mu"].values[0])
            phi = float(county_marg["phi"].values[0])
            L[i] = tweedie.qtweedie(ro.FloatVector(u[i].flatten()), power=p, mu=mu, phi=phi).reshape(T, S)
            for t in range(T):
                scenarios.append({"county": counties[i], "T": t, "loss": json.dumps(L[i][t].tolist())})
        df = pd.DataFrame(scenarios)
        df.to_csv(state_file, index=False)

    # Compute initial premium and demand function for each county
    policy_csv = pd.read_csv("final_data/policy_per_county.csv")
    state_policy = policy_csv[policy_csv['propertyState'] == problem['state']]
    gamma = problem['rate_capping_parameter']
    cmin = problem['minimum_demand']
    p0 = np.full(N, np.inf)
    pwl_x, pwl_y = np.zeros((N, 4)), np.ones(4)
    pwl_y[2:] = cmin
    for i, county in enumerate(counties):
        county_policy = state_policy[state_policy['countyCode'] == county]
        p0[i] = county_policy['totalPremium'].min()
        pmax = county_policy['totalPremium'].max()
        pwl_x[i, 1] = problem['minimum_premium_factor']*pmax
        pwl_x[i, 2] = (1-cmin)*pmax/problem['demand_rate'] + pwl_x[i, 1]
        pwl_x[i, 3] = np.ceil(p0[i]*(1+gamma)**T)
        
    # SAA
    norm = np.max(p0)**(1/2)
    model = gp.Model('SAA')
    p = model.addVars(N, T, name='p')
    for i in range(N):
        model.setAttr('LB', p[i,0], (1-gamma)*p0[i]/norm)
        model.setAttr('UB', p[i,0], (1+gamma)*p0[i]/norm)
    for i in range(N):
        for t in range(1, T):
            model.addConstr((1-gamma)*p[i, t-1] <= p[i, t])
            model.addConstr(p[i, t] <= (1+gamma)*p[i, t-1])
    y = model.addVars(S, name='y')
    beta = model.addVar(lb=-GRB.INFINITY, name='beta')
    # for s in range(S):
    #     model.addConstr(y[s] >= np.sum(L[:,:,s])/norm
    #                     - problem['permissible_loss_ratio']*gp.quicksum(p[i, t] for i in range(N) for t in range(T))
    #                     - beta)
    # model.addConstr(beta + gp.quicksum(y[s] for s in range(S))/((1-problem['confidence_level'])*S) <= 0)
    # model.setObjective(gp.quicksum(p[i, t] for i in range(N) for t in range(T)), GRB.MINIMIZE)

    d = model.addVars(N, T, name='d')
    for i in range(N):
        print(pwl_x[i]/norm)
        for t in range(T):
            model.addGenConstrPWL(p[i, t], d[i, t], pwl_x[i]/norm, pwl_y)
    pp = model.addVar(name='pp')
    model.addConstr(pp == gp.quicksum(d[i, t] * p[i, t] for i in range(N) for t in range(T)))
    for s in range(S):
        model.addConstr(y[s] >= gp.quicksum(d[i, t] * L[i, t, s] for i in range(N) for t in range(T))/norm 
                        - problem['permissible_loss_ratio']*pp - beta)
    model.addConstr(beta + gp.quicksum(y[s] for s in range(S))/((1-problem['confidence_level'])*S) <= 0)
    model.setObjective(pp, GRB.MINIMIZE)
    
    # model.setParam('OutputFlag', 0)
    model.setParam('MIPGap', 0)
    model.setParam('FeasibilityTol', 1e-9)
    model.setParam('ScaleFlag', 2)
    model.setParam('NumericFocus', 3)
    model.optimize()
    # Z0 = model.ObjVal
    # # RS
    # delta = 0.01
    # tau = Z0*(1+delta)
    # model = gp.Model('RS')
    # p = model.addVars(N, T, name='p')
    # for i in range(N):
    #     model.setAttr('LB', p[i,0], (1-gamma)*p0[i]/norm)
    #     model.setAttr('UB', p[i,0], (1+gamma)*p0[i]/norm)
    # for i in range(N):
    #     for t in range(1, T):
    #         model.addConstr((1-gamma)*p[i, t-1] <= p[i, t])
    #         model.addConstr(p[i, t] <= (1+gamma)*p[i, t-1])
    # d = model.addVars(N, T, name='d')
    # for i in range(N):
    #     for t in range(T):
    #         model.addGenConstrPWL(p[i, t], d[i, t], pwl_x[i]/norm, pwl_y)
    # pp = model.addVar(name='pp')
    # model.addConstr(pp == gp.quicksum(d[i, t] * p[i, t] for i in range(N) for t in range(T)))
    # model.addConstr(pp <= tau)
    # y = model.addVars(S, name='y')
    # beta = model.addVar(lb=float('-inf'), name='beta')
    # model.addConstr(beta + gp.quicksum(y[s] for s in range(S))/((1-problem['confidence_level'])*S) <= 0)
    # eta = model.addVars(N, T, S, name='eta')
    # for s in range(S):
    #     model.addConstr(y[s] >= gp.quicksum(eta[i, t, s]*L[i, t, s] for i in range(N) for t in range(T))/norm
    #                     - problem['permissible_loss_ratio']*pp
    #                     - beta)
    #     for i in range(N):
    #         for t in range(T):
    #             model.addConstr(eta[i, t, s] >= d[i, t])
    # k = model.addVar(name='k')
    # for i in range(N):
    #     for t in range(T):
    #         for s in range(S):
    #             model.addConstr(k >= eta[i, t, s])
    # model.setObjective(k, GRB.MINIMIZE)
    # model.setParam('MIPGap', 0)
    # model.setParam('FeasibilityTol', 1e-9)
    # model.setParam('ScaleFlag', 2)
    # model.setParam('NumericFocus', 3)
    # model.optimize()
    
    if model.status == GRB.OPTIMAL:
        optimal_p = model.getAttr('x', p)
        optimal_p = np.array([[optimal_p[i, t] for t in range(T)] for i in range(N)])*norm
        # print("Z0:", Z0*norm, "Tau:", tau*norm, "Obj:", pp.X*norm)
        delta = np.zeros(T)
        for i in range(N):
            delta[0] = (optimal_p[i, 0] - p0[i]) / p0[i]
            for t in range(1, T):
                delta[t] = (optimal_p[i, t] - optimal_p[i, t-1]) / optimal_p[i, t-1]
            delta = np.round(delta, 4)
            print(p0[i], '|', np.mean(L[i])/p0[i], '|', np.std(L[i]), end=' | ')
            check(delta)
        optimal_y = model.getAttr('x', y)
        optimal_y = np.array([optimal_y[s] for s in range(S)])
        print(optimal_y)
        # eta_ = model.getAttr('x', eta)
        # eta_ = np.array([[[eta_[i, t, s] for s in range(S)] for t in range(T)] for i in range(N)])
        # ids = set()
        # for i in range(N):
        #     for t in range(T):
        #         for s in range(S):
        #             if eta_[i, t, s] >= k.X:
        #                 ids.add(i)
        # print(ids)
        print(model.ObjVal*norm, beta.X)
    else:
        print("No optimal solution found.")
    # print(pp.X*norm, beta.X)

problem = {
    # Input
    'state': 'FL',
    'number_of_scenarios': 30,
    'rate_capping_parameter': 0.1,
    'permissible_loss_ratio': 0.1,
    'confidence_level': 0.99,
    # Hyperparameters
    'minimum_demand': 0.2,
    'minimum_premium_factor': 0.1,
    'demand_rate': 1.0
}
solve(problem)

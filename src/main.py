import configparser, os, json
from scipy.stats import gmean
import numpy as np
import pandas as pd
import rpy2.robjects as ro
import gurobipy as gp
from gurobipy import GRB
from tqdm import tqdm

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
    alpha = problem['confidence_level']
    p0 = np.zeros(N)
    for i, county in enumerate(counties):
        county_policy = state_policy[state_policy['countyCode'] == county]
        p0[i] = county_policy['totalPremium'].min()
    # SAA
    norm = np.quantile(p0, 1.0)**(1/2)
    # norm = 1
    model = gp.Model('SAA')
    p = model.addVars(N, T, name='p')
    pen = model.addVars(N, T, name='pen')
    for i in range(N):
        model.setAttr('LB', p[i, 0], (1-gamma)*p0[i]/norm)
        for t in range(T):
            model.addConstr(pen[i, t] >= p[i, t] - (1+gamma)**(t+1)*p0[i]/norm)
    for i in range(N):
        for t in range(1, T):
            model.addConstr((1-gamma)*p[i, t-1] <= p[i, t])
            model.addConstr(p[i, t] <= (1+gamma)*p[i, t-1])
    mu = np.mean(L, axis=2)
    mu = mu / mu.sum(axis=0)
    lb_pen = model.addVars(N, T, name='lb_pen')
    ub_pen = model.addVars(N, T, name='ub_pen')
    for i in range(N):
        for t in range(T):
            model.addConstr(lb_pen[i, t] >= mu[i, t] * gp.quicksum(p[i, t] for i in range(N)) - p[i, t]) 
            model.addConstr(ub_pen[i, t] >= p[i, t] - mu[i, t] * gp.quicksum(p[i, t] for i in range(N)))
    y = model.addVars(S, name='y')
    beta = model.addVar(lb=-GRB.INFINITY, name='beta')
    for s in range(S):
        model.addConstr(y[s] >= gp.quicksum(L[i, t, s] for i in range(N) for t in range(T))/norm 
                        - problem['permissible_loss_ratio']*gp.quicksum(p[i, t] for i in range(N) for t in range(T)) 
                        - beta)
    model.addConstr(beta + gp.quicksum(y[s] for s in range(S))/((1-alpha)*S) <= 0)
    model.setObjective(gp.quicksum(p[i, t]+pen[i, t]+lb_pen[i, t]+ub_pen[i, t] for i in range(N) for t in range(T)), GRB.MINIMIZE)
    
    model.setParam('OutputFlag', 0)
    model.setParam('MIPGap', 0)
    model.setParam('FeasibilityTol', 1e-9)
    model.setParam('ScaleFlag', 2)
    model.setParam('NumericFocus', 3)
    model.optimize()
    if model.status != GRB.OPTIMAL:
        print("SAA is infeasible")
        return None
    Z0 = model.ObjVal

    tau = Z0*(1+problem['delta'])
    model = gp.Model('RS')
    p = model.addVars(N, T, name='p')
    pen = model.addVars(N, T, name='pen')
    for i in range(N):
        model.setAttr('LB', p[i, 0], (1-gamma)*p0[i]/norm)
        for t in range(T):
            model.addConstr(pen[i, t] >= p[i, t] - (1+gamma)**(t+1)*p0[i]/norm)
    for i in range(N):
        for t in range(1, T):
            model.addConstr((1-gamma)*p[i, t-1] <= p[i, t])
            model.addConstr(p[i, t] <= (1+gamma)*p[i, t-1])
    lb_pen = model.addVars(N, T, name='lb_pen')
    ub_pen = model.addVars(N, T, name='ub_pen')
    for i in range(N):
        for t in range(T):
            model.addConstr(lb_pen[i, t] >= mu[i, t] * gp.quicksum(p[i, t] for i in range(N)) - p[i, t]) 
            model.addConstr(ub_pen[i, t] >= p[i, t] - mu[i, t] * gp.quicksum(p[i, t] for i in range(N)))
    model.addConstr(gp.quicksum(p[i, t]+pen[i, t]+lb_pen[i, t]+ub_pen[i, t] for i in range(N) for t in range(T)) <= tau)
    y = model.addVars(S, name='y')
    beta = model.addVar(lb=-GRB.INFINITY, name='beta')
    model.addConstr(beta + gp.quicksum(y[s] for s in range(S))/((1-alpha)*S) <= 0)
    eta = model.addVars(N, T, S, lb=1.0, name='eta')
    theta = model.addVars(N, T, S, name='theta')
    for s in range(S):
        model.addConstr(y[s] >= gp.quicksum(eta[i, t, s]*L[i, t, s] for i in range(N) for t in range(T))/norm
                        - problem['permissible_loss_ratio']*gp.quicksum(p[i, t] for i in range(N) for t in range(T))
                        - beta)
        model.addConstr(y[s] >= gp.quicksum(theta[i, t, s]*L[i, t, s] for i in range(N) for t in range(T))/norm)
    k = model.addVar(name='k')
    is_minimize = False
    if is_minimize:
        for i in range(N):
            for t in range(T):
                for s in range(S):
                    model.addConstr(k*(1-alpha) >= eta[i, t, s])
                    model.addConstr(k*(1-alpha) >= theta[i, t, s])
        model.setObjective(k, GRB.MINIMIZE)
    else:
        for i in range(N):
            for t in range(T):
                for s in range(S):
                    model.addConstr(k*(1-alpha) <= eta[i, t, s])
                    model.addConstr(k*(1-alpha) <= theta[i, t, s])
        model.setObjective(k, GRB.MAXIMIZE)
    
    model.setParam('MIPGap', 0)
    model.setParam('FeasibilityTol', 1e-9)
    model.setParam('ScaleFlag', 2)
    model.setParam('NumericFocus', 3)
    model.setParam('Presolve', 2)
    model.optimize()
    
    if model.status == GRB.OPTIMAL:
        optimal_p = model.getAttr('x', p)
        optimal_p = np.array([[optimal_p[i, t] for t in range(T)] for i in range(N)])*norm
        optimal_pen = model.getAttr('x', pen)
        optimal_pen = np.array([[optimal_pen[i, t] for t in range(T)] for i in range(N)])*norm
        optimal_lb_pen = model.getAttr('x', lb_pen)
        optimal_lb_pen = np.array([[optimal_lb_pen[i, t] for t in range(T)] for i in range(N)])*norm
        optimal_ub_pen = model.getAttr('x', ub_pen)
        optimal_ub_pen = np.array([[optimal_ub_pen[i, t] for t in range(T)] for i in range(N)])*norm
        print("Z0:", Z0*norm, "Tau:", tau*norm, "Obj:", np.sum(optimal_p), "+", np.sum(optimal_pen), "+", np.sum(optimal_lb_pen), "+", np.sum(optimal_ub_pen), "=", np.sum(optimal_p)+np.sum(optimal_pen)+np.sum(optimal_lb_pen)+np.sum(optimal_ub_pen))
        delta = np.zeros(T)
        for i in range(N):
            delta[0] = (optimal_p[i, 0] - p0[i]) / p0[i]
            for t in range(1, T):
                delta[t] = (optimal_p[i, t] - optimal_p[i, t-1]) / optimal_p[i, t-1]
            delta = np.round(delta, 4)
            print(p0[i], '|', np.mean(L[i])/p0[i], '|', np.std(L[i])/p0[i], end=' | ')
            formatted_array = [
                f"\033[91m{x}\033[0m" if x > 0 else f"{x}"  # 91m = red
                for t, x in enumerate(delta)
            ]
            print("  ".join(formatted_array))
        print(beta.X)
        # for i in range(N):
        #     print(beta[i].X, end=' ')
        # print()
    else:
        print("No optimal solution found.")

problem = {
    # Input
    'state': 'FL',
    'number_of_scenarios': 300,
    'rate_capping_parameter': 0.2,
    'permissible_loss_ratio': 0.3,
    'confidence_level': 0.99,
    'delta': 0.1,
}
solve(problem)

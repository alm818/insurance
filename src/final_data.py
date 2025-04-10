import json, os
import pandas as pd
import numpy as np
import configparser
from scipy.optimize import minimize
from scipy.special import gammaln, digamma
from scipy.stats import norm
from tqdm import tqdm
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
numpy2ri.activate()
tweedie = importr("tweedie")

folder = "final_data"
policy_file = os.path.join(folder, "policy.csv")
claim_file = os.path.join(folder, "claim.csv")
per_county_file = os.path.join(folder, "per_county.csv")
per_state_file = os.path.join(folder, "per_state.csv")

def aggregate_data():
    chunk_size = 100000  # Reads 100,000 rows at a time
    
    # Create an empty DataFrame to store cumulative results
    aggregated_results = pd.DataFrame()
    # Process file in chunks
    for df in tqdm(pd.read_csv("data/FimaNfipPolicies.csv", chunksize=chunk_size, low_memory=False), total=69489458 // chunk_size + 1, desc="Processing Chunks"):
        # Convert policyEffectiveDate to datetime and extract the year
        df['policyEffectiveDate'] = pd.to_datetime(df['policyEffectiveDate'], errors='coerce')
        df['year'] = df['policyEffectiveDate'].dt.year
        df['state'] = df['propertyState'].str.upper()
        df = df.rename(columns={'countyCode': 'county'})
        
        # Group by state, county, and year, then aggregate total premium and policy count
        chunk_aggregated = df.groupby(['state', 'county', 'year']).agg(
            policy_amount=('totalInsurancePremiumOfThePolicy', 'sum'),
            policy_count=('policyCount', 'sum')
        ).reset_index()
    
        # Append chunk results to aggregated_results
        aggregated_results = pd.concat([aggregated_results, chunk_aggregated], ignore_index=True)
    
    # Final aggregation across all chunks
    final_aggregated_data = aggregated_results.groupby(['state', 'county', 'year']).agg(
        policy_amount=('policy_amount', 'sum'),
        policy_count=('policy_count', 'sum')
    ).reset_index()
    
    # Fill missing values with 0
    final_aggregated_data.fillna(0, inplace=True)
    
    # Save the result to CSV
    final_aggregated_data.to_csv(policy_file, index=False)
    
    # Display the first few rows
    print(final_aggregated_data.head())
    print(final_aggregated_data.shape)

    aggregated_results = pd.DataFrame()
    
    # Process file in chunks with progress bar
    for df in tqdm(pd.read_csv("data/FimaNfipClaims.csv", chunksize=chunk_size, low_memory=False), 
                   total=2709121 // chunk_size + 1, desc="Processing Claims Chunks"):
    
        # Convert dateOfLoss to datetime and extract the year
        df['dateOfLoss'] = pd.to_datetime(df['dateOfLoss'], errors='coerce')
        df['year'] = df['dateOfLoss'].dt.year
        df['state'] = df['state'].str.upper()
        df = df.rename(columns={'countyCode': 'county'})

        # Compute total claim payments
        df['claim_amount'] = df[['amountPaidOnBuildingClaim', 
                                    'amountPaidOnContentsClaim', 
                                    'amountPaidOnIncreasedCostOfComplianceClaim']].sum(axis=1, min_count=1)
    
        # Aggregate by state, county, and year
        chunk_aggregated = df.groupby(['state', 'county', 'year']).agg(
            claim_amount=('claim_amount', 'sum'),
            claim_count=('id', 'count')  # Counting unique claim IDs
        ).reset_index()
    
        # Append chunk results to aggregated_results
        aggregated_results = pd.concat([aggregated_results, chunk_aggregated], ignore_index=True)
    
    # Final aggregation across all chunks
    final_aggregated_data = aggregated_results.groupby(['state', 'county', 'year']).agg(
        claim_amount=('claim_amount', 'sum'),
        claim_count=('claim_count', 'sum')
    ).reset_index()
    
    # Fill missing values with 0
    final_aggregated_data.fillna(0, inplace=True)
    
    # Save the result to CSV
    final_aggregated_data.to_csv(claim_file, index=False)
    
    # Display the first few rows
    print(final_aggregated_data.head())
    print(final_aggregated_data.shape)

def mle_estimates(c, l):
    """
    Compute MLE for a compound Poisson-Gamma model with Jacobian.
    
    Parameters:
    c : array-like, length n
        Number of claims per year.
    l : array-like, length n
        Total loss per year (l[i] = 0 if c[i] = 0, l[i] > 0 if c[i] >= 1).
    
    Returns:
    lambda_hat : float
        MLE of Poisson parameter λ.
    alpha_hat : float
        MLE of Gamma shape parameter α.
    theta_hat : float
        MLE of Gamma scale parameter θ.
    """
    n = len(c)
    l = np.maximum(l, 1e-4)
    # MLE for λ: average number of claims per year
    lambda_hat = np.mean(c)
    
    # Indices of years with at least one claim
    I = np.where(c >= 1)[0]
    
    # If no years have claims, cannot estimate α and θ
    if len(I) == 0:
        return lambda_hat, np.nan, np.nan
    
    # Compute sums for years with claims
    li = l[I]
    ci = c[I]
    S = np.sum(li)  # Total loss over years with claims
    C = np.sum(ci)  # Total number of claims over years with claims
    
    # Define negative log-likelihood function for α
    def neg_L2(alpha):
        """Negative log-likelihood for α, given θ(α) = S / (α * C)."""
        if alpha <= 0:
            return np.inf  # Enforce α > 0
        log_likelihood = np.sum(-gammaln(ci * alpha)
                                +ci * alpha * np.log(alpha)      
                                - ci * alpha * np.log(S)          
                                + ci * alpha * np.log(C)           
                                + (ci * alpha - 1) * np.log(li)    
                                - (li * alpha * C / S))           
        return -log_likelihood  # Minimize negative log-likelihood
    
    # Define Jacobian (first derivative) of neg_L2 with respect to α
    def jacobian(alpha):
        """Jacobian of neg_L2 with respect to α."""
        if alpha <= 0:
            return np.inf  # Enforce α > 0
        jac = np.sum(ci * (
                digamma(ci * alpha)         
                - np.log(alpha)                 
                - 1                           
                + np.log(S)                    
                - np.log(C)                 
                - np.log(li)                   
            ))
        # Add the term that does not depend on α
        jac += np.sum(li * C / S)
        return jac
    
    # Optimize to find α_hat using Jacobian
    result = minimize(
        neg_L2,
        x0=1.0,                    # Initial guess for α
        method='L-BFGS-B',         # Bounded optimization method
        bounds=[(1e-6, None)],     # α > 0
        jac=jacobian               # Provide Jacobian
    )
    
    # Check optimization success
    if result.success:
        alpha_hat = result.x[0]
        theta_hat = S / (alpha_hat * C)  # θ = S / (α * C)
    else:
        alpha_hat = np.nan
        theta_hat = np.nan
    
    return lambda_hat, alpha_hat, theta_hat

def per_county_data():
    df = pd.read_csv(claim_file)
    config = configparser.ConfigParser()
    config.read('config.txt')
    years = list(range(int(config['data']['start_year']), int(config['data']['end_year'])))
    T = int(config['data']['time_horizon'])
    county_data = {}
    
    for (state, county), group in df.groupby(["state", "county"]):
        claim_counts = {year: 0 for year in years}
        claim_amounts = {year: 0 for year in years}
        
        for _, row in group.iterrows():
            year = row["year"]
            if year in claim_counts:
                claim_counts[year] = row["claim_count"]
                claim_amounts[year] = row["claim_amount"]
        claim_counts = np.array([claim_counts[year] for year in years])
        claim_amounts = np.array([claim_amounts[year] for year in years])
        lambda_hat, alpha_hat, theta_hat = mle_estimates(claim_counts[:-T], claim_amounts[:-T])
        if lambda_hat == 0:
            p, mu, phi = np.nan, 0, 0
        else:
            p, mu = (2+alpha_hat)/(1+alpha_hat), lambda_hat*alpha_hat*theta_hat
            phi = (1+alpha_hat) * lambda_hat**(1-p) * alpha_hat**(1-p) * theta_hat**(2-p)
        county_data[(state, county)] = {
            "train_claim_count": json.dumps(claim_counts[:-T].tolist()),
            "train_claim_amount": json.dumps(claim_amounts[:-T].tolist()),
            "test_claim_amount": json.dumps(claim_amounts[-T:].tolist()),
            "p": p,
            "mu": mu,
            "phi": phi
        }
    
    df = pd.DataFrame.from_dict(county_data, orient="index")
    df.index = pd.MultiIndex.from_tuples(df.index, names=["state", "county"])
    df.reset_index(inplace=True)
    df.to_csv(per_county_file, index=False)
    
    print(df.head())
    print(df.shape)

def per_state_data():
    eps = 1e-6
    df = pd.read_csv(per_county_file)
    df['train_claim_amount'] = df['train_claim_amount'].apply(lambda x: np.array(eval(x)) if isinstance(x, str) else np.array(x))
    df = df[df['train_claim_amount'].apply(np.var) > 0]
    state_data = {}
    for state, group in df.groupby("state"):
        group_sorted = group.sort_values(by="county")
        claim_amount_matrix = np.array([cl for cl in group_sorted["train_claim_amount"]])
        tweedie_matrix = group_sorted[["p", "mu", "phi"]].to_numpy()
        cdf_matrix = np.empty_like(claim_amount_matrix)
        for i, tw in enumerate(tweedie_matrix):
            cdf_matrix[i] = tweedie.ptweedie(ro.FloatVector(claim_amount_matrix[i]), power=tw[0], mu=tw[1], phi=tw[2])
        cdf_matrix = np.clip(cdf_matrix, eps, 1 - eps)
        z = norm.ppf(cdf_matrix)
        corr_matrix = np.corrcoef(z)
        state_data[state] = {
            "state": state,
            "counties": json.dumps(list(group_sorted['county'])),
            "correlation": json.dumps(corr_matrix.tolist())
        }
    df = pd.DataFrame.from_dict(state_data, orient="index")
    df.to_csv(per_state_file, index=False)
    print(df.head())
    print(df.shape)
    
if __name__ == "__main__":
    aggregate_data()
    per_county_data()
    per_state_data()
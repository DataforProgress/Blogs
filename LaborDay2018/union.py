import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from scipy.special import logit,expit,gamma
sns.set(color_codes=True)
import pymc3 as pm
import us
from theano import tensor as tt
from scipy.optimize import fsolve

#Data sources
#http://election.princeton.edu/
#http://unionstats.gsu.edu/MonthlyLaborReviewArticle.html
#https://www.chartbookofeconomicinequality.com/inequality-by-country/usa/

def joinFig(name):
    return os.path.join("Figures",name)

def standardize(a):
    return (a - np.mean(a))/np.std(a)

def hierarchical_normal(name, shape, mu=0.,cs=5.,sigma=None):
    delta = pm.Normal('delta_{}'.format(name), 0., 1., shape=shape)
    if sigma is None:
        sigma = pm.HalfCauchy('sigma_{}'.format(name), cs)
    
    return pm.Deterministic(name, mu + delta * sigma)

#Election data compiled by the Princeton Election Consortium http://election.princeton.edu/
df = pd.read_csv("Elections.csv")
df["RepVotes"] = df["RepVotes"].str.replace(",","")
df["DemVotes"] = df["DemVotes"].str.replace(",","")
df["RepVotes"] = df["RepVotes"].str.replace("Unopposed","0")
df["DemVotes"] = df["DemVotes"].str.replace("Unopposed","0")
df["RepVotes"] = df["RepVotes"].astype(np.int64)
df["DemVotes"] = df["DemVotes"].astype(np.int64)
df = df.rename(columns={"raceYear":"Year"})
df = df.groupby(["State","Year"]).apply(lambda x: x[["DemVotes","RepVotes"]].sum(axis=0)).reset_index()
df["TotVotes"] = df["DemVotes"]+df["RepVotes"]
df["DemShare"] = 100.*df["DemVotes"]/df["TotVotes"]
states = np.unique(df["State"])

#Union Membership data from Hirsh et al http://unionstats.gsu.edu/MonthlyLaborReviewArticle.html
dfu = pd.read_excel("State_Union_Membership_Density_1964-2017.xlsx")
rn = {"%Mem"+str(x)[-2:]:x for x in range(1964,2017)}
dfa = dfu[dfu["State Name"]=="All States"]
dfu = dfu[dfu["State Name"].isin(states)]
cols = [x for x in range(1964,2017)]
dfu = dfu.rename(columns=rn)
dfa = dfa.rename(columns=rn)
dfu = pd.melt(dfu,id_vars=["State Name"],value_vars=cols,var_name="Year",value_name="UnionMembership")
dfu = dfu.rename(columns={"State Name":"State"})

dfa = pd.melt(dfa,id_vars=["State Name"],value_vars=cols,var_name="Year",value_name="UnionMembership")
df = pd.merge(df,dfu,on=["State","Year"])

#Plot pooled data 
ax = sns.regplot(x="UnionMembership",y="DemShare",data=df)
plt.savefig("pooled.png")
plt.close()

#Plot unpooled data
g = sns.FacetGrid(df,col="Year",col_wrap=5)
df["% Union Membership"]    = df["UnionMembership"]
df["% Democrat Vote Share"] = df["DemShare"]
g = g.map(sns.regplot,"% Union Membership","% Democrat Vote Share")
for i,ax in enumerate(g.axes.flat):
    ax.set_ylim((0.,100.))
plt.savefig("time.png")
plt.close()

#Inequality metrics from https://www.chartbookofeconomicinequality.com/inequality-by-country/usa/
dfl = pd.read_excel("AllData_ChartbookOfEconomicInequality.xlsx",sheet_name="US",skiprows=4,skipfooter=12)
dfl = dfl.rename(columns={"Unnamed: 0":"Year"})
dfl = dfl[dfl.Year>=1964]

#Plot inequality metrics and nationwide union membership
fig,ax = plt.subplots(1,1)
ax.plot(dfl["Year"],dfl["Share of top 1 per cent in total net wealth (individuals) (*)"],label="Share of total net wealth held by the top 1%")
ax.plot(dfl["Year"],dfl["Share of top 1 per cent in gross income (tax units, excluding capital gains) (*)"],label="Share of gross income paid to the top 1%")
ax.plot(dfa["Year"],dfa["UnionMembership"],label="Union Membership")
ax.set_xlim((1964,2016))
ax.legend(loc=2)
ax.set_ylabel("Percent")
plt.savefig("Inequality.png")
plt.close()

stateDict = {name:i for i,name in enumerate(states)}
df["Year"] = df["Year"].astype(np.int64)
uyears = np.unique(df["Year"])
uyearsStd = standardize(uyears)
yearDict = {name:i for i,name in enumerate(uyears)}
df["state"]  = df["State"].apply(lambda x: stateDict[x])
df["year"]   = df["Year"].apply(lambda x: yearDict[x])
stdU = np.std(df["UnionMembership"])
stdD = np.std(df["DemShare"])
df["UnionMembership"] = standardize(df["UnionMembership"])
df["DemShare"]        = standardize(df["DemShare"])
df["Year"]            = standardize(df["Year"])

state     = df["state"].values
union     = df["UnionMembership"].values
demShare  = df["DemShare"].values
year      = df["year"].values

NUTS_KWARGS = {'target_accept': 0.99,'max_treedepth':30}
SEED = 4260026 # from random.org, for reproducibility

np.random.seed(SEED)
ndraws = 1000
ntune = int(ndraws)
nyear = 23

lower = np.min(np.diff(uyearsStd))*2.
drange = np.max(uyearsStd)-np.min(uyearsStd)
upper = drange/2.

def getResid(x):
    a = x[0]
    b = x[1]
    c1 = b/(a-1)
    c2 = 3.*b/np.sqrt((a-1.)**2*(a-2.))
    rL = lower-c1+c2
    rU = upper-c1-c2
    return [rL,rU]

#Estimate inverse gamma paramters for the gp length scale
x0 = [5.,5.]
alpha,beta = fsolve(getResid,x0)
x = np.linspace(0.,drange)
def IGammaPDF(x,alpha,beta):
    return beta**alpha/gamma(alpha)*x**(-alpha-1.)*np.exp(-beta/x)

plt.figure()
plt.plot(x,IGammaPDF(x,alpha,beta))
plt.plot(x,IGammaPDF(x,5.,5.))
plt.savefig("Gammas.png")
plt.close()

with pm.Model() as hierarchical_model:
    #Simple linear regression for intercepts
    a        = pm.Normal("a",mu=0.,sd=10.)
    ay       = pm.Normal("ay",mu=0.,sd=10.)
    mu_a_year = a+ay*uyearsStd

    #Use gaussian process prior for the slopes
    rho = pm.InverseGamma("rho_b", alpha=alpha, beta=beta)
    alpha = pm.HalfNormal("alpha_b", sd=1)
    cov = alpha**2 * pm.gp.cov.ExpQuad(1, rho)
    gp = pm.gp.Latent(cov_func=cov)
    mu_b_year = gp.prior("f_b", X=uyearsStd[:,None])

    a_year = hierarchical_normal("a_year",nyear,mu=mu_a_year)
    b_year = hierarchical_normal("b_year",nyear,mu=mu_b_year)

    mu_DemShare = a_year[year] + b_year[year]*union
    sd_glob  = pm.HalfCauchy("sdGlob",1)
    eps         = pm.HalfCauchy("eps",sd_glob,shape=nyear)
    likelihood = pm.Normal("DemShare",mu=mu_DemShare,sd=eps[year],observed=demShare)
    hierarchical_trace = pm.sample(draws=ndraws,tune=ntune,init="adapt_diag",chains=3, random_seed=SEED,nuts_kwargs=NUTS_KWARGS)

pm.traceplot(hierarchical_trace)
plt.savefig("hierarchical_trace.png")
plt.close()

fig,ax = plt.subplots(1,1)
means = np.mean(hierarchical_trace["b_year"],axis=0)*stdD/stdU
pL = np.percentile(hierarchical_trace["b_year"],2.5,axis=0)*stdD/stdU
pU = np.percentile(hierarchical_trace["b_year"],97.5,axis=0)*stdD/stdU
ax.plot(uyears,means)
ax.fill_between(uyears,pL,pU,alpha=0.25)
ax.set_xlim((1972,2016))
ax.set_ylabel("% Change in Dem Share Associated with\n a 1% Increase in Union Membership")
ax.set_title("State Union Membership has Become More Strongly Associated\n with Democrat's Statewide Vote Share in House Elections")
plt.tight_layout()
plt.savefig("slopes.png")
plt.close()

with pm.Model() as unpooled_model:
    a_year = pm.Normal("a_year",shape=nyear,mu=0.,sd=50.)
    b_year = pm.Normal("b_year",shape=nyear,mu=0.,sd=50.)

    mu_DemShare = a_year[year] + b_year[year]*union
    eps         = pm.HalfCauchy("eps",5,shape=nyear)
    likelihood = pm.Normal("DemShare",mu=mu_DemShare,sd=eps[year],observed=demShare)
    unpooled_trace = pm.sample(draws=ndraws,tune=ntune,init="adapt_diag",chains=3, random_seed=SEED,nuts_kwargs=NUTS_KWARGS)

pm.traceplot(unpooled_trace)
plt.savefig("unpooled_trace.png")
plt.close()

fig,ax = plt.subplots(1,1)

ax.plot(uyears,means,color='b',label="Hierarchical")
ax.fill_between(uyears,pL,pU,alpha=0.25,color='b')

means = np.mean(unpooled_trace["b_year"],axis=0)*stdD/stdU
pL = np.percentile(unpooled_trace["b_year"],2.5,axis=0)*stdD/stdU
pU = np.percentile(unpooled_trace["b_year"],97.5,axis=0)*stdD/stdU
ax.plot(uyears,means,color='g',label="Unpooled")
ax.fill_between(uyears,pL,pU,alpha=0.25,color='g')
ax.set_xlim((1972,2016))
ax.legend(loc=2)
ax.set_ylabel("% Change in Dem Share Associated with\n a 1% Increase in Union Membership")
ax.set_title("State Union Membership has Become More Strongly Associated\n with Democrat's Statewide Vote Share in House Elections")
plt.tight_layout()
plt.savefig("slopesUnpooled.png")
plt.close()

with pm.Model() as pooled_model:
    a = pm.Normal("a",mu=0.,sd=50.)
    b = pm.Normal("b",mu=0.,sd=50.)

    mu_DemShare = a + b*union
    eps         = pm.HalfCauchy("eps",5)
    likelihood = pm.Normal("DemShare",mu=mu_DemShare,sd=eps,observed=demShare)
    pooled_trace = pm.sample(draws=ndraws,tune=ntune,init="adapt_diag",chains=3, random_seed=SEED,nuts_kwargs=NUTS_KWARGS)

pm.traceplot(pooled_trace)
plt.savefig("pooled_trace.png")
plt.close()

pooled_model.name="Pooled"
unpooled_model.name="Unpooled"
hierarchical_model.name="Hierarchical"

dfComp = pm.compare({hierarchical_model: hierarchical_trace, pooled_model: pooled_trace, unpooled_model: unpooled_trace},ic="LOO")
print(dfComp)
pm.compareplot(dfComp)
plt.tight_layout()
plt.savefig("compare.png")
plt.close()

g = sns.FacetGrid(df,col="Year",col_wrap=5)
g = g.map(plt.scatter,"UnionMembership","DemShare")
x = np.linspace(-2,2,100)
for i,ax in enumerate(g.axes.flat):
    p_state = hierarchical_trace["a_year"][:,i] + hierarchical_trace["b_year"][:,i]*x[:,None]
    p_mean  = np.mean(p_state,axis=1)
    ax.plot(x,p_mean,color='r')
    p_state = unpooled_trace["a_year"][:,i] + unpooled_trace["b_year"][:,i]*x[:,None]
    p_mean  = np.mean(p_state,axis=1)
    ax.plot(x,p_mean,color='g')

plt.savefig("reg.png")
plt.close()

fig,axShrink = plt.subplots(1,1,figsize=(8,8))
for i in range(nyear):
    hint,hslp = np.mean(hierarchical_trace["a_year"][:,i]),np.mean(hierarchical_trace["b_year"][:,i])
    uint,uslp = np.mean(unpooled_trace["a_year"][:,i]),np.mean(unpooled_trace["b_year"][:,i])

    axShrink.plot([uint,hint],[uslp,hslp],color='b')
    axShrink.scatter([uint],[uslp],color='r')
    axShrink.scatter([hint],[hslp],color='b')


pint,pslp = np.mean(pooled_trace["a"]),np.mean(pooled_trace["b"])
axShrink.scatter([pint],[pslp],color='g')
fig.savefig("shrink.png")

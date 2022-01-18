import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import numpy as np

stocks = pd.read_csv(r'new_stocks.csv')

stocks = stocks.rename(columns={'Unnamed: 0': 'Date'})
stocks.set_index('Date')
stocks = stocks.dropna(subset=['Unibail-Rodamco',
                               'LOréal',
                               'Worldline SA',
                               'EssilorLuxottica',
                               'Bouygues',
                               'LEGRAND',
                               'SAFRAN',
                               'Dassault Systèmes',
                               'Sodexo',
                               'Publicis',
                               'Schneider Electric',
                               'Hermès (Hermes International',
                               'Accor',
                               'LVMH Moet Hennessy Louis Vuitton',
                               'Air Liquide'])
stocks = stocks.set_index('Date')
for s in stocks:
    stocks[s] = np.log(stocks[s] / stocks[s].shift(1, axis=0))

np.random.seed(42)
num_ports = 6000
all_weights = np.zeros((num_ports, len(stocks.columns)))
ret_arr = np.zeros(num_ports)
vol_arr = np.zeros(num_ports)
sharpe_arr = np.zeros(num_ports)

for x in range(num_ports):
    weights = np.array(np.random.random(15))
    weights = weights / np.sum(weights)

    all_weights[x, :] = weights

    ret_arr[x] = np.sum((stocks.mean() * weights * 252))

    vol_arr[x] = np.sqrt(np.dot(weights.T, np.dot(stocks.cov() * 252, weights)))

    # Sharpe Ratio
    sharpe_arr[x] = ret_arr[x] / vol_arr[x]

# best portfolio index and sharpe ratio
print('Max Sharpe Ratio: {}'.format(sharpe_arr.max()))
print('Its location in the array: {}'.format(sharpe_arr.argmax()))

max_sr_ret = ret_arr[sharpe_arr.argmax()]
max_sr_vol = vol_arr[sharpe_arr.argmax()]

print('Le meilleur portefeuille se compose de:')
print(all_weights[1212, 0] * 100, '% Unibail-Rodamco')
print(all_weights[1212, 1] * 100, '% LOréal')
print(all_weights[1212, 2] * 100, '% Worldline SA')
print(all_weights[1212, 3] * 100, '% EssilorLuxottica')
print(all_weights[1212, 4] * 100, '% Bouygues')
print(all_weights[1212, 5] * 100, '% LEGRAND')
print(all_weights[1212, 6] * 100, '% SAFRAN')
print(all_weights[1212, 7] * 100, '% Dassault Systèmes')
print(all_weights[1212, 8] * 100, '% Sodexo')
print(all_weights[1212, 9] * 100, '% Publicis')
print(all_weights[1212, 10] * 100, '% Schneider Electric')
print(all_weights[1212, 11] * 100, '% Hermès (Hermes International')
print(all_weights[1212, 12] * 100, '% Accor')
print(all_weights[1212, 13] * 100, '% LVMH')
print(all_weights[1212, 14] * 100, '% Air Liquide')

plt.figure(figsize=(12, 8))
plt.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='viridis')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.scatter(max_sr_vol, max_sr_ret, c='red', s=50)  # red dot
plt.show()


def get_ret_vol_sr(weights):
    weights = np.array(weights)
    ret = np.sum(stocks.mean() * weights) * 252
    vol = np.sqrt(np.dot(weights.T, np.dot(stocks.cov() * 252, weights)))
    sr = ret / vol
    return np.array([ret, vol, sr])


def neg_sharpe(weights):
    # the number 2 is the sharpe ratio index from the get_ret_vol_sr
    return get_ret_vol_sr(weights)[2] * -1


def check_sum(weights):
    # return 0 if sum of the weights is 1
    return np.sum(weights) - 1


cons = ({'type': 'eq', 'fun': check_sum})
bounds = (
    (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1),
    (0, 1))
init_guess = [1 / 15, 1 / 15, 1 / 15, 1 / 15, 1 / 15, 1 / 15, 1 / 15, 1 / 15, 1 / 15, 1 / 15, 1 / 15, 1 / 15, 1 / 15,
              1 / 15, 1 / 15]

opt_results = minimize(neg_sharpe, init_guess, bounds=bounds, constraints=cons)
print(opt_results)

print('Après optimisation, le meilleur portefeuille se compose de:')
print(opt_results.x[0] * 100, '% Unibail-Rodamco')
print(opt_results.x[1] * 100, '% LOréal')
print(opt_results.x[2] * 100, '% Worldline SA')
print(opt_results.x[3] * 100, '% EssilorLuxottica')
print(opt_results.x[4] * 100, '% Bouygues')
print(opt_results.x[5] * 100, '% LEGRAND')
print(opt_results.x[6] * 100, '% SAFRAN')
print(opt_results.x[7] * 100, '% Dassault Systèmes')
print(opt_results.x[8] * 100, '% Sodexo')
print(opt_results.x[9] * 100, '% Publicis')
print(opt_results.x[10] * 100, '% Schneider Electric')
print(opt_results.x[11] * 100, '% Hermès (Hermes International')
print(opt_results.x[12] * 100, '% Accor')
print(opt_results.x[13] * 100, '% LVMH')
print(opt_results.x[14] * 100, '% Air Liquide')

get_ret_vol_sr(opt_results.x)

frontier_y = np.linspace(0, 0.06, 1000)


def minimize_volatility(weights):
    return get_ret_vol_sr(weights)[1]


frontier_x = []

for possible_return in frontier_y:
    cons = ({'type': 'eq', 'fun': check_sum},
            {'type': 'eq', 'fun': lambda w: get_ret_vol_sr(w)[0] - possible_return})

    result = minimize(minimize_volatility, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
    frontier_x.append(result['fun'])

plt.figure(figsize=(12, 8))
plt.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='viridis')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.plot(frontier_x, frontier_y, 'r--', linewidth=3)
plt.show()

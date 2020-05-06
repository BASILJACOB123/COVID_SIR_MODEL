# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 13:35:38 2020

@author: DELL
"""

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
import math

S_0 = 15000
I_0 = 14
R_0 = 0
df_new=pd.DataFrame()
#Date when confirmed cases were positive

START_DATE = {
  'Beijing': '1/22/20',
  'Heilongjiang': '1/23/20',
  'Qinghai': '1/25/20'
}
province='Beijing'
def loss(point, data, recovered):
    size = len(data)
    beta, gamma = point
    def SIR(t, y):
        S = y[0]
        I = y[1]
        R = y[2]
        return [-beta*S*I, beta*S*I -gamma*I, gamma*I]
    solution = solve_ivp(SIR, [0, size], [S_0,I_0,R_0], t_eval=np.arange(0, size, 1), vectorized=True)
    l1 = np.sqrt(np.mean((solution.y[1] - data)**2))
    l2 = np.sqrt(np.mean((solution.y[2] - recovered)**2))
    alpha = 0.1
    return alpha * l1 + (1 - alpha) * l2

class Learner:
    def __init__(self, province, loss):
        self.province = province
        self.loss = loss

    def load_confirmed(self, province):
      df = pd.read_csv('time_series_covid_19_confirmed.csv')
      province_df = df[df['Province/State'] == province]
      return province_df.iloc[0].loc[START_DATE[province]:]

    def load_recovered(self, province):
      df = pd.read_csv('time_series_covid_19_recovered.csv')
      province_df = df[df['Province/State'] == province]
      return province_df.iloc[0].loc[START_DATE[province]:]

    def extend_index(self, index, new_size):
        values = index.values
        current = datetime.strptime(index[-1], '%m/%d/%y')
        while len(values) < new_size:
            current = current + timedelta(days=1) #calculatng the current date for a span of 150 days from the larget date in dataset
            values = np.append(values, datetime.strftime(current, '%m/%d/%y'))
        return values

    def predict(self, beta, gamma, data, recovered, province):
        predict_range = 150 #over as span of so many days
        new_index = self.extend_index(data.index, predict_range)
        size = len(new_index)
        def SIR(t, y):
            S = y[0]
            I = y[1]
            R = y[2]
            return [-beta*S*I, beta*S*I -gamma*I, gamma*I]
        extended_actual = np.concatenate((data.values, [None] * (size - len(data.values))))
        extended_recovered = np.concatenate((recovered.values, [None] * (size - len(recovered.values))))
        return new_index, extended_actual, extended_recovered, solve_ivp(SIR, [0, size], [S_0,I_0,R_0], t_eval=np.arange(0, size, 1))

    def train(self):
        data = self.load_confirmed(self.province)
        recovered = self.load_recovered(self.province)
        optimal = minimize(loss, [0.001, 0.001], args=(data, recovered), method='L-BFGS-B', bounds=[(0.00000001, 0.4), (0.00000001, 0.4)])
        print(optimal)
        beta, gamma = optimal.x
        new_index, extended_actual, extended_recovered, prediction = self.predict(beta, gamma, data, recovered, self.province)
        df = pd.DataFrame({'Confirmed': extended_actual, 'Recovered': extended_recovered, 'S': prediction.y[0], 'I': prediction.y[1], 'R': prediction.y[2]}, index=new_index)
        df_new=pd.DataFrame({'Confirmed': extended_actual, 'Recovered': extended_recovered, 'S': prediction.y[0], 'I': prediction.y[1], 'R': prediction.y[2]}, index=new_index)
        
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.set_title(self.province)
        df.plot(ax=ax)
        print(f"country={self.province}, beta={beta:.8f}, gamma={gamma:.8f}, r_0:{(beta/gamma):.8f}")
        fig.savefig(f"{self.province}.png")



learner = Learner('Beijing', loss)
learner.train()

learner = Learner('Heilongjiang', loss)
learner.train()

learner = Learner('Qinghai', loss)
learner.train()
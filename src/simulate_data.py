import pandas as pd 
import numpy as np
np.random.seed(42) 

# to ensure reproducibility of experiments
n_timesteps = 10000 
failure_events = [2500,5000,7500,9500]
failure_window = 200
time = np.arange(n_timesteps)
temperature = 70 + np.random.normal(0,2,n_timesteps)
np.random.normal(0,2,n_timesteps)

vibration = 20 + np.random.normal(0,1,n_timesteps)
pressure = 100 + np.random.normal(0,5,n_timesteps)

failure = np.zeros(n_timesteps)
failure_warning = np.zeros(n_timesteps)
for failure_time in failure_events:
    start = failure_time - failure_window
    end = failure_time
    failure_warning[start:end] = 1
failure[failure_time] = 1

trend = np.linspace(0,1,failure_window)
temperature[start:end] += trend * 25
vibration[start:end] += trend * 15
pressure[start:end] += trend * 30

df = pd.DataFrame({
    "time": time,
    "temperature": temperature,
    "vibration": vibration,
    "pressure": pressure,
    "failure_warning": failure_warning,
    "failure": failure
})
df.to_csv("data/sensor_data.csv", index=False)
df.to_parquet("data/sensor_data.parquet", index=False)
print(df.head())
print()
print(df["failure_warning"].value_counts())
print()
print("Dataset saved to data/sensor_data.csv and data/sensor_data.parquet")
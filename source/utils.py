import os
import math
import scipy.stats as st

def store_preprocessed_data(df_train, df_test, df_val, data_dir):
    print(data_dir)
    os.system(f"mkdir {data_dir}")
    if not os.path.exists(data_dir):
    # If it doesn't exist, create the directory
        os.makedirs(data_dir)

    path_to_train_file = os.path.join(data_dir,"train_preprocessed.csv")
    df_train_without_end_activity = df_train.copy()
    df_train_without_end_activity = df_train_without_end_activity[df_train_without_end_activity['activity_name'] != 'zzz_end']
    df_train_without_end_activity.to_csv(path_to_train_file, index=False)

    # save test data
    path_to_test_file = os.path.join(data_dir,"test_preprocessed.csv")
    df_test_without_end_activity = df_test.copy()
    df_test_without_end_activity = df_test_without_end_activity[df_test_without_end_activity['activity_name'] != 'zzz_end']
    df_test_without_end_activity.to_csv(path_to_test_file, index=False)

    return df_train_without_end_activity


def store_simulated_log(data_dir, simulated_log, index):
    # # remove agent column
    # simulated_log = simulated_log.drop(columns=['agent', 'TimeStep'])
    # # rename activity column to activity_name
    # simulated_log = simulated_log.rename(columns={'activity_name': 'activity'})
    # # rename start_timestamp to Start_Time and end_timestamp to End_Time
    # simulated_log = simulated_log.rename(columns={'start_timestamp': 'Start_Time', 'end_timestamp': 'End_Time'})
    path_to_file = os.path.join(data_dir,f"simulated_log_{index}.csv")
    simulated_log.to_csv(path_to_file, index=False)
    print(f"Simulated logs are stored in {path_to_file}")


def sample_from_distribution(distribution):
    if distribution.type.value == "expon":
        scale = distribution.mean - distribution.min
        if scale < 0.0:
            print("Warning! Trying to generate EXPON sample with 'mean' < 'min', using 'mean' as scale value.")
            scale = distribution.mean
        sample = st.expon.rvs(loc=distribution.min, scale=scale, size=1)
    elif distribution.type.value == "gamma":
        # If the distribution corresponds to a 'gamma' with loc!=0, the estimation is done wrong
        # dunno how to take that into account
        sample = st.gamma.rvs(
            pow(distribution.mean, 2) / distribution.var,
            loc=0,
            scale=distribution.var / distribution.mean,
            size=1,
        )
    elif distribution.type.value == "norm":
        sample = st.norm.rvs(loc=distribution.mean, scale=distribution.std, size=1)
    elif distribution.type.value == "uniform":
        sample = st.uniform.rvs(loc=distribution.min, scale=distribution.max - distribution.min, size=1)
    elif distribution.type.value == "lognorm":
        # If the distribution corresponds to a 'lognorm' with loc!=0, the estimation is done wrong
        # dunno how to take that into account
        pow_mean = pow(distribution.mean, 2)
        phi = math.sqrt(distribution.var + pow_mean)
        mu = math.log(pow_mean / phi)
        sigma = math.sqrt(math.log(phi ** 2 / pow_mean))
        sample = st.lognorm.rvs(sigma, loc=0, scale=math.exp(mu), size=1)
    elif distribution.type.value == "fix":
        sample = [distribution.mean] * 1

    return sample[0]
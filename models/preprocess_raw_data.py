from tqdm import *
import pickle

def excute(file_name):
    name = '../data/old_data/raw_data/' + file_name
    with open(name, 'rb') as f:
        df = pickle.load(f)
    def raw_to_list(df):
        import pickle
        final_data = {}
        for i in trange(len(df)):
            execution_times = df[i][1]
            # Calculating the average per run
            run_averages = [sum(run) / len(run) for run in execution_times]
            # Calculating the overall average
            overall_average = sum(run_averages) / len(run_averages)
            # print("Average execution time per run:", run_averages)
            # print("Overall average execution time:", overall_average)
            final_data[tuple(df[i][0])] = overall_average
        return final_data
    df = raw_to_list(df)
    # print(df)
    name = '../data/old_data/cpu/' + file_name
    with open(name, 'wb') as f:
        pickle.dump(df, f)

    print(df)


if __name__ == '__main__':
    excute('desktop-cpu-core-i9-13900k-fp32.pickle')
    excute('desktop-cpu-core-i7-12700h-fp32.pickle')


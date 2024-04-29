#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt

def plot_data(df):
    df.columns = ['Diagnosis', 'Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness', 'Compactness', 'Concavity', 'Concave_points', 'Symmetry', 'Fractal_dimension']

    pairplot = sns.pairplot(df, hue='Diagnosis', diag_kind='kde', corner=True)

    pairplot.set(xticklabels=[], yticklabels=[])

    plt.subplots_adjust(wspace=0, hspace=0)

    for ax in pairplot.axes.flat:
        if ax != None:
            if ax.get_ylabel() != '':
                ax.set_ylabel(ax.get_ylabel(), rotation=0)

    plt.show()

def preprocess_data(df, scale_type='standard', impute_strategy='mean'):
    imputer = SimpleImputer(strategy=impute_strategy)

    if scale_type == 'standard':
        scaler = StandardScaler()
    elif scale_type == 'minmax':
        scaler = MinMaxScaler()
    elif scale_type == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError("Invalid scale type")

    X = df.iloc[:, 1:].values  # Features (excluding Diagnosis)
    Y = df.iloc[:, 0].values   # Labels (Diagnosis)

    # Apply imputation
    X_imputed = imputer.fit_transform(X)

    # Encode the categorical labels to integers
    label_encoder = LabelEncoder()
    Y_encoded = label_encoder.fit_transform(Y)

    # Scale the features
    X_scaled = scaler.fit_transform(X_imputed)


    return train_test_split(X_scaled, Y_encoded, test_size=0.2, random_state=42)

def main():
    parser = argparse.ArgumentParser(description="Preprocess dataset")
    parser.add_argument('--scale_type', type=str, choices=['standard', 'minmax', 'robust'], default='robust',
                        help='Type of scaling to use (standard, minmax, or robust)')
    parser.add_argument('--impute_strategy', type=str, choices=['mean', 'median', 'most_frequent'], default='mean',
                        help='Imputation strategy to use for handling missing values')
    parser.add_argument('--src', type=str, default='data.csv',
                        help='Source data file')
    parser.add_argument('--train_dst', type=str, default='train.csv',
                        help='Destination for train data')
    parser.add_argument('--val_dst', type=str, default='val.csv',
                        help='Destination for validation data')
    parser.add_argument('--plot', action='store_true',
                        help='Plot the data')


    args = parser.parse_args()

    data = pd.read_csv(args.src, header=None)

    columns = pd.MultiIndex.from_tuples([
        ('ID', 'id'),
        ('Diagnosis', 'diagnosis'),
        ('Radius', 'mean'), ('Radius', 'se'), ('Radius', 'worst'),
        ('Texture', 'mean'), ('Texture', 'se'), ('Texture', 'worst'),
        ('Perimeter', 'mean'), ('Perimeter', 'se'), ('Perimeter', 'worst'),
        ('Area', 'mean'), ('Area', 'se'), ('Area', 'worst'),
        ('Smoothness', 'mean'), ('Smoothness', 'se'), ('Smoothness', 'worst'),
        ('Compactness', 'mean'), ('Compactness', 'se'), ('Compactness', 'worst'),
        ('Concavity', 'mean'), ('Concavity', 'se'), ('Concavity', 'worst'),
        ('Concave_points', 'mean'), ('Concave_points', 'se'), ('Concave_points', 'worst'),
        ('Symmetry', 'mean'), ('Symmetry', 'se'), ('Symmetry', 'worst'),
        ('Fractal_dimension', 'mean'), ('Fractal_dimension', 'se'), ('Fractal_dimension', 'worst')
    ])

    data.columns = columns

    #only mean/ first value of each feature but keep diagnosis
    df_mean  = data.iloc[:, [1, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29]]
    df_se    = data.iloc[:, [1, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30]]
    df_worst = data.iloc[:, [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31]]

    df_mean = df_mean.dropna()

    if args.plot:
        plot_data(df_mean)

    X_train, X_test, Y_train, Y_test = preprocess_data(df_mean, scale_type=args.scale_type, impute_strategy=args.impute_strategy)

    # Convert arrays back to DataFrame for saving
    train_df = pd.DataFrame(np.column_stack((Y_train, X_train)), columns=['Diagnosis'] + list(df_mean.columns[1:]))
    val_df = pd.DataFrame(np.column_stack((Y_test, X_test)), columns=['Diagnosis'] + list(df_mean.columns[1:]))

    # Save to CSV files
    train_df.to_csv(args.train_dst, index=False)
    val_df.to_csv(args.val_dst, index=False)

    print(f"Train data saved to {args.train_dst}")
    print(f"Validation data saved to {args.val_dst}")

if __name__ == "__main__":
    main()


'''
data structure of data.csv:
ID, Diagnosis followed by the mean, standard error, and worst (mean of the three largest values) of the following features:
         Radius , Texture , Perimeter , Area , Smoothness , Compactness , Concavity , Concave points , Symmetry , Fractal dimension
842302,M,17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189
'''

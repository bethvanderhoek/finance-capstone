def clean_model_data(df):

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # # Rename columns and replace NSD values as None type

    df = df.rename(columns={"Drug Overdose Deaths": "DrugOverdoseDeaths", "Unemployment Percentage Rate": "UnemploymentPercentageRate", "Property Crime Rates": "PropertyCrimeRates", "Violent Crime Rates": "ViolentCrimeRates"})
    df = df.replace({'NSD': None})


    # Changing numeric objects to float

    df.Year = df.Year.astype(float)
    df.GeneralRevenue = df.GeneralRevenue.astype(float)
    df.PropertyTax = df.PropertyTax.astype(float)
    df.AlcoholTax = df.AlcoholTax.astype(float)
    df.InsurancePremiumTax = df.InsurancePremiumTax.astype(float)
    df.MotorFuelsTax = df.MotorFuelsTax.astype(float)
    df.PublicUtilityTax = df.PublicUtilityTax.astype(float)
    df.TobaccoTax = df.TobaccoTax.astype(float)
    df.IndividualIncomeTax = df.IndividualIncomeTax.astype(float)
    df.LiquorStoreRevenue = df.LiquorStoreRevenue.astype(float)
    df.UnemploymentRevenue = df.UnemploymentRevenue.astype(float)
    df.UnempPayrollTax = df.UnempPayrollTax.astype(float)
    df.PoliceFireExpenses = df.PoliceFireExpenses.astype(float)
    df.TotalCorrectionalExpenses = df.TotalCorrectionalExpenses.astype(float)
    df.TotalHigherEdExpenses = df.TotalHigherEdExpenses.astype(float)
    df.TotalEducationExpenses = df.TotalEducationExpenses.astype(float)
    df.EducationAssistanceExpenses = df.EducationAssistanceExpenses.astype(float)
    df.HealthExpenses = df.HealthExpenses.astype(float)
    df.PublicWelfareExpenses = df.PublicWelfareExpenses.astype(float)
    df.UnemploymentExpenses = df.UnemploymentExpenses.astype(float)
    df.TotalDebt = df.TotalDebt.astype(float)
    df.DrugOverdoseDeaths = df.DrugOverdoseDeaths.astype(float)
    df.ViolentCrimeRates = df.ViolentCrimeRates.astype(float)
    df.UnemploymentPercentageRate = df.UnemploymentPercentageRate.astype(float)
    df.PovertyPerc = df.PovertyPerc.astype(float)

    # # Feature Engineering

    # Initial check for nulls:

    # print(df.isnull().sum())

    # Imputing mean into DrugOverdoseDeaths, PropertyCrimeRates, and ViolentCrimeRates columns' data with values 'NSD' (not sufficient data):

    df = df.fillna(df.mean())

    # Dropping years with missing DrugOverdoseDeaths data (1995-1998):

    df = df[df.Year != 1995]
    df = df[df.Year != 1996]
    df = df[df.Year != 1997]
    df = df[df.Year != 1998]

    # Dropping District of Colombia:

    df = df.dropna()

    # Final Check for nulls:

    df.isnull().sum()

    # Looking at data info, corr. matrix does not include states or party

    # print(df.info())

    # df.hist(figsize = (20, 20))

    # plt.figure(dpi=300)
    # f, ax = plt.subplots(figsize=(20, 15))
    # corr = df.corr()
    # mask = np.triu(np.ones_like(corr, dtype=bool))
    # sns.heatmap(corr, annot=True, linewidths=.5, ax=ax, mask=mask)
    # plt.show()

    # # Creating Model

    # Making dummy variables and dropping party/state columns:

    party_dummies = pd.get_dummies(df.Party, prefix='Party').iloc[:, 1:]
    state_dummies = pd.get_dummies(df.State, prefix='State').iloc[:, 1:]
    new_df_1 = pd.concat([df, party_dummies], axis=1).drop("Party", axis = 1)
    new_df = pd.concat([new_df_1, state_dummies], axis=1).drop("State", axis = 1)
    # print(len(new_df.columns))
    # new_df.columns

    # Standardizing the data:

    scaler = StandardScaler().fit(new_df)
    df_scaled = pd.DataFrame(scaler.transform(new_df))
    df_scaled.columns = new_df.columns
    # df_scaled

    # Declaring variables and getting training/test data:

    X = df_scaled.drop("PovertyPerc", axis = 1)
    y = df_scaled["PovertyPerc"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
    return X_train, X_test, y_train, y_test
  






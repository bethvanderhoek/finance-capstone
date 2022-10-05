# Exploring factors that influence poverty in the United States (1995-2020)


We researched factors that contribute to poverty across the United States to predict levels of poverty by state and year. We explored the national U.S. poverty level between 1995 and 2020 to see how inflation, federal government spending, and political parties in control of the House, Senate, and Executive office can affect poverty.  We then narrowed our focus to state-level for the same time period, investigating whether the following factors affected state poverty rates:
 
- Drug overdose rate 
- Violent crime rate
- Property crime rate
- Unemployment rate
- Party control of state government 
- State alcohol revenue
- State tobacco revenue
- State debt
- State property tax 
- State spending on education 
- State spending on social services 


Based on our findings and model predictions, we hope to provide recommendations to government officials that can inform improved future policies to reduce poverty rates across the U.S.

### Exploratory Questions

**Federal Level**
1. How has the national U.S. poverty level changed between 1995-2020?
2. Is there a relationship between the U.S. national poverty level and: party control of the Senate, House, and President, inflation, or federal spending allocated to social services categories?

**State Level**
1. What factors influence poverty rates between 1995-2020 at the state-level?
2. Is state tobacco revenue, state alcohol revenue, or drug overdose rates related to state poverty levels?
3. Are state poverty levels and state education spending related?
4. Is there a relationship between state poverty levels and state social services spending?
5. Are we able to predict state poverty levels based on all the factors explored in Question 3?

## Table of Contents

1. [Project & File Structure](#project-structure)
2. [SQL Database](#sql-database)
3. [Machine Learning](#machine-learning)
4. [Visualizations](#visualizatoins)
5. [Dash Deployment](#dash-deployment)
6. [Results](#results)
7. [Refrences](#refrences)


<a name = "project-structure"></a>
## **Project & File Structure**

Below is a diagram of all Cloud ETL components and how they work with other elements of the project.

![cloud-diagram](https://github.com/bethvanderhoek/finance-capstone/blob/master/CloudETLDiagram.png)

### Data Extraction, Transformation, and Loading

A full ETL description can be found in the [`RepeatableETLReport.pdf`](https://github.com/bethvanderhoek/finance-capstone/blob/master/RepeatableETLReport.pdf) file.

Each dataset was either downloaded as a csv, accessed trhough an API, or scraped from the web. All data transforamtion was done via Azure Databricks. All files for the ETL Process can be found in the [`/code/`](https://github.com/bethvanderhoek/finance-capstone/tree/master/code) directory. In order to run the following files, import them into Azure Databricks. In order to write to a database, you will need to create a SQL Database with a username and password. 

- [`master-cleaning-databrick.ipynb`:](https://github.com/bethvanderhoek/finance-capstone/blob/master/code/master-cleaning-databrick.ipynb) Contains code used to transform all datasets.

- [`write-to-SQL.ipynb`:](https://github.com/bethvanderhoek/finance-capstone/blob/master/code/write-to-SQL.ipynb) Contains code used to load the data into a SQL database.

### Streaming Data

All streaming Data was handled via Azure Data Factory. Code for the Kafka producer and consumer can be found in the [`/code/`](https://github.com/bethvanderhoek/finance-capstone/tree/master/code) directory. In order to run the following files, import them into Azure Databricks.

- [`capstone-producer.ipynb`:](https://github.com/bethvanderhoek/finance-capstone/blob/master/code/capstone-producer.ipynb) Contains code for setting up the Kafka producer.

- [`capstone-consumer.ipynb`:](https://github.com/bethvanderhoek/finance-capstone/blob/master/code/capstone-consumer.ipynb) Contains code for setting up the Kafka consumer.


### Reporting


Below is a brief summary of each report
- [`ExecutiveSummary.pdf`:](https://github.com/bethvanderhoek/finance-capstone/blob/master/ExecutiveSummary.pdf) High-level overview of the porject, including an introduction and exploratory questions.

- [`ExploratoryQuestions.pdf`:](https://github.com/bethvanderhoek/finance-capstone/blob/master/ExploratoryQuestions.pdf) List of exploratory questions that are answered in this project.

- [`RepeatableETLReport.pdf`:](https://github.com/bethvanderhoek/finance-capstone/blob/master/RepeatableETLReport.pdf) Detailed description of the steps taken to extract, transfrom and load the data in to a SQL Database.

- [`VisualizationsNapkinsAndFeedback.pdf`:](https://github.com/bethvanderhoek/finance-capstone/blob/master/VisualizationsNapkinsAndFeedback.pdf) Initial napkin drawings, feedback, and final napkin drawings for visualizations.

- [`DashboardNapkinsAndFeedback.pdf`:](https://github.com/bethvanderhoek/finance-capstone/blob/master/DashboardNapkinsAndFeedback.pdf) Initial napkin drawings, feedback, and final napkin drawings for the dashboard.

- [`ProjectTechnicalReport.pdf`:](https://github.com/bethvanderhoek/finance-capstone/blob/master/ProjectTechnicalReport.pdf) Detailed description of the entire project including answers to exploratory questions, details of the machine learning model and recommendations based on our conclusions.

- [`CapstonePresentationSlides.pdf`:](https://github.com/bethvanderhoek/finance-capstone/blob/master/CapstonePresentationSlides.pdf) Slide deck describing results of the project. 

<a name = "sql-database"></a>
## **SQL Database**
Below is the Entity Relationship Diagram for the SQL Database. This can also be viewed in the [`ERD.pdf`](https://github.com/bethvanderhoek/finance-capstone/blob/master/CloudETLDiagram.pdf) file. 

![erd](https://github.com/bethvanderhoek/finance-capstone/blob/master/ERD.png)

Below are descriptions of the tables in our SQL database

- `Party`: Contains a unique `Id` corresponding with each `PartyName` in the datasets.

- `Year`: Contains a unique `Id` corresponding to each `Year` in the datasets.

- `State`: Contains a unique `Id` corresponding to each `StateName` in the datasets.

- `StateYear`: Contains a unique `Id` corresponding all unique combinations of `StateId` and `YearId`.

- `StateFinances`: Each row of this table contains data from each state for every year. It contains a `StateId` as a foreign key. Other columns are `GeneralRevenue`, `PropertyTax`, `InsurancePremiumTax`, `MotorFuelsTax`, `PublicUtilityTax`, `IndividualIncomeTax`, `LiquorStoreRevenue`, `UnemploymentRevenue`, `UnemploymentPayrollTax`, `PoliceFireExpenses`, `TotalCorrectionalExpenses`, `TotalEducationExpenses`, `TotalHigherEdExpenses`, `EducationAssistanceExpenses`, `PublicWelfareExpenses`, `UnemploymentExpenses`, and `TotalDebt`. All of these columns are reported in dollars per capita. All dollar amounts are adjusted for inflation and are reported in 2020 dollars.

- `StateDrug`: Each row of this table contains data from each state for every year. It contains a `StateId` as a foreign key. The other column, `OverdoseRate`, is reported in overdose deaths per 100,000 people.

- `StatePoverty`: Each row of this table contains data from each state for every year. It contains a `StateId` as a foreign key. The column `PovertyPercent` represents the percent of the total population that is under the poverty line.

- `StatePolitics`: Each row of this table contains data from each state for every year. It contains a `StateId` as a foreign key. The other column, `PartyId` is also a foreign key and represents what political party controls the state legislature.

- `StateCrime`: Each row of this table contains data from each state for every year. It contains a `StateId` as a foreign key.

- `StateUnemployment`: Each row of this table contains data from each state for every year. It contains a `StateId` as a foreign key. `UnemploymentRate` is the percent of the population who are unemployed. 

- `FedFinances`: Each row of this table represents a year. It contains `YearId` as a foreign key. Other columns are `TotalSpending`, `NetInterest`, `SocialSecurityMedicare`, `OtherIndividualPayments`, `OtherFederal`, `StateAndLocal`, `Cpi`. All dollar amounts are adjusted for inflation and are reported in 2020 dollars.

- `FedPolitics`: Each row of this table represents a year. It contains `YearId` as a foreign key. `HousePartyId`,`SenatePartyId`, and `PresidentPartyId` are also all foreign keys representing the party that controls each chamber of congress and the party that aligns with the president.

- `FedPoverty`: Each row of this table represents a year. It contains `YearId` as a foreign key. The column `PovertyPercent` represents the percent of the total population that us under the poverty line.


<a name = "machine-learning"></a>
## **Machine Learning**

One of our key questions was if we could use a machine learning model to predict poverty rates. Code for the machine learning model can be found in the [`/code/`](https://github.com/bethvanderhoek/finance-capstone/tree/master/code) directory. This models uses data from a SQL Database, so one must be created and loaded before running these files.

- [`LassoMLModel.ipynb`:](https://github.com/bethvanderhoek/finance-capstone/tree/master/code/LassoMLModel.ipynb) Contains the code for all model implementation and feature engineering.

- [`Poverty-ML-Model.model`:](https://github.com/bethvanderhoek/finance-capstone/tree/master/code/Poverty-ML-Model.ipynb) An exported version of the model that can be loaded into a dashboard.

<a name = "visualizations"></a>
## **Visualizations**

All visuals were made using plotly and can be seen in the dashboard. For napkin drawings, see the [`VisualizationsNapkinsAndFeedback.pdf`](https://github.com/bethvanderhoek/finance-capstone/blob/master/VisualizationsNapkinsAndFeedback.pdf) file.

<a name = "dash-deployment"></a>
## **Dash Deployment**

The script for the dash deployment can be found in the [`/code/`](https://github.com/bethvanderhoek/finance-capstone/tree/master/code) directory. THe file is called [`dashboard.py`](https://github.com/bethvanderhoek/finance-capstone/tree/master/code/dashboard.py)This dashboard uses data from a SQL Database, so one must be created and loaded before running this file.

<a name = "results"></a>
## **Results**

For a description of our results, see the following 2 files.

- [`ProjectTechnicalReport.pdf`:](https://github.com/bethvanderhoek/finance-capstone/blob/master/ProjectTechnicalReport.pdf) Detailed description of the entire project including answers to exploratory questions, details of the machine learning model and recommendations based on our conclusions.

- [`CapstonePresentationSlides.pdf`:](https://github.com/bethvanderhoek/finance-capstone/blob/master/CapstonePresentationSlides.pdf) Slide deck describing results of the project. 

<a name = "refrences"></a>
## **Refrences**
[1] History, Art & Archives: United States House of Representatives(2021). “Party Government Since 1857.” https://history.house.gov/Institution/Presidents-Coinciding/Party-Government/ Accessed on 09/21/2022.

[2] Iowa Community Indicators Program, Iowa State University (2021). Annual Unemployment Rates by State [Data set]. https://www.icip.iastate.edu/tables/employment/unemployment-states. Accessed 09/21/2022.

[3] Kaiser Family Foundation (2022). Drug Overdose Death Rate (per 100,000 population) [Data set]. https://www.kff.org/other/state-indicator/drug-overdose-death-rate-per-100000-population/?currentTimeframe=20&sortModel=%7B%22colId%22:%22Location%22,%22sort%22:%22asc%22%7D Accessed 09/21/2022.

[4] National Conference of State Legislatures(2022). Legislative Partisan Composition Table [Data set]. https://www.ncsl.org/research/about-state-legislatures/partisan-composition.aspx Accessed 09/21/2022.

[5] Statista(2021). Property crime rate in the United States in 2020, by state. https://www.statista.com/statistics/232575/property-crime-rate-in-the-us-by-state/ Accessed 9/29/22.

[6] Statista(2021). Reported violent crime rate in the United States in 2020, by state. https://www.statista.com/statistics/200445/reported-violent-crime-rate-in-the-us-states/ Accessed 9/29/22. 

[7] Unified Crime Reporting Statistics (2021). State Crime: From the CORGIS Dataset Project [Data set]. https://corgis-edu.github.io/corgis/csv/state_crime/ Accessed 09/20/2022.

[8] Urban Institute (2020). State and Local Finance Data: Exploring the Census of Governments [Data set]. https://state-local-finance-data.taxpolicycenter.org/pages.cfm Accessed 09/19/2022.

[9] U.S. Bureau of Labor Statistics(2022). American Consumer Price Index: 1913 to 2022. https://www.rateinflation.com/consumer-price-index/usa-historical-cpi/ Accessed 09/20/2022.

[10] U.S. Bureau of Labor Statistics(2022). Unemployment Rates for States, 2019 Annual Averages. https://www.bls.gov/lau/lastrk19.htm Accessed 09/27/2022.

[11] U.S. Bureau of Labor Statistics(2022). Unemployment Rates for States, 2020 Annual Averages. https://www.bls.gov/lau/lastrk20.htm Accessed 09/27/2022.

[12] U.S. Census Bureau (2022). Poverty Statistics: CPS & SAIPE (Time Series: various years) [Data set]. https://www.census.gov/data/developers/data-sets/Poverty-Statistics.html Accessed 09/20/2022.

[13] White House Office of Management and Budget(2021). Table 14.4–Total Government Expenditures by Major Category of Expenditure: 1948-2021[Data set]. https://www.whitehouse.gov/omb/budget/historical-tables/ Accessed 09/21/2022.

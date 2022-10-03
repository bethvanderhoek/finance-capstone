import dash
import dash as dcc
from dash import Dash, dcc, html, callback, Output, Input
import plotly.express as px
from plotly import graph_objects as go
import pandas as pd
import pymssql
import dash_bootstrap_components as dbc
import plotly.figure_factory as ff
import numpy as np
import config
from joblib import load

from config import database
from config import table
from config import table2
from config import table3
from config import table4
from config import username
from config import password
from config import server

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

CONTENT_STYLE = {
    "marginLeft": "2rem",
    "marginRight" : "2rem",
    "padding" : "1rem 1rem",
}

conn = pymssql.connect(server,username, password,database)

cursor = conn.cursor()

#Federal level

query_1 = f'select PercentPoverty, Year from {table} as fp inner join {table2} as year on year.YearID = fp.YearID'

df_1 = pd.read_sql(query_1, conn)

fed_fig1 = px.line(df_1, x='Year', y= 'PercentPoverty', labels={
                     "Year": "Year",
                     "PercentPoverty": "Percent of U.S. Population Under Poverty Line"
                 },
                 title='<b>U.S. National Poverty Rates between 1995 to 2020<b>')
fed_fig1.update_layout(title = {'font':{'size':15}}, title_x=0.5)



query_2 = f'select [Social Security and Medicare], [Other Payments for Individuals], Year from {table3} as fp inner join {table2} as year on year.YearID = fp.YearID'

df_2 = pd.read_sql(query_2, conn)

df_2['Government Spending'] = df_2['Social Security and Medicare'] + df_2['Other Payments for Individuals']

fed_fig2 = px.line(df_2, x='Year', y= 'Government Spending', labels={
                     "Year": "Year",
                     "Government Spending": "Federal Spending (in billions of dollars)"
                 },
                 title='<b>Federal Government Spending on Medicare, Social Security, <br>and Other Payments to Individuals (1995 to 2020)</b> <br>(adjusted for inflation)')
fed_fig2.update_layout(title = {'font':{'size':15}}, title_x=0.5)



query3 = f'select Cpi, Year from {table3} as fp inner join {table2} as year on year.YearID = fp.YearID'

df_3 = pd.read_sql(query3, conn)


fed_fig3 = px.line(df_3, x='Year', y= 'Cpi', labels={
                     "Year": "Year",
                     "Cpi": "Consumer Price Index (CPI)"
                 },
                 title='<b>Consumer Price Index (CPI) in the U.S.<br>(1995 to 2020)</b>')
fed_fig3.update_layout(title = {'font':{'size':15}}, title_x=0.5)



query4 = f'''select PresidentId, HouseId, SenateId, PercentPoverty 
from {table4}
inner join {table2} on {table2}.YearID = {table4}.YearID
inner join {table} on {table}.YearID = {table4}.YearID
order by Year'''

df_4 = pd.read_sql(query4, conn)

df_4 = df_4.rename(columns={'PresidentId': 'President Party', 'HouseId': 'House Party', 'SenateId': 'Senate Party', 'PercentPoverty': 'U.S. Poverty Rate (%)'})

df_corr = df_4.corr() 
mask = np.triu(np.ones_like(df_corr, dtype=np.bool))
df_corr = df_corr.mask(mask)

fed_fig4 = ff.create_annotated_heatmap(
        z = df_corr.to_numpy().round(2),
        x = list(df_corr.index.values),
        y = list(df_corr.columns.values),
        xgap=3, ygap=3,
        zmin=-1, zmax=1,
        colorscale=px.colors.sequential.Reds,
    )

fed_fig4.update_layout(title_text='''<b>Correlation Matrix: <br> U.S. Poverty Rate and Party Control of President, House, and Senate 
<br><br><br>Democrat Party = 0, Republican Party = 2, Mixed = 1 ''',
                  title = {'font':{'size':15}},
                  title_x=0.5,
                  xaxis_showgrid=False,
                  xaxis={'side': 'bottom'},
                  yaxis_showgrid=False,
                  yaxis_autorange='reversed',                  
                  )

query5 = f'''select Year, State, PovertyPerc, HighSchoolDiploma, BachelorsDegree, GeneralRevenue, TotalCorrectionalExpenses, PoliceFireExpenses, HealthExpenses, PartyId, EducationAssistanceExpenses, TotalHigherEdExpenses, UnempPayrollTax, IndividualIncomeTax, [Unemployment Percentage Rate], [Drug Overdose Deaths], AlcoholTax, TobaccoTax, TotalDebt, TotalEducationExpenses, PublicWelfareExpenses
from dbo.[State]
left join dbo.StateYear on dbo.StateYear.StateId = dbo.State.StateId
left join dbo.StatePoverty on dbo.StatePoverty.StateYearId = dbo.StateYear.StateYearId
left join dbo.Year on dbo.StateYear.YearId = dbo.Year.YearId
left join dbo.StatePolitics on dbo.StatePolitics.StateYearId = dbo.StateYear.StateYearId
left join dbo.StateUnemployment on dbo.StateUnemployment.StateYearId = dbo.StateYear.StateYearId
left join dbo.StateDrug on dbo.StateDrug.StateYearId = dbo.StateYear.StateYearId
left join dbo.StateFinances on dbo.StateFinances.StateYearId = dbo.StateYear.StateYearId
left join dbo.Education on dbo.Education.StateYearId = dbo.StateYear.StateYearId
order by State, Year'''

df_5 = pd.read_sql(query5, conn)

df_5 = df_5.replace(0, 'None')

df_5 = df_5.dropna()


df_bar = df_5.corrwith(df_5['PovertyPerc']).round(2)

df_bar = pd.DataFrame({'Poverty Factors':df_bar.index, 'Correlation':df_bar.values})

df_bar = df_bar[df_bar.Correlation != 1]

fed_fig6 = px.bar(df_bar, x='Poverty Factors', y= 'Correlation', labels={
                     "Poverty Factors": "Factors Related to Poverty",
                     "Correlation": "Correlation Coefficient"
                 },
                 title='<b>Correlation Coefficients of Factors Related to Poverty with Poverty Rates in U.S. States 1995-2020<b>',
                 text_auto= True)
fed_fig6.update_layout(title = {'font':{'size':15}}, title_x=0.5, yaxis_range=[-1,1])


query = f'''select State, Year, PovertyPerc, HighSchoolDiploma, BachelorsDegree
            from dbo.State
            join dbo.StateYear on dbo.StateYear.StateId = dbo.State.StateId
            join dbo.Year on dbo.Year.YearId = dbo.StateYear.YearId
            left join dbo.Education on dbo.Education.StateYearId = dbo.StateYear.StateYearId
            left join dbo.StatePoverty on dbo.StatePoverty.StateYearId = dbo.StateYear.StateYearId
            order by Year, State'''

df_8 = pd.read_sql(query, conn)
fed_fig8 = px.scatter(df_8, x='PovertyPerc', y= 'HighSchoolDiploma', opacity = 0.5, trendline='ols', trendline_color_override='red', labels={
                        'PovertyPerc': 'Population Under Poverty Level (%)',
                        'HighSchoolDiploma': 'High School Diploma Rate (%)'
                        },
                        title= '<b>Percentage under Poverty Line vs. <br>High School Diploma Rate by State')
fed_fig8.update_layout(title= {'font': {'size': 15}}, title_x = 0.5)

# fed_fig9 = px.scatter(df_8, x='PovertyPerc', y= 'BachelorsDegree', opacity = 0.5, trendline='ols', trendline_color_override='red', labels={
#                         'PovertyPerc': 'Population Under Poverty Level (%)',
#                         'BachelorsDegree': 'Bachelor\'s Degree Rate (%)'
#                         },
#                         title= '<b>Percentage under Poverty Line vs. <br>Bachelor\'s Degree Rate by State')
# fed_fig9.update_layout(title= {'font': {'size': 15}}, title_x = 0.5)

#state level

query = f"select * from StateYear join StateDrug on StateYear.StateYearId=StateDrug.StateYearId join StateFinances on StateYear.StateYearId=StateFinances.StateYearId join StatePolitics on StateYear.StateYearId=StatePolitics.StateYearId join StatePoverty on StateYear.StateYearId=StatePoverty.StateYearId join StateUnemployment on StateYear.StateYearId=StateUnemployment.StateYearId join Year on StateYear.YearId=Year.YearId join State on StateYear.StateId=State.StateId join Party on StatePolitics.PartyId=Party.PartyId"
df = pd.read_sql(query,conn)

df=df.drop(columns=['StateId','YearId','PartyId','StateYearId','Id'])
states = str = sorted(list(df['State'].unique()))

# fed_fig7 = px.scatter(df, x='PovertyPerc', y='Unemployment Percentage Rate', trendline = 'ols', trendline_color_override= "red", labels={
#                         "PovertyPerc" : "Population Under Poverty Level (%)",
#                         "Unemployment Percentage Rate" : 'Unemployment Rate (%)'
#                     },
#                     title = '<b>Percentage under Poverty Line by State vs. <br>Unemployment Rate by State')

# fed_fig7.update_layout(title = {'font':{'size':15}}, title_x = 0.5)

fig1 = px.scatter(df, 
                x='AlcoholTax',
                y='PovertyPerc',
                title='<b>Percent of Population in Poverty vs. Alcohol Tax Revenue per Capita</b>',
                labels = {'AlcoholTax':'Alcohol Tax Revenue per Capita','PovertyPerc': 'Percent of Population in Poverty'}, 
                hover_data = ['State','Year'],
                trendline='ols',
                trendline_color_override='red')
fig2 = px.scatter(df, 
                x='TobaccoTax',
                y='PovertyPerc',
                title='<b>Percent of Population in Poverty vs. Tobacco Tax Revenue per Capita</b>', 
                labels = {'TobaccoTax':'Tobacco Tax Revenue per Capita','PovertyPerc': 'Percent of Population in Poverty'},
                hover_data = ['State','Year'],
                trendline ='ols',
                trendline_color_override='red')
fig3 = px.scatter(df, 
                x='Drug Overdose Deaths',
                y='PovertyPerc',
                title='<b>Percent of Population in Poverty vs. Drug Overdose Deaths per 100,000 people</b>', 
                labels = {'Drug Overdose Deaths':'Drug Overdose Deaths per 100,000 people','PovertyPerc': 'Percent of Population in Poverty'},
                hover_data = ['State','Year'],
                trendline='ols',
                trendline_color_override='red')
fig4 = px.scatter(df,
                x='TotalEducationExpenses',
                y='PovertyPerc',
                title='Percent of Population in Poverty vs. Total Education Expenses per Capita',
                labels = {'TotalEducationExpenses':'Total Education Expenses per Capita','PovertyPerc': 'Percent of Population in Poverty'},
                hover_data = ['State','Year'],
                trendline='ols',
                trendline_color_override='red')
fig5 = px.scatter(df, 
                x='TotalHigherEdExpenses',
                y='PovertyPerc',
                title='Percent of Population in Poverty vs. Higher Education Expenses per Captia',
                labels = {'TotalHigherEdExpenses':'Total Higher Education Expenses per Captia','PovertyPerc': 'Percent of Population in Poverty'}, 
                hover_data = ['State','Year'],
                trendline='ols',
                trendline_color_override='red')
fig6 = px.scatter(df, 
                x='HealthExpenses',
                y='PovertyPerc',
                title='Percent of Population in Poverty vs. Total Health Expenses per Capita', 
                labels = {'HealthExpenses':'Total Health Expenses per Capita','PovertyPerc': 'Percent of Population in Poverty'},
                hover_data = ['State','Year'],
                trendline='ols',
                trendline_color_override='red')
fig7 = px.scatter(df, 
                x='PublicWelfareExpenses',
                y='PovertyPerc',
                title='Percent of Population in Poverty vs. Public Welfare Expenses per Capita', 
                labels = {'PublicWelfareExpenses':'Public Welfare Expenses per Capita','PovertyPerc': 'Percent of Population in Poverty'},
                hover_data = ['State','Year'],
                trendline='ols',
                trendline_color_override='red')

#model visualizations

query = f'''SELECT State, Year, GeneralRevenue, PropertyTax, AlcoholTax, InsurancePremiumTax, MotorFuelsTax, PublicUtilityTax, TobaccoTax, IndividualIncomeTax, LiquorStoreRevenue, UnemploymentRevenue, UnempPayrollTax, PoliceFireExpenses, TotalCorrectionalExpenses, TotalEducationExpenses, TotalHigherEdExpenses, EducationAssistanceExpenses, HealthExpenses, PublicWelfareExpenses, UnemploymentExpenses, TotalDebt, Party, [Property Crime Rates], [Violent Crime Rates], [Drug Overdose Deaths], [Unemployment Percentage Rate], PovertyPerc 
FROM StateYear k 
JOIN State s on s.StateId = k.StateId 
JOIN Year y on y.YearId = k.YearId 
LEFT JOIN StateFinances f on f.StateYearId = k.StateYearId 
LEFT JOIN StatePolitics p on p.StateYearId = k.StateYearId 
LEFT JOIN Party x on x.PartyId = p.PartyId 
LEFT JOIN StateDrug d on d.StateYearId = k.StateYearId 
LEFT JOIN StateUnemployment u on u.StateYearId = k.StateYearId 
LEFT JOIN StatePoverty z on z.StateYearId = k.StateYearId 
LEFT JOIN StateCrime c on c.StateYearId = k.StateYearId'''

df = pd.read_sql(query, conn)

#import function from python script to clean and transform data for model
from functionScript import clean_model_data

X_train, X_test, y_train, y_test = clean_model_data(df)

#import and load Lasso machine learning model

lasso = load("Poverty-ML-Model.model")

#Run the model using X_test

fit = lasso.fit(X_train, y_train)
y_pred = lasso.predict(X_test)
r2 = lasso.score(X_test, y_test)

#predicted vs actual

model_fig1 = px.scatter(x = y_test, y= y_pred, title = '<b>Predicted vs Actual Values', 
                        opacity = 0.5,
                        labels = {'x': 'Actual', 'y': 'Predicted'}, 
                        trendline='ols',
                        trendline_color_override='red')
model_fig1.update_layout(title = {'font': {'size': 15}}, title_x = 0.5)

#predicted vs residuals

residuals = y_test - y_pred 

model_fig2 = px.scatter(x = y_pred, y= residuals, title = '<b>Predicted Values vs. Residuals', 
                        opacity= 0.5, 
                        labels = {'x': 'Predicted', 'y': 'Residuals'})
model_fig2.add_hline(y=0, line_dash="dash", line_color="red")
model_fig2.update_layout(title = {'font': {'size': 15}}, title_x = 0.5)


#dashboard layout

app.layout = (html.Div(children=[    
    
    dbc.Row([
        dbc.Col([
            html.H2(children= 'Investigating Poverty Rates in U.S. States (1995-2020)', style={'textAlign': 'center'}),

            html.H3(children='Financial Services Group 1 Capstone Dashboard', style={'textAlign': 'center', 'marginBottom': '1.5em'}),

            html.H5(children = '''
            By: Shannon Bayless, Beth Vander Hoek, Jerad Ipsen, Joel Garcia
            ''',
            ),

            ]),
        
        dbc.Col(
            html.Img(src='/assets/poverty.png', width = '400', height = 'auto'),
            style={'textAlign': 'center'})
    ], style={'textAlign': 'center', 'marginBottom': '1em'}),

    html.Div([
        html.P('''For our capstone project, we investigated factors that contributed to U.S. poverty rates,
        both at the federal and state level, 
        to predict state poverty levels.
        We focused on the years 1995 to 2020 to get the most data as possible from available datasets.'''),
        html.P("From our findings, we hope to advise policymakers on where to focus their efforts to effectively reduce poverty")
    ], style = {'marginBottom': '0.5em', 'textAlign': 'center'}),

    html.H4(children = '''
    Federal Level
    '''),
    html.H5(children = '''
    1. How has the national U.S. poverty level changed between 1995-2020?
    '''),

    dcc.Graph(
        id='percent-poverty-fed',
        figure = fed_fig1
    ),
    html.H5(children = '''
    2. Does the U.S. national poverty level and the following trend similarly : (a) party control of the Senate, House, & President, (b) inflation, or (c) federal spending allocated to social services categories?
    '''),
    dbc.Row([
            dbc.Col(
                dcc.Graph(
                    id='fed-spending',
                    figure=fed_fig2,
                    style={'width':'100%'}
                )
            ),
            dbc.Col(
                dcc.Graph(
                    id='cpi',
                    figure=fed_fig3,
                    style={'width':'100%'}
                )
            )
        ]),
    
    dcc.Graph(
        id= 'federal-politics',
        figure= fed_fig4
    ),   

    html.Div([
        html.P('''In the figure above, both the President and Senate had a moderate negative correlation
                with the U.S. poverty rate. Thus, when the President and Senate are Democrat (coded as 0), 
                the poverty rate is higher. When they are Republican (coded as 2), it is lower.''', style={'textAlign': 'center'}),
        html.P('''However, this does not account for any policy affecting poverty rates enacted by previous
        administrations.''',
        style={'textAlign': 'center'}),
    ]),

## State Level

    html.H4(children = '''
    State Level
    '''),

    html.H6(children='''
        Select a state to filter all state-level charts below by:
    '''),
    dcc.Dropdown(
    id='drop', 
    options=[{'label': 'All States', 'value': 'all_values'}]+[{'label':x, 'value':x} for x in states], 
    value='all_values',  
    multi=False,
    ),

    html.H5(children = ''' 
    3a. What factors influence poverty rates between 1995-2020 at the state-level?
    '''),
    
    dcc.Graph(
        id = 'state-poverty-factors-bar', 
        figure = fed_fig6,
        style = {'marginBottom': '1em'}
    ),
    html.H5(children='''
        3b. Are unemployment or education attainment rates directly related to state poverty levels?
    '''),
    
    dbc.Row([
        dbc.Col(
            dcc.Graph(
                id = 'state-hs',
                figure = fed_fig8,
                style = {'width': '100%'}
            )
        ),
        # dbc.Col( 
        #     dcc.Graph(
        #         id = 'state-ba', 
        #         figure = fed_fig9,
        #         style = {'width': '100%'}
        #     )
        # ),
    ]),
        

    html.H5(children='''
        4. Is state tobacco revenue, state alcohol revenue, or drug overdose rates directly related to state poverty levels?
    '''),
    dbc.Row([
        dbc.Col(
            dcc.Graph(
                id='alcohol',
                figure=fig1,
                style={'width':'100%'}
            )
        ),
        dbc.Col(
            dcc.Graph(
                id='tobacco',
                figure=fig2,
                style={'width':'100%'}
            )
        ),

    ]),
    dcc.Graph(
        id='overdose',
        figure=fig3,
        style={'width':'100%'}
    ),
        html.H5(children='''
        5. Is state spending on education directly related to state poverty levels?
    '''),
    dbc.Row([
        dbc.Col(
            dcc.Graph(
                id='TotalEd',
                figure=fig4
            )
        ),
        dbc.Col(
            dcc.Graph(
                id='HigherEd',
                figure=fig5
            )
        )
    
    ]),
        html.H5(children='''
        6. Is state spending on social services directly related to state poverty levels?
    '''),
    dbc.Row([
        dbc.Col(
            dcc.Graph(
                id='Health',
                figure=fig6
            )
        ),
        dbc.Col(
            dcc.Graph(
                id='Welfare',
                figure=fig7
            )
        )
    ]),

    html.H5(children = '''
        7. Using a Lasso Machine Learning model, can we predict state poverty levels based on all the factors explored above in Question 3?
        '''),

    dbc.Row([
        dbc.Col(
            dcc.Graph(
                id = 'predict vs actual',
                figure = model_fig1
            )
        ),
        dbc.Col(
            dcc.Graph(
                id = 'predict vs residuals',
                figure = model_fig2
            )
        )
    ]),

    html.Div([
        html.P('''These figures show that our Lasso model accounts for 90%% of variance in our dataset of 
        state poverty levels between 1999 and 2020, using the factors correlated with poverty rates in Question 3. 
        We shortened the time period to account for large amounts of null values between 1995 to 1998.
        ''', style = {'textAlign': 'center'}),
        html.P("From our findings, we recommend policymakers focus on these issues to reduce state poverty levels: "),
        html.Ol([
            html.Li('''High School Diploma and Bachelor\'s Degree education attainment'''),
            html.Li('Unemployment Rates'),
            html.Li('Drug Usage and Overdose Rates')
        ]),
        html.P('''Poverty is a complex issue: no one related factor correlates strongly as all factors have at least some effect
        on poverty rates.''', style={'textAlign': 'center'}),
        html.H5('Please see our technical report for a thorough discussion of our results', style={'textAlign': 'center'})
    ], style = {'marginTop': '1em'}),

], style= CONTENT_STYLE))





@app.callback(
    [Output(component_id='state-poverty-factors-bar', component_property='figure'),
    # Output(component_id='state-ba', component_property='figure'),
    Output(component_id='state-hs', component_property='figure'),
    Output(component_id='alcohol', component_property='figure'),
    Output(component_id='tobacco', component_property='figure'),
    Output(component_id='overdose', component_property='figure'),
    Output(component_id='TotalEd', component_property='figure'),
    Output(component_id='HigherEd', component_property='figure'),
    Output(component_id='Health', component_property='figure'),
    Output(component_id='Welfare', component_property='figure')],
    Input(component_id='drop', component_property='value')
)

def update_graph(label_selected):
    if label_selected == 'all_values':
        dff1 = df
        dff2 = df_bar
        dff3 = df_8
        
    else: 
        dff1 = df[df['State'].eq(label_selected)]
        dff2 = df_5[df_5['State'].eq(label_selected)]
        dff2 = dff2.corrwith(dff2['PovertyPerc']).round(2)
        dff2 = pd.DataFrame({'Poverty Factors':dff2.index, 'Correlation':dff2.values})
        dff2 = dff2[dff2.Correlation != 1]
        dff3 = df_8[df_8['State'].eq(label_selected)]
        
    fed_fig6 = px.bar(dff2, x='Poverty Factors', y= 'Correlation', labels={
                        "Poverty Factors": "Factors Related to Poverty",
                        "Correlation": "Correlation Coefficient"
                    },
                    title='<b>Correlation Coefficients of Factors Related to Poverty with Poverty Rates in U.S. States 1995-2020<b>',
                    text_auto= True)
    fed_fig6.update_layout(title = {'font':{'size':15}}, title_x=0.5, yaxis_range=[-1,1])

    # fed_fig7 = px.scatter(dff1, x='PovertyPerc', y='Unemployment Percentage Rate', opacity = 0.5,
    #                         trendline = 'ols', trendline_color_override= "red", labels={
    #                         "PovertyPerc" : "Population Under Poverty Level (%)",
    #                         "Unemployment Percentage Rate" : 'Unemployment Rate (%)'
    #                         },
    #                         title = '<b>Percentage under Poverty Line vs. <br>Unemployment Rate by State')

    # fed_fig7.update_layout(title = {'font':{'size':15}}, title_x = 0.5)

    fed_fig8 = px.scatter(dff3, x='PovertyPerc', y= 'HighSchoolDiploma', opacity = 0.5, trendline='ols', trendline_color_override='red', 
                        labels={
                        'PovertyPerc': 'Population Under Poverty Level (%)',
                        'HighSchoolDiploma': 'High School Diploma Rate (%)'
                        },
                        title= '<b>Percentage under Poverty Line vs. <br>High School Diploma Rate by State')
    fed_fig8.update_layout(title= {'font': {'size': 15}}, title_x = 0.5)

    # fed_fig9 = px.scatter(dff3, x='PovertyPerc', y= 'BachelorsDegree', opacity = 0.5, trendline='ols', trendline_color_override='red', labels={
    #                     'PovertyPerc': 'Population Under Poverty Level (%)',
    #                     'BachelorsDegree': 'Bachelor\'s Degree Rate (%)'
    #                     },
    #                     title= '<b>Percentage under Poverty Line vs. <br>Bachelor\'s Degree Rate by State')
    # fed_fig9.update_layout(title= {'font': {'size': 15}}, title_x = 0.5)


    fig1 = px.scatter(dff1, 
                    x='AlcoholTax',
                    y='PovertyPerc',
                    opacity = 0.5,
                    title='<b>Percent of Population in Poverty vs. <br>Alcohol Tax Revenue per Capita',
                    labels = {'AlcoholTax':'Alcohol Tax Revenue per Capita','PovertyPerc': 'Percent of Population in Poverty'}, 
                    hover_data = ['State','Year'],
                    trendline='ols',
                    trendline_color_override='red')
    fig2 = px.scatter(dff1, 
                    x='TobaccoTax',
                    y='PovertyPerc',
                    opacity = 0.5,
                    title='<b>Percent of Population in Poverty vs. <br>Tobacco Tax Revenue per Capita', 
                    labels = {'TobaccoTax':'Tobacco Tax Revenue per Capita','PovertyPerc': 'Percent of Population in Poverty'},
                    hover_data = ['State','Year'],
                    trendline ='ols',
                    trendline_color_override='red')
    fig3 = px.scatter(dff1, 
                    x='Drug Overdose Deaths',
                    y='PovertyPerc',
                    opacity = 0.5,
                    title='<b>Percent of Population in Poverty vs. <br>Drug Overdose Deaths per 100,000 people', 
                    labels = {'Drug Overdose Deaths':'Drug Overdose Deaths per 100,000 people','PovertyPerc': 'Percent of Population in Poverty'},
                    hover_data = ['State','Year'],
                    trendline='ols',
                    trendline_color_override='red')
    fig4 = px.scatter(dff1,
                    x='TotalEducationExpenses',
                    y='PovertyPerc',
                    opacity = 0.5,
                    title='<b>Percent of Population in Poverty vs. <br>Total Education Expenses per Capita',
                    labels= {'TotalEducationExpenses':'Total Education Expenses per Capita','PovertyPerc': 'Percent of Population in Poverty'},
                    hover_data = ['State','Year'],
                    trendline='ols',
                    trendline_color_override='red')
    fig5 = px.scatter(dff1, 
                    x='TotalHigherEdExpenses',
                    y='PovertyPerc',
                    opacity = 0.5,
                    title='<b>Percent of Population in Poverty vs. <br>Higher Education Expenses per Captia',
                    labels = {'TotalHigherEdExpenses':'Total Higher Education Expenses per Captia','PovertyPerc': 'Percent of Population in Poverty'}, 
                    hover_data = ['State','Year'],
                    trendline='ols',
                    trendline_color_override='red')
    fig6 = px.scatter(dff1, 
                    x='HealthExpenses',
                    y='PovertyPerc',
                    opacity = 0.5,
                    title='<b>Percent of Population in Poverty vs. <br>Total Health Expenses per Capita', 
                    labels = {'HealthExpenses':'Total Health Expenses per Capita','PovertyPerc': 'Percent of Population in Poverty'},
                    hover_data = ['State','Year'],
                    trendline='ols',
                    trendline_color_override='red')
    fig7 = px.scatter(dff1, 
                    x='PublicWelfareExpenses',
                    y='PovertyPerc',
                    opacity = 0.5,
                    title='<b>Percent of Population in Poverty vs. <br>Public Welfare Expenses per Capita', 
                    labels = {'PublicWelfareExpenses':'Public Welfare Expenses per Capita','PovertyPerc': 'Percent of Population in Poverty'},
                    hover_data = ['State','Year'],
                    trendline='ols',
                    trendline_color_override='red')
    fig1.update_layout(
        title = {'font':{'size':15}},
        title_x=0.5
    )
    fig2.update_layout(
        title = {'font':{'size':15}},
        title_x=0.5
    )
    fig3.update_layout(
        title = {'font':{'size':15}},
        title_x=0.5
    )
    fig4.update_layout(
        title = {'font':{'size':15}},
        title_x=0.5
    )
    fig5.update_layout(
        title = {'font':{'size':15}},
        title_x=0.5
    )
    fig6.update_layout(
        title = {'font':{'size':15}},
        title_x=0.5
    )
    fig7.update_layout(
        title = {'font':{'size':15}},
        title_x=0.5
    )
    return fed_fig6, fed_fig8, fig1, fig2, fig3, fig4, fig5, fig6, fig7

if __name__ == '__main__':
    app.run_server(debug=True)


import warnings
import os
import joblib
import pydotplus

import pickle
import numpy as np
import pandas as pd
import optuna
import seaborn as sns
from matplotlib import pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import scipy.stats as stats

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve


pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)
warnings.simplefilter(action='ignore', category=Warning)

class DataAnalysis:
    def __init__(self):
        pass

    def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=10):
        train_score, test_score = validation_curve(
            model, X=X, y=y, param_name=param_name, param_range=param_range, scoring=scoring, cv=cv)
        mean_train_score = np.mean(train_score, axis=1)
        mean_test_score = np.mean(test_score, axis=1)

        plt.plot(param_range, mean_train_score,
                 label="Training Score", color="b")
        plt.plot(param_range, mean_test_score,
                 label="Test Score", color="g")
        plt.title(f"Validation Curve for {type(model).__name__}")
        plt.xlabel(f"Number of {param_name}")
        plt.ylabel(f"{scoring}")
        plt.tight_layout()
        plt.legend(loc="best")
        plt.show(block=True)
        # Implementation of val_curve_params function

    def plot_importance(model, features, num=10, save=False):
        feature_imp = pd.DataFrame({'Value': model.feature_importances_,
                                    'Feature': features.columns})
        plt.figure(figsize=(10, 10))
        sns.set(font_scale=1)
        sns.barplot(x='Value', y='Feature', data=feature_imp.sort_values(by="Value",
                                                                         ascending=False)[0:num])
        plt.title("Features")
        plt.tight_layout()
        plt.show()
        if save:
            plt.savefig("importances.png")
        # Implementation of plot_importance function

    def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.99):
        quartile1 = dataframe[col_name].quantile(q1)
        quartile3 = dataframe[col_name].quantile(q3)
        interquantile_range = quartile3 - quartile1
        up_limit = quartile3 + 1.5 * interquantile_range
        low_limit = quartile1 - 1.5 * interquantile_range
        return low_limit, up_limit
        # Implementation of outlier_thresholds function
    def check_outlier(dataframe, col_name):
        low_limit, up_limit = outlier_thresholds(dataframe, col_name)
        if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
            return True
        else:
            return False
        # Implementation of check_outlier function
    def replace_with_thresholds(dataframe, variable):
        low_limit, up_limit = outlier_thresholds(dataframe, variable)
        dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
        dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
        # Implementation of replace_with_thresholds function

    def target_summary_with_num(dataframe, target, numerical_col):
        print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")
        # Implementation of target_summary_with_num function
    def plot_numerical_col(dataframe, numerical_col):
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.show(block=True)
        # Implementation of plot_numerical_col function
    def check_df(dataframe, head=5):
        print("############## Shape ##############")
        print(dataframe.shape)
        print("############## Types ##############")
        print(dataframe.dtypes)
        print("############## Head ##############")
        print(dataframe.head(head))
        print("############## Tail ##############")
        print(dataframe.tail(head))
        print("############## NA ##############")
        print(dataframe.isnull().sum())
        print("############## Quantiles ##############")
        print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
        # Implementation of check_df function

    def grab_col_names(dataframe, cat_th=10, car_th=20):
        """

        Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
        Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

        Parameters
        ------
            dataframe: dataframe
                    Değişken isimleri alınmak istenilen dataframe
            cat_th: int, optional
                    numerik fakat kategorik olan değişkenler için sınıf eşik değeri
            car_th: int, optinal
                    kategorik fakat kardinal değişkenler için sınıf eşik değeri

        Returns
        ------
            cat_cols: list
                    Kategorik değişken listesi
            num_cols: list
                    Numerik değişken listesi
            cat_but_car: list
                    Kategorik görünümlü kardinal değişken listesi

        Examples
        ------
            import seaborn as sns
            df = sns.load_dataset("iris")
            print(grab_col_names(df))


        Notes
        ------
            cat_cols + num_cols + cat_but_car = toplam değişken sayısı
            num_but_cat cat_cols'un içerisinde.
            Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

        """

        # cat_cols, cat_but_car
        cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
        num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                       dataframe[col].dtypes != "O"]
        cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                       dataframe[col].dtypes == "O"]
        cat_cols = cat_cols + num_but_cat
        cat_cols = [col for col in cat_cols if col not in cat_but_car]

        # num_cols
        num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
        num_cols = [col for col in num_cols if col not in num_but_cat]

        # print(f"Observations: {dataframe.shape[0]}")
        # print(f"Variables: {dataframe.shape[1]}")
        # print(f'cat_cols: {len(cat_cols)}')
        # print(f'num_cols: {len(num_cols)}')
        # print(f'cat_but_car: {len(cat_but_car)}')
        # print(f'num_but_cat: {len(num_but_cat)}')
        return cat_cols, num_cols, cat_but_car
        # Implementation of grab_col_names function

    def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
        dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
        return dataframe

#######################################################
#  Data
#######################################################
df_ = pd.read_excel('/Users/yigitcankarakoylu/PycharmProjects/po_lead_time_tahmin/In_house_order_lead_time.xlsx')
df = df_.copy()
df.head()

#with open('po_lead_time_original_df.pickle', 'wb') as file:
    #pickle.dump(df, file)
##############################################################################
# Data preprocessing
##############################################################################
df.columns = ['ORDER_ID', 'MATERIAL_ID', 'MATERIAL_GROUP', 'MATERIAL_TYPE', 'PRODUCTION_CENTER', 'MATERIAL_TYPE_CAT', 'MATERIAL_GROUP_CAT',
              'QUANTITY', 'ORDER_TYPE', 'ORDER_PLAN', 'PROCESS_NUMBER', 'DIFF_STATION_NUM', 'NATIONAL_INDICATOR', 'ORDER_PRIORITY',
              'IS_PROJECT', 'IS_ALL_COMPONENTS_AVAILABLE', 'CONTRACTOR_INVOLVED', 'MANUAL_LABOUR_HOUR', 'MACHINE_LABOUR_HOUR', 'Z4_NOTIF_NUM',
              'Z6_NOTIF_NUM', 'LEAD_TIME']

DataAnalysis.check_df(df)

df.fillna("XXXX-XXXX-XXXX",inplace=True)

########################### Target'ın Analizi ####################################
#Explore the dependent variable
# Fit a normal distribution to the SalePrice data
mu, sigma = stats.norm.fit(df['LEAD_TIME'])

# Create a histogram of the SalePrice column
hist_data = go.Histogram(x=df['LEAD_TIME'], nbinsx=50, name="Histogram", opacity=0.75, histnorm='probability density', marker=dict(color='purple'))

# Calculate the normal distribution based on the fitted parameters
x_norm = np.linspace(df['LEAD_TIME'].min(), df['LEAD_TIME'].max(), 60000)
y_norm = stats.norm.pdf(x_norm, mu, sigma)

# Create the normal distribution overlay
norm_data = go.Scatter(x=x_norm, y=y_norm, mode="lines", name=f"Normal dist. (μ={mu:.2f}, σ={sigma:.2f})", line=dict(color="green"))

# Combine the histogram and the overlay
fig = go.Figure(data=[hist_data, norm_data])

# Set the layout for the plot
fig.update_layout(
    title="Lead Time Distribution",
    xaxis_title="Lead_Time",
    yaxis_title="Density",
    legend_title_text="Fitted Normal Distribution",
    plot_bgcolor='rgba(32, 32, 32, 1)',
    paper_bgcolor='rgba(32, 32, 32, 1)',
    font=dict(color='white')
)

# Create a Q-Q plot
qq_data = stats.probplot(df['LEAD_TIME'], dist="norm")
qq_fig = px.scatter(x=qq_data[0][0], y=qq_data[0][1], labels={'x': 'Theoretical Quantiles', 'y': 'Ordered Values'}, color_discrete_sequence=["purple"])
qq_fig.update_layout(
    title="Q-Q plot",
    plot_bgcolor='rgba(32, 32, 32, 1)',
    paper_bgcolor='rgba(32, 32, 32, 1)',
    font=dict(color='white')
)
# Calculate the line of best fit
slope, intercept, r_value, p_value, std_err = stats.linregress(qq_data[0][0], qq_data[0][1])
line_x = np.array(qq_data[0][0])
line_y = intercept + slope * line_x

# Add the line of best fit to the Q-Q plot
line_data = go.Scatter(x=line_x, y=line_y, mode="lines", name="Normal Line", line=dict(color="green"))

# Update the Q-Q plot with the normal line
qq_fig.add_trace(line_data)

# Show the plots
fig.show()
qq_fig.show()

#Store the plots
#pio.write_image(fig, 'lead_time_distribution.png')
#pio.write_image(qq_fig, 'lead_time_qq_plot.png')

########################### Target Analysis ####################################

########################### Categorical Feature Analysis ####################################

############################################
# 1. Analysis on Production Center #
############################################
production_center_types = df['PRODUCTION_CENTER'].value_counts()
production_center_lead_times = df.groupby('PRODUCTION_CENTER')['LEAD_TIME'].mean()

# Create bar charts
fig1 = go.Figure(data=[go.Bar(
    x=production_center_types.index,
    y=production_center_types.values,
    marker_color='rgb(76, 175, 80)',
    text=production_center_types.values,
    textposition='outside',
    width=100,
    marker=dict(line=dict(width=2, color='rgba(0,0,0,1)'), opacity=1)
)])
fig1.update_layout(
    title='Distribution of Production Centers',
    xaxis_title='Production Center',
    yaxis_title='Order Count',
    plot_bgcolor='rgba(34, 34, 34, 1)',
    paper_bgcolor='rgba(34, 34, 34, 1)',
    font=dict(color='white')
)

fig2 = go.Figure(data=[go.Bar(
    x=production_center_lead_times.index,
    y=production_center_lead_times.values,
    marker_color='rgb(156, 39, 176)',
    text=production_center_lead_times,
    textposition='outside',
    width=100,
    marker=dict(line=dict(width=2, color='rgba(0,0,0,1)'), opacity=1)
)])
fig2.update_layout(
    title='Average Lead Time by Production Center',
    xaxis_title='Production Center',
    yaxis_title='Lead Time',
    plot_bgcolor='rgba(34, 34, 34, 1)',
    paper_bgcolor='rgba(34, 34, 34, 1)',
    font=dict(color='white')
)
# Show the figures
fig1.show()
fig2.show()

#Store the plots
#pio.write_image(fig1, 'production_center_barchart.png')
#pio.write_image(fig2, 'production_center_vs_lead_time.png')

############################################
# Analysis on Material Type #
############################################
material_types = df['MATERIAL_TYPE_CAT'].value_counts()
material_type_lead_times = df.groupby('MATERIAL_TYPE_CAT')['LEAD_TIME'].mean()

# Create bar charts
fig3 = go.Figure(data=[go.Bar(
    x=material_types.index,
    y=material_types.values,
    marker_color='rgb(76, 175, 80)',
    text=material_types.values,
    textposition='outside',
    width=0.8,
    marker=dict(line=dict(width=2, color='rgba(0,0,0,1)'), opacity=1)
)])
fig3.update_layout(
    title='Distribution of Material Types',
    xaxis_title='Material Types',
    yaxis_title='Order Count',
    plot_bgcolor='rgba(34, 34, 34, 1)',
    paper_bgcolor='rgba(34, 34, 34, 1)',
    font=dict(color='white')
)

fig4 = go.Figure(data=[go.Bar(
    x=material_type_lead_times.index,
    y=material_type_lead_times.values,
    marker_color='rgb(156, 39, 176)',
    text=material_type_lead_times,
    textposition='outside',
    width=0.8,
    marker=dict(line=dict(width=2, color='rgba(0,0,0,1)'), opacity=1)
)])
fig4.update_layout(
    title='Average Lead Time by Material Type',
    xaxis_title='Material Type',
    yaxis_title='Lead Time',
    plot_bgcolor='rgba(34, 34, 34, 1)',
    paper_bgcolor='rgba(34, 34, 34, 1)',
    font=dict(color='white')
)
# Show the figures
fig3.show()
fig4.show()

#Store the plots
#pio.write_image(fig1, 'material_type_barchart.png')
#pio.write_image(fig2, 'material_type_vs_lead_time.png')

############################################
# Analysis on Material Group #
############################################
material_groups = df['MATERIAL_GROUP_CAT'].value_counts()
material_group_lead_times = df.groupby('MATERIAL_GROUP_CAT')['LEAD_TIME'].mean()

# Create bar charts
fig5 = go.Figure(data=[go.Bar(
    x=material_groups.index,
    y=material_groups.values,
    marker_color='rgb(76, 175, 80)',
    text=material_groups.values,
    textposition='outside',
    width=0.8,
    marker=dict(line=dict(width=2, color='rgba(0,0,0,1)'), opacity=1)
)])
fig5.update_layout(
    title='Distribution of Material Groups',
    xaxis_title='Material Group',
    yaxis_title='Order Count',
    plot_bgcolor='rgba(34, 34, 34, 1)',
    paper_bgcolor='rgba(34, 34, 34, 1)',
    font=dict(color='white')
)

fig6 = go.Figure(data=[go.Bar(
    x=material_group_lead_times.index,
    y=material_group_lead_times.values,
    marker_color='rgb(156, 39, 176)',
    text=material_group_lead_times,
    textposition='outside',
    width=0.8,
    marker=dict(line=dict(width=2, color='rgba(0,0,0,1)'), opacity=1)
)])
fig6.update_layout(
    title='Average Lead Time by Material Group',
    xaxis_title='Material Group',
    yaxis_title='Lead Time',
    plot_bgcolor='rgba(34, 34, 34, 1)',
    paper_bgcolor='rgba(34, 34, 34, 1)',
    font=dict(color='white')
)
# Show the figures
fig5.show()
fig6.show()

#Store the plots
#pio.write_image(fig5, 'material_group_barchart.png')
#pio.write_image(fig6, 'material_group_vs_lead_time.png')

############################################
# Analysis on Order Plan #
############################################
order_plans = df['ORDER_PLAN'].value_counts()
order_plan_lead_times = df.groupby('ORDER_PLAN')['LEAD_TIME'].mean()

# Create bar charts
fig7 = go.Figure(data=[go.Bar(
    x=order_plans.index,
    y=order_plans.values,
    marker_color='rgb(76, 175, 80)',
    text=order_plans.values,
    textposition='outside',
    width=0.8,
    marker=dict(line=dict(width=2, color='rgba(0,0,0,1)'), opacity=1)
)])
fig7.update_layout(
    title='Distribution of Order Plans',
    xaxis_title='Order Plan',
    yaxis_title='Order Count',
    plot_bgcolor='rgba(34, 34, 34, 1)',
    paper_bgcolor='rgba(34, 34, 34, 1)',
    font=dict(color='white')
)

fig8 = go.Figure(data=[go.Bar(
    x=order_plan_lead_times.index,
    y=order_plan_lead_times.values,
    marker_color='rgb(156, 39, 176)',
    text=order_plan_lead_times,
    textposition='outside',
    width=0.8,
    marker=dict(line=dict(width=2, color='rgba(0,0,0,1)'), opacity=1)
)])
fig8.update_layout(
    title='Average Lead Time by Order Plan',
    xaxis_title='Order Plan',
    yaxis_title='Lead Time',
    plot_bgcolor='rgba(34, 34, 34, 1)',
    paper_bgcolor='rgba(34, 34, 34, 1)',
    font=dict(color='white')
)
# Show the figures
fig7.show()
fig8.show()

#Store the plots
#pio.write_image(fig7, 'order_plan_barchart.png')
#pio.write_image(fig8, 'order_plan_vs_lead_time.png')

############################################
# Analysis on NATIONALIZATION INDICATOR #
############################################
nat_ind = df['NATIONALIZATION INDICATOR'].value_counts()
national_ind = df.groupby('NATIONALIZATION INDICATOR')['LEAD_TIME'].mean()

# Create bar charts
fig10 = go.Figure(data=[go.Bar(
    x=nat_ind.index,
    y=nat_ind.values,
    marker_color='rgb(76, 175, 80)',
    text=nat_ind.values,
    textposition='outside',
    width=0.8,
    marker=dict(line=dict(width=2, color='rgba(0,0,0,1)'), opacity=1)
)])
fig10.update_layout(
    title='Distribution of Nationalization Indicator',
    xaxis_title='Nationalization Indicator',
    yaxis_title='Order Count',
    plot_bgcolor='rgba(34, 34, 34, 1)',
    paper_bgcolor='rgba(34, 34, 34, 1)',
    font=dict(color='white')
)
fig10.show()
#pio.write_image(fig10, 'national_ind_barchart.png')

fig9 = px.bar(x=national_ind.index, y=national_ind.values, title='Average Sale Price by Nationalization Indicator',
              color_discrete_sequence=['purple', 'green'], text=national_ind.values,
              template='plotly_dark')

fig9.update_traces(texttemplate='%{text:,.2f}', textposition='outside')
fig9.update_yaxes(title='Lead Time', tickformat=',')
fig9.update_xaxes(title='Nationalization Indicator')
fig9.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

fig9.show()
#pio.write_image(fig9, 'national_ind_vs_lead_time.png')

############################################
# Analysis on ORDER_PRIORITY #
############################################
priority = df['ORDER_PRIORITY'].value_counts()
priority_lead_times = df.groupby('ORDER_PRIORITY')['LEAD_TIME'].mean()

# Create bar charts
fig11 = go.Figure(data=[go.Bar(
    x=priority.index,
    y=priority.values,
    marker_color='rgb(76, 175, 80)',
    text=priority.values,
    textposition='outside',
    width=0.8,
    marker=dict(line=dict(width=2, color='rgba(0,0,0,1)'), opacity=1)
)])
fig11.update_layout(
    title='Distribution of Order Priority',
    xaxis_title='Order Priority',
    yaxis_title='Order Count',
    plot_bgcolor='rgba(34, 34, 34, 1)',
    paper_bgcolor='rgba(34, 34, 34, 1)',
    font=dict(color='white')
)
fig11.show()
#pio.write_image(fig11, 'priority_barchart.png')

fig12 = px.bar(x=priority_lead_times.index, y=priority_lead_times.values, title='Average Sale Price by Order Priority',
              color_discrete_sequence=['purple', 'green'], text=priority_lead_times.values,
              template='plotly_dark')

fig12.update_traces(texttemplate='%{text:,.2f}', textposition='outside')
fig12.update_yaxes(title='Lead Time', tickformat=',')
fig12.update_xaxes(title='Order Priority')
fig12.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

fig12.show()
#pio.write_image(fig12, 'priority_vs_lead_time.png')

############################################
# Analysis on IS_ALL_COMPONENTS_AVAILABLE #
############################################
components_available = df['IS_ALL_COMPONENTS_AVAILABLE'].value_counts()
components_available_lead_times = df.groupby('IS_ALL_COMPONENTS_AVAILABLE')['LEAD_TIME'].mean()

# Create bar charts
fig13 = go.Figure(data=[go.Bar(
    x=components_available.index,
    y=components_available.values,
    marker_color='rgb(76, 175, 80)',
    text=components_available.values,
    textposition='outside',
    width=0.8,
    marker=dict(line=dict(width=2, color='rgba(0,0,0,1)'), opacity=1)
)])
fig13.update_layout(
    title='Distribution of All the Components Ready or Not',
    xaxis_title='Component Availability',
    yaxis_title='Order Count',
    plot_bgcolor='rgba(34, 34, 34, 1)',
    paper_bgcolor='rgba(34, 34, 34, 1)',
    font=dict(color='white')
)
fig13.show()
#pio.write_image(fig13, 'com_availability_barchart.png')

fig14 = px.bar(x=priority_lead_times.index, y=priority_lead_times.values, title='Average Sale Price by Component Availability',
              color_discrete_sequence=['purple', 'green'], text=priority_lead_times.values,
              template='plotly_dark')

fig14.update_traces(texttemplate='%{text:,.2f}', textposition='outside')
fig14.update_yaxes(title='Lead Time', tickformat=',')
fig14.update_xaxes(title='Component Availability')
fig14.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

fig14.show()
#pio.write_image(fig14, 'com_availability_vs_lead_time.png')

###########################################
# Analysis on CONTRACTOR_INVOLVED #
###########################################
contractor_available = df['CONTRACTOR_INVOLVED'].value_counts()
contractor_available_lead_times = df.groupby('CONTRACTOR_INVOLVED')['LEAD_TIME'].mean()

# Create bar charts
fig15 = go.Figure(data=[go.Bar(
    x=contractor_available.index,
    y=contractor_available.values,
    marker_color='rgb(76, 175, 80)',
    text=contractor_available.values,
    textposition='outside',
    width=0.8,
    marker=dict(line=dict(width=2, color='rgba(0,0,0,1)'), opacity=1)
)])
fig15.update_layout(
    title='Distribution of Contractor Availability',
    xaxis_title='Contractor Availability',
    yaxis_title='Order Count',
    plot_bgcolor='rgba(34, 34, 34, 1)',
    paper_bgcolor='rgba(34, 34, 34, 1)',
    font=dict(color='white')
)
fig15.show()
#pio.write_image(fig15, 'contractor_availability_barchart.png')

fig16 = px.bar(x=priority_lead_times.index, y=priority_lead_times.values, title='Average Sale Price by Contractor Availability',
              color_discrete_sequence=['purple', 'green'], text=priority_lead_times.values,
              template='plotly_dark')

fig16.update_traces(texttemplate='%{text:,.2f}', textposition='outside')
fig16.update_yaxes(title='Lead Time', tickformat=',')
fig16.update_xaxes(title='Contractor Availability')
fig16.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

fig16.show()
#pio.write_image(fig16, 'contractor_availability_vs_lead_time.png')

###########################################
# Analysis on IS_PROJECT #
###########################################
under_project_lead_times = df.groupby('IS_PROJECT')['LEAD_TIME'].mean()
fig17 = px.bar(x=priority_lead_times.index, y=priority_lead_times.values, title='Average Sale Price by Project Availability',
              color_discrete_sequence=['purple', 'green'], text=priority_lead_times.values,
              template='plotly_dark')

fig17.update_traces(texttemplate='%{text:,.2f}', textposition='outside')
fig17.update_yaxes(title='Lead Time', tickformat=',')
fig17.update_xaxes(title='Project Availability')
fig17.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

fig17.show()
#pio.write_image(fig17, 'project_availability_vs_lead_time.png')
########################### Categorical Feature Analysis ####################################

########################### Numerical Feature Analysis ####################################

# Since all the variables are numerical
cols = [col for col in numeric_cols if "LEAD_TIME" not in col]

for col in cols:
    DataAnalysis.plot_numerical_col(df, col)
#All the numerical variables exponentially distributed

# Assuming 'df' is your dataframe and 'numerical_columns' is a list of numerical column names
target_column = 'LEAD_TIME'
# Create a directory to save the plots
os.makedirs('target_analysis_plots', exist_ok=True)

for col in cols:
    # Create scatter plot
    fig = px.scatter(df, x=col, y=target_column, trendline='lowess', title=f'{col} vs {target_column} Analysis')
    fig.update_layout(xaxis_title=col, yaxis_title=target_column)

    # Save plot as PNG
    file_path = os.path.join('target_analysis_plots', f'{col}_vs_{target_column}_analysis.png')
    fig.write_image(file_path)
########################### Numerical Feature Analysis ####################################

###########################################################################################
# Feature engineering
###########################################################################################
df["PROCESS_NUMBER"] = df["PROCESS_NUMBER"].apply(lambda x: x + 1)
df['NATIONAL_INDICATOR'] = df["NATIONAL_INDICATOR"].apply(lambda x: 1 if x == 'X        ' else 0)

df.drop(columns=["ORDER_ID","MATERIAL_ID","MATERIAL_GROUP",
                 "MATERIAL_TYPE","ORDER_TYPE","MATERIAL_TYPE_CAT",
                 'DIFF_STATION_NUM', 'CONTRACTOR_INVOLVED',
                 "PRODUCTION_CENTER", "MATERIAL_GROUP_CAT"],inplace=True)

#categoric_cols, numeric_cols, cat_but_car = DataAnalysis.grab_col_names(df, cat_th=10, car_th=20)

#categoric_cols = ['PRODUCTION_CENTER','MATERIAL_GROUP_CAT']

#df = DataAnalysis.one_hot_encoder(df,categoric_cols)

categoric_cols, numeric_cols, cat_but_car = DataAnalysis.grab_col_names(df, cat_th=10, car_th=20)

df[categoric_cols] = df[categoric_cols].astype(int)

#with open('data.pickle', 'rb') as f:
    #df = pickle.load(f)

# Standartlaştırma
#num_scaled = QuantileTransformer(output_distribution='uniform').fit_transform(df[numeric_cols])
#df[numeric_cols] = pd.DataFrame(num_scaled, columns=df[numeric_cols].columns)

#Correlation Matrix
corr_matrix = df.corr()

# Remove columns that are 0.8 or more correlated
cols_to_drop = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) >= 0.8:
            colname = corr_matrix.columns[j]
            if colname not in cols_to_drop:
                cols_to_drop.append(colname)

y = df["LEAD_TIME"]
X = df.drop(["LEAD_TIME"], axis=1)

################################################################
# 3. Modeling & Hyperparameter Optimization
################################################################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)

#categoric_cols_train, numeric_cols_train, cat_but_car = DataAnalysis.grab_col_names(X_train, cat_th=10, car_th=20)
#DataAnalysis.replace_with_thresholds(X_train, "QUANTITY")
#DataAnalysis.replace_with_thresholds(X_train, "PROCESS_NUMBER")
#DataAnalysis.replace_with_thresholds(X_train, "MANUAL_LABOUR_HOUR")
#DataAnalysis.replace_with_thresholds(X_train, "MACHINE_LABOUR_HOUR")
#DataAnalysis.replace_with_thresholds(X_train, "Z4_NOTIF_NUM")
#DataAnalysis.replace_with_thresholds(X_train, "Z6_NOTIF_NUM")


############################################# XGBoost with Optuna R2 #############################################
# Assuming X_train, X_test, y_train, y_test are your training and test sets
def objective(trial):
    params = {
        'objective': 'reg:squarederror',  # Use 'reg:squarederror' for regression
        'tree_method': 'hist',
        'n_estimators': 100,
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10)
    }

    model = XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10, verbose=False)

    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    return r2

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

best_params = study.best_params
print(f"Best Hyperparameters: {best_params}")

# Train the final model with the best hyperparameters
final_model = XGBRegressor(**best_params)
final_model.fit(X_train, y_train)
joblib.dump(final_model, 'final_model_v4.joblib')

# Evaluate the final model
predictions = final_model.predict(X_test)
final_r2 = r2_score(y_test, predictions)
print(f"Final Model R2: {final_r2}")
#Final Model R2: 0.34014222520082116

control = pd.DataFrame({'y': y_test,'pred':predictions})

best_indexes = [35513, 356496, 223536, 184796, 359567, 131496, 324036, 218528, 214152, 321389,
                356895, 290601, 3135, 91997, 263604, 182440, 81677, 90362]




# external libraries for visualizations and data manipulation
import pandas as pd        
import numpy as np      
# external libraries used for visualization            
import matplotlib.pyplot as plt    
import seaborn as sns              # pretty plotting
# Additional library installed for added visuals
import plotly.offline              # scatterplot matrix
import plotly.figure_factory as ff

# correlation heat map setup for seaborn provided by imported py code in zip file
def corr_chart(df_corr):
    corr=df_corr.corr()
    #screen top half to get a triangle
    top = np.zeros_like(corr, dtype=np.bool)
    top[np.triu_indices_from(top)] = True
    fig=plt.figure()
    fig, ax = plt.subplots(figsize=(12,12))
    sns.heatmap(corr, mask=top, cmap='coolwarm', 
        center = 0, square=True, 
        linewidths=.5, cbar_kws={'shrink':.5}, 
        annot = True, annot_kws={'size': 9}, fmt = '.3f')           
    plt.xticks(rotation=45) # rotate variable labels on columns (x axis)
    plt.yticks(rotation=0) # use horizontal variable labels on rows (y axis)
    plt.title('Correlation Heat Map')   
    plt.savefig('plot-corr-map.pdf', 
        bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
        orientation='portrait', papertype=None, format=None, 
        transparent=True, pad_inches=0.25, frameon=None)      

np.set_printoptions(precision=3)

# Options
np.set_printoptions(precision=3)
sns.set(style="whitegrid")

# Functions
def gather(df, key, value, cols):
    # Combine multiple columns into key/value columns
    id_vars = [col for col in df.columns if col not in cols]
    return pd.melt(df, id_vars, cols, key, value)

# read in comma-delimited text file, creating a pandas DataFrame object
# note that IPAddress is formatted as an actual IP address
# but is actually a random-hash of the original IP address
valid_survey_input = pd.read_csv('mspa_survey_data.csv')

# use the RespondentID as label for the rows... the index of DataFrame
valid_survey_input.set_index('RespondentID', drop = True, inplace = True)

# examine the structure of the DataFrame object
print('\nContents of initial survey data ---------------')

# could use len() or first index of shape() to get number of rows/observations
print('\nNumber of Respondents =', len(valid_survey_input)) 

# show the column/variable names of the DataFrame
# note that RespondentID is no longer present
print(valid_survey_input.columns)

# abbreviated printing of the first five rows of the data frame
print(pd.DataFrame.head(valid_survey_input)) 

# shorten the variable/column names for software preference variables
survey_df = valid_survey_input.rename(index=str, columns={
    'Personal_JavaScalaSpark': 'My_Java',
    'Personal_JavaScriptHTMLCSS': 'My_JS',
    'Personal_Python': 'My_Python',
    'Personal_R': 'My_R',
    'Personal_SAS': 'My_SAS',
    'Professional_JavaScalaSpark': 'Prof_Java',
    'Professional_JavaScriptHTMLCSS': 'Prof_JS',
    'Professional_Python': 'Prof_Python',
    'Professional_R': 'Prof_R',
    'Professional_SAS': 'Prof_SAS',
    'Industry_JavaScalaSpark': 'Ind_Java',
    'Industry_JavaScriptHTMLCSS': 'Ind_JS',
    'Industry_Python': 'Ind_Python',
    'Industry_R': 'Ind_R',
    'Industry_SAS': 'Ind_SAS'})
    
# Convert course columns to 1/0 to check total courses
course_cols = [col for col in list(survey_df) if 
               col.startswith('PREDICT') or col.startswith('Other')]
survey_df[course_cols] = survey_df[course_cols].fillna(0)
for col in course_cols:
    survey_df[col] = (
      pd.to_numeric(survey_df[col], errors='coerce').fillna(1)
    )
    survey_df.loc[survey_df[col] > 1, col] = 1

# Find total courses taken by respondent if survey value is NA
survey_df['total_courses'] = survey_df[course_cols].sum(axis=1)
survey_df['Courses_Completed'] = survey_df['Courses_Completed'].fillna(
        survey_df['total_courses'])

print(survey_df.info())
print('\nDescriptive statistics for survey data ---------------')
print(survey_df.describe())

# define subset DataFrame for analysis of software preferences 
software_df = survey_df.loc[:, 'My_Java':'Ind_SAS']
software_df['total_courses'] = survey_df['Courses_Completed']
software_df = pd.DataFrame(software_df.to_records())

# define subset data frame for potiential new courses 
potiential_new_courses_df = survey_df.loc[:, 'Python_Course_Interest':'Systems_Analysis_Course_Interest'].dropna(how='all')

# descriptive statistic for potiential new courses
print("\n------Descriptive statistics for potiential new courses ---------------\n{}"
      .format(potiential_new_courses_df.describe().transpose()))


# Make a dot plot of survey responses with titles
g = sns.PairGrid(software_df.sort_values("total_courses", ascending=True),
                 x_vars=software_df.columns[1:-1], y_vars=["total_courses"],
                 size=10, aspect=.25)
g.map(sns.stripplot, size=10, orient="h",
      palette="GnBu_d", edgecolor="black")
g.set(xlim=(0, 100), xlabel="Preference", ylabel="")
titles = ['java_per', 'js_per', 'python_per', 'r_per', 'sas_per',
          'java_pro', 'js_pro', 'python_pro', 'r_pro', 'sas_pro',
          'java_ind', 'js_ind', 'python_ind', 'r_ind', 'sas_ind']

# Scatter plot of Python/R
fig, ax = plt.subplots()
ax.set_xlabel('Personal Preference for R')
ax.set_ylabel('Personal Preference for Python')
plt.title('R and Python Perferences')
scatter_plot = ax.scatter(survey_df['My_R'], 
    survey_df['My_Python'],
    facecolors = 'none', 
    edgecolors = 'blue') 
plt.show() 

# This is where we create the offline scatter plot matrix
pref_df = gather(software_df.copy(), 'software', 'pref',  
       ['My_Java', 'My_Js', 'My_Python', 'My_R', 'My_SAS',
        'Prof_Java', 'Prof_Js', 'Prof_Python', 'Prof_R', 'Prof_SAS',
        'Ind_Java', 'Ind_Js', 'Ind_Python', 'Ind_R', 'Ind_SAS'])
pref_df[['software','use']] = pref_df['software'].str.split('_', expand=True)   
fig = ff.create_scatterplotmatrix(
        pref_df.iloc[:, 1:], diag='histogram', index='software',
        height=800, width=800
)
plotly.offline.plot(fig, filename='scatter_matrix.html')

# Boxplot of software preferences    
sns.factorplot(x="software", y="pref", col="use", 
               data=pref_df, kind="box")  

pref_stats = pref_df.iloc[:, 2:].groupby(['software','use'], as_index=False)


# examine intercorrelations among software preference variables
# with correlation matrix/heat map
corr_chart(df_corr = software_df) 

# descriptive statistics for software preference variables
print('\nDescriptive statistics for survey data ---------------')
print(software_df.describe())

# descriptive statistics for one variable
print('\nDescriptive statistics for courses completed ---------------')
print(survey_df['Courses_Completed'].describe())

# ----------------------------------------------------------
# transformation code added with version v005
# ----------------------------------------------------------
# transformations a la Scikit Learn
# documentation at http://scikit-learn.org/stable/auto_examples/
#                  preprocessing/plot_all_scaling.html#sphx-glr-auto-
#                  examples-preprocessing-plot-all-scaling-py
# transformations a la Scikit Learn
# select variable to examine, eliminating missing data codes
X = survey_df['Courses_Completed'].dropna()

# Seaborn provides a convenient way to show the effects of transformations
# on the distribution of values being transformed
# Documentation at https://seaborn.pydata.org/generated/seaborn.distplot.html

unscaled_fig, ax = plt.subplots()
sns.distplot(X).set_title('Unscaled')
unscaled_fig.savefig('Transformation-Unscaled' + '.pdf', 
    bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
    orientation='portrait', papertype=None, format=None, 
    transparent=True, pad_inches=0.25, frameon=None)  
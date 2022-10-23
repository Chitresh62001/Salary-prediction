import pandas as pd


df = pd.read_csv("/home/chitresh/Desktop/data science/project/glassdoor_jobs.csv")


'''
salary parsing
company name text only
state field
age of company
parsing of job description
'''
#salary parsing
df['hourly'] = df['Salary Estimate'].apply(lambda x: 1 if 'per hour' in x.lower() else 0)
df['Employer_provided'] = df['Salary Estimate'].apply(lambda x: 1 if 'employer provided salary' in x.lower() else 0)


df = df[df['Salary Estimate'] != '-1']
salary = df['Salary Estimate'].apply(lambda x: x.split("(")[0].replace('K', '').replace('$', ''))
min_hr = salary.apply(lambda x: x.lower().replace('per hour', '').replace('employer provided salary:', ''))

df['min_salary'] = min_hr.apply(lambda x: int(x.split('-')[0]))
df['max_salary'] = min_hr.apply(lambda x: int(x.split('-')[1]))
df['avg_salary'] = (df['min_salary']+df['max_salary'])/2


#text only in company name
df['company_txt'] = df.apply(lambda x: x['Company Name'] if x['Rating']<0 else x['Company Name'][:-3], axis = 1)


#job location in which state
df['job_state'] = df['Location'].apply(lambda x: x.split(',')[1])
df['same_sate'] = df.apply(lambda x: 1 if x.Location == x.Headquarters else 0, axis = 1)

#age of the company
df['age'] = df.Founded.apply(lambda x: x if x<1 else 2022-x)

#parsing job description
df['Job Description'][0]

#python
df['python_yn'] = df['Job Description'].apply(lambda x: 1 if 'python' in x.lower() else 0) 
df.python_yn.value_counts() 

#rstudio
df['r_yn'] = df['Job Description'].apply(lambda x: 1 if 'r studio' in x.lower() else 0)
df.r_yn.value_counts() 

#spark
df['spark_yn'] = df['Job Description'].apply(lambda x: 1 if 'spark' in x.lower() else 0) 
df.spark_yn.value_counts() 

#aws
df['aws_yn'] = df['Job Description'].apply(lambda x: 1 if 'aws' in x.lower() else 0) 
df.aws_yn.value_counts() 

#excel
df['excel_yn'] = df['Job Description'].apply(lambda x: 1 if 'excel' in x.lower() else 0) 
df.excel_yn.value_counts() 

df_out = df.drop(['Unnamed: 0'], axis = 1)
df_out.to_csv("Salary_data_cleaned.csv", index = False)

pd.read_csv("Salary_data_cleaned.csv")





















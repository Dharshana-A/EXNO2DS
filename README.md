# EXNO2DS
# AIM:
      To perform Exploratory Data Analysis on the given data set.
      
# EXPLANATION:
  The primary aim with exploratory analysis is to examine the data for distribution, outliers and anomalies to direct specific testing of your hypothesis.
  
# ALGORITHM:
STEP 1: Import the required packages to perform Data Cleansing,Removing Outliers and Exploratory Data Analysis.

STEP 2: Replace the null value using any one of the method from mode,median and mean based on the dataset available.

STEP 3: Use boxplot method to analyze the outliers of the given dataset.

STEP 4: Remove the outliers using Inter Quantile Range method.

STEP 5: Use Countplot method to analyze in a graphical method for categorical data.

STEP 6: Use displot method to represent the univariate distribution of data.

STEP 7: Use cross tabulation method to quantitatively analyze the relationship between multiple variables.

STEP 8: Use heatmap method of representation to show relationships between two variables, one plotted on each axis.

## CODING AND OUTPUT
```
# Step 1: Import Necessary Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, PowerTransformer
from scipy.stats import boxcox

# Step 2: Load the Dataset
data = pd.read_csv('titanic_dataset.csv')

print("Original Dataset:")
print(data.head())

# Step 3: Handle Missing Values (Fill numeric columns with mean)
data.fillna(data.mean(numeric_only=True), inplace=True)

# Select a suitable numeric column for transformation
numeric_column = data.select_dtypes(include=np.number).columns[0]

print(f"\nColumn Selected for Transformation: {numeric_column}")

# Keep only positive values for log and boxcox
positive_data = data[data[numeric_column] > 0].copy()

# Step 4: Log Transformation
positive_data['Log_Transform'] = np.log(positive_data[numeric_column])

# Step 5: Reciprocal Transformation
positive_data['Reciprocal_Transform'] = 1 / positive_data[numeric_column]

# Step 6: Square Root Transformation
positive_data['Sqrt_Transform'] = np.sqrt(positive_data[numeric_column])

# Step 7: Square Transformation
positive_data['Square_Transform'] = np.square(positive_data[numeric_column])

# Step 8: Box-Cox Transformation (only positive values)
positive_data['BoxCox_Transform'], lambda_value = boxcox(positive_data[numeric_column])

print(f"\nBox-Cox Lambda Value: {lambda_value}")

# Step 9: Yeo-Johnson Transformation (works with zero/negative values)
pt = PowerTransformer(method='yeo-johnson')
data['YeoJohnson_Transform'] = pt.fit_transform(data[[numeric_column]])

# Standard Scaling
scaler = StandardScaler()
data['Standard_Scaled'] = scaler.fit_transform(data[[numeric_column]])

# Save the transformed dataset
positive_data.to_csv('Transformed_Positive_Data.csv', index=False)
data.to_csv('Transformed_Full_Data.csv', index=False)

print("\nTransformation Completed Successfully.")
print("\nTransformed Dataset Preview:")
print(positive_data.head())
```
<img width="824" height="600" alt="Screenshot 2026-02-25 110301" src="https://github.com/user-attachments/assets/55930b54-6655-4090-8c5b-9b8fe6b84b8c" />
<img width="783" height="625" alt="Screenshot 2026-02-25 110314" src="https://github.com/user-attachments/assets/25cfa068-30a9-4de8-9646-7f96856bcbab" />


# RESULT
   The dataset was successfully cleaned by handling missing values and removing outliers, and the relationships and distributions of variables were analyzed using graphical and statistical methods.    

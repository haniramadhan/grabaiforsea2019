# grabaiforsea2019

Summary of work on AI for SEA 2019 - Safety Challenge

Name: Hani Ramadhan

Libraries: 
- Data processing: pandas, numpy
- Plotting: matplotlib, seaborn
- Machine learning: 
    + sklearn
    + xgboost, can be obtained in pip
    + hyperopt (optimization), source: [http://hyperopt.github.io/hyperopt/]

Work summary:
- Data cleaning
    + Duplicate labels
    + Negative speeds
    + Non-consecutive seconds in trips
- Data preprocessing
    + Data smoothing using simple moving window average
    + Creating new features
- Exploratory Data Analysis
    + Investigating the label distribution on each features
    + Checks out the trip length distribution without outliers
- Experiments and Evaluation
    + Optimizing the Random Forest and XGBoost
    + Comparing the performance between the unsmoothed and smoothed data
    + Evaluation on selected features
- Description of real-time safe/unsafe driving detection use case
- Final thoughts

Full description of the codes: [Notebook](./Hani Ramadhan - AI for SEA - Safety.ipynb)

Only code: [./main.py]

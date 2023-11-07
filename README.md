# Life Expectancy, Healthcare Expenditures, and Universal Healthcare Coverage

## Overview

Thank you for checking out my work! Analyzing the life expectancy and health expenditure of various health care systems is important to understanding policy decisions to promote public health equity and outcomes. This application provides exploratory analysis and visualization of the relation between life expectancy at birth and health expenditure. Additionally, regression is done to forecast this relation. Finally, there is comparison between countries with universal healthcare coverage (universal government-funded health systems, often referred to as single-payer health care) and those without.

![Recording 2023-11-07 at 09 47 51](https://github.com/CarruthT/Health-Expenditure-and-Life-Expectancy-Data-Analysis-and-Web-Scraping/assets/97051391/c95e5e2f-49ed-410d-a645-98f004688cd7)


## Usage

To utlize this application you must download the excel file titled "CountryLE_EXP_UHC.csv" and place it in a directory with the dowloaded application "LEHE.py". Then through terminal activate a python version with the required packages (described below) and navigate to the relevant directory with the two files. From here run the command "streamlit run LEHE.py" and the application should appear in a browser window.

## Features

The application displays two buttons. One button to display the health expenditure and life expectancy of countries with those having universal health care coverage colored differently. The other displays the same information with a log scaling of the health expenditure. Additionally both plots have a regression between the life expectancy and health expenditure data. Below these plots there are statistics regarding universal health care coverage to compare life expectancy and health expenditure in countries with the different health systems and model information such as the performance and coefficients.

## Packages

This application runs on streamlit using the python programming language. The following packages are utilized in the application: pandas, numpy, streamlit, matplotlib.pyplot, plotly.express, plotly.graph_objs, sklearn.linear_model, and sklearn.metrics. 

The webscraping notebook makes use of the following packages: bs4, requests, pandas, and numpy. 

There is a relevant requirements.txt file also included to review the exact packages in use on my machine the time of upload.
## Data Sources

Data was webscraped using beautifulsoup4 from wikipedia pages as seen in the datacollection.ipynb. 
Source links are provided below:

Data on Life expectancy can be found here from the World Health Organization:https://apps.who.int/gho/data/node.main.688

Data on health expenditure can be found here from the World Health Organization:https://apps.who.int/nha/database/Select/Indicators/en

Data for universal health care coverage was from multiple sources all cited on the wikipedia page: https://en.wikipedia.org/wiki/Health_care_systems_by_country

It should be noted that some data was excluded for the purpose of comparison. Regions not present on all data sets were filtered out.

## Contributing

All work was done by Trystan Carruth

## Contact

If you wish to contact me, my email is: trystaned@gmail.com

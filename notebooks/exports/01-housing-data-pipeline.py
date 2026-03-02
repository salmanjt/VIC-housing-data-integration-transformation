# %% [markdown]
# # <a id='toc1_'></a>[Data Integration & Transformation](#toc0_)
#
# **Author:** Salman Tahir
# **Environment:** Conda 23.7.2, Python 3.10.12
#
# ---
#

# %% [markdown]
# **Table of contents**<a id='toc0_'></a>
#
# - [Introduction](#toc2_)
# - [Pre-defined Information](#toc3_)
# - [Importing Libraries](#toc4_)
# - [Data Loading](#toc5_)
#   - [Loading Sample data](#toc5_1_)
#   - [Loading XML data](#toc5_2_)
#   - [Loading JSON data](#toc5_3_)
#   - [Loading Shapefile data](#toc5_4_)
#   - [Loading GTFS data](#toc5_5_)
#   - [Loading PDF data](#toc5_6_)
# - [Data Integration](#toc6_)
#   - [Integrating XML & JSON data](#toc6_1_)
#   - [Integrating Data from Sources](#toc6_2_)
#   - [Scraping Website for Housing Data](#toc6_3_)
#   - [Addressing Duplication Issues](#toc6_4_)
# - [Data Export](#toc7_)
# - [Data Transformation](#toc8_)
#   - [Preparing Data for Analysis](#toc8_1_)
#   - [Standardization](#toc8_2_)
#   - [Normalization](#toc8_3_)
#   - [Log Transformation](#toc8_4_)
#   - [Power Transformation](#toc8_5_)
#   - [Box-Cox Transformation](#toc8_6_)
#   - [Effect of Data Transformation on Linear Regression Models](#toc8_7_)
# - [Summary](#toc9_)
# - [References](#toc10_)
#
# <!-- vscode-jupyter-toc-config
# 	numbering=false
# 	anchor=true
# 	flat=false
# 	minLevel=1
# 	maxLevel=2
# 	/vscode-jupyter-toc-config -->
# <!-- THIS CELL WILL BE REPLACED ON TOC UPDATE. DO NOT WRITE YOUR TEXT IN THIS CELL -->
#

# %% [markdown]
# # <a id='toc2_'></a>[Introduction](#toc0_)
#
# Data integration is a crucial process that involves merging data from multiple sources to create a unified and comprehensive view. In this project, we will focus on integrating diverse datasets pertaining to housing information in Victoria, Australia. These datasets originate from various sources and are presented in different formats.
#
# Our primary objective is to merge these datasets into a cohesive mediated schema or dataset. Throughout the integration process, we will prioritise ensuring the validity and integrity of the data by cross-referencing it with the provided sample output. Additionally, we will address data cleaning requirements where necessary. Lastly, we explore the impact of employing various data transformation methods and techniques on the final dataset, highlighting their effects and benefits.
#

# %% [markdown]
# # <a id='toc3_'></a>[Pre-defined Information](#toc0_)
#
# We are provided with the following data sources:
#
# - CSV file (Sample Output)
#   - `sample_output.csv`
# - XML file
#   - `properties.xml`
# - JSON file
#   - `properties.json`
# - Vic_suburb_boundary (directory), which contains the following files:
#   - `VIC_LOCALITY_POLYGON_shp.dbf`
#   - `VIC_LOCALITY_POLYGON_shp.prj`
#   - `VIC_LOCALITY_POLYGON_shp.shp`
#   - `VIC_LOCALITY_POLYGON_shp.shx`
# - Vic_GTFS_data (directory) -> metropolitan (subdirectory), which contains the following files:
#   - `agency.txt`
#   - `calendar_dates.txt`
#   - `calendar.txt`
#   - `routes.txt`
#   - `shapes.txt`
#   - `stop_times.txt`
#   - `stops.txt`
#   - `trips.txt`
# - PDF file containing information about the LGA to suburb mapping in Victoria, Australia.
#   - `Lga_to_suburb.pdf`
# - Webpage containing housing information that we need to scrape.
#   - `http://house.speakingsame.com/`
#
# We are required to integrate data from these sources into a single mediated dataset that contains the same attributes as in the sample output file.
#
# Furthermore, we have the following notes to consider from the project brief:
#
# > If you decide not to calculate any of the required columns, then you must still have that column in your final DataFrame with all the values as the Default value: "`NA`" (string).
#
# > Direct journey means that you can reach Melbourne Central Station without changing your train at any point in the journey. So, when you board the train at the closest station, you can directly go to the Melbourne Central Station without moving to another vehicle.
#
# > The only external source of information is the webpage we need to scrape. All other information is provided in the input files.
#
# > For Haversine distance, use 6378 km as the radius of the earth.
#

# %% [markdown]
# # <a id='toc4_'></a>[Importing Libraries](#toc0_)
#
# In this section, we import several libraries that are essential for our data integration and analysis tasks.
#
# - `folium`: visualising geospatial data.
# - `geopandas`: working with geospatial data, allowing us to handle geographic data and perform spatial operations.
# - `matplotlib.pyplot`: plotting library to create graphs and charts.
# - `numpy`: numerical computing that provides efficient mathematical operations and array manipulations.
# - `os`: provides functions for interacting with the operating system
# - `pandas`: provides high-performance, easy-to-use data structures and data analysis tools.
# - `re`: regular expression operations.
# - `requests`: making HTTP requests.
# - `time`: time-related functions.
# - `warnings`: used for handling warnings.
# - `xml.etree.ElementTree`: parsing and manipulating XML files.
# - `BeautifulSoup`: parsing HTML and XML documents, extracting data from web pages.
# - `math`: provides mathematical functions and constants, such as for calculating the Haversine distance.
# - `pdfminer.high_level`: extracting text from PDF documents.
# - `shapely.geometry.Point`: defines geometric objects and operations related to points in space.
# - `sklearn`: several libraries were imported for machine learning algorithms and tools for data mining.
#
# Additionally, we set the plotting style `plt.style.use('seaborn-v0_8')` to enhance the visual appearance of our plots.
#
# We also adjust the display options of Pandas using `pd.set_option` to show all rows and columns in our DataFrames.
#

# %%
# Importing libraries
import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import requests
import seaborn as sns
import time
import warnings
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from math import radians, cos, sin, asin, sqrt
from pdfminer.high_level import extract_text
from shapely.geometry import Point
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Plotting style
plt.style.use("seaborn-v0_8")
sns.set_palette("muted")

# Pandas display options
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

# %% [markdown]
# # <a id='toc5_'></a>[Data Loading](#toc0_)
#
# We import data from various sources into Pandas `DataFrames` that can be used for further processing or analysis.
#
# Each dataset format requires a specific approach for loading the data.
#
# - <span style="color:green">CSV</span> _(Sample Output)_
#   - The pandas `read_csv` function is used to load data from CSV files into a DataFrame.
# - <span style="color:green">XML</span>
#   - The `xml.etree.ElementTree` library is used to parse the XML file.
#   - The XML data is then cleaned and loaded into a DataFrame using pandas' `read_xml` function.
# - <span style="color:green">JSON</span>
#   - The pandas `read_json` function is used to load data from JSON files into a DataFrame.
# - <span style="color:green">Shapefile</span>
#   - The geopandas `read_file` function is used to load data from Shapefiles into a GeoDataFrame.
# - <span style="color:green">GTFS</span>
#   - The pandas `read_csv` function is used to load data from GTFS files into multiple DataFrames.
# - <span style="color:green">PDF</span>
#   - The pdfminer library is used to extract text from the PDF file.
#   - The extracted text is then processed and loaded into a DataFrame.
#

# %% [markdown]
# ## <a id='toc5_1_'></a>[Loading Sample data](#toc0_)
#

# %%
# Set path to sample file
FILE_PATH = "../data/sample/sample_output.csv"

# Read CSV file
df_sample = pd.read_csv(FILE_PATH)

# %%
print(df_sample.shape)
df_sample.head()

# %% [markdown]
# ## <a id='toc5_2_'></a>[Loading XML data](#toc0_)
#

# %%
# Set path to XML file
FILE_PATH = "../data/input/properties.xml"

# Read XML file
with open(FILE_PATH, "r") as file:
    try:
        tree = ET.parse(file)
    except Exception as e:
        print(e)

# %% [markdown]
# We have come across an error associated with the XML file as it is not valid.
#
# After examining the structure of the file, we have identified the following:
#
# - The file is missing a root element
# - The file is missing an XML declaration
# - The ampersand character `'&'` is not escaped
#
# We fix these issues and read the XML file to a DataFrame.
#

# %%
# Read XML file
with open(FILE_PATH, "r") as file:
    xml_data = file.read()

# Replace ampersand with entity reference
xml_data = xml_data.replace("&", "&amp;")

# Add root element and XML declaration
xml_data = f"<root>\n{xml_data}\n</root>"
xml_data = '<?xml version="1.0" encoding="UTF-8"?>\n' + xml_data

# Read XML file to DataFrame
df_xml = pd.read_xml(xml_data, xpath="//property")

# %%
print(df_xml.shape)
df_xml.head()

# %% [markdown]
# ## <a id='toc5_3_'></a>[Loading JSON data](#toc0_)
#

# %%
# Set path to JSON file
FILE_PATH = "../data/input/properties.json"

# Read JSON file to DataFrame
df_json = pd.read_json(FILE_PATH)

# %%
print(df_json.shape)
df_json.head()

# %% [markdown]
# ## <a id='toc5_4_'></a>[Loading Shapefile data](#toc0_)
#

# %%
# Set path to  Shapefile
FILE_PATH = "../data/input/Vic_suburb_boundary/VIC_LOCALITY_POLYGON_shp.shp"

# Read shapefile to GeoDataFrame
gdf_shapefile = gpd.read_file(FILE_PATH)

# %%
print(gdf_shapefile.crs)
print(gdf_shapefile.shape)
gdf_shapefile.head()

# %% [markdown]
# We can also plot the data on a map to display the boundaries of the suburbs, ensuring that the data is loaded correctly and validates our expectations regarding data belonging to Victoria, Australia.
#

# %%
# Plot shapefile
gdf_shapefile.plot()

# Customise plot
plt.title("Victorian Suburbs Shapefile")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

# %% [markdown]
# ## <a id='toc5_5_'></a>[Loading GTFS data](#toc0_)
#
# For each file in the GTFS directory, we set the `FILE_PATH` variable to the name and location of the file and read it into a DataFrame.
#

# %%
# Set path to GTFS file
FILE_PATH = "../data/input/Vic_GTFS_data/metropolitan/agency.txt"

# Read GTFS file to DataFrame
df_agency = pd.read_csv(FILE_PATH)

print(df_agency.shape)
df_agency.head()

# %%
# Set path to GTFS file
FILE_PATH = "../data/input/Vic_GTFS_data/metropolitan/calendar_dates.txt"

# Read GTFS file to DataFrame
df_calendar_dates = pd.read_csv(FILE_PATH)

print(df_calendar_dates.shape)
df_calendar_dates.head()

# %%
# Set path to GTFS file
FILE_PATH = "../data/input/Vic_GTFS_data/metropolitan/calendar.txt"

# Read GTFS file to DataFrame
df_calendar = pd.read_csv(FILE_PATH)

print(df_calendar.shape)
df_calendar.head()

# %%
# Set path to GTFS file
FILE_PATH = "../data/input/Vic_GTFS_data/metropolitan/routes.txt"

# Read GTFS file to DataFrame
df_routes = pd.read_csv(FILE_PATH)

print(df_routes.shape)
df_routes.head()

# %%
# Set path to GTFS file
FILE_PATH = "../data/input/Vic_GTFS_data/metropolitan/shapes.txt"

# Read GTFS file to DataFrame
df_shapes = pd.read_csv(FILE_PATH)

print(df_shapes.shape)
df_shapes.head()

# %%
# Set path to GTFS file
FILE_PATH = "../data/input/Vic_GTFS_data/metropolitan/stop_times.txt"

# Read GTFS file to DataFrame
df_stop_times = pd.read_csv(FILE_PATH)

print(df_stop_times.shape)
df_stop_times.head()

# %%
# Set path to GTFS file
FILE_PATH = "../data/input/Vic_GTFS_data/metropolitan/stops.txt"

# Read GTFS file to DataFrame
df_stops = pd.read_csv(FILE_PATH)

print(df_stops.shape)
df_stops.head()

# %%
# Set path to GTFS file
FILE_PATH = "../data/input/Vic_GTFS_data/metropolitan/trips.txt"

# Read GTFS file to DataFrame
df_trips = pd.read_csv(FILE_PATH)

print(df_trips.shape)
df_trips.head()

# %% [markdown]
# ## <a id='toc5_6_'></a>[Loading PDF data](#toc0_)
#
# Now, we'll use regular expressions to find all matches in the extracted PDF text that follow the pattern:
#
# - One or more uppercase letters, spaces, or hyphens
# - Followed by a colon and spaces
# - Followed by a sequence of word characters, spaces, commas, or single quotes enclosed in square brackets
#
# For example in the text `"SUBURB NAME : ['Suburb1', 'Suburb2', 'Suburb3']"`
#
# - First capturing group
#   - `SUBURB NAME`
# - Second capturing group
#   - `'Suburb1', 'Suburb2', 'Suburb3'`.
#
# After extracting the matches, we split the second capturing group by the comma character and remove the square brackets and single quotes from each element in the resulting list, while also removing any leading or trailing whitespace. The resulting list is then used to create a lookup DataFrame that maps each suburb to its corresponding LGA.
#

# %%
# Set path to PDF file
FILE_PATH = "../data/input/Lga_to_suburb.pdf"

# Extract text from PDF
pdf_text = extract_text(FILE_PATH)

# Extract suburb data from text using regular expressions
suburb_data = re.findall(r"([A-Z\s-]+)\s+:\s+\[([\w\s,']+)\]", pdf_text)

# Clean suburb names and create list of tuples
pdf_data = []
for lga, suburbs in suburb_data:
    cleaned_suburbs = []
    for suburb in suburbs.split(","):
        cleaned_suburb = suburb.replace("'", "").strip()
        cleaned_suburbs.append(cleaned_suburb)
    pdf_data.append((lga.strip(), cleaned_suburbs))

# Create a lookup format DataFrame from list of tuples
df_pdf = pd.DataFrame(
    [(suburb, lga) for lga, suburbs in pdf_data for suburb in suburbs],
    columns=["suburb", "lga"],
)

# %%
print(df_pdf.shape)
df_pdf.head()

# %% [markdown]
# # <a id='toc6_'></a>[Data Integration](#toc0_)
#

# %% [markdown]
# ## <a id='toc6_1_'></a>[Integrating XML & JSON data](#toc0_)
#
# We start by integrating the XML and JSON data into a single DataFrame.
#
# First, we identify any missing values in both DataFrames.
#

# %%
# Count of missing values in df_xml
print("Missing values in df_xml:")
print(df_xml.isnull().sum())
print()

# Count of missing values in df_json
print("Missing values in df_json:")
print(df_json.isnull().sum())

# %% [markdown]
# We validate that both DataFrames have the same column names and data types ensuring that we can concatenate them into a single DataFrame.
#

# %%
# Check if column names are the same
print("Column names are the same:", df_xml.columns.equals(df_json.columns))

# Check if data types are the same
print("Data types are the same:", df_xml.dtypes.equals(df_json.dtypes))

# %% [markdown]
# Now, we look for any duplicate values in `df_xml`.
#

# %%
# Show duplicated rows grouped by property_id
df_xml[df_xml.duplicated(keep=False)].sort_values(by="property_id")

# %% [markdown]
# After examining the duplicate values, we find that they are true duplicates as all columns have the same values. Therefore, we can safely remove them from the DataFrame.
#

# %%
# Remove duplicated rows
df_xml = df_xml.drop_duplicates()

# %% [markdown]
# Now, we look for any duplicate values in `df_json`.
#

# %%
# Show duplicated rows grouped by property_id
df_json[df_json.duplicated(keep=False)].sort_values(by="property_id")

# %% [markdown]
# After examining the duplicate values, we find that they are true duplicates as all columns have the same values. Therefore, we can safely remove them from the DataFrame.
#

# %%
# Remove duplicated rows
df_json = df_json.drop_duplicates()

# %% [markdown]
# Now, we concatenate both DataFrames into a single integrated DataFrame.
#

# %%
# Concatenate both DataFrames
df_final = pd.concat([df_xml, df_json])

print(df_final.shape)
df_final.head()

# %% [markdown]
# After concatenating the two DataFrames, we identify and remove any duplicate rows.
#

# %%
# Show duplicated rows grouped by property_id
df_final[df_final.duplicated(keep=False)].sort_values(by="property_id")

# %% [markdown]
# As we can see above, the duplicate rows from both DataFrames are exactly the same. Therefore, we can safely remove them from the DataFrame.
#

# %%
# Remove duplicated rows
df_final = df_final.drop_duplicates(subset="property_id")

df_final.shape

# %% [markdown]
# Now, we can also check for duplicates based on the `addr_street` column, although we will require more information to determine whether these are actual duplicates or not, such as validating the `suburb` names for these properties, once we have integrated the `suburb` data into the DataFrame.
#

# %%
# Show duplicated rows grouped by addr_street
df_final[df_final.duplicated(subset="addr_street", keep=False)].sort_values(
    by="addr_street"
)

# %%
# Store property_id of duplicated rows into a list for later use
addr_duplicates = df_final[df_final.duplicated(subset="addr_street", keep=False)][
    "property_id"
].tolist()
addr_duplicates

# %% [markdown]
# Now, we have a single DataFrame that contains information from both the XML and JSON files.
#

# %%
# Reset index
df_final = df_final.reset_index(drop=True)

print(df_final.shape)

display(df_final.head())

display(df_final.tail())

# %% [markdown]
# ## <a id='toc6_2_'></a>[Integrating Data from Sources](#toc0_)
#

# %% [markdown]
# ### Integrating `suburb`
#
# To integrate the `Suburb` column, we will refer to the `VIC_LOCALITY_POLYGON_shp` GeoDataFrame that we loaded earlier as `gdf_shapefile`.
#
# Instead of manually iterating over the shapes in the shapefile to check if each property's coordinates are within each shape, we can use the `geopandas.sjoin` function to perform a spatial join between the two GeoDataFrames. This function will return a GeoDataFrame that contains the properties from the first GeoDataFrame (`gdf_final`) that intersect with the shapes from the second GeoDataFrame (`gdf_shapefile`). This approach is more efficient than manually iterating over the shapes in the shapefile.
#
# Hence, we first ensure that our `df_final` columns `lat` and `lng` are of type `float`.
#

# %%
df_final.dtypes

# %% [markdown]
# We then convert the `df_final` DataFrame to a GeoDataFrame and create a `geometry` column as a Point object that contains the coordinates of each property using the `lat` and `lng` column values.
#

# %%
# Create a geometry column from lng and lat
geometry = [Point(xy) for xy in zip(df_final["lng"], df_final["lat"])]

# Create a GeoDataFrame from df_final
gdf_final = gpd.GeoDataFrame(df_final, geometry=geometry)

# Validate our created point object
print(geometry[0:1])
print()
print(df_final[["lng", "lat"]].head(1).round(3))

# %% [markdown]
# We spatially join both GeoDataFrames using a left join and create a new column `suburb` in `df_final` that contains the suburb values from the `VIC_LOCA_2` column in `gdf_merged`. Note that we also handle the case where a property is not located in any suburb by setting the `suburb` value to `NA`.
#

# %%
try:
    # Ensure both GeoDataFrames use same CRS
    gdf_final.set_crs(gdf_shapefile.crs, inplace=True)

    # Spatially join both GeoDataFrames
    gdf_merged = gpd.sjoin(gdf_final, gdf_shapefile, how="left", predicate="within")

    # Add 'suburb' column to df_final
    df_final["suburb"] = gdf_merged["VIC_LOCA_2"]

    # Replace any NaN values in 'suburb' column with 'NA'
    df_final["suburb"].fillna("NA", inplace=True)

except Exception as e:
    print(e)
    df_final["suburb"] = "NA"

# %%
# Reset index
df_final = df_final.reset_index(drop=True)

# Check if 'NA' in 'suburb' column
print("NA" in df_final["suburb"].unique())

# %%
print(df_final.shape)
df_final.head()

# %% [markdown]
# ### Integrating `lga`
#
# We refer to the `df_pdf` lookup DataFrame that we created earlier by extracting LGA and suburb information form the provided PDF file.
#

# %%
# Convert suburb column to uppercase
df_pdf["suburb"] = df_pdf["suburb"].str.upper()

df_pdf.head()

# %%
try:
    # Merge df_final with df_pdf based on 'suburb'
    df_final = pd.merge(df_final, df_pdf, on="suburb", how="left")

    # Replace any NaN values in 'lga' column with 'NA'
    df_final["lga"].fillna("NA", inplace=True)

except Exception as e:
    print(e)
    df_final["lga"] = "NA"

# %%
# Check if 'NA' in 'lga' column
print("NA" in df_final["lga"].unique())

# %%
print(df_final.shape)
df_final.head()

# %% [markdown]
# ### Integrating `closest_train_station_id` & `distance_to_closest_train_station`
#
# First, we need to find the closest train station using the Haversine distance formula. We define a custom haversine function with a radius of 6378 km, as specified in the assessment. The haversine function calculates the distance between two points on the Earth's surface using their latitude and longitude coordinates.
#


# %%
def haversine(lon1, lat1, lon2, lat2, radius=6378):
    """
    Calculates the distance between two points on the Earth's surface
    @param lat1: Latitude of first point
    @param lon1: Longitude of first point
    @param lat2: Latitude of second point
    @param lon2: Longitude of second point
    @param radius: Radius of earth in kilometers. Using 6378 km as specified
    @return: Distance between the two points in kilometers
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))

    # Round to 6 decimal places to keep same format as sample data
    return round(radius * c, 6)


# %% [markdown]
# We then calculate the distance from each property in `df_final` to each train station in the `df_stops` DataFrame and select the train station ID to be included in the new `closest_train_station_id` column in `df_final`.
#

# %%
# Initialise columns
df_final["closest_train_station_id"] = np.nan
df_final["distance_to_closest_train_station"] = np.nan

# Iterate over each property in df_final
for i, property in df_final.iterrows():
    # Calculate the distance to all train stations using haversine function
    distances = df_stops.apply(
        lambda station: haversine(
            property["lng"], property["lat"], station["stop_lon"], station["stop_lat"]
        ),
        axis=1,
    )

    # Find the closest station ID and its distance
    closest_station_id = df_stops.iloc[distances.idxmin()]["stop_id"]
    min_distance = distances.min()

    # Assign values to each property
    df_final.at[i, "closest_train_station_id"] = closest_station_id
    df_final.at[i, "distance_to_closest_train_station"] = min_distance

# %% [markdown]
# We now replace any NaN values in the new column with `NA` and convert the `closest_train_station_id` column to an integer type, as specified in the sample output file.
#

# %%
# Replace any NaN values with 'NA' as specified
df_final.update(
    df_final[["closest_train_station_id", "distance_to_closest_train_station"]].fillna(
        "NA"
    )
)

# Convert 'closest_train_station_id' column to int
df_final["closest_train_station_id"] = df_final["closest_train_station_id"].astype(
    "int64"
)

# %%
# Check if 'NA' in 'closest_train_station_id' and 'distance_to_closest_train_station' columns
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    print("NA" in df_final["closest_train_station_id"].unique())
    print("NA" in df_final["distance_to_closest_train_station"].unique())

# %%
print(df_final.shape)
df_final.head()

# %% [markdown]
# ### Integrating `travel_min_to_MC` & `direct_journey_flag`
#
# We have the following information regarding the `travel_min_to_MC` column from the project brief:
#
# - `MC`= Melbourne Central
# - Rounded average travel time in minutes
# - Direct journeys mean:
#   > You can reach Melbourne Central Station without changing your train at any point in the journey. So, when you board the train at the closest station, you can directly go to the Melbourne Central Station without moving to another vehicle.
# - Direct journeys are:
#   - From the closest train station to "Melbourne Central" station
#   - On all weekdays (Monday to Friday) departing between 7am to 9am
# - If there are no direct journeys, then the value should be set to `no direct trip is available`.
# - If the closest station to a property is Melbourne Central station itself, then the value should be set to `0`.
# - Finally, default value should be set to `NA`.
#
# We also have the following information regarding the `direct_journey_flag` column:
#
# - If there is a direct journey from the closest station to the Melbourne Central station, the flag is set to 1.
# - If there is no direct journey, the flag is set to 0.
# - If the closest station is the Melbourne Central station itself, the flag is also set to 1.
# - Finally, default value should be set to `NA`.
#
# Let's have a look at the DataFrames that we will be working with to calculate the `travel_min_to_MC` column. We loaded these DataFrames earlier in the notebook in the Data Loading section.
#

# %%
display(df_agency.head(2))
display(df_calendar_dates.head(2))
display(df_calendar.head(2))
display(df_routes.head(2))
display(df_shapes.head(2))
display(df_stop_times.head(2))
display(df_stops.head(2))
display(df_trips.head(2))

# %% [markdown]
# We first identify the `Melbourne Central` station in the `df_stops` DataFrame and extract its `stop_id`.
#

# %%
# Check if 'Melbourne Central' is in 'stop_name' of df_stops DataFrame
display(df_stops[df_stops["stop_name"].str.contains("Melbourne Central")])

# Extract the 'stop_id' of 'Melbourne Central Railway Station'
MC_STATION_ID = df_stops[df_stops["stop_name"] == "Melbourne Central Railway Station"][
    "stop_id"
].values[0]

print(MC_STATION_ID)

# %% [markdown]
# We can see from the `df_agency` DataFrame that there is only one agency running the train services in Victoria.
#

# %%
# Get `agency_id` from df_agency
AGENCY_ID = df_agency["agency_id"].values[0]

print(AGENCY_ID)

# %% [markdown]
# Now, using the `agency_id` we can identify the `service_id` for the train services that run on weekdays using the `df_calendar` DataFrame.
#

# %%
WEEKDAY_SERVICE_ID = df_calendar[
    (df_calendar["monday"] == AGENCY_ID)
    & (df_calendar["tuesday"] == AGENCY_ID)
    & (df_calendar["wednesday"] == AGENCY_ID)
    & (df_calendar["thursday"] == AGENCY_ID)
    & (df_calendar["friday"] == AGENCY_ID)
]["service_id"].values

print(WEEKDAY_SERVICE_ID)

# %% [markdown]
# Next, we initialise columns `travel_min_to_MC` in `df_final` with the default value as `NA`.
#

# %%
# Initialise column with 'NA' values
df_final["travel_min_to_MC"] = "NA"
df_final["direct_journey_flag"] = "NA"

# %% [markdown]
# Utilising the information we have thus far, we can now calculate the `travel_min_to_MC` and `direct_journey_flag` columns as follows:
#
# 1. Firstly, identify all trips that operate on all weekdays and belong to the routes listed in `df_routes`
# 2. Then we, iterate over each row (property) in our `df_final` DataFrame and get the `closest_train_station_id` for each property
# 3. For the trips we identified earlier, we get the stop times at the closest station between 7-9 am `stop_times_closest` and the stop times at the Melbourne Central station `stop_times_mc`
# 4. Find the direct journeys by merging the two DataFrames on the `trip_id` column
# 5. Calculate the travel time for each direct journey by subtracting the departure time at the closest station from the arrival time at the Melbourne Central station
# 6. Filter out any journeys where the calculated travel time is negative
# 7. Lastly, we calculate the average travel time as `travel_min_to_MC` and apply the `direct_journey_flag` based on the pre-defined information as specified
#

# %%
# Identify trips that operate on weekdays only and are in df_routes
trips = df_trips[
    (df_trips["service_id"].isin(WEEKDAY_SERVICE_ID))
    & (df_trips["route_id"].isin(df_routes["route_id"]))
]

# Iterate over each property in df_final
for i, property in df_final.iterrows():
    # Get closest station ID
    closest_station_id = property["closest_train_station_id"]

    # Get stop times for trips at the closest station
    stop_times_closest = df_stop_times[
        (df_stop_times["trip_id"].isin(trips["trip_id"]))
        & (df_stop_times["stop_id"] == closest_station_id)
        & (df_stop_times["departure_time"].between("07:00:00", "09:00:00"))
    ]

    # Get stop times for trips at the "Melbourne Central" station
    stop_times_mc = df_stop_times[
        (df_stop_times["trip_id"].isin(trips["trip_id"]))
        & (df_stop_times["stop_id"] == MC_STATION_ID)
    ]

    # Merge both stop times to find direct journeys
    direct_journeys = pd.merge(
        stop_times_closest, stop_times_mc, on="trip_id", suffixes=("_closest", "_mc")
    )

    # Calculate travel times for direct journeys
    direct_journeys["travel_time"] = (
        pd.to_datetime(direct_journeys["arrival_time_mc"], format="%H:%M:%S")
        - pd.to_datetime(direct_journeys["departure_time_closest"], format="%H:%M:%S")
    ).dt.total_seconds() / 60

    # Filter out journeys where travel time is negative (Found by inspection)
    direct_journeys = direct_journeys[direct_journeys["travel_time"] >= 0]

    if not direct_journeys.empty:
        # If there are direct journeys, calculate the average travel time
        avg_travel_time = round(direct_journeys["travel_time"].mean())
        df_final.at[i, "travel_min_to_MC"] = avg_travel_time
        # Set flag to 1 for direct journeys
        df_final.at[i, "direct_journey_flag"] = 1

    elif closest_station_id == MC_STATION_ID:
        # If the closest station is 'Melbourne Central Railway Station', set the value to 0
        df_final.at[i, "travel_min_to_MC"] = 0
        # Set flag to 1 for direct journeys
        df_final.at[i, "direct_journey_flag"] = 1
    else:
        # If there are no direct journeys, set the value to 'no direct trip is available'
        df_final.at[i, "travel_min_to_MC"] = "no direct trip is available"
        # Set flag to 0 for no direct journeys
        df_final.at[i, "direct_journey_flag"] = 0

# Convert only numeric values in `travel_min_to_MC` column to float
df_final["travel_min_to_MC"] = df_final["travel_min_to_MC"].apply(
    lambda x: float(x) if isinstance(x, (int, float)) else x
)

# Convert `direct_journey_flag` column to int
df_final["direct_journey_flag"] = df_final["direct_journey_flag"].astype("int64")

# %%
# Check if 'NA' in 'travel_min_to_MC' and 'direct_journey_flag' columns
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    print("NA" in df_final["travel_min_to_MC"].unique())
    print("NA" in df_final["direct_journey_flag"].unique())

# %%
print(df_final.shape)
df_final.head()

# %% [markdown]
# ## <a id='toc6_3_'></a>[Scraping Website for Housing Data](#toc0_)
#
# We will now scrape the website `house.speakingsame.com` to obtain additional housing data for each property suburb in our `df_final` DataFrame.
#
# Since the website can rate limit our requests as our DataFrame is quite large, we save the DataFrame to a CSV file and load it back in to continue scraping from where we left off. Furthermore, we cut down on the number of requests by only scraping data for unique property suburbs in our DataFrame. Additionally, we introduce a delay of 5 second between each request to avoid being rate limited by the website.
#

# %%
# Load saved data if it exists
if os.path.exists("../data/output/scraped_data.csv"):
    df_scraped = pd.read_csv("../data/output/scraped_data.csv")
# Create an new DataFrame if it doesn't exist
else:
    df_scraped = pd.DataFrame(
        columns=[
            "suburb",
            "number_of_houses",
            "number_of_units",
            "municipality",
            "aus_born_perc",
            "median_income",
            "median_house_price",
            "population",
        ]
    )

# %%
# Get a list of unique suburbs
suburbs = df_final["suburb"].unique()

# Count of unique suburbs
print(f"There are {len(suburbs)} unique suburbs")

# Get a list of suburbs that have already been scraped
scraped_suburbs = df_scraped["suburb"].unique()

# Count of scraped suburbs
print(f"Data already scraped for {len(scraped_suburbs)} suburbs")

# %%
# Remove the scraped suburbs from the list of all suburbs
suburbs_to_scrape = [i for i in suburbs if i not in scraped_suburbs]

# Count of suburbs to scrape
print(f"Data needs to be scraped for {len(suburbs_to_scrape)} remaining suburbs")

# %% [markdown]
# Through inspecting the website and the soup object, we can see that the data we are interested in is contained in a table with the class `mainT`.
#
# However, since we are only interested in scraping specific data for each suburb from the website, we can use the `find_all` function with the text attribute to find the table rows that contain the data we are interested in.
#
# - We then access the parent element of the specific row, until we reach the table row that contains the data we are interested in.
# - Finally, we extract the data from that row by accessing the `td` elements in the row which contain the data we are interested in.
#
# Hence, by applying the above approach, we can extract the required data for each suburb, while also ensuring we account for rate limits as described above.
#

# %%
for suburb in suburbs_to_scrape:
    # Create URL for each suburb
    url = f"http://house.speakingsame.com/profile.php?q={suburb}%2C+VIC"

    # Send GET request to URL
    response = requests.get(url)

    # Parse HTML from response object
    soup = BeautifulSoup(response.text, "html.parser")

    try:
        # Initialize a dictionary to store data
        data = {}

        # Set suburb name
        data["suburb"] = suburb

        # Get municipality name
        data["municipality"] = (
            soup.find_all(text="Municipality")[0]
            .parent.parent.parent.find("a")
            .text.strip()
        )

        # Get number of houses and units
        numbers = (
            soup.find_all(text="Number of houses/units")[0]
            .parent.parent.parent.find_all("td")[2]
            .text.strip()
        )
        # Split the numbers into houses and units
        data["number_of_houses"], data["number_of_units"] = numbers.split(" / ")

        # Get Australian born percentage
        data["aus_born_perc"] = (
            soup.find_all(text="Australian Born")[0]
            .parent.parent.parent.find_all("td")[1]
            .text.strip()
        )

        # Get median income
        data["median_income"] = (
            soup.find_all(text="Weekly income")[0]
            .parent.parent.parent.find_all("td")[1]
            .text.strip()
        )

        # Get population
        data["population"] = (
            soup.find_all(text="All People")[0]
            .parent.parent.find_all("td")[1]
            .text.strip()
        )

        # Get median house price
        data["median_house_price"] = (
            soup.find_all(text="House")[0].parent.parent.find_all("td")[1].text.strip()
        )

        # Append the data to the DataFrame
        df_scraped = pd.concat([df_scraped, pd.DataFrame([data])], ignore_index=True)

    except Exception as e:
        print(f"{suburb}: {e}")

    # Save the new DataFrame to a file after each request
    df_scraped.to_csv("scraped_data.csv", index=False)

    # Wait for 5 seconds before the next request
    # (Adjust to 15-20 seconds to avoid rate limiting)
    time.sleep(5)

# %% [markdown]
# We verify that the data has been scraped correctly by checking the first 5 rows of the `df_scraped` DataFrame and checking for any null values.
#

# %%
display(df_scraped.head())

df_scraped.isna().sum()

# %% [markdown]
# Now, we convert the data types of the columns in `df_scraped` to match the data types in the sample output file after which we merge the scraped data with our `df_final` DataFrame.
#

# %%
# Convert `number_of_houses` and `number_of_units` columns to float
df_scraped["number_of_houses"] = df_scraped["number_of_houses"].astype(float)
df_scraped["number_of_units"] = df_scraped["number_of_units"].astype(float)

# Convert `population` column to float
df_scraped["population"] = df_scraped["population"].astype(float)

# %%
# Join df_scraped to df_final
df_final = pd.merge(df_final, df_scraped, on="suburb", how="left")

# %% [markdown]
# ### Validating Scraped Data
#
# We can also perform validation on the scraped data by comparing the scraped data with the data in the sample output file. We do this by comparing data for property suburbs that are present in both DataFrames.
#

# %%
# Select columns for comparison
TEMP_DF1 = df_sample[
    [
        "suburb",
        "number_of_houses",
        "number_of_units",
        "municipality",
        "aus_born_perc",
        "median_income",
        "median_house_price",
        "population",
    ]
]

# Keep only unique suburbs
TEMP_DF1 = TEMP_DF1.drop_duplicates(subset=["suburb"])

# Sort by suburb
TEMP_DF1 = TEMP_DF1.sort_values(by=["suburb"]).reset_index(drop=True)

# %%
# Create a list of suburbs from df_sample and sort it
sub_list = sorted(df_sample["suburb"].unique())

# Select rows from df_final where suburb is in sub_list
TEMP_DF2 = df_final[df_final["suburb"].isin(sub_list)]

# Select columns for comparison
TEMP_DF2 = TEMP_DF2[
    [
        "suburb",
        "number_of_houses",
        "number_of_units",
        "municipality",
        "aus_born_perc",
        "median_income",
        "median_house_price",
        "population",
    ]
]

# Keep only unique suburbs
TEMP_DF2 = TEMP_DF2.drop_duplicates(subset=["suburb"])

# Sort by suburb and reset index
TEMP_DF2 = TEMP_DF2.sort_values(by="suburb").reset_index(drop=True)

# %%
# Compare Both DataFrames to see differences
TEMP_DF1.compare(TEMP_DF2, keep_shape=True)

# %% [markdown]
# Here we can see that the scraped data for all columns except for `median_house_price` matches the data in the sample output file.
#
# Note that there are slight discrepancies in the `median_house_price` column. This is due to the website data being updated after the sample output file was created.
#

# %% [markdown]
# ## <a id='toc6_4_'></a>[Addressing Duplication Issues](#toc0_)
#
# Previously, we identified that there were duplicate property addresses in our `df_final` DataFrame when we concatenated the `XML` and `JSON` data. Now that we have more information about each property, such as the `suburb`, we can use this information to identify and remove duplicate properties.
#

# %%
# Property ID for duplicates when concatenating XML and JSON data
addr_duplicates

# %%
# Rows in df_final where property_id is in addr_duplicates
df_final[df_final["property_id"].isin(addr_duplicates)].sort_values(by="addr_street")

# %% [markdown]
# Here we can see that property_id 10440 and 10439 are identical in all aspects except for minor differences in the `lat` and `lng` columns. We can explore this further by plotting the two properties on a map to see if they are in fact the same property.
#

# %%
# Get latitude and longitude values from df_final for property_id 10440 and 10439
property_ids = [10440, 10439]
df_markers = df_final[df_final["property_id"].isin(property_ids)][
    ["lat", "lng", "property_id"]
]

# Create map figure
map_figure = folium.Figure(width=800, height=600)

# Create map object with center coordinates and zoom level
temp_map = folium.Map(
    location=[df_markers["lat"].mean(), df_markers["lng"].mean()],
    zoom_start=19,
    control_scale=True,
    max_zoom=20,
)

# Add markers for both property_id's
for index, row in df_markers.iterrows():
    lat, lng, property_id = row["lat"], row["lng"], row["property_id"]
    folium.Marker([lat, lng], popup=str(property_id)).add_to(temp_map)

# Add map to figure and display
temp_map.add_to(map_figure)
map_figure

# %% [markdown]
# Our suspicions are confirmed as we can see that the two properties are in fact the same property. We can now remove the duplicate property from our DataFrame.
#

# %%
# Remove one of the duplicates from df_final
df_final = df_final[df_final["property_id"] != 10439]

# Reset index
df_final = df_final.reset_index(drop=True)

# %%
print(df_final.shape)
df_final.head()

# %% [markdown]
# # <a id='toc7_'></a>[Data Export](#toc0_)
#
# We have finished integrating all data into our `df_final` DataFrame. We can now export the DataFrame to a CSV file.
#
# First ensure that:
#
# - The column names in `df_final` match the column names in the provided sample output file
# - The data types of the columns in `df_final` match the data types in the sample output file
# - The order of the columns in `df_final` match the order of the columns in the sample output file
#   - (Not required but makes it easier to compare the two files for validation purposes)
#

# %%
df_sample.info()

# %%
# Rearrange columns in df_final to match df_sample
df_final = df_final[
    [
        "property_id",
        "lat",
        "lng",
        "addr_street",
        "suburb",
        "number_of_houses",
        "number_of_units",
        "municipality",
        "population",
        "aus_born_perc",
        "median_income",
        "median_house_price",
        "lga",
        "closest_train_station_id",
        "distance_to_closest_train_station",
        "travel_min_to_MC",
        "direct_journey_flag",
    ]
]
df_final.info()

# %%
# Compare column names of both DataFrames
if set(df_final.columns) == set(df_sample.columns):
    print("Column names are the same.")

# %%
# Compare data types of both DataFrames
if df_final.dtypes.equals(df_sample.dtypes):
    print("Data types are the same.")

# %% [markdown]
# Now we can export the DataFrame to a CSV file.
#

# %%
# Export df_final to CSV
df_final.to_csv("../data/output/properties_solution.csv", index=False)

# %% [markdown]
# # <a id='toc8_'></a>[Data Transformation](#toc0_)
#
# In this section, we will perform some data transformation on our `df_final` DataFrame to study the effect different normalisation/transformation techniques on specific columns of the DataFrame that were scraped from the `house.speakingsame.com` website.
#
# As per the project specification, we only focus on the following columns:
#
# - `number_of_houses`
# - `number_of_units`
# - `population`
# - `aus_born_perc`
# - `median_income`
# - `median_house_price` (_target variable_)
#
# First, we make a copy of the `df_final` DataFrame and drop all columns except for the columns listed above.
#

# %%
# Create a copy of df_final with selected columns
df_trans = df_final[
    [
        "number_of_houses",
        "number_of_units",
        "population",
        "aus_born_perc",
        "median_income",
        "median_house_price",
    ]
].copy()

# %% [markdown]
# ## <a id='toc8_1_'></a>[Preparing Data for Analysis](#toc0_)
#

# %%
df_trans.info()

# %%
df_trans.head()

# %% [markdown]
# We can see that some of the columns are not in a numeric format. We can convert these columns to a numeric format for analysis.
#

# %%
# Remove '%' from aus_born_perc and convert to float
df_trans["aus_born_perc"] = df_trans["aus_born_perc"].str.replace("%", "").astype(float)

# Remove '$' and ',' from median_income and median_house_price and convert to float
df_trans["median_income"] = (
    df_trans["median_income"].str.replace("$", "").str.replace(",", "").astype(float)
)
df_trans["median_house_price"] = (
    df_trans["median_house_price"]
    .str.replace("$", "")
    .str.replace(",", "")
    .astype(float)
)

# %%
df_trans.head()

# %%
df_trans.info()

# %%
# Check for duplicates in df_trans
df_trans.duplicated().sum()

# %% [markdown]
# As seen above, we have a large number of duplicate rows in our DataFrame. This is because our dataset contains multiple properties in the same suburb. We can remove these duplicates to ensure that we only have unique suburbs in our dataset for further analysis.
#

# %%
# Drop duplicates
df_trans = df_trans.drop_duplicates()

df_trans.shape

# %% [markdown]
# Now, lets plot histograms of the columns in our dataset to get an initial understanding of the data distribution.
#

# %%
# Set figure size
plt.figure(figsize=(20, 20))


# Define plot colours
colors = sns.color_palette("muted", len(df_trans.columns))

for i, column in enumerate(df_trans.columns):
    plt.subplot(3, 2, i + 1)

    # Plot histogram with kde and assign a different colour for each plot
    sns.histplot(
        data=df_trans, x=column, bins=30, kde=True, color=colors[i % len(colors)]
    )

    # Customise plot
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.title(f"Histogram of {column}", color=colors[i % len(colors)])
    plt.grid(True)

plt.tight_layout()
plt.show()

# %% [markdown]
# Based on the facet visualisation above, we can make the following observations:
#
# **Histogram of `number_of_houses`:**
#
# - This distribution is right-skewed (positively skewed), meaning there are many regions with a lower number of houses, but a few regions have a significantly larger number.
# - Most regions have between 0 to 5000 houses, with a long tail extending towards higher values.
#
# **Histogram of `number_of_units`:**
#
# - This is also right-skewed, similar to the number of houses.
# - Most regions have a small number of units, with only a few areas having a large number of units (ranging from 0 to 5000 units).
#
# **Histogram of `population`:**
#
# - The population distribution appears to be bimodal with two peaks.
# - There’s a slight right-skewness, with a tail indicating some regions have much larger populations.
#
# **Histogram of `aus_born_perc`:**
#
# - The distribution of the percentage of Australian-born residents shows a multimodal pattern with multiple peaks.
# - There is no strong skewness, but the distribution is slightly left-skewed, with a few regions having higher percentages of Australian-born residents.
#
# **Histogram of `median_income`:**
#
# - The distribution of median income appears to be roughly symmetric, although with some irregularity and minor deviations.
#
# **Histogram of `median_house_price`:**
#
# - This histogram is right-skewed, with most house prices concentrated between $500,000 and $1.5 million.
# - The tail indicates that a few regions have significantly higher house prices (up to $2.5 million and beyond).
#
# These distributions suggest that the dataset likely represents diverse areas with varying levels of development, population density and economic conditions.
#

# %% [markdown]
# ## <a id='toc8_2_'></a>[Standardization](#toc0_)
#

# %% [markdown]
# Lets observe the min and max values of the columns to determine if any scaling or transformation is necessary.
#

# %%
df_trans.describe().loc[["min", "max"]]

# %% [markdown]
# Observing the minimum and maximum values of each column (feature), we note the features have different scales. While this does not affect ordinary least squares linear regression (which is scale-invariant), it can cause issues for distance-based and gradient-based algorithms (e.g., KNN, SVM, neural networks). Standardization is still useful for interpreting coefficient magnitudes on a common scale.
#
# We can use standardization or min-max normalization for this purpose.
#

# %%
# Instantiate StandardScaler
scaler_std = StandardScaler()

# Standardize data
df_standardized = pd.DataFrame(
    scaler_std.fit_transform(df_trans), columns=df_trans.columns
)

df_standardized.describe().loc[["mean", "std"]]

# %%
df_standardized.head()

# %% [markdown]
# The standardisation was performed using the `StandardScaler` class from the `sklearn.preprocessing` module.
#
# Now, all features have a **mean of approximately 0** and a **standard deviation of approximately 1**, making them suitable for algorithms that require standardised inputs.
#

# %% [markdown]
# ## <a id='toc8_3_'></a>[Normalization](#toc0_)
#
# We can also apply min-max normalization to the features and observe the results. Min-max normalization will transform the features to fall within the range [0, 1].
#

# %%
# Instantiate MinMaxScaler
scaler_mm = MinMaxScaler()

# Normalize data
df_normalized = pd.DataFrame(
    scaler_mm.fit_transform(df_trans), columns=df_trans.columns
)

df_normalized.describe().loc[["min", "max"]]

# %%
df_normalized.head()

# %% [markdown]
# As we can see from the output, all variables have been successfully normalized to a range of values between 0 and 1.
#
# We can also plot the normalized variables to observe the changes visually.
#

# %%
# Set figure size
plt.figure(figsize=(10, 5))

# Define x-axis range
x = range(len(df_normalized))

# Column names and labels
columns = [
    "number_of_houses",
    "number_of_units",
    "population",
    "aus_born_perc",
    "median_income",
    "median_house_price",
]
labels = [
    "Number of Houses",
    "Number of Units",
    "Population",
    "Australian Born Percentage",
    "Median Income",
    "Median House Price",
]

# Loop through columns and plot
for col, label in zip(columns, labels):
    plt.plot(x, df_normalized[col], label=label, linewidth=1.5)

# Add dynamic limits
plt.xlim(0, len(df_normalized))
plt.ylim(df_normalized.min().min() - 0.1, df_normalized.max().max() + 0.1)

# Set labels, title and legend
plt.xlabel("Data Point")
plt.ylabel("Normalized Value")
plt.title("Normalized Variables")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

# Add grid and show plot
plt.grid(True)
plt.show()

# %% [markdown]
# ## <a id='toc8_4_'></a>[Log Transformation](#toc0_)
#
# As we observed earlier, variables like the Number of Houses, Number of Units and Median House Price showed right-skewed distributions. Log transformation would help reduce this right-skew, making the distributions more symmetric and closer to normal.
#
# The log function compresses the scale of large values while stretching out smaller values. This would be particularly noticeable for variables with wide ranges or outliers, like Number of Units and Median House Price.
#
# The extreme high values we saw in the normalized data (spikes to 1.0) would be brought closer to the rest of the data points. This could make patterns in the data more visible and reduce the disproportionate influence of outliers.
#

# %%
# Apply log transformation
df_log_transformed = df_trans.apply(np.log1p)

# Set color palette
colors = sns.color_palette("muted", len(df_trans.columns))

# Set figure size
plt.figure(figsize=(20, 15))

for i, column in enumerate(df_log_transformed.columns):
    plt.subplot(3, 2, i + 1)

    # Plot histogram with kde
    sns.histplot(
        data=df_log_transformed,
        x=column,
        bins=30,
        kde=True,
        color=colors[i % len(colors)],
    )

    # Customise plot
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.title(f"Histogram of {column}")
    plt.grid(True)

plt.tight_layout()
plt.show()

# %% [markdown]
# As we can see from the visualisation above, the distributions of `number_of_houses`, `number_of_units` and `median_house_price` have become more symmetric after the log transformation. Note that `median_income`, which was already roughly symmetric, does not benefit as much from log transformation — applying log to an already symmetric distribution can introduce left-skew.
#

# %% [markdown]
# ## <a id='toc8_5_'></a>[Power Transformation](#toc0_)
#
# Power transformation aims to map data from any distribution to as close to a Gaussian distribution as possible in order to stabilise variance and minimize skewness. The Yeo-Johnson transformation is a type of power transformation that supports both positive and negative data. We apply the Yeo-Johnson transformation to the features and observe the results.
#

# %%
# Instantiate PowerTransformer
pt_yj = PowerTransformer(method="yeo-johnson")

# Apply power transformation
df_power_transformed = pd.DataFrame(
    pt_yj.fit_transform(df_trans), columns=df_trans.columns
)

# %% [markdown]
# ## <a id='toc8_6_'></a>[Box-Cox Transformation](#toc0_)
#
# The Box-Cox transformation is a power transformation that aims to stabilise variance and make data more normal-like. Unlike Yeo-Johnson, it only works with strictly positive data. Since all our data is positive, we can apply it here.
#

# %%
# Instantiate PowerTransformer
pt_bc = PowerTransformer(method="box-cox")

# Apply power transformation
df_box_cox_transformed = pd.DataFrame(
    pt_bc.fit_transform(df_trans), columns=df_trans.columns
)

# %% [markdown]
# The Yeo-Johnson transformation and Box-Cox transformation both aim to transform the data to a more normal distribution. The Box-Cox transformation is a special case of the Yeo-Johnson transformation that only works with positive data. The Yeo-Johnson transformation is more flexible and can handle both positive and negative data. Since the result is quite similar for both transformations, we can visualise one of them to observe the changes.
#

# %%
# Set color palette
colors = sns.color_palette("muted", len(df_trans.columns))

# Set figure size
plt.figure(figsize=(20, 15))

for i, column in enumerate(df_box_cox_transformed.columns):
    plt.subplot(3, 2, i + 1)

    # Plot histogram with kde
    sns.histplot(
        data=df_box_cox_transformed,
        x=column,
        bins=30,
        kde=True,
        color=colors[i % len(colors)],
    )

    # Customise plot
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.title(f"Histogram of {column}")
    plt.grid(True)

plt.tight_layout()
plt.show()

# %% [markdown]
# As we can see from the visualisation above, the Box-Cox transformation has successfully transformed the data to more normal-like distributions. Note that linear regression assumes _residuals_ (errors) are normally distributed, not the features themselves. However, transforming skewed features can help reduce heteroscedasticity and the influence of outliers, which may indirectly improve model assumptions.
#

# %% [markdown]
# ## <a id='toc8_7_'></a>[Effect of Data Transformation on Linear Regression Models](#toc0_)
#
# In this section, we study the effect of data transformation on linear regression models. To ensure correct methodology, the transformer is **fitted on training data only** and then applied to both train and test sets — avoiding data leakage. Predictions are then **inverse-transformed back to the original scale** so that RMSE and R² are directly comparable across all methods.
#
# The RMSE measures the average magnitude of residuals or prediction errors, while the R-squared measures the proportion of variance in the dependent variable that is predictable from our independent variables.
#


# %%
def evaluate_linear_regression(df_original, target_column, transformer=None):
    """
    Evaluates linear regression model with proper train/test methodology.
    The transformer is fitted on training data only, then used to transform both sets.
    Predictions are inverse-transformed back to original scale for fair comparison.
    @param df_original: untransformed DataFrame
    @param target_column: target column name (median_house_price)
    @param transformer: 'log' for log1p, sklearn transformer instance, or None (original)
    @return: Tuple of RMSE and R2 score (both in original scale)
    """
    # Split original data into train/test FIRST (before any transformation)
    X = df_original.drop(target_column, axis=1)
    y = df_original[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if transformer == "log":
        # Log transform (element-wise, no fitting needed — no leakage concern)
        X_train_t = np.log1p(X_train)
        X_test_t = np.log1p(X_test)
        y_train_t = np.log1p(y_train)
        inv_func = lambda y_vals: np.expm1(np.array(y_vals))

    elif transformer is not None:
        # Sklearn transformer — fit on training data only
        train_combined = pd.concat([X_train, y_train], axis=1)
        test_combined = pd.concat([X_test, y_test], axis=1)

        transformer.fit(train_combined)

        train_transformed = pd.DataFrame(
            transformer.transform(train_combined), columns=train_combined.columns
        )
        test_transformed = pd.DataFrame(
            transformer.transform(test_combined), columns=test_combined.columns
        )

        X_train_t = train_transformed.drop(target_column, axis=1)
        X_test_t = test_transformed.drop(target_column, axis=1)
        y_train_t = train_transformed[target_column]

        # Build inverse function for the target column
        target_idx = list(train_combined.columns).index(target_column)
        col_names = list(train_combined.columns)

        def make_inv_func(tfm, t_idx, cols):
            def inv_func(y_vals):
                y_arr = np.array(y_vals).ravel()
                # Use a DataFrame with proper column names to avoid sklearn warnings
                dummy = pd.DataFrame(np.zeros((len(y_arr), len(cols))), columns=cols)
                dummy.iloc[:, t_idx] = y_arr
                result = tfm.inverse_transform(dummy)
                return result[:, t_idx]

            return inv_func

        inv_func = make_inv_func(transformer, target_idx, col_names)
    else:
        # No transformation — use original data
        X_train_t, X_test_t = X_train, X_test
        y_train_t = y_train
        inv_func = None

    # Fit model and make predictions (in transformed space)
    model = LinearRegression()
    model.fit(X_train_t, y_train_t)
    y_pred_t = model.predict(X_test_t)

    # Inverse-transform predictions back to original scale
    if inv_func is not None:
        y_pred = inv_func(y_pred_t)
    else:
        y_pred = y_pred_t

    # Metrics are always computed against original-scale y_test
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return rmse, r2


# Evaluate each transformation using fresh transformer instances
# (fitted on training data only inside the function)
datasets = {
    "Original": None,
    "Standardized": StandardScaler(),
    "Normalized": MinMaxScaler(),
    "Log Transformed": "log",
    "Yeo-Johnson Transformed": PowerTransformer(method="yeo-johnson"),
    "Box-Cox Transformed": PowerTransformer(method="box-cox"),
}

# Create a list to store results
results = []

# Evaluate linear regression for each transformation
for name, transformer in datasets.items():
    rmse, r2 = evaluate_linear_regression(df_trans, "median_house_price", transformer)
    results.append({"Dataset": name, "RMSE": rmse, "R²": r2})

# Create DataFrame from results
results_df = pd.DataFrame(results)
results_df

# %%
# Set the figure size
plt.figure(figsize=(14, 6))

# Create a bar plot for RMSE (all datasets, now on the same original scale)
plt.subplot(1, 2, 1)
rmse_plot = sns.barplot(
    data=results_df, x="Dataset", y="RMSE", palette="mako", hue="Dataset", dodge=False
)
plt.title("RMSE Across Different Transformations")
plt.xticks(rotation=45)
plt.ylabel("Root Mean Square Error (RMSE)")

# Add data labels on bars
for p in rmse_plot.patches:
    rmse_plot.annotate(
        f"{p.get_height():.0f}",
        (p.get_x() + p.get_width() / 2.0, p.get_height()),
        ha="center",
        va="bottom",
        fontsize=11,
    )

# Create a bar plot for R²
plt.subplot(1, 2, 2)
r2_plot = sns.barplot(
    data=results_df, x="Dataset", y="R²", palette="mako", hue="Dataset", dodge=False
)
plt.title("R² Across Different Transformations")
plt.xticks(rotation=45)
plt.ylabel("R-squared (R²)")

# Add numbers on top of bars
for p in r2_plot.patches:
    r2_plot.annotate(
        f"{p.get_height():.2f}",
        (p.get_x() + p.get_width() / 2.0, p.get_height()),
        ha="center",
        va="bottom",
        fontsize=10,
    )

# Show plots
plt.tight_layout()
plt.show()

# %% [markdown]
# **Note on Fair Comparison**
#
# All predictions have been inverse-transformed back to the original house price scale before computing RMSE and R². This ensures the metrics are directly comparable across all transformations.
#
# **RMSE Across Different Transformations**
#
# - Original, Standardized, and Normalized
#   - These three produce **identical** RMSE (~302,891) and R² (0.64) values. This is expected because standardization and min-max normalization are linear transformations, and ordinary least squares linear regression is scale-invariant.
# - Log Transformed
#   - The log transformation achieved the **best performance**, with the lowest RMSE (~270,880) and highest R² (0.71). By compressing the scale of large values, the log-space linear fit better captures the feature-target relationships in this dataset, particularly for right-skewed variables like house prices.
# - Yeo-Johnson and Box-Cox Transformed
#   - Both performed **slightly worse** than the original data, with RMSE of ~308,300 and R² of ~0.62. Making features more Gaussian-like did not help the linear model here — the power transformations may have distorted relationships that were already reasonably linear.
#
# **R-squared Across Different Transformations**
#
# - Original, Standardized, and Normalized all share the same R² (0.64) — confirming that linear scaling does not change the model's explanatory power.
# - Log transformation improved R² to 0.71, indicating that the log-space linearisation captures more variance in house prices when mapped back to original scale.
# - Yeo-Johnson and Box-Cox reduced R² to ~0.62, meaning their transformed linear fits translate back slightly worse than the untransformed model.
#
# **Overall Observations**
#
# - Linear transformations (standardization, normalization) do not improve or degrade linear regression performance — they are purely cosmetic for OLS. Their value lies in interpretability and in benefiting other algorithms (e.g., regularized regression, KNN, SVM).
# - Among non-linear transformations, **log transformation was the most effective**, likely because it addresses the right-skew in house prices while preserving monotonic relationships. The power transformations (Yeo-Johnson, Box-Cox), despite making distributions more Gaussian, slightly degraded performance.
#

# %% [markdown]
# # <a id='toc9_'></a>[Summary](#toc0_)
#
# This project aimed to integrate diverse datasets related to housing in Victoria, Australia, into a unified format for predictive analysis. The objective was to assess how different data transformation techniques affect the performance of linear regression models for housing price prediction.
#
# Data from various sources, including CSV, XML, JSON, shapefiles, PDFs and web scraping, were integrated to include property details, geographic boundaries, public transport information and additional housing statistics. Key steps included mapping properties to suburbs using spatial joins, calculating distances to the nearest train stations and extracting housing-related data from web sources.
#
# After data integration, we applied several transformation techniques, including z-score standardisation, min-max normalization, logarithmic scaling, Yeo-Johnson and Box-Cox power transformations. By inverse-transforming predictions back to the original scale, we showed that **linear transformations (standardization, normalization) produce identical results** for OLS linear regression, as expected, since OLS is scale-invariant. Among non-linear transformations, **log transformation yielded the best performance** (lowest RMSE, highest R²), while Yeo-Johnson and Box-Cox power transformations slightly degraded performance compared to the original data.
#
# This project highlights the importance of understanding how transformations interact with specific model types, and suggests that future work could explore non-linear models (e.g. tree-based methods, regularized regression) where scaling choices have a more meaningful impact.
#

# %% [markdown]
# # <a id='toc10_'></a>[References](#toc0_)
#
# [1] https://geopandas.org/en/stable/docs.html
# [2] https://developers.google.com/transit/gtfs/reference
# [3] https://developers.google.com/transit/gtfs/reference#stop_timestxt
# [4] https://stackoverflow.com/questions/4913349/
# [5] https://stackoverflow.com/questions/14463277/how-to-disable-python-warnings
# [6] https://python-visualization.github.io/folium/quickstart.html#Markers
# [7] https://en.wikipedia.org/wiki/Power_transform
#

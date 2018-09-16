# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------# ------ 
# Step 1: Import necessary modules and environment
# ----------------------------------------------------------------------

# import libraries
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# This file contains all Twitter-related actions, so no need to import here
import zka as tp
# ----------------------------------------------------------------------
# Step 2: Call API, get tweets, and parse tweets into a dataframe+CSV
# ----------------------------------------------------------------------

# create list of target news organizations' Twitter handles
targetNewsOrg_list = ["Bitcoin","cryptocurrency"]

# create and set color palette for all charts
orgPalette = sns.color_palette("bright", len(targetNewsOrg_list))
sns.set_palette(orgPalette)

# define number of tweets we want to pull from each org
numTweets = 150

# break into increments of 10
numCycles = int(round(numTweets/10))

# create dict to store dictionaries generated during analysis
completeResults_df = tp.parseTweets(targetNewsOrg_list, numCycles)

# rearrange columns to be more sensible
completeResults_df = completeResults_df[["handle", "count", "compound",
                                         "positive", "negative", "neutral",
                                         "date", "text"]]
completeResults_df.to_csv("TweetsAnalyzed.csv")

completeResults_df


from preprocessor import *
from eda import *
from training import *

laliga = merge_get_data()
# For analyzing the full time result of the match
# The betting data about total goals, asian handicap and closing odds are not related to the result of the game
# Therefore, those data will be deleted.
laliga = laliga.iloc[:, 3:41]
laliga = displayNullAndDrop(laliga)

# Reset Index
laliga = laliga.reset_index().drop(["index"], axis=1)

print(laliga.shape)
with pd.option_context('display.max_columns', None):
    print(laliga)

# Exploratory Data Analysis (EDA)
data = pd.DataFrame(set(laliga['FTR']))

# For EDA of the statistical data of the match itself,
# Betting columns will be split out in this part.
statisticalResultData = SplitResultMatchStat(laliga)

# Count for goal of each team against
df_goal = statisticalResultData.pivot_table(values=['FTHG', 'FTAG'], index='HomeTeam', columns='AwayTeam',
                                            aggfunc='sum')

# Discover the basic statistical information by describe function in pandas
with pd.option_context('display.max_columns', None):
    print(statisticalResultData.describe())

# List of teams and match count
plt_match_count(statisticalResultData)

# List all winner from each match
statisticalResultData['Winner'] = listWinner(statisticalResultData)
print(statisticalResultData['Winner'])
print(" ")

# What is the percentage of winners in Home and Away Ground during all three seasons of La Liga?
pltWinPercent(statisticalResultData)

# Is there any correlation between number of winning a match and total goal?
plt_corr_score_win(statisticalResultData, df_goal)

# What is the Percentage on Home Team ground of Win, Loss, Draw?
# What is the Percentage on Away Team ground of Win, Loss, Draw?
# Which team is winning the most of the match?
plt_performing_home_away(statisticalResultData)

# How many goal made in Home Ground and Away Ground by team?
plt_goal_count(df_goal)

# Which team score how many goals against which team?
plt_score_home_away(df_goal)

# Correlation between different variables
piv_h = pivH(statisticalResultData)
piv_a = pivA(statisticalResultData)
m_ha = pd.concat([piv_h, piv_a], axis=0, ignore_index=False)
m_ha_2 = pd.concat([piv_h, piv_a], axis=0, ignore_index=True)
plt_corr_other(m_ha_2)

# Playing style analysis of Teams: Average Corner, Shot on Target,Goals
plt_goal_corner_target(m_ha)

# what is the percentage of changing the result after half of game?
plt_result_changed(statisticalResultData)

# Encode string data to numerical data
laliga = encode_string(laliga)

# The relation between Betting Data and final result of the game
plt_corr_betting_result(laliga)

# The relation between Betting Data and final result of the game
plt_corr_goals_result(laliga)
laliga = laliga.drop(columns=['FTHG', 'FTAG', 'HTHG', 'HTAG', 'HTR'])

finalDataset = meanOfOdds(laliga)
finalDataset = finalDataset.drop(finalDataset.iloc[:, 15:33], axis=1)
with pd.option_context('display.max_columns', None):
    print(finalDataset)

# number of PCA explained variance analysis
plt_PCA(finalDataset)

finalDataset = decode_ftr(finalDataset)

X = finalDataset.drop(columns=['FTR'])
y = finalDataset['FTR']

# Run models_evaluation function
print(models_evaluation_training(X, y))

print(summary_mlp())

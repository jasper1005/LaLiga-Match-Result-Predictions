import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

# Global Variable
colors = ['g', 'r', 'b']
colors_pha = ['c', 'm', 'y', 'k', 'r', 'g', 'b', 'darkorange', 'navy', 'turquoise', 'fuchsia', 'olivedrab']


# List of teams and match count
def listMatchCount(statisticalResultData):
    df_match_count = pd.DataFrame()
    df_match_count['FTHC'] = statisticalResultData.groupby('HomeTeam').count()['FTR'].sort_values(ascending=False)
    df_match_count['FTAC'] = statisticalResultData.groupby('AwayTeam').count()['FTR'].sort_values(ascending=False)
    print("Number of Match of Home Team:")
    print(statisticalResultData.groupby('HomeTeam').count()['FTR'].sort_values(ascending=False))
    print("Number of Match of Away Team:")
    print(statisticalResultData.groupby('AwayTeam').count()['FTR'].sort_values(ascending=False))
    return df_match_count


# List all winner from each match
def listWinner(statisticalResultData):
    pd.options.mode.chained_assignment = None  # default='warn'
    statisticalResultData['Winner'] = 'No winner'
    statisticalResultData['Winner'][statisticalResultData['FTR'] == 'H'] = statisticalResultData['HomeTeam']
    statisticalResultData['Winner'][statisticalResultData['FTR'] == 'A'] = statisticalResultData['AwayTeam']
    return statisticalResultData['Winner']


# What is the percentage of winners in Home and Away Ground during all three seasons of La Liga?
def pltWinPercent(statisticalResultData):
    x = (statisticalResultData.groupby('FTR').count()['HomeTeam'] / len(statisticalResultData)) * 100
    print(x)
    plt.pie(x, explode=[.05, 0, .05], labels=['Away Team Win = ' + str(x[0].round(1)) + '%',
                                              'Draw Game = ' + str(x[1].round(1)) + '%',
                                              'Home Team Win = ' + str(x[2].round(1)) + '%'],
            startangle=5, shadow=True, colors=colors)
    plt.title(' Percentage Home and Away Win', fontsize=14, fontweight='bold')
    plt.show()


def plt_match_count(statisticalResultData):
    df_match_count = listMatchCount(statisticalResultData)
    pd.set_option('display.max_columns', 500)
    plt.figure(figsize=(10, 10))
    plt.subplots_adjust(wspace=.6)
    plt.suptitle('Total Match Count', fontsize=20)
    plt.subplot(1, 2, 1)
    df_match_count['FTHC'].plot(kind='barh', color=colors_pha)
    plt.title('Home Ground Total match')
    plt.subplot(1, 2, 2)
    df_match_count['FTAC'].plot(kind='barh', color=colors)
    plt.title('Away Ground Total match')
    plt.show()


# How many goal made in Home Ground and Away Ground by team?
def plt_goal_count(df_goal):
    pd.set_option('display.max_columns', 500)
    plt.figure(figsize=(10, 10))
    plt.subplots_adjust(wspace=.6)
    plt.suptitle('Total goals', fontsize=20)
    plt.subplot(1, 2, 1)
    df_goal.xs('FTHG', level=0, axis=1).sum(axis=1). \
        sort_values(ascending=True).plot(kind='barh', color=colors_pha)
    plt.title('Home Ground Total goal')
    plt.subplot(1, 2, 2)
    df_goal.xs('FTAG', level=0, axis=1).sum(axis=0).sort_values().plot(kind='barh', color=colors)
    plt.title('Away Ground Total goal')
    plt.show()


# Which team score how many goals against which team on both Home and Away Ground
def plt_score_home_away(df_goal):
    plt.figure(figsize=(20, 10))
    plt.subplots_adjust(wspace=0.1)
    plt.suptitle('Goal against each team', fontsize=20)
    plt.subplot(1, 2, 1)
    sns.heatmap(df_goal.xs('FTHG', level=0, axis=1), cmap='coolwarm', annot=True)
    plt.title('Home Ground Score of Teams', fontsize=16)
    plt.subplot(1, 2, 2)
    sns.heatmap(df_goal.xs('FTAG', level=0, axis=1), cmap='Blues', annot=True)
    plt.title('Away Ground Score of Teams', fontsize=16)
    plt.show()


# What is the Percentage on Home Team ground of Win, Loss, Draw?
# What is the Percentage on Away Team ground of Win, Loss, Draw?
# Which team is winning the most of the match?
def plt_performing_home_away(statisticalResultData):
    plt.figure(figsize=(18, 10))
    plt.suptitle("Matches Won by Team Insight", fontsize=18, fontweight='bold')
    plt.subplots_adjust(wspace=0.4, hspace=0.2)

    plt.subplot(1, 3, 1)
    home_team_df = statisticalResultData.pivot_table(values='AwayTeam',
                                                     index='HomeTeam', columns='FTR',
                                                     aggfunc='count')
    total = home_team_df.sum(axis=1)
    for i in home_team_df.columns.tolist():
        home_team_df['Percent:' + i] = 100 * home_team_df[i] / total
    sns.heatmap(home_team_df[['Percent:A', 'Percent:D', 'Percent:H']],
                cmap='Spectral', annot=True, cbar=False)
    plt.title('Percentage of Result at Home Ground')

    plt.subplot(1, 3, 2)
    away_team_df = statisticalResultData.pivot_table(values='HomeTeam',
                                                     index='AwayTeam', columns='FTR',
                                                     aggfunc='count')
    total = away_team_df.sum(axis=1)
    for i in away_team_df.columns.tolist():
        away_team_df['Percent:' + i] = 100 * away_team_df[i] / total
    sns.heatmap(away_team_df[['Percent:A', 'Percent:D', 'Percent:H']],
                cmap='Spectral', annot=True)
    plt.title('Percentage of Result at Away Ground')

    plt.subplot(1, 3, 3)
    statisticalResultData.groupby('Winner').count()['FTR'].sort_values(ascending=True)[1:-1].plot(kind='barh',
                                                                                                  grid=True,
                                                                                                  color=colors_pha)
    plt.axvline(np.mean(statisticalResultData.groupby('Winner').count()['FTR']), color='r')
    plt.title('All three season combined winners')
    plt.xlabel('Count of matches')
    plt.tight_layout()
    plt.show()


# Is there any correlation between number of winning a match and total goal?
def plt_corr_score_win(statisticalResultData, df_goal):
    total_goals_df = df_goal.xs('FTHG', level=0, axis=1).sum(axis=1) + \
                     df_goal.xs('FTAG', level=0, axis=1).sum(axis=0)
    win_df = statisticalResultData.groupby('Winner').count()['FTR']
    win_goal_df = pd.concat({'Win Count': win_df, 'Total Goal': total_goals_df}, axis=1).dropna()
    sns.regplot('Win Count', 'Total Goal', win_goal_df)
    plt.title('Correlation of Win the match vs total score', fontsize=16)
    plt.show()


# pivot table for home team
def pivH(statisticalResultData):
    piv_h = statisticalResultData.pivot_table(values=['FTHG', 'HST', 'HF', 'HC', 'HY', 'HR'],
                                              index='HomeTeam', aggfunc=np.mean)
    piv_h['Team'] = 'HomeTeam'
    piv_h.rename(index=str,
                 columns={'FTHG': 'Goal', 'HC': 'Corner', 'HF': 'Foul', 'HR': 'Red Card', 'HST': 'Shot Target',
                          'HY': 'Yellow Card'}, inplace=True)
    return piv_h


# pivot table for away team
def pivA(statisticalResultData):
    piv_a = statisticalResultData.pivot_table(values=['FTAG', 'AST', 'AF', 'AC', 'AY', 'AR'],
                                              index='AwayTeam', aggfunc=np.mean)
    piv_a['Team'] = 'AwayTeam'
    piv_a.rename(index=str,
                 columns={'FTAG': 'Goal', 'AC': 'Corner', 'AF': 'Foul', 'AR': 'Red Card', 'AST': 'Shot Target',
                          'AY': 'Yellow Card'}, inplace=True)
    return piv_a


# Correlation between different variables
def plt_corr_other(m_ha):
    sns.pairplot(m_ha, hue='Team', palette='husl', kind='scatter')
    plt.show()


# Playing style analysis of Teams
def plt_goal_corner_target(m_ha):
    m_ha = m_ha.drop('Team', axis=1)
    m_ha = m_ha.groupby(m_ha.index).mean()
    m_ha.sort_values('Goal', ascending=False, inplace=True)
    m_ha[['Corner', 'Shot Target', 'Goal']].plot(kind='bar', figsize=(20, 8),
                                                 label=m_ha.index)
    plt.xlabel("Team Name")
    a = np.arange(len(m_ha))
    plt.xticks(a, m_ha.index, rotation=90)
    plt.title('Corner/Shot Target/Average Goal per Match', fontsize=18)
    plt.show()


# what is the percentage of changing the result after half of game?
def plt_result_changed(statisticalResultData):
    result = statisticalResultData.loc[:, ['FTR', 'HTR']]
    sameResult = result[result['FTR'] == result['HTR']].reset_index().sort_index()
    sameResult = sameResult.drop(columns='index')  # Dropping the result Same column
    result.index += 1  # starting from the index of 1 instead of 0
    sameResult.index += 1
    sameResult = sameResult.index.values[-1]  # Finding the last index (the last game)
    result = result.index.values[-1]  # Finding the last index (the last game)
    result = result - sameResult  # Finding the number of games where the result was different

    # Getting the parameters ready for the pie chart
    labels_pie = ['Result Stayed the Same', 'Result Changed']
    list_pie = [sameResult, result]
    colors_rc = ['g', 'y']

    # Plotting the pie chart
    fig = plt.figure(figsize=(10, 7))
    plt.pie(list_pie, explode=[0, 0.1], autopct='%.2f%%', colors=colors_rc, shadow=True)
    plt.style.use('fivethirtyeight')
    plt.title('What Percentage changed in Result after Halftime?', fontsize=20)
    plt.legend(labels_pie, loc='lower left')
    plt.show()

# The relation between Betting Data and final result of the game
def plt_corr_betting_result(laliga):
    plt.figure(figsize=(15, 15))
    plt.suptitle("Relation between Betting Result from different company and Game Result",
                 fontsize=20, fontweight='bold')
    plt.subplots_adjust(wspace=0.8, hspace=0.8)
    df_2 = laliga.drop(laliga.iloc[:, 0:4], axis=1)
    df_2 = df_2.drop(df_2.iloc[:, 1:16], axis=1)
    print(df_2)
    df_2['FTR'] = laliga['FTR']
    df_plot = df_2.corr()
    ax = sns.heatmap(df_plot, annot=True, square=True)
    plt.show()

# The relation between final result of the game and other statistical data
def plt_corr_goals_result(laliga):
    plt.figure(figsize=(15, 15))
    plt.suptitle("Relation between result and other result related data",
                 fontsize=20, fontweight='bold')
    plt.subplots_adjust(wspace=0.8, hspace=0.8)
    df_2 = laliga.drop(laliga.iloc[:, 20:], axis=1)
    df_2 = df_2.drop(laliga.iloc[:, 0:2], axis=1)
    print(df_2)
    df_2['FTR'] = laliga['FTR']
    df_plot = df_2.corr()
    ax = sns.heatmap(df_plot, annot=True, square=True)
    plt.show()


# number of PCA explained variance analysis
def plt_PCA(laliga):
    pca = PCA().fit(laliga)
    print(np.cumsum(pca.explained_variance_ratio_))
    plt.figure(figsize=(15, 10))
    plt.suptitle("Explained variance of number of PCA",
                 fontsize=20, fontweight='bold')
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.show()


# EDA refers to different websites:
# https://medium.com/nerd-for-tech/premier-league-predictions-using-artificial-intelligence-7421dddc8778
# https://medium.com/analytics-vidhya/using-data-science-to-analyze-the-premier-league-b468c5b836ba
# https://github.com/HanDarkholme/EDA-on-football
# https://www.kaggle.com/sayakchakraborty/eda-of-football-league-data/notebook
# https://towardsdatascience.com/lets-learn-exploratory-data-analysis-practically-4a923499b779
# https://nycdatascience.com/blog/student-works/college-football-eda-r/
# https://www.kaggle.com/pasinduranasinghe123/fifa-2022-eda-prediction-model

# 1. Background

Football (called soccer in the US) is one of the most popular and well-known sports globally. Hence, football match prediction has caught many people's interests, such as betting crowds and companies, researchers, sports teams' management, etc. However, plenty of factors like weather, location, teams’ lineup, yellow or red cards, home advantage or any other factor could be affected the result of the football match. Therefore, the probability of the result will keep changing within 90 minutes of a football match.
Traditionally, the experts had been predicted the results by different statistical models. One of the most popular statistical models, Markov Chain Monte Carlo [1], attempts to evaluate the game's results by analyzing the strength difference between players, the psychological impact of under-estimate the opponent, calculating the attack intensity, Etc. This mathematical model can reflect a certain extent of the match information. Nevertheless, with the increasing number of football-related information referenced, statistical models are hard to handle a huge amount of data with too many other factors.
Due to the limitation of the statistical models, this research attempted to estimate the abilities of three different machine learning models, SVC, RF and MLP, for predicting football match results which shows a better performance for predicting football matches [2]. The dataset for training the models was acquired by https://www.football-data.co.uk/spainm.php.
A literature review of the previous related work in Section 2. The detail of the dataset description in Section 3. The experiments, exploration data analysis and model evaluation are in Section 4. The comparison of different model and data performances in Section 5.

# 2. Dataset description

There are three seasons of La Liga from 2019-2020 to 2021-2022 had been extracted from the website Football-Data.co.uk. The 2019-2020, 2020-2021, and 2021-2022 files contain 380, 380 and 258 matches. Each match originally had 105 columns of data such as match results and statistics, result odds (Home Win, Draw, Away Team), total goals odds, Asian handicap odds and closing odds provided by six different companies. Except for result odds, the odds data were not related to predicting the result by observation. In addition, since the matches were independent of each other rather than a time series relationship, the date, time, and the unrelated betting odds had been dropped first. The shape of the initial dataset is 1015 matches and 38 columns of data after combining all dataset.

# 3.Summary

This study refers to the model design of the literature [3] and [4] and finds out the advantages and disadvantages of the related paper. At the same time, The research refers to the book's content [7] to find out the model that the researchers think is the most suitable for predicting the outcome of the football match and conducting experiments. By analysing the experimental results of three models, including SVM, RF and MLP, it is found that SVM is the most accurate in training this dataset, with an accuracy rate and f1 score of 60.09% and 57.17%. The experimental results show that SVM is more suitable for training football data than RF and MLP.
In the process of EDA, the research found that many data have a certain correlation. For example, Shot target and the number of corners have a particular relationship with the number of goals; the odds of the gambling company will have a significant correlation with the player's result; the home field has a more significant win rate than the away game, etc.

# 4. References

[1] M. J. Dixon and S. G. Coles, “Modelling association football scores and inefficiencies in the football betting market,” Journal of the Royal Statistical Society: Series C (Applied Statistics), vol. 46, no. 2, pp. 265–280, 1997.

[2] M. P. da Silva, F. Gonçalves, and L. Ramos, “Football Classification Predications,” GitHub. [Online]. Available: https://github.com/motapinto/football-classification- predications/blob/master/src/Supervised%20Learning%20Models.ipynb. [Accessed: 10-Mar-2022].

[3] N. Razali, A. Mustapha, F. A. Yatim, and R. Ab Aziz, “Predicting football matches results using Bayesian Networks for English Premier League (EPL),” IOP Conference Series: Materials Science and Engineering, vol. 226, p. 012099, 2017.

[4] S. Samba, “Football result prediction by deep learning algorithms.” [Online]. Available: https://www.researchgate.net/profile/Stefan- Samba/publication/334415630_Football_Result_Prediction_by_Deep_Learning_Algorithms/links/5d2 834b9458515c11c273ba3/Football-Result-Prediction-by-Deep-Learning-Algorithms.pdf. [Accessed: 10-Mar-2022].

[5] H. R. Azhari, Y. Widyaningsih, and D. Lestari, “Predicting final result of football match using Poisson regression model,” Journal of Physics: Conference Series, vol. 1108, p. 012066, 2018.

[6] T. M. Mitchell, Machine learning. New York: McGraw Hill, 2017.

[7] A. Boz, “Large Scale Machine Learning using NVIDIA CUDA,” CodeProject, 09-Mar-2012. [Online]. Available: https://www.codeproject.com/Articles/336147/Large-Scale-Machine-Learning-using- NVIDIA-CUDA. [Accessed: 10-Mar-2022].

[8] E. M. Condon, B. L. Golden, and E. A. Wasil, “Predicting the success of nations at the summer olympics using neural networks,” Computers & Operations Research, vol. 26, no. 13, pp. 1243–1265, 1999.

[9] A. P. Rotshtein, M. Posner, and A. B. Rakityanskaya, “Football predictions based on a fuzzy model with genetic and neural tuning - cybernetics and Systems Analysis,” SpringerLink. [Online]. Available: https://link.springer.com/article/10.1007/s10559-005-0098-4. [Accessed: 11-Mar-2022].

[10] A. J. Silva, A. M. Costa, P. M. Oliveira, V. M. Reis, J. Saavedra, J. Perl, A. Rouboa, and D. A. Marinho, “The use of neural network technology to model swimming performance,” Journal of sports science & medicine, 01-Mar-2007. [Online]. Available: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3778687/. [Accessed: 11-Mar-2022].

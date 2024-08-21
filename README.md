# ads_bandit

main problem: There are some problems with the bandits results. (details listed below)

Traceback (most recent call last):
  File "/logistic_bandit/logistic_contextual_bandit_UCB.py", line 163, in <module>
    alg.learn(context, best_arm, reward)
  File "/logistic_bandit/logistic_contextual_bandit_UCB.py", line 78, in learn
    raise ValueError(msg)
ValueError:  Oops. ContextualECOLog has a problem: the data-dependent condition was not met. This is rare; try increasing the regularization (self.l2reg) 

consumer dataset: audienceInfoOutput.csv

ads dataset: adsInfo_lite.csv

### Noted that this code can only be processed under the environemnt of the original Logistic_Bandit repo: https://github.com/louisfaury/logistic_bandit

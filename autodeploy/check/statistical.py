import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import ks_2samp


def test_corr(x1, x2):
    """
      Test correlation matrix similarity using
      signed distances.

      Parameters
      ----------
      x1 : DataFrame.
          Numerical features of the original data.
      x2 : DataFrame.
          Numerical features of the new data.

      Returns
      -------
      corr_df : DataFrame
          Dataframe with the distances.
      explain_df : DataFrame
          Dataframe with the explanations.
    """

    def signed_distance(x1, x2):
        df_t = (x1.corr().abs() - x2.corr().abs()).abs()
        df_t = df_t.where((x1.corr() * x2.corr()) > 0, -df_t)

        return df_t

    def explain_distance(df):
        conditions = [
            (df > 0) & (df <= 0.1),
            (df > 0.1) & (df <= 0.2),
            (df > 0.2),
            (df < 0)]

        choices = ['good', 'medium', 'bad', 'diff direction']
        df = np.select(conditions, choices, default='diagonal')

        return pd.DataFrame(df)

    corr_df = signed_distance(x1, x2)
    explain_df = explain_distance(corr_df)

    return corr_df, explain_df


def test_ks(x1, x2):
    """
      Test Kolmogorov-Smirnov.
      The null hypothesis (H0) is that these two variables
      are drawn from same continuous distribution.

      Parameters
      ----------
      x1 : DataFrame.
          Numerical features of the original data.
      x2 : DataFrame.
          Numerical features of the new data.

      Returns
      -------
      ks_df : DataFrame
          Dataframe with the p-values and explanations.
    """

    def explainer_ks(p_value):
        explain = ''
        # interpretation
        alpha = 0.1
        if p_value < alpha:
            explain = 'Different'  # reject H0
        elif (p_value >= alpha) and (p_value <= 0.4):
            explain = 'Slightly different'  # fail to reject H0
        else:
            explain = 'Comparable'  # fail to reject H0
        return explain

    p_value_dict = {}
    for col in x1:
        stat, p_value = ks_2samp(x1[col], x2[col])
        p_value_dict[col] = [p_value, explainer_ks(p_value)]

    ks_df = pd.DataFrame(p_value_dict, index=['p_value', 'explain'])

    return ks_df


def kolmogorov(x1, x2, test=['ks'], cols=None, verbose=False):
    print(f'########## Comparing two numerical dataframes ##########')

    if 'ks' in test:
        print()
        print(f'######## Kolmogorov-Smirnov test ########')
        n_cols = 3
        if cols is not None:
            df_ks = test_ks(x1[cols], x2[cols])
        else:
            df_ks = test_ks(x1[x1.columns[:n_cols]], x2[x2.columns[:n_cols]])

        if verbose:
            #       x1.plot(kind='density', title='Old matrix')
            fig = plt.figure(figsize=[7, 7])
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212)
            plt.subplots_adjust(hspace=0.3)
            if cols is not None:
                for col in cols:
                    _ = sns.kdeplot(x1[col], ax=ax1).set_title("Old matrix")
                for col in cols:
                    _ = sns.kdeplot(x2[col], ax=ax2).set_title("New matrix")
            else:
                for col in x1.columns[:n_cols]:
                    _ = sns.kdeplot(x1[col], ax=ax1).set_title("Old matrix")
                for col in x2.columns[:n_cols]:
                    _ = sns.kdeplot(x2[col], ax=ax2).set_title("New matrix")

            #       _ = sns.distplot(x1).set_title("Old matrix")
            #       x2.plot(kind='density', title='New matrix')
            plt.show()

        print(df_ks)

    if 'corr' in test:
        print()
        print(f'######## Correlation matrix similarity test ########')
        corr_df, explain_df = test_corr(x1[x1.columns[:n_cols]],
                                        x2[x2.columns[:n_cols]])
        if verbose:
            fig = plt.figure(figsize=[7, 7])
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212)

            plt.subplots_adjust(hspace=0.3)
            _ = sns.heatmap(x1[x1.columns[:n_cols]].corr(), vmin=-1, vmax=1,
                            annot=True, ax=ax1).set_title("Old matrix")
            pp1 = sns.pairplot(x1, height=1.5)
            pp1.fig.suptitle("Old matrix", y=1.02)

            _ = sns.heatmap(x2[x2.columns[:n_cols]].corr(), vmin=-1, vmax=1,
                            annot=True, ax=ax2).set_title("New matrix")
            pp2 = sns.pairplot(x2, height=1.5)
            pp2.fig.suptitle("New matrix", y=1.02)
            plt.show()
            print(corr_df)

        print(explain_df)

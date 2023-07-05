from sklearn.preprocessing import StandardScaler
import math
import numpy as np


def returns(model, data, amount, window=5, vol=40, conf=0.5, exp=2.4, alpha=0.5):
    """
    Function to execute the strategy and compute the returns using a trained classifier.

    Args:
        model (sklearn model): The model to use for predictions.
        data (pd.DataFrame): The dataframe containing the data.
        amount (float): The amount of money to use for trading.
        window (int, optional): The window size to use for predictions. Defaults to 5.
        vol (int, optional): The maximum volume of shares are allowed to trade in a single day. Defaults to 40.
        conf (float, optional): The initial confidence multiplier that decides the volume to trade. Defaults to 0.5.
        exp (float, optional): The exponent to use for the confidence multiplier. A higher value tends to bring a lower risk, but the overall returns might reduce. Defaults to 2.4.
        alpha (float, optional): The moving average coefficient for confidence. Defaults to 0.5.

    Returns:
        float: The returns as a decimal.
    """
    i = 0 # window index
    holding = 0 # number of shares held
    bal = amount # balance
    scaler = StandardScaler()
    prediction = np.zeros(window) # initial prediction, this value is used to calculate confidence
    data["pnl"] = amount # profit/loss column
    while True:
        df_train = data.iloc[i * window : (i * window) + window]

        if len(df_train) < window:
            break

        features = [x for x in df_train.columns if x not in ["pred", "pnl"]]
        X = df_train[features]
        X = scaler.fit_transform(X)
        new_prediction = model.predict(X)
        for j in range(window):
            cur_rate = df_train["close"].values[j]
            if i >= 1:
                prev_rate = data.iloc[(i - 1) * window + j]["close"]
                result = prediction[j] == (cur_rate > prev_rate) # 1 if earlier prediction was correct, 0 if wrong
                conf = (1 - alpha) * conf + alpha * result # confidence as a moving average
            shares = math.floor(vol * conf**exp) # volume to trade
            max_shares = math.floor(bal / cur_rate) # maximum shares that can be bought
            if new_prediction[j] == 1:
                ex = min(shares, max_shares) # number of shares to buy
                bal -= ex * cur_rate # reduce balance
                holding += ex # increase holding
            else:
                ex = min(shares, holding) # number of shares to sell
                holding -= ex # reduce holding
                bal += ex * cur_rate # increase balance
            data.iloc[i * window + j, data.columns.get_loc("pnl")] = (
                bal + holding * cur_rate # update profit/loss
            )
        prediction = new_prediction # update prediction
        i += 1

    # print(f"balance = {bal}")
    # print(f"final holding = {holding}")
    closing_rate = data["close"].values[-1]
    # print(f"closing rate = {closing_rate}")
    # print(f"total = {bal + holding*closing_rate}")
    # print(f"return % = {ret}")
    ret = (bal + holding * closing_rate) / amount
    return ret


def calculate_max_drawdown(pnl):
    """
    Function to calculate the maximum drawdown from a dataframe column 'pnl'.

    Args:
        pnl (pd.Series): A pandas series representing the profit/loss values.

    Returns:
        float: The maximum drawdown value as a decimal.
    """
    cummax = pnl.cummax()  # Calculate cumulative maximum
    drawdown = (
        pnl - cummax
    ) / cummax  # Calculate drawdown as a percentage of the cumulative maximum
    max_drawdown = drawdown.min()  # Get the minimum drawdown value
    return max_drawdown


def calculate_sharpe_ratio(pnl, risk_free_rate):
    """
    Function to calculate the Sharpe ratio from a dataframe column 'pnl' and a risk-free rate.

    Args:
        pnl (pd.Series): A pandas series representing the profit/loss values.
        risk_free_rate (float): The risk-free rate for the given period.

    Returns:
        float: The Sharpe ratio.
    """
    returns = pnl.pct_change()  # Calculate the returns as the percentage change of pnl
    annualized_returns = (
        np.mean(returns) * 252
    )  # Calculate the annualized returns assuming 252 trading days in a year
    volatility = np.std(returns) * np.sqrt(
        252
    )  # Calculate the volatility (standard deviation of returns)
    sharpe_ratio = (
        annualized_returns - risk_free_rate
    ) / volatility  # Calculate the Sharpe ratio
    return sharpe_ratio

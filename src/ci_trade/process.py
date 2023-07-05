from finta import TA


def _get_indicator_data(df):
    df['SMA5'] = TA.SMA(df, 5)
    df['SMA10'] = TA.SMA(df, 10)
    df['SMA15'] = TA.SMA(df, 15)
    df['SMA20'] = TA.SMA(df, 20)

    # exponential moving averages
    df['EMA5'] = TA.EMA(df, 5)
    df['EMA10'] = TA.EMA(df, 10)
    df['EMA15'] = TA.EMA(df, 15)
    df['EMA20'] = TA.EMA(df, 20)

    # Bollinger Bands
    df['upperband'] = TA.BBANDS(df)['BB_UPPER']
    df['middleband'] = TA.BBANDS(df)['BB_MIDDLE']
    df['lowerband'] = TA.BBANDS(df)['BB_LOWER']

    # Kaufman's Adaptive Moving Average
    df['KAMA10'] = TA.KAMA(df, 10)
    df['KAMA20'] = TA.KAMA(df, 20)
    # df['KAMA30'] = TA.KAMA(df, 30)
    df['SAR'] = TA.SAR(df, 0.02, 0.2)
    df['TRIMA5'] = TA.TRIMA(df, 5)
    df['TRIMA10'] = TA.TRIMA(df, 10)
    df['TRIMA20'] = TA.TRIMA(df, 20)
    df['ADX5'] = TA.ADX(df, 5)
    df['ADX10'] = TA.ADX(df, 10)
    df['ADX20'] = TA.ADX(df, 20)
    df['CCI5'] = TA.CCI(df, 5)
    df['CCI10'] = TA.CCI(df, 10)
    df['CCI15'] = TA.CCI(df, 15)
    df['MACD510'] = TA.MACD(df, period_fast=5, period_slow=10).iloc[:, 0]
    df['MACD520'] = TA.MACD(df, period_fast=5, period_slow=20).iloc[:, 0]
    df['MACD1020'] = TA.MACD(df, period_fast=10, period_slow=20).iloc[:, 0]
    df['MACD1520'] = TA.MACD(df, period_fast=15, period_slow=20).iloc[:, 0]
    df['MACD1226'] = TA.MACD(df, period_fast=12, period_slow=26).iloc[:, 0]
    df['MOM10'] = TA.MOM(df, 10)
    df['MOM15'] = TA.MOM(df, 15)
    df['MOM20'] = TA.MOM(df, 20)
    df['ROC5'] = TA.ROC(df, 5)
    df['ROC10'] = TA.ROC(df, 10)
    df['ROC20'] = TA.ROC(df, 20)
    df['PPO'] = TA.PPO(df).iloc[:, 0]
    df['RSI14'] = TA.RSI(df, 14)
    df['RSI8'] = TA.RSI(df, 8)

    df['fastk'] = TA.STOCH(df)
    df['fastd'] = TA.STOCHD(df)
    df['fastrsi'] = TA.STOCHRSI(df)
    # Ultimate Oscillator (ULTOSC)
    df['ULTOSC'] = TA.UO(df)
    # Williams %R (WILLR) - 14 period
    df['WILLR'] = TA.WILLIAMS(df)
    # Average True Range (ATR)
    df['ATR7'] = TA.ATR(df, 7)
    df['ATR14'] = TA.ATR(df, 14)
    # True Range (TRange)
    df['Trange'] = TA.TR(df)
    # Typical Price (TYPPRICE)
    df['TYPPRICE'] = TA.TP(df)
    df['VIm'] = TA.VORTEX(df)['VIm']
    df['VIp'] = TA.VORTEX(df)['VIp']
    # Money Flow Volume (MFV) (When the data has the volume)
    if 'volume' in df.columns:
        df['MFV'] = TA.MFI(df)
        del (df['volume'])
    del (df['open'])
    del (df['high'])
    del (df['low'])
    return df


def _exponential_smooth(data, alpha):
    """
    Function that exponentially smooths dataset so values are less 'rigid'
    :param alpha: weight factor to weight recent values more
    """

    return data.ewm(alpha=alpha).mean()


def _produce_prediction(data, lookahead):
    """
    Function that produces the 'truth' values
    At a given row, it looks 'window' rows ahead to see if the price increased (1) or decreased (0)
    :param window: number of days, or rows to look ahead to see what the price did
    """
    
    prediction = (data.shift(-lookahead)['close'] >= data['close'])
    prediction = prediction.iloc[:-lookahead]
    data['pred'] = prediction.astype(int)
    
    return data
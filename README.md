# trading-using-ml-ci
This repository contains the code for an exploratory project created under our semester course Computation Intelligence. It is based on creating a trading strategy using `scikit-learn` classifiers that are trained to predict whether the price of a stock (or derivative) goes up or down based on the current trend in the market. 

### How To Run
The dependencies can be installed using the requirements file by running
```sh
pip install -r requirements.txt
```
You can then run the cells in `src/example.ipynb` and view the outputs corresponding to some sample parameters. The default dataset is the 22 years of NIFTY data that is split into 18 years to train and 4 years to test the model. The parameters in the `returns` function correspond to the coefficients associated with the trading function. The heart of this strategy determines a confidence value that combines with the prediction of our model to trade a volume of shares. This is done to ensure that the drawdown is minimized and the Sharpe ratio is as high as possible.

### Documentation
You can refer to the reports/presentations that we submitted for the project evaluation to learn more about the theory that goes behind the project.

### Contributors
- @[vihaanthora](https://github.com/vihaanthora)
- @[gjain-7](https://github.com/gjain-7/)

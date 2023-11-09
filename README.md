# Unveiling commonalities and differences in genetic regulations via two-way fusion
This is a `Python` implementation of the following paper:
Mei, B., Jian, Y., Sun, Y. (2023). Unveiling commonalities and differences in genetic regulations via two-way fusion.

# Introduction
This code implements the iterative algorithm introduced in the article to solve the  proposed  objective function:

$$H(\tilde\beta)=	\sum_{m=1}^M\frac1{n^m}
	    	\sum\limits_{
	    i=v_m+1
	    	}^{v_{m+1}}
    	\rho_\delta\left((\tilde y-\tilde{X}\tilde\beta)_i\right)+\lambda_1\|\tilde\beta\|_1 +\lambda_{2}\sum_{m_1< m_2}\sum_{k=1}^{q}\|\beta_{\cdot k}^{m_{1}}-\beta_{\cdot k}^{m_{2}}\|_{2}+\lambda_{3}\sum_{m_1<m_2}\sum_{j=1}^{p}\|\beta_{j.}^{m_{1}}-\beta_{j.}^{m_{2}}\|_{2}
$$
The algorithm starts from an initial estimate $\tilde{\beta^0}$ , and then update $\tilde{\beta}^{t+1}$ by solving the following convex optimization problem at t-th iteration.
$$
\tilde{\beta}^{t+1}=\arg\min\left(\sum_{m=1}^M\frac1{n^m}
		\sum\limits_{
			i=v_m+1
		}^{v_{m+1}}
		\rho_\delta\left((\tilde y-\tilde{ X}\tilde \beta)_i\right)\right)+\frac{1}{2}\tilde \beta^\top {S}^{(t)}\tilde\beta+\frac{1}{2}\tilde \beta^\top{F}^{(t)}\tilde\beta.$$

After obtaining $\tilde\beta^{(t+1)}$, we update $S^{(t+1)}$ and  $F^{(t+1)}$, respectively. We repeat this two-step procedure until convergence.

# Requirements

* Python3
* Package numpy; panads; seaborn; matplotlib; scipy;

# Contents
* All usable function and class:
   TWF, heatmap.

# Demo
* Generate simulated data under "S2" setting
```python
#Generate simulated data based on the three given coefficient matrix
B1 =pd.read_csv(r'B1.csv', index_col=0).values
B2 =pd.read_csv(r'B2.csv', index_col=0).values
B3 =pd.read_csv(r'B3.csv', index_col=0).values
B = [B1, B2, B3]
cov = np.zeros((100, 100))
mean  = np.zeros(100)
for i in range(100):
    for j in range(100):
        cov[i][j] = pow(0.3, np.abs(i - j))
X = []
X.append(np.random.multivariate_normal(mean=mean, cov=cov, size=60))
X.append(np.random.multivariate_normal(mean=mean, cov=cov, size=80))
X.append(np.random.multivariate_normal(mean=mean, cov=cov, size=100))
Y = []
for k in range(3):
    err = np.random.standard_t(3, size=X[k].shape)
    Y.append(X[k] @ B[k] + err)
#The ratio of training set to test set is 4:1
X_train = [X[0][:48, :], X[1][:64, :], X[2][:80, :]]
X_test = [X[0][48:, :], X[1][64:, :], X[2][80:, :]]
Y_train = [Y[0][:48, :], Y[1][:64, :], Y[2][:80, :]]
Y_test = [Y[0][48:, :], Y[1][64:, :], Y[2][80:, :]]
```
* Implement algorithm
```python
lam = np.array([0.2, 0.25, 0.25])
model = TWF(max_iter = 30, lam_exc=0.2, lam_fus=np.array([0.25, 0.25]))
model.fit(X_train, Y_train)
```
* Evaluation
```python
Y_pred = model.pre(X_test)
W = np.stack([model.beta[0], model.beta[1], model.beta[2]])
Pmse = np.mean((np.vstack(Y_pred) - np.vstack(Y_test))**2)
heatmap(W, col = 'seismic')
# Calculate pmse and draw a heatmap of the estimated results
```

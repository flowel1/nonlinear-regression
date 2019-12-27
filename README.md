# Levenberg-Marquardt algorithm for nonlinear regression

## Nonlinear regression: generic statement
Assume that we have N observations

<a href="https://www.codecogs.com/eqnedit.php?latex=(\underline{x}_1,&space;y_1),&space;...,&space;(\underline{x}_N,&space;y_N)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?(\underline{x}_1,&space;y_1),&space;...,&space;(\underline{x}_N,&space;y_N)" title="(\underline{x}_1, y_1), ..., (\underline{x}_N, y_N)" /></a>

where <a href="https://www.codecogs.com/eqnedit.php?latex=\underline{x}_i&space;\in&space;\mathbb{R}^n" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\underline{x}_i&space;\in&space;\mathbb{R}^n" title="\underline{x}_i \in \mathbb{R}^n" /></a> are observable predictors and <a href="https://www.codecogs.com/eqnedit.php?latex=y_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y_i" title="y_i" /></a> are target real variables (i.e., the variables that must be predicted).

We would like to estimate a nonlinear model of the form

<a href="https://www.codecogs.com/eqnedit.php?latex=y&space;=&space;f(\underline{x};\underline{\theta})&space;&plus;&space;\epsilon" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y&space;=&space;f(\underline{x};\underline{\theta})&space;&plus;&space;\epsilon" title="y = f(\underline{x};\underline{\theta}) + \epsilon" /></a>

where <ins>&theta;</ins>  is a vector of _k_ unknown real parameters, _f_ is a known function nonlinear in <ins>&theta;</ins> and <a href="https://www.codecogs.com/eqnedit.php?latex=\epsilon&space;\sim&space;N(0,&space;\sigma^2)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\epsilon&space;\sim&space;N(0,&space;\sigma^2)" title="\epsilon \sim N(0, \sigma^2)" /></a> for some positive value of &sigma;.

Setting

<a href="https://www.codecogs.com/eqnedit.php?latex=X&space;=\begin{bmatrix}&space;\underline{x}_1&space;\\&space;...&space;\\&space;\underline{x}_N&space;\\&space;\end{bmatrix}&space;\in&space;\mathbb{R}^{N&space;\times&space;n},&space;\hspace{10}&space;\underline{y}&space;=&space;\begin{bmatrix}&space;y_1&space;\\&space;...&space;\\&space;y_N&space;\end{bmatrix}&space;\in&space;\mathbb{R}^N,&space;\hspace{10}&space;\underline{\epsilon}&space;=&space;\begin{bmatrix}&space;\epsilon_1&space;\\&space;...&space;\\&space;\epsilon_N&space;\end{bmatrix}&space;\in&space;\mathbb{R}^N" target="_blank"><img src="https://latex.codecogs.com/gif.latex?X&space;=\begin{bmatrix}&space;\underline{x}_1&space;\\&space;...&space;\\&space;\underline{x}_N&space;\\&space;\end{bmatrix}&space;\in&space;\mathbb{R}^{N&space;\times&space;n},&space;\hspace{10}&space;\underline{y}&space;=&space;\begin{bmatrix}&space;y_1&space;\\&space;...&space;\\&space;y_N&space;\end{bmatrix}&space;\in&space;\mathbb{R}^N,&space;\hspace{10}&space;\underline{\epsilon}&space;=&space;\begin{bmatrix}&space;\epsilon_1&space;\\&space;...&space;\\&space;\epsilon_N&space;\end{bmatrix}&space;\in&space;\mathbb{R}^N" title="X =\begin{bmatrix} \underline{x}_1 \\ ... \\ \underline{x}_N \\ \end{bmatrix} \in \mathbb{R}^{N \times n}, \hspace{10} \underline{y} = \begin{bmatrix} y_1 \\ ... \\ y_N \end{bmatrix} \in \mathbb{R}^N, \hspace{10} \underline{\epsilon} = \begin{bmatrix} \epsilon_1 \\ ... \\ \epsilon_N \end{bmatrix} \in \mathbb{R}^N" /></a>

and assuming that the N observations are independent, the log-likelihood of the model given our _N_ observations is

<a href="https://www.codecogs.com/eqnedit.php?latex=L(\underline{\theta}&space;|&space;X,&space;\underline{y})&space;=&space;\prod_{i&space;=&space;1}^N&space;\frac{1}{\sigma\sqrt{2\pi}}\exp&space;\left&space;\{&space;-&space;\left(&space;\frac{y_i-f(\underline{x}_i;\underline{\theta})}{\sigma}&space;\right)&space;^&space;2&space;\right&space;\}&space;\Rightarrow&space;\log&space;L(\underline{\theta}&space;|&space;X,&space;\underline{y})&space;=&space;-\frac{1}{(2\pi\sigma^2)^\frac{N}{2}}&space;\sum_{i=1}^N&space;\left(&space;\frac{y_i-f(\underline{x}_i;\underline{\theta})}{\sigma}&space;\right)&space;^&space;2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?L(\underline{\theta}&space;|&space;X,&space;\underline{y})&space;=&space;\prod_{i&space;=&space;1}^N&space;\frac{1}{\sigma\sqrt{2\pi}}\exp&space;\left&space;\{&space;-&space;\left(&space;\frac{y_i-f(\underline{x}_i;\underline{\theta})}{\sigma}&space;\right)&space;^&space;2&space;\right&space;\}&space;\Rightarrow&space;\log&space;L(\underline{\theta}&space;|&space;X,&space;\underline{y})&space;=&space;-\frac{1}{(2\pi\sigma^2)^\frac{N}{2}}&space;\sum_{i=1}^N&space;\left(&space;\frac{y_i-f(\underline{x}_i;\underline{\theta})}{\sigma}&space;\right)&space;^&space;2" title="L(\underline{\theta} | X, \underline{y}) = \prod_{i = 1}^N \frac{1}{\sigma\sqrt{2\pi}}\exp \left \{ - \left( \frac{y_i-f(\underline{x}_i;\underline{\theta})}{\sigma} \right) ^ 2 \right \} \Rightarrow \log L(\underline{\theta} | X, \underline{y}) = -\frac{1}{(2\pi\sigma^2)^\frac{N}{2}} \sum_{i=1}^N \left( \frac{y_i-f(\underline{x}_i;\underline{\theta})}{\sigma} \right) ^ 2" /></a>

We estimate the model parameters by maximizing the log-likelihood with respect to <ins>&theta;</ins>. This is equivalent to minimizing the following objective function (sum of squared residuals):

<a href="https://www.codecogs.com/eqnedit.php?latex=\text{Obj}(\underline{\theta})&space;=&space;\left&space;\|&space;\underline{y}&space;-&space;f(X,&space;\underline{\theta})&space;\right&space;\|&space;^&space;2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\text{Obj}(\underline{\theta})&space;=&space;\left&space;\|&space;\underline{y}&space;-&space;f(X,&space;\underline{\theta})&space;\right&space;\|&space;^&space;2" title="\text{Obj}(\underline{\theta}) = \left \| \underline{y} - f(X, \underline{\theta}) \right \| ^ 2" /></a>

## The Levenberg-Marquardt algorithm

Start from <a href="https://www.codecogs.com/eqnedit.php?latex=\underline{\theta}^{(0)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\underline{\theta}^{(0)}" title="\underline{\theta}^{(0)}" /></a> and approximate the objective function around <a href="https://www.codecogs.com/eqnedit.php?latex=\underline{\theta}^{(0)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\underline{\theta}^{(0)}" title="\underline{\theta}^{(0)}" /></a> with the following quadratic function:

<a href="https://www.codecogs.com/eqnedit.php?latex=\phi(\underline{\delta})&space;=&space;\left&space;\|&space;\underline{y}&space;-&space;f(X,&space;\underline{\theta}^{(0)})&space;-&space;J_{\underline{\theta}}f(X,&space;\underline{\theta})_{|_{\underline{\theta}&space;=&space;\underline{\theta}^{(0)}}}&space;\cdot&space;\underline{\delta}&space;\right&space;\|&space;^2&space;\sim&space;\text{Obj}(\underline{\theta}^{(0)}&space;&plus;&space;\underline{\delta})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\phi(\underline{\delta})&space;=&space;\left&space;\|&space;\underline{y}&space;-&space;f(X,&space;\underline{\theta}^{(0)})&space;-&space;J_{\underline{\theta}}f(X,&space;\underline{\theta})_{|_{\underline{\theta}&space;=&space;\underline{\theta}^{(0)}}}&space;\cdot&space;\underline{\delta}&space;\right&space;\|&space;^2&space;\sim&space;\text{Obj}(\underline{\theta}^{(0)}&space;&plus;&space;\underline{\delta})" title="\phi(\underline{\delta}) = \left \| \underline{y} - f(X, \underline{\theta}^{(0)}) - J_{\underline{\theta}}f(X, \underline{\theta})_{|_{\underline{\theta} = \underline{\theta}^{(0)}}} \cdot \underline{\delta} \right \| ^2 \sim \text{Obj}(\underline{\theta}^{(0)} + \underline{\delta})" /></a>

Thanks to the objective function's special form, we can calculate a local quadratic approximation by taking the first order expansion of _f_ instead of the second-order expansion of the objective function itself.

Defining for simplicity

<a href="https://www.codecogs.com/eqnedit.php?latex=J^{(0)}&space;:=&space;J_{\underline{\theta}}f(X,&space;\underline{\theta})_{|_{\underline{\theta}&space;=&space;\underline{\theta}^{(0)}}}&space;\in&space;\mathbb{R}^{N&space;\times&space;k},&space;\hspace{10}&space;\underline{\epsilon}^{(0)}&space;:=&space;\underline{y}&space;-&space;f(X,&space;\underline{\theta}^{(0)})&space;\in&space;\mathbb{R}^N" target="_blank"><img src="https://latex.codecogs.com/gif.latex?J^{(0)}&space;:=&space;J_{\underline{\theta}}f(X,&space;\underline{\theta})_{|_{\underline{\theta}&space;=&space;\underline{\theta}^{(0)}}}&space;\in&space;\mathbb{R}^{N&space;\times&space;k},&space;\hspace{10}&space;\underline{\epsilon}^{(0)}&space;:=&space;\underline{y}&space;-&space;f(X,&space;\underline{\theta}^{(0)})&space;\in&space;\mathbb{R}^N" title="J^{(0)} := J_{\underline{\theta}}f(X, \underline{\theta})_{|_{\underline{\theta} = \underline{\theta}^{(0)}}} \in \mathbb{R}^{N \times k}, \hspace{10} \underline{\epsilon}^{(0)} := \underline{y} - f(X, \underline{\theta}^{(0)}) \in \mathbb{R}^N" /></a>,

we have that this quadratic approximation reaches its minimum when

<a href="https://www.codecogs.com/eqnedit.php?latex=\nabla_{\underline{\delta}}&space;\hspace{2}&space;\phi(\underline{\delta})&space;=&space;-2&space;\cdot&space;J^{(0)}^T&space;\cdot&space;(\underline{\epsilon}^{(0)}&space;-&space;J^{(0)}&space;\cdot&space;\underline{\delta})&space;=&space;\underline{0}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\nabla_{\underline{\delta}}&space;\hspace{2}&space;\phi(\underline{\delta})&space;=&space;-2&space;\cdot&space;J^{(0)}^T&space;\cdot&space;(\underline{\epsilon}^{(0)}&space;-&space;J^{(0)}&space;\cdot&space;\underline{\delta})&space;=&space;\underline{0}" title="\nabla_{\underline{\delta}} \hspace{2} \phi(\underline{\delta}) = -2 \cdot J^{(0)}^T \cdot (\underline{\epsilon}^{(0)} - J^{(0)} \cdot \underline{\delta}) = \underline{0}" /></a>

which is satisfied when the displacement <ins>&delta;</ins> solves the following linear system:

<a href="https://www.codecogs.com/eqnedit.php?latex=(J^{(0)}^T&space;\cdot&space;J^{(0)})&space;\cdot&space;\underline{\delta}&space;=&space;J^{(0)}&space;\cdot&space;\underline{\epsilon}^{(0)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?(J^{(0)}^T&space;\cdot&space;J^{(0)})&space;\cdot&space;\underline{\delta}&space;=&space;J^{(0)}&space;\cdot&space;\underline{\epsilon}^{(0)}" title="(J^{(0)}^T \cdot J^{(0)}) \cdot \underline{\delta} = J^{(0)} \cdot \underline{\epsilon}^{(0)}" /></a>

![alt-text](https://github.com/flowel1/nonlinear-regression/blob/master/pictures/quadratic-approx.png)

## Extension: weighted observations and priors

<a href="https://www.codecogs.com/eqnedit.php?latex=\text{Gaussian&space;prior:&space;}&space;p(\theta_j)&space;=&space;\frac{\beta_j}{\sqrt{2&space;\pi}}&space;\cdot&space;\exp{&space;\left&space;(&space;-\frac{1}{2}&space;\beta_j^2&space;(\theta_j&space;-&space;\mu_j)^2&space;\right&space;)&space;}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\text{Gaussian&space;prior:&space;}&space;p(\theta_j)&space;=&space;\frac{\beta_j}{\sqrt{2&space;\pi}}&space;\cdot&space;\exp{&space;\left&space;(&space;-\frac{1}{2}&space;\beta_j^2&space;(\theta_j&space;-&space;\mu_j)^2&space;\right&space;)&space;}" title="\text{Gaussian prior: } p(\theta_j) = \frac{\beta_j}{\sqrt{2 \pi}} \cdot \exp{ \left ( -\frac{1}{2} \beta_j^2 (\theta_j - \mu_j)^2 \right ) }" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\text{Lognormal&space;prior:&space;}&space;p(\theta_j)&space;=&space;\frac{\beta_j}{\theta_j&space;\cdot&space;\sqrt{2&space;\pi}}&space;\cdot&space;\exp{&space;\left&space;(&space;-\frac{1}{2}&space;\beta_j^2&space;(\log\theta_j&space;-&space;\log\mu_j)^2&space;\right&space;)&space;}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\text{Lognormal&space;prior:&space;}&space;p(\theta_j)&space;=&space;\frac{\beta_j}{\theta_j&space;\cdot&space;\sqrt{2&space;\pi}}&space;\cdot&space;\exp{&space;\left&space;(&space;-\frac{1}{2}&space;\beta_j^2&space;(\log\theta_j&space;-&space;\log\mu_j)^2&space;\right&space;)&space;}" title="\text{Lognormal prior: } p(\theta_j) = \frac{\beta_j}{\theta_j \cdot \sqrt{2 \pi}} \cdot \exp{ \left ( -\frac{1}{2} \beta_j^2 (\log\theta_j - \log\mu_j)^2 \right ) }" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=L(\underline{\theta})&space;=p(\underline{\theta}&space;|&space;X,&space;\underline{y})&space;=&space;\frac{p(\underline{y}&space;|&space;X,&space;\underline{\theta})&space;\cdot&space;p(\underline{\theta})}{p(\underline{y}&space;|&space;X)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?L(\underline{\theta})&space;=p(\underline{\theta}&space;|&space;X,&space;\underline{y})&space;=&space;\frac{p(\underline{y}&space;|&space;X,&space;\underline{\theta})&space;\cdot&space;p(\underline{\theta})}{p(\underline{y}&space;|&space;X)}" title="L(\underline{\theta}) =p(\underline{\theta} | X, \underline{y}) = \frac{p(\underline{y} | X, \underline{\theta}) \cdot p(\underline{\theta})}{p(\underline{y} | X)}" /></a>

Since

<a href="https://www.codecogs.com/eqnedit.php?latex=p(\underline{y}&space;|&space;X,&space;\underline{\theta})&space;=&space;\prod_{i&space;=&space;1}^N&space;p(y_i&space;|&space;\underline{x}_i,&space;\underline{\theta})&space;=&space;\prod_{i&space;=&space;1}^N&space;\frac{1}{\sigma&space;\sqrt{\frac{2&space;\pi}{w_i}}}&space;\cdot&space;\exp{&space;\left(&space;-\frac{1}{2}&space;w_i&space;\frac{(y_i&space;-&space;f(\underline{x}_i,&space;\underline{\theta}))^2}{\sigma^2}{}\right)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p(\underline{y}&space;|&space;X,&space;\underline{\theta})&space;=&space;\prod_{i&space;=&space;1}^N&space;p(y_i&space;|&space;\underline{x}_i,&space;\underline{\theta})&space;=&space;\prod_{i&space;=&space;1}^N&space;\frac{1}{\sigma&space;\sqrt{\frac{2&space;\pi}{w_i}}}&space;\cdot&space;\exp{&space;\left(&space;-\frac{1}{2}&space;w_i&space;\frac{(y_i&space;-&space;f(\underline{x}_i,&space;\underline{\theta}))^2}{\sigma^2}{}\right)}" title="p(\underline{y} | X, \underline{\theta}) = \prod_{i = 1}^N p(y_i | \underline{x}_i, \underline{\theta}) = \prod_{i = 1}^N \frac{1}{\sigma \sqrt{\frac{2 \pi}{w_i}}} \cdot \exp{ \left( -\frac{1}{2} w_i \frac{(y_i - f(\underline{x}_i, \underline{\theta}))^2}{\sigma^2}{}\right)}" /></a>

setting <a href="https://www.codecogs.com/eqnedit.php?latex=W&space;=&space;\text{diag}\{\sqrt{w_1},&space;...&space;\sqrt{w_N}&space;\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?W&space;=&space;\text{diag}\{\sqrt{w_1},&space;...&space;\sqrt{w_N}&space;\}" title="W = \text{diag}\{\sqrt{w_1}, ... \sqrt{w_N} \}" /></a>, we have

<a href="https://www.codecogs.com/eqnedit.php?latex=\log&space;L(\underline{\theta})&space;=&space;\text{const}&space;-\frac{N}{2}&space;\log(\sigma^2)&space;-\frac{1}{2\sigma^2}&space;\left&space;\|&space;W&space;\cdot&space;(\underline{y}&space;-&space;f(X,&space;\underline{\theta}))&space;\right&space;\|^2&space;&plus;&space;\newline&space;-\frac{1}{2}&space;\cdot&space;\sum_{\{j&space;|&space;p(\theta_j)&space;\text{&space;gaussian}\}}&space;\beta_j^2&space;(\theta_j&space;-&space;\mu_j)^2&space;\newline&space;-\frac{1}{2}&space;\cdot&space;\sum_{\{j&space;|&space;p(\theta_j)&space;\text{&space;lognormal}\}}&space;\{&space;\log\theta_j&space;&plus;&space;\beta_j^2&space;(\log\theta_j&space;-&space;\log\mu_j)^2\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\log&space;L(\underline{\theta})&space;=&space;\text{const}&space;-\frac{N}{2}&space;\log(\sigma^2)&space;-\frac{1}{2\sigma^2}&space;\left&space;\|&space;W&space;\cdot&space;(\underline{y}&space;-&space;f(X,&space;\underline{\theta}))&space;\right&space;\|^2&space;&plus;&space;\newline&space;-\frac{1}{2}&space;\cdot&space;\sum_{\{j&space;|&space;p(\theta_j)&space;\text{&space;gaussian}\}}&space;\beta_j^2&space;(\theta_j&space;-&space;\mu_j)^2&space;\newline&space;-\frac{1}{2}&space;\cdot&space;\sum_{\{j&space;|&space;p(\theta_j)&space;\text{&space;lognormal}\}}&space;\{&space;\log\theta_j&space;&plus;&space;\beta_j^2&space;(\log\theta_j&space;-&space;\log\mu_j)^2\}" title="\log L(\underline{\theta}) = \text{const} -\frac{N}{2} \log(\sigma^2) -\frac{1}{2\sigma^2} \left \| W \cdot (\underline{y} - f(X, \underline{\theta})) \right \|^2 + \newline -\frac{1}{2} \cdot \sum_{\{j | p(\theta_j) \text{ gaussian}\}} \beta_j^2 (\theta_j - \mu_j)^2 \newline -\frac{1}{2} \cdot \sum_{\{j | p(\theta_j) \text{ lognormal}\}} \{ \log\theta_j + \beta_j^2 (\log\theta_j - \log\mu_j)^2\}" /></a>

The following contributions are added to the gradient of _&phi;_ in the two cases:
<a href="https://www.codecogs.com/eqnedit.php?latex=\phi_j(\delta_j)&space;=&space;-\frac{1}{2}&space;\cdot&space;\beta_j^2&space;(\theta_j&space;&plus;&space;\delta_j&space;-&space;\mu_j)^2&space;\Rightarrow&space;\frac{d}{d\delta_j}&space;\phi_j(\delta_j)&space;=&space;-\beta_j^2&space;\delta_j&space;-\beta_j^2(\theta_j&space;-&space;\mu_j)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\phi_j(\delta_j)&space;=&space;-\frac{1}{2}&space;\cdot&space;\beta_j^2&space;(\theta_j&space;&plus;&space;\delta_j&space;-&space;\mu_j)^2&space;\Rightarrow&space;\frac{d}{d\delta_j}&space;\phi_j(\delta_j)&space;=&space;-\beta_j^2&space;\delta_j&space;-\beta_j^2(\theta_j&space;-&space;\mu_j)" title="\phi_j(\delta_j) = -\frac{1}{2} \cdot \beta_j^2 (\theta_j + \delta_j - \mu_j)^2 \Rightarrow \frac{d}{d\delta_j} \phi_j(\delta_j) = -\beta_j^2 \delta_j -\beta_j^2(\theta_j - \mu_j)" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\newline&space;\phi_j(\delta_j)&space;=&space;-\frac{1}{2}&space;\beta_j^2(\log\theta_j&space;-&space;\log\mu_j)^2&space;-&space;\log\theta_j&space;\sim&space;\newline&space;\sim&space;-\frac{1}{2}&space;\beta_j^2(\log\theta_j&space;&plus;\frac{\delta_j}{\theta_j}&space;-&space;\log\mu_j)^2&space;-\log\theta_j&space;-\frac{\delta_j}{\theta_j}&space;\Rightarrow&space;\newline&space;\newline&space;\frac{d}{d\delta_j}&space;\phi_j(\delta_j)&space;=&space;-\frac{\beta_j^2}{\theta_j^2}&space;\cdot&space;\delta_j&space;&plus;&space;\{-&space;\frac{1}{\theta_j}&space;-\frac{\beta_j^2}{\theta_j}\log\frac{\theta_j}{\mu_j}&space;\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\newline&space;\phi_j(\delta_j)&space;=&space;-\frac{1}{2}&space;\beta_j^2(\log\theta_j&space;-&space;\log\mu_j)^2&space;-&space;\log\theta_j&space;\sim&space;\newline&space;\sim&space;-\frac{1}{2}&space;\beta_j^2(\log\theta_j&space;&plus;\frac{\delta_j}{\theta_j}&space;-&space;\log\mu_j)^2&space;-\log\theta_j&space;-\frac{\delta_j}{\theta_j}&space;\Rightarrow&space;\newline&space;\newline&space;\frac{d}{d\delta_j}&space;\phi_j(\delta_j)&space;=&space;-\frac{\beta_j^2}{\theta_j^2}&space;\cdot&space;\delta_j&space;&plus;&space;\{-&space;\frac{1}{\theta_j}&space;-\frac{\beta_j^2}{\theta_j}\log\frac{\theta_j}{\mu_j}&space;\}" title="\newline \phi_j(\delta_j) = -\frac{1}{2} \beta_j^2(\log\theta_j - \log\mu_j)^2 - \log\theta_j \sim \newline \sim -\frac{1}{2} \beta_j^2(\log\theta_j +\frac{\delta_j}{\theta_j} - \log\mu_j)^2 -\log\theta_j -\frac{\delta_j}{\theta_j} \Rightarrow \newline \newline \frac{d}{d\delta_j} \phi_j(\delta_j) = -\frac{\beta_j^2}{\theta_j^2} \cdot \delta_j + \{- \frac{1}{\theta_j} -\frac{\beta_j^2}{\theta_j}\log\frac{\theta_j}{\mu_j} \}" /></a>

The linear system is now

<a href="https://www.codecogs.com/eqnedit.php?latex=\left(&space;\frac{J^T&space;J}{\hat{\sigma}^2}&space;&plus;&space;R&space;\right)&space;\cdot&space;\underline{\delta}&space;=&space;\frac{1}{\hat{\sigma}}J&space;\cdot&space;\underline{\epsilon}&space;&plus;&space;\underline{r}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\left(&space;\frac{J^T&space;J}{\hat{\sigma}^2}&space;&plus;&space;R&space;\right)&space;\cdot&space;\underline{\delta}&space;=&space;\frac{1}{\hat{\sigma}}J&space;\cdot&space;\underline{\epsilon}&space;&plus;&space;\underline{r}" title="\left( \frac{J^T J}{\hat{\sigma}^2} + R \right) \cdot \underline{\delta} = \frac{1}{\hat{\sigma}}J \cdot \underline{\epsilon} + \underline{r}" /></a>

where

<a href="https://www.codecogs.com/eqnedit.php?latex=\hat{\sigma}^2&space;=&space;\frac{1}{N}&space;\sum_{i=1}^N&space;w_i&space;(y_i&space;-f(\underline{x_i},&space;\underline{\theta}))^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat{\sigma}^2&space;=&space;\frac{1}{N}&space;\sum_{i=1}^N&space;w_i&space;(y_i&space;-f(\underline{x_i},&space;\underline{\theta}))^2" title="\hat{\sigma}^2 = \frac{1}{N} \sum_{i=1}^N w_i (y_i -f(\underline{x_i}, \underline{\theta}))^2" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=R_{jj}&space;=&space;\left\{\begin{matrix}&space;\beta_j^2&space;&&space;&&space;\text{if&space;}&space;p(\theta_j)&space;=&space;N(\mu_j,&space;\frac{1}{\beta_j^2})&space;\\&space;\newline&space;\\&space;\frac{\beta_j^2}{\theta_j^2}&space;&&space;&&space;\text{if&space;}&space;p(\theta_j)&space;=&space;\text{lognorm}(\log\mu_j,&space;\frac{1}{\beta_j^2})&space;\end{matrix}\right.,&space;R_{ij}&space;=&space;0&space;\text{&space;if&space;}&space;i&space;\neq&space;j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?R_{jj}&space;=&space;\left\{\begin{matrix}&space;\beta_j^2&space;&&space;&&space;\text{if&space;}&space;p(\theta_j)&space;=&space;N(\mu_j,&space;\frac{1}{\beta_j^2})&space;\\&space;\newline&space;\\&space;\frac{\beta_j^2}{\theta_j^2}&space;&&space;&&space;\text{if&space;}&space;p(\theta_j)&space;=&space;\text{lognorm}(\log\mu_j,&space;\frac{1}{\beta_j^2})&space;\end{matrix}\right.,&space;R_{ij}&space;=&space;0&space;\text{&space;if&space;}&space;i&space;\neq&space;j" title="R_{jj} = \left\{\begin{matrix} \beta_j^2 & & \text{if } p(\theta_j) = N(\mu_j, \frac{1}{\beta_j^2}) \\ \newline \\ \frac{\beta_j^2}{\theta_j^2} & & \text{if } p(\theta_j) = \text{lognorm}(\log\mu_j, \frac{1}{\beta_j^2}) \end{matrix}\right., R_{ij} = 0 \text{ if } i \neq j" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=r_j&space;=&space;\left\{\begin{matrix}&space;-\beta_j^2&space;(\theta_j&space;-&space;\mu_j)&&space;&&space;\text{if&space;}&space;p(\theta_j)&space;=&space;N(\mu_j,&space;\frac{1}{\beta_j^2})&space;\\&space;\newline&space;\\&space;-\frac{1}{\theta_j}&space;-&space;\frac{\beta_j^2}{\theta_j}&space;\log&space;\frac{\theta_j}{\mu_j}&space;&&space;&&space;\text{if&space;}&space;p(\theta_j)&space;=&space;\text{lognorm}(\log\mu_j,&space;\frac{1}{\beta_j^2})&space;\end{matrix}\right." target="_blank"><img src="https://latex.codecogs.com/gif.latex?r_j&space;=&space;\left\{\begin{matrix}&space;-\beta_j^2&space;(\theta_j&space;-&space;\mu_j)&&space;&&space;\text{if&space;}&space;p(\theta_j)&space;=&space;N(\mu_j,&space;\frac{1}{\beta_j^2})&space;\\&space;\newline&space;\\&space;-\frac{1}{\theta_j}&space;-&space;\frac{\beta_j^2}{\theta_j}&space;\log&space;\frac{\theta_j}{\mu_j}&space;&&space;&&space;\text{if&space;}&space;p(\theta_j)&space;=&space;\text{lognorm}(\log\mu_j,&space;\frac{1}{\beta_j^2})&space;\end{matrix}\right." title="r_j = \left\{\begin{matrix} -\beta_j^2 (\theta_j - \mu_j)& & \text{if } p(\theta_j) = N(\mu_j, \frac{1}{\beta_j^2}) \\ \newline \\ -\frac{1}{\theta_j} - \frac{\beta_j^2}{\theta_j} \log \frac{\theta_j}{\mu_j} & & \text{if } p(\theta_j) = \text{lognorm}(\log\mu_j, \frac{1}{\beta_j^2}) \end{matrix}\right." /></a>

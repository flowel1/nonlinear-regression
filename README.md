# Levenberg-Marquardt algorithm for nonlinear regression

## Nonlinear regression: generic problem statement
Let us consider a regression problem where a scalar target variable _y_ must be predicted based on a vector of observables <a href="https://www.codecogs.com/eqnedit.php?latex=\underline{x}&space;\in&space;\mathbb{R}^n" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\underline{x}&space;\in&space;\mathbb{R}^n" title="\underline{x} \in \mathbb{R}^n" /></a>.

We assume that the dynamics are **nonlinear** and, specifically, that

<a href="https://www.codecogs.com/eqnedit.php?latex=y&space;=&space;f(\underline{x};\underline{\theta})&space;&plus;&space;\epsilon" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y&space;=&space;f(\underline{x};\underline{\theta})&space;&plus;&space;\epsilon" title="y = f(\underline{x};\underline{\theta}) + \epsilon" /></a>

where <a href="https://www.codecogs.com/eqnedit.php?latex=\underline{\theta}&space;\in&space;\mathbb{R}^k" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\underline{\theta}&space;\in&space;\mathbb{R}^k" title="\underline{\theta} \in \mathbb{R}^k" /></a>  is a vector of unknown real parameters, _f_ is a known deterministic function nonlinear in <ins>&theta;</ins> and &epsilon; is a random noise with distribution <a href="https://www.codecogs.com/eqnedit.php?latex=\epsilon&space;\sim&space;N(0,&space;\sigma^2)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\epsilon&space;\sim&space;N(0,&space;\sigma^2)" title="\epsilon \sim N(0, \sigma^2)" /></a> for some positive and unknown value of &sigma;.

If we have N independent observations <a href="https://www.codecogs.com/eqnedit.php?latex=(\underline{x}_1,&space;y_1),&space;...,&space;(\underline{x}_N,&space;y_N)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?(\underline{x}_1,&space;y_1),&space;...,&space;(\underline{x}_N,&space;y_N)" title="(\underline{x}_1, y_1), ..., (\underline{x}_N, y_N)" /></a>, we can estimate the value of <ins>&theta;</ins> by maximizing the log-likelihood. We can optionally choose to weight some observations more or less that others by choosing weights <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{100}&space;w_i,&space;i&space;=&space;1,&space;...,&space;n" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\dpi{100}&space;w_i,&space;i&space;=&space;1,&space;...,&space;n" title="w_i, i = 1, ..., n" /></a> and assuming that <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{100}&space;\small&space;y_i&space;\sim&space;N(f(\underline{x}_i,&space;\underline{\theta}),&space;\frac{\sigma^2}{w_i})" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\dpi{100}&space;\small&space;y_i&space;\sim&space;N(f(\underline{x}_i,&space;\underline{\theta}),&space;\frac{\sigma^2}{w_i})" title="\small y_i \sim N(f(\underline{x}_i, \underline{\theta}), \frac{\sigma^2}{w_i})" /></a> for each i.  

Under these assumptions, the log-likelihood is given by

<a href="https://www.codecogs.com/eqnedit.php?latex=L(\underline{\theta}&space;|&space;X,&space;\underline{y})&space;=&space;\prod_{i&space;=&space;1}^N&space;\frac{\sqrt{w_i}}{\sigma\sqrt{2\pi}}\exp&space;\left&space;\{&space;-&space;w_i&space;\cdot&space;\left(&space;\frac{y_i-f(\underline{x}_i;\underline{\theta})}{\sigma}&space;\right)&space;^&space;2&space;\right&space;\}&space;\Rightarrow&space;\newline&space;\log&space;L(\underline{\theta}&space;|&space;X,&space;\underline{y})&space;=&space;\frac{1}{2}\sum_{i=1}^N&space;\log&space;w_i&space;-\frac{N}{2}&space;\log(2\pi)&space;-&space;\frac{N}{2}&space;\log(\sigma^2)&space;-\sum_{i=1}^N&space;w_i&space;\cdot&space;\left(&space;\frac{y_i-f(\underline{x}_i;\underline{\theta})}{\sigma}&space;\right)&space;^&space;2" target="_blank"><img src="https://latex.codecogs.com/svg.latex?L(\underline{\theta}&space;|&space;X,&space;\underline{y})&space;=&space;\prod_{i&space;=&space;1}^N&space;\frac{\sqrt{w_i}}{\sigma\sqrt{2\pi}}\exp&space;\left&space;\{&space;-&space;w_i&space;\cdot&space;\left(&space;\frac{y_i-f(\underline{x}_i;\underline{\theta})}{\sigma}&space;\right)&space;^&space;2&space;\right&space;\}&space;\Rightarrow&space;\newline&space;\log&space;L(\underline{\theta}&space;|&space;X,&space;\underline{y})&space;=&space;\frac{1}{2}\sum_{i=1}^N&space;\log&space;w_i&space;-\frac{N}{2}&space;\log(2\pi)&space;-&space;\frac{N}{2}&space;\log(\sigma^2)&space;-\sum_{i=1}^N&space;w_i&space;\cdot&space;\left(&space;\frac{y_i-f(\underline{x}_i;\underline{\theta})}{\sigma}&space;\right)&space;^&space;2" title="L(\underline{\theta} | X, \underline{y}) = \prod_{i = 1}^N \frac{\sqrt{w_i}}{\sigma\sqrt{2\pi}}\exp \left \{ - w_i \cdot \left( \frac{y_i-f(\underline{x}_i;\underline{\theta})}{\sigma} \right) ^ 2 \right \} \Rightarrow \newline \log L(\underline{\theta} | X, \underline{y}) = \frac{1}{2}\sum_{i=1}^N \log w_i -\frac{N}{2} \log(2\pi) - \frac{N}{2} \log(\sigma^2) -\sum_{i=1}^N w_i \cdot \left( \frac{y_i-f(\underline{x}_i;\underline{\theta})}{\sigma} \right) ^ 2" /></a>
  
Setting for simplicity of notation

<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{100}&space;\small&space;X&space;=\begin{bmatrix}&space;\underline{x}_1&space;\\&space;...&space;\\&space;\underline{x}_N&space;\\&space;\end{bmatrix}&space;\in&space;\mathbb{R}^{N&space;\times&space;n},&space;\hspace{10}&space;\underline{y}&space;=&space;\begin{bmatrix}&space;y_1&space;\\&space;...&space;\\&space;y_N&space;\end{bmatrix}&space;\in&space;\mathbb{R}^N,&space;\hspace{10}&space;W&space;=&space;\text{diag}&space;\left&space;\{&space;\sqrt{w_1},&space;...,&space;\sqrt{w_N}&space;\right&space;\}&space;\in&space;\mathbb{R}^{N&space;\times&space;N}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\dpi{100}&space;\small&space;X&space;=\begin{bmatrix}&space;\underline{x}_1&space;\\&space;...&space;\\&space;\underline{x}_N&space;\\&space;\end{bmatrix}&space;\in&space;\mathbb{R}^{N&space;\times&space;n},&space;\hspace{10}&space;\underline{y}&space;=&space;\begin{bmatrix}&space;y_1&space;\\&space;...&space;\\&space;y_N&space;\end{bmatrix}&space;\in&space;\mathbb{R}^N,&space;\hspace{10}&space;W&space;=&space;\text{diag}&space;\left&space;\{&space;\sqrt{w_1},&space;...,&space;\sqrt{w_N}&space;\right&space;\}&space;\in&space;\mathbb{R}^{N&space;\times&space;N}" title="\small X =\begin{bmatrix} \underline{x}_1 \\ ... \\ \underline{x}_N \\ \end{bmatrix} \in \mathbb{R}^{N \times n}, \hspace{10} \underline{y} = \begin{bmatrix} y_1 \\ ... \\ y_N \end{bmatrix} \in \mathbb{R}^N, \hspace{10} W = \text{diag} \left \{ \sqrt{w_1}, ..., \sqrt{w_N} \right \} \in \mathbb{R}^{N \times N}" /></a>

we see that maximizing the log-likelihood is equivalent to minimizing the following objective function (weighted sum of squared residuals):

<a href="https://www.codecogs.com/eqnedit.php?latex=\text{Obj}(\underline{\theta})&space;=&space;\left&space;\|&space;W&space;\cdot&space;\left(&space;\underline{y}&space;-&space;f(X,&space;\underline{\theta})&space;\right)&space;\right&space;\|&space;^&space;2" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\text{Obj}(\underline{\theta})&space;=&space;\left&space;\|&space;W&space;\cdot&space;\left(&space;\underline{y}&space;-&space;f(X,&space;\underline{\theta})&space;\right)&space;\right&space;\|&space;^&space;2" title="\text{Obj}(\underline{\theta}) = \left \| W \cdot \left( \underline{y} - f(X, \underline{\theta}) \right) \right \| ^ 2" /></a>

## The Levenberg-Marquardt algorithm

The Levenberg-Marquardt algorithm calculates the minimum of Obj in an iterative way calculating a series of local quadratic approximations.

The algorithm requires that an initial guess <a href="https://www.codecogs.com/eqnedit.php?latex=\underline{\theta}^{(0)}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\underline{\theta}^{(0)}" title="\underline{\theta}^{(0)}" /></a> is provided for the unknown vector of parameters. The objective function Obj is then approximated locally in a neighbourhood of <a href="https://www.codecogs.com/eqnedit.php?latex=\underline{\theta}^{(0)}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\underline{\theta}^{(0)}" title="\underline{\theta}^{(0)}" /></a> with the following quadratic function:

<a href="https://www.codecogs.com/eqnedit.php?latex=\text{Obj}(\underline{\theta}^{(0)}&space;&plus;&space;\underline{\delta})&space;\sim&space;\phi(\underline{\delta})&space;:=&space;\left&space;\|&space;W&space;\left&space;\(&space;\underline{y}&space;-&space;f(X,&space;\underline{\theta}^{(0)})&space;-&space;J_{\underline{\theta}}f(X,&space;\underline{\theta})_{|_{\underline{\theta}&space;=&space;\underline{\theta}^{(0)}}}&space;\cdot&space;\underline{\delta}&space;\right&space;\)&space;\right&space;\|&space;^2&space;\hspace{10}&space;\text{for&space;small&space;values&space;of&space;}&space;\left&space;\|&space;\underline{\delta}&space;\right&space;\|" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\text{Obj}(\underline{\theta}^{(0)}&space;&plus;&space;\underline{\delta})&space;\sim&space;\phi(\underline{\delta})&space;:=&space;\left&space;\|&space;W&space;\left&space;\(&space;\underline{y}&space;-&space;f(X,&space;\underline{\theta}^{(0)})&space;-&space;J_{\underline{\theta}}f(X,&space;\underline{\theta})_{|_{\underline{\theta}&space;=&space;\underline{\theta}^{(0)}}}&space;\cdot&space;\underline{\delta}&space;\right&space;\)&space;\right&space;\|&space;^2&space;\hspace{10}&space;\text{for&space;small&space;values&space;of&space;}&space;\left&space;\|&space;\underline{\delta}&space;\right&space;\|" title="\text{Obj}(\underline{\theta}^{(0)} + \underline{\delta}) \sim \phi(\underline{\delta}) := \left \| W \left \( \underline{y} - f(X, \underline{\theta}^{(0)}) - J_{\underline{\theta}}f(X, \underline{\theta})_{|_{\underline{\theta} = \underline{\theta}^{(0)}}} \cdot \underline{\delta} \right \) \right \| ^2 \hspace{10} \text{for small values of } \left \| \underline{\delta} \right \|" /></a>

The peculiarity here is that, thanks to the objective function's special form, we can calculate a local quadratic approximation by taking the first order expansion of _f_ instead of the second-order expansion of the objective function itself (as we would be forced to do in the general case).

Defining for simplicity of notation

<a href="https://www.codecogs.com/eqnedit.php?latex=J&space;:=&space;J_{\underline{\theta}}f(X,&space;\underline{\theta})_{|_{\underline{\theta}&space;=&space;\underline{\theta}^{(0)}}}&space;\in&space;\mathbb{R}^{N&space;\times&space;k},&space;\hspace{10}&space;\underline{\epsilon}&space;:=&space;\underline{y}&space;-&space;f(X,&space;\underline{\theta}^{(0)})&space;\in&space;\mathbb{R}^N" target="_blank"><img src="https://latex.codecogs.com/svg.latex?J&space;:=&space;J_{\underline{\theta}}f(X,&space;\underline{\theta})_{|_{\underline{\theta}&space;=&space;\underline{\theta}^{(0)}}}&space;\in&space;\mathbb{R}^{N&space;\times&space;k},&space;\hspace{10}&space;\underline{\epsilon}&space;:=&space;\underline{y}&space;-&space;f(X,&space;\underline{\theta}^{(0)})&space;\in&space;\mathbb{R}^N" title="J := J_{\underline{\theta}}f(X, \underline{\theta})_{|_{\underline{\theta} = \underline{\theta}^{(0)}}} \in \mathbb{R}^{N \times k}, \hspace{10} \underline{\epsilon} := \underline{y} - f(X, \underline{\theta}^{(0)}) \in \mathbb{R}^N" /></a>

we have that this quadratic has a unique global minimum satisfying the equation

<a href="https://www.codecogs.com/eqnedit.php?latex=\nabla_{\underline{\delta}}&space;\hspace{2}&space;\phi(\underline{\delta})&space;=&space;-2&space;\cdot&space;J^T&space;W^T&space;\cdot&space;(\underline{\epsilon}&space;-&space;W&space;J\cdot&space;\underline{\delta})&space;=&space;\underline{0}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\nabla_{\underline{\delta}}&space;\hspace{2}&space;\phi(\underline{\delta})&space;=&space;-2&space;\cdot&space;J^T&space;W^T&space;\cdot&space;(\underline{\epsilon}&space;-&space;W&space;J\cdot&space;\underline{\delta})&space;=&space;\underline{0}" title="\nabla_{\underline{\delta}} \hspace{2} \phi(\underline{\delta}) = -2 \cdot J^T W^T \cdot (\underline{\epsilon} - W J\cdot \underline{\delta}) = \underline{0}" /></a>

which is equivalent to requiring that the **displacement** <ins>&delta;</ins> solves the following linear system:

<a href="https://www.codecogs.com/eqnedit.php?latex=(J^T&space;W^T&space;W&space;J)&space;\cdot&space;\underline{\delta}&space;=&space;J^T&space;W^T&space;\cdot&space;\underline{\epsilon}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?(J^T&space;W^T&space;W&space;J)&space;\cdot&space;\underline{\delta}&space;=&space;J^T&space;W^T&space;\cdot&space;\underline{\epsilon}" title="(J^T W^T W J) \cdot \underline{\delta} = J^T W^T \cdot \underline{\epsilon}" /></a>

This picture illustrates the calculation of <ins>&delta;</ins> in a simple one-dimensional case:

![alt-text](https://github.com/flowel1/nonlinear-regression/blob/master/pictures/quadratic-approx.png)

In the picture, the displacement has been applied as it is. Actually, in practice, since the quadratic approximation is generally only valid locally, <ins>&delta;</ins> will just provide the displacement **direction**, while its module will be re-scaled according to a small positive number _h_ (**step**) when updating <ins>&theta;</ins>:

<a href="https://www.codecogs.com/eqnedit.php?latex=\underline{\theta}^{(1)}&space;=&space;\underline{\theta}^{(0)}&space;&plus;&space;h&space;\cdot&space;\frac{\underline{\delta}}{\left&space;\|&space;\underline{\delta}&space;\right&space;\|}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\underline{\theta}^{(1)}&space;=&space;\underline{\theta}^{(0)}&space;&plus;&space;h&space;\cdot&space;\frac{\underline{\delta}}{\left&space;\|&space;\underline{\delta}&space;\right&space;\|}" title="\underline{\theta}^{(1)} = \underline{\theta}^{(0)} + h \cdot \frac{\underline{\delta}}{\left \| \underline{\delta} \right \|}" /></a>


## Regularization

The Levenberg-Marquardt algorithm can be extended to incorporate regularization terms. Seeing the problem in a Bayesian perspective, we can decide to provide a prior distribution on <ins>&theta;</ins>. For simplicity, we assume that the prior distributions on the different parameter components are independent, so that the global prior distribution can be factorized:

<a href="https://www.codecogs.com/eqnedit.php?latex=p(\underline{\theta})&space;=&space;\prod_{j=1}^k&space;p_j(\theta_j)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?p(\underline{\theta})&space;=&space;\prod_{j=1}^k&space;p_j(\theta_j)" title="p(\underline{\theta}) = \prod_{j=1}^k p_j(\theta_j)" /></a>

We assume moreover that the one-dimensional priors on the single components are either gaussian or lognormal:

<a href="https://www.codecogs.com/eqnedit.php?latex=\text{Gaussian&space;prior:&space;}&space;p_j(\theta_j)&space;=&space;\frac{\beta_j}{\sqrt{2&space;\pi}}&space;\cdot&space;\exp{\left\{&space;-\frac{1}{2}&space;\beta_j^2&space;(&space;\theta_j&space;-&space;\mu_j)^2&space;\right\}}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\text{Gaussian&space;prior:&space;}&space;p_j(\theta_j)&space;=&space;\frac{\beta_j}{\sqrt{2&space;\pi}}&space;\cdot&space;\exp{\left\{&space;-\frac{1}{2}&space;\beta_j^2&space;(&space;\theta_j&space;-&space;\mu_j)^2&space;\right\}}" title="\text{Gaussian prior: } p_j(\theta_j) = \frac{\beta_j}{\sqrt{2 \pi}} \cdot \exp{\left\{ -\frac{1}{2} \beta_j^2 ( \theta_j - \mu_j)^2 \right\}}" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\text{Lognormal&space;prior:&space;}&space;p_j(\theta_j)&space;=&space;\frac{\beta_j}{\theta_j&space;\cdot&space;\sqrt{2&space;\pi}}&space;\cdot&space;\exp{&space;\left&space;\{&space;-\frac{1}{2}&space;\beta_j^2&space;(\log\theta_j&space;-&space;\log\mu_j)^2&space;\right&space;\}&space;}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\text{Lognormal&space;prior:&space;}&space;p_j(\theta_j)&space;=&space;\frac{\beta_j}{\theta_j&space;\cdot&space;\sqrt{2&space;\pi}}&space;\cdot&space;\exp{&space;\left&space;\{&space;-\frac{1}{2}&space;\beta_j^2&space;(\log\theta_j&space;-&space;\log\mu_j)^2&space;\right&space;\}&space;}" title="\text{Lognormal prior: } p_j(\theta_j) = \frac{\beta_j}{\theta_j \cdot \sqrt{2 \pi}} \cdot \exp{ \left \{ -\frac{1}{2} \beta_j^2 (\log\theta_j - \log\mu_j)^2 \right \} }" /></a>

Reasoning in Bayesian terms, this time
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

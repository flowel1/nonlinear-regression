# Levenberg-Marquardt algorithm for nonlinear regression

## Nonlinear regression: generic problem statement
Let us consider a regression problem where a scalar variable _y_ (target variable) must be predicted based on a vector of observables <a href="https://www.codecogs.com/eqnedit.php?latex=\underline{x}&space;\in&space;\mathbb{R}^n" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\underline{x}&space;\in&space;\mathbb{R}^n" title="\underline{x} \in \mathbb{R}^n"/></a>.

We assume that the dynamics are **nonlinear** and, specifically, that

<a href="https://www.codecogs.com/eqnedit.php?latex=y&space;=&space;f(\underline{x};\underline{\theta})&space;&plus;&space;\epsilon" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y&space;=&space;f(\underline{x};\underline{\theta})&space;&plus;&space;\epsilon" title="y = f(\underline{x};\underline{\theta}) + \epsilon" /></a>

where <a href="https://www.codecogs.com/eqnedit.php?latex=\underline{\theta}&space;\in&space;\mathbb{R}^k" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\underline{\theta}&space;\in&space;\mathbb{R}^k" title="\underline{\theta} \in \mathbb{R}^k" /></a>  is a vector of unknown real parameters, _f_ is a known deterministic function nonlinear in <ins>&theta;</ins> and &epsilon; is a random noise with distribution <a href="https://www.codecogs.com/eqnedit.php?latex=\epsilon&space;\sim&space;N(0,&space;\sigma^2)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\epsilon&space;\sim&space;N(0,&space;\sigma^2)" title="\epsilon \sim N(0, \sigma^2)" /></a> for some positive and unknown value of &sigma;.

If we have N independent observations <a href="https://www.codecogs.com/eqnedit.php?latex=(\underline{x}_1,&space;y_1),&space;...,&space;(\underline{x}_N,&space;y_N)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?(\underline{x}_1,&space;y_1),&space;...,&space;(\underline{x}_N,&space;y_N)" title="(\underline{x}_1, y_1), ..., (\underline{x}_N, y_N)" /></a>, we can estimate the value of <ins>&theta;</ins> by maximizing the log-likelihood. We can optionally choose to weight some observations more or less than others by choosing weights <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{100}&space;w_i,&space;i&space;=&space;1,&space;...,&space;n" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\dpi{100}&space;w_i,&space;i&space;=&space;1,&space;...,&space;n" title="w_i, i = 1, ..., n" /></a> and assuming that <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{100}&space;\small&space;y_i&space;\sim&space;N(f(\underline{x}_i,&space;\underline{\theta}),&space;\frac{\sigma^2}{w_i})" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\dpi{100}&space;\small&space;y_i&space;\sim&space;N(f(\underline{x}_i,&space;\underline{\theta}),&space;\frac{\sigma^2}{w_i})" title="\small y_i \sim N(f(\underline{x}_i, \underline{\theta}), \frac{\sigma^2}{w_i})" /></a> for each i.  

Under these assumptions and setting for simplicity of notation

<a href="https://www.codecogs.com/eqnedit.php?latex=X&space;=&space;\begin{bmatrix}&space;\underline{x}_1&space;\\&space;...&space;\\&space;\underline{x}_N&space;\end{bmatrix}&space;\in&space;\mathbb{R}^{N},&space;\underline{y}&space;=&space;\begin{bmatrix}&space;y_1&space;\\&space;...\\&space;y_N&space;\end{bmatrix}\in&space;\mathbb{R}^N,&space;W&space;=&space;\text{diag}&space;\left\{&space;\sqrt{w_1},&space;...,&space;\sqrt{w_N}\right\}&space;\in&space;\mathbb{R}^{N}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?X&space;=&space;\begin{bmatrix}&space;\underline{x}_1&space;\\&space;...&space;\\&space;\underline{x}_N&space;\end{bmatrix}&space;\in&space;\mathbb{R}^{N},&space;\underline{y}&space;=&space;\begin{bmatrix}&space;y_1&space;\\&space;...\\&space;y_N&space;\end{bmatrix}\in&space;\mathbb{R}^N,&space;W&space;=&space;\text{diag}&space;\left\{&space;\sqrt{w_1},&space;...,&space;\sqrt{w_N}\right\}&space;\in&space;\mathbb{R}^{N}" title="X = \begin{bmatrix} \underline{x}_1 \\ ... \\ \underline{x}_N \end{bmatrix} \in \mathbb{R}^{N}, \underline{y} = \begin{bmatrix} y_1 \\ ...\\ y_N \end{bmatrix}\in \mathbb{R}^N, W = \text{diag} \left\{ \sqrt{w_1}, ..., \sqrt{w_N}\right\} \in \mathbb{R}^{N}" /></a>

we have that the log-likelihood is given by

<a href="https://www.codecogs.com/eqnedit.php?latex=L(\underline{\theta}&space;|&space;X,&space;\underline{y})&space;=&space;\prod_{i&space;=&space;1}^N&space;\frac{\sqrt{w_i}}{\sigma\sqrt{2\pi}}\exp&space;\left&space;\{&space;-&space;\frac{1}{2}&space;w_i&space;\cdot&space;\left(&space;\frac{y_i-f(\underline{x}_i;\underline{\theta})}{\sigma}&space;\right)&space;^&space;2&space;\right&space;\}&space;\Rightarrow&space;\newline&space;\log&space;L(\underline{\theta}&space;|&space;X,&space;\underline{y})&space;=&space;\frac{1}{2}\sum_{i=1}^N&space;\log&space;w_i&space;-\frac{N}{2}&space;\log(2\pi)&space;-&space;\frac{N}{2}&space;\log(\sigma^2)&space;-&space;\frac{1}{2}&space;\sum_{i=1}^N&space;w_i&space;\cdot&space;\left(&space;\frac{y_i-f(\underline{x}_i;\underline{\theta})}{\sigma}&space;\right)&space;^&space;2" target="_blank"><img src="https://latex.codecogs.com/svg.latex?L(\underline{\theta}&space;|&space;X,&space;\underline{y})&space;=&space;\prod_{i&space;=&space;1}^N&space;\frac{\sqrt{w_i}}{\sigma\sqrt{2\pi}}\exp&space;\left&space;\{&space;-&space;\frac{1}{2}&space;w_i&space;\cdot&space;\left(&space;\frac{y_i-f(\underline{x}_i;\underline{\theta})}{\sigma}&space;\right)&space;^&space;2&space;\right&space;\}&space;\Rightarrow&space;\newline&space;\log&space;L(\underline{\theta}&space;|&space;X,&space;\underline{y})&space;=&space;\frac{1}{2}\sum_{i=1}^N&space;\log&space;w_i&space;-\frac{N}{2}&space;\log(2\pi)&space;-&space;\frac{N}{2}&space;\log(\sigma^2)&space;-&space;\frac{1}{2}&space;\sum_{i=1}^N&space;w_i&space;\cdot&space;\left(&space;\frac{y_i-f(\underline{x}_i;\underline{\theta})}{\sigma}&space;\right)&space;^&space;2" title="L(\underline{\theta} | X, \underline{y}) = \prod_{i = 1}^N \frac{\sqrt{w_i}}{\sigma\sqrt{2\pi}}\exp \left \{ - \frac{1}{2} w_i \cdot \left( \frac{y_i-f(\underline{x}_i;\underline{\theta})}{\sigma} \right) ^ 2 \right \} \Rightarrow \newline \log L(\underline{\theta} | X, \underline{y}) = \frac{1}{2}\sum_{i=1}^N \log w_i -\frac{N}{2} \log(2\pi) - \frac{N}{2} \log(\sigma^2) - \frac{1}{2} \sum_{i=1}^N w_i \cdot \left( \frac{y_i-f(\underline{x}_i;\underline{\theta})}{\sigma} \right) ^ 2" /></a>

Maximizing the log-likelihood is equivalent to minimizing the following objective function (weighted sum of squared residuals):

<a href="https://www.codecogs.com/eqnedit.php?latex=\text{Obj}(\underline{\theta})&space;=&space;\frac{1}{2}\left&space;\|&space;W&space;\cdot&space;\left(&space;\underline{y}&space;-&space;f(X,&space;\underline{\theta})&space;\right)&space;\right&space;\|&space;^&space;2" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\text{Obj}(\underline{\theta})&space;=&space;\frac{1}{2}\left&space;\|&space;W&space;\cdot&space;\left(&space;\underline{y}&space;-&space;f(X,&space;\underline{\theta})&space;\right)&space;\right&space;\|&space;^&space;2" title="\text{Obj}(\underline{\theta}) = \frac{1}{2}\left \| W \cdot \left( \underline{y} - f(X, \underline{\theta}) \right) \right \| ^ 2" /></a>

Moreover, the maximum likelihood estimate for &sigma; is

<a href="https://www.codecogs.com/eqnedit.php?latex=\hat{\sigma}^2&space;=&space;\frac{1}{N}&space;\sum_{i=1}^N&space;w_i&space;(y_i&space;-f(\underline{x_i},&space;\underline{\theta}))^2" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\hat{\sigma}^2&space;=&space;\frac{1}{N}&space;\sum_{i=1}^N&space;w_i&space;(y_i&space;-f(\underline{x_i},&space;\underline{\theta}))^2" title="\hat{\sigma}^2 = \frac{1}{N} \sum_{i=1}^N w_i (y_i -f(\underline{x_i}, \underline{\theta}))^2" /></a>.

## The Levenberg-Marquardt algorithm

The Levenberg-Marquardt algorithm calculates the minimum of Obj in an iterative way calculating a series of local quadratic approximations.

The algorithm requires that an initial guess <a href="https://www.codecogs.com/eqnedit.php?latex=\underline{\theta}^{(0)}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\underline{\theta}^{(0)}" title="\underline{\theta}^{(0)}" /></a> is provided for the unknown vector of parameters. The objective function Obj is then approximated locally in a neighbourhood of <a href="https://www.codecogs.com/eqnedit.php?latex=\underline{\theta}^{(0)}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\underline{\theta}^{(0)}" title="\underline{\theta}^{(0)}" /></a> with the following quadratic function (the approximation is only valid for small values of ||&delta;||:

<a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;\begin{matrix}&space;\text{Obj}(\underline{\theta}^{(0)}&space;&plus;&space;\underline{\delta},&space;\sigma)&space;\sim&space;&\phi(\underline{\delta})&space;:=&space;\frac{N}{2}\log(\sigma^2)&space;&plus;&space;\frac{1}{2\sigma^2}&space;\left&space;\|&space;W&space;\cdot&space;\right(&space;\underline{y}&space;-&space;f(X,&space;\underline{\theta}^{(0)})&space;-&space;J_{\underline{\theta}}&space;f(X,&space;\underline{\theta})_{|_{\underline{\theta}&space;=&space;\underline{\theta}^{(0)}}}&space;\cdot&space;\underline{\delta}&space;\left)\;&space;\right&space;\|&space;^&space;2&space;\end{matrix}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\small&space;\begin{matrix}&space;\text{Obj}(\underline{\theta}^{(0)}&space;&plus;&space;\underline{\delta},&space;\sigma)&space;\sim&space;&\phi(\underline{\delta})&space;:=&space;\frac{N}{2}\log(\sigma^2)&space;&plus;&space;\frac{1}{2\sigma^2}&space;\left&space;\|&space;W&space;\cdot&space;\right(&space;\underline{y}&space;-&space;f(X,&space;\underline{\theta}^{(0)})&space;-&space;J_{\underline{\theta}}&space;f(X,&space;\underline{\theta})_{|_{\underline{\theta}&space;=&space;\underline{\theta}^{(0)}}}&space;\cdot&space;\underline{\delta}&space;\left)\;&space;\right&space;\|&space;^&space;2&space;\end{matrix}" title="\small \begin{matrix} \text{Obj}(\underline{\theta}^{(0)} + \underline{\delta}, \sigma) \sim &\phi(\underline{\delta}) := \frac{N}{2}\log(\sigma^2) + \frac{1}{2\sigma^2} \left \| W \cdot \right( \underline{y} - f(X, \underline{\theta}^{(0)}) - J_{\underline{\theta}} f(X, \underline{\theta})_{|_{\underline{\theta} = \underline{\theta}^{(0)}}} \cdot \underline{\delta} \left)\; \right \| ^ 2 \end{matrix}" /></a>

The peculiarity here is that, thanks to the objective function's special form, we can calculate a local quadratic approximation by taking the first order expansion of _f_ instead of the second-order expansion of the objective function itself (as we would be forced to do in the general case).

Defining for simplicity of notation

<a href="https://www.codecogs.com/eqnedit.php?latex=J&space;:=&space;J_{\underline{\theta}}f(X,&space;\underline{\theta})_{|_{\underline{\theta}&space;=&space;\underline{\theta}^{(0)}}}&space;\in&space;\mathbb{R}^{N&space;\times&space;k},&space;\hspace{10}&space;\underline{\epsilon}&space;:=&space;\underline{y}&space;-&space;f(X,&space;\underline{\theta}^{(0)})&space;\in&space;\mathbb{R}^N" target="_blank"><img src="https://latex.codecogs.com/svg.latex?J&space;:=&space;J_{\underline{\theta}}f(X,&space;\underline{\theta})_{|_{\underline{\theta}&space;=&space;\underline{\theta}^{(0)}}}&space;\in&space;\mathbb{R}^{N&space;\times&space;k},&space;\hspace{10}&space;\underline{\epsilon}&space;:=&space;\underline{y}&space;-&space;f(X,&space;\underline{\theta}^{(0)})&space;\in&space;\mathbb{R}^N" title="J := J_{\underline{\theta}}f(X, \underline{\theta})_{|_{\underline{\theta} = \underline{\theta}^{(0)}}} \in \mathbb{R}^{N \times k}, \hspace{10} \underline{\epsilon} := \underline{y} - f(X, \underline{\theta}^{(0)}) \in \mathbb{R}^N" /></a>

we have that this quadratic has a unique global minimum satisfying the equation

<a href="https://www.codecogs.com/eqnedit.php?latex=\nabla_{\underline{\delta}}&space;\text{&space;}&space;\phi(\underline{\delta})&space;=&space;-\frac{1}{\sigma^2}&space;J^T&space;W^T&space;\cdot&space;(\underline{\epsilon}&space;-&space;W&space;J\cdot&space;\underline{\delta})&space;=&space;\underline{0}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\nabla_{\underline{\delta}}&space;\text{&space;}&space;\phi(\underline{\delta})&space;=&space;-\frac{1}{\sigma^2}&space;J^T&space;W^T&space;\cdot&space;(\underline{\epsilon}&space;-&space;W&space;J\cdot&space;\underline{\delta})&space;=&space;\underline{0}" title="\nabla_{\underline{\delta}} \text{ } \phi(\underline{\delta}) = -\frac{1}{\sigma^2} J^T W^T \cdot (\underline{\epsilon} - W J\cdot \underline{\delta}) = \underline{0}" /></a>

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

<a href="https://www.codecogs.com/eqnedit.php?latex=\text{Gaussian&space;prior:&space;}&space;p_j(\theta_j)&space;=&space;\sqrt{\frac{\beta_j}{2&space;\pi}}&space;\cdot&space;\exp{\left\{&space;-\frac{1}{2}&space;\beta_j&space;(&space;\theta_j&space;-&space;\mu_j)^2&space;\right\}}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\text{Gaussian&space;prior:&space;}&space;p_j(\theta_j)&space;=&space;\sqrt{\frac{\beta_j}{2&space;\pi}}&space;\cdot&space;\exp{\left\{&space;-\frac{1}{2}&space;\beta_j&space;(&space;\theta_j&space;-&space;\mu_j)^2&space;\right\}}" title="\text{Gaussian prior: } p_j(\theta_j) = \sqrt{\frac{\beta_j}{2 \pi}} \cdot \exp{\left\{ -\frac{1}{2} \beta_j ( \theta_j - \mu_j)^2 \right\}}" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\text{Lognormal&space;prior:&space;}&space;p_j(\theta_j)&space;=&space;\frac{1}{\theta_j}&space;\sqrt{\frac{\beta_j}{2&space;\pi}}&space;\cdot&space;\exp{&space;\left&space;\{&space;-\frac{1}{2}&space;\beta_j&space;(\log\theta_j&space;-&space;\log\mu_j)^2&space;\right&space;\}&space;}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\text{Lognormal&space;prior:&space;}&space;p_j(\theta_j)&space;=&space;\frac{1}{\theta_j}&space;\sqrt{\frac{\beta_j}{2&space;\pi}}&space;\cdot&space;\exp{&space;\left&space;\{&space;-\frac{1}{2}&space;\beta_j&space;(\log\theta_j&space;-&space;\log\mu_j)^2&space;\right&space;\}&space;}" title="\text{Lognormal prior: } p_j(\theta_j) = \frac{1}{\theta_j} \sqrt{\frac{\beta_j}{2 \pi}} \cdot \exp{ \left \{ -\frac{1}{2} \beta_j (\log\theta_j - \log\mu_j)^2 \right \} }" /></a>

Reasoning in Bayesian terms, this time we estimate <ins>&theta;</ins> via **maximum posterior** instead of maximum likelihood. The posterior distribution of <ins>&theta;</ins> is

<a href="https://www.codecogs.com/eqnedit.php?latex=p(\underline{\theta}&space;|&space;X,&space;\underline{y})&space;=&space;\frac{p(\underline{y}&space;|&space;X,&space;\underline{\theta})&space;\cdot&space;p(\underline{\theta})}{p(\underline{y}&space;|&space;X)}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?p(\underline{\theta}&space;|&space;X,&space;\underline{y})&space;=&space;\frac{p(\underline{y}&space;|&space;X,&space;\underline{\theta})&space;\cdot&space;p(\underline{\theta})}{p(\underline{y}&space;|&space;X)}" title="p(\underline{\theta} | X, \underline{y}) = \frac{p(\underline{y} | X, \underline{\theta}) \cdot p(\underline{\theta})}{p(\underline{y} | X)}" /></a>

(using the fact that <a href="https://www.codecogs.com/eqnedit.php?latex=p(\underline{\theta}&space;|&space;X)&space;=&space;p(\underline{\theta})" target="_blank"><img src="https://latex.codecogs.com/svg.latex?p(\underline{\theta}&space;|&space;X)&space;=&space;p(\underline{\theta})" title="p(\underline{\theta} | X) = p(\underline{\theta})" /></a>).

Keeping the same notations as before, the objective function to minimize is now

<a href="https://www.codecogs.com/eqnedit.php?latex=\text{Obj}(\underline{\theta})&space;=&space;-\log&space;p(\underline{\theta}&space;|&space;X,&space;\underline{y})&space;=&space;\newline&space;\newline&space;=&space;\text{const}&space;&plus;\frac{N}{2}&space;\log(\sigma^2)&space;&plus;\frac{1}{2\sigma^2}&space;\left&space;\|&space;W&space;\cdot&space;(\underline{y}&space;-&space;f(X,&space;\underline{\theta}))&space;\right&space;\|^2&space;&plus;&space;\newline&space;\newline&space;&plus;\frac{1}{2}&space;\cdot&space;\sum_{\{j&space;|&space;p(\theta_j)&space;\text{&space;gaussian}\}}&space;\beta_j&space;(\theta_j&space;-&space;\mu_j)^2&space;\newline&space;\newline&space;&plus;\sum_{\{j&space;|&space;p(\theta_j)&space;\text{&space;lognormal}\}}&space;\{&space;\log\theta_j&space;&plus;\frac{1}{2}&space;\beta_j&space;(\log\theta_j&space;-&space;\log\mu_j)^2\}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\text{Obj}(\underline{\theta})&space;=&space;-\log&space;p(\underline{\theta}&space;|&space;X,&space;\underline{y})&space;=&space;\newline&space;\newline&space;=&space;\text{const}&space;&plus;\frac{N}{2}&space;\log(\sigma^2)&space;&plus;\frac{1}{2\sigma^2}&space;\left&space;\|&space;W&space;\cdot&space;(\underline{y}&space;-&space;f(X,&space;\underline{\theta}))&space;\right&space;\|^2&space;&plus;&space;\newline&space;\newline&space;&plus;\frac{1}{2}&space;\cdot&space;\sum_{\{j&space;|&space;p(\theta_j)&space;\text{&space;gaussian}\}}&space;\beta_j&space;(\theta_j&space;-&space;\mu_j)^2&space;\newline&space;\newline&space;&plus;\sum_{\{j&space;|&space;p(\theta_j)&space;\text{&space;lognormal}\}}&space;\{&space;\log\theta_j&space;&plus;\frac{1}{2}&space;\beta_j&space;(\log\theta_j&space;-&space;\log\mu_j)^2\}" title="\text{Obj}(\underline{\theta}) = -\log p(\underline{\theta} | X, \underline{y}) = \newline \newline = \text{const} +\frac{N}{2} \log(\sigma^2) +\frac{1}{2\sigma^2} \left \| W \cdot (\underline{y} - f(X, \underline{\theta})) \right \|^2 + \newline \newline +\frac{1}{2} \cdot \sum_{\{j | p(\theta_j) \text{ gaussian}\}} \beta_j (\theta_j - \mu_j)^2 \newline \newline +\sum_{\{j | p(\theta_j) \text{ lognormal}\}} \{ \log\theta_j +\frac{1}{2} \beta_j (\log\theta_j - \log\mu_j)^2\}" /></a>

(the term const incorporates terms that are independent of both <ins>&theta;</ins> and &sigma;). This corresponds to minimizing a weighted sum of squared residuals plus a series of regularization terms.

The following contributions are added to _&phi;_ and to the j-th component of its gradient due to the presence of prior distributions:

Gaussian prior:

<a href="https://www.codecogs.com/eqnedit.php?latex=\phi(\underline{\delta})&space;\text{:&space;}&space;\text{&space;}&space;\frac{1}{2}&space;\cdot&space;\beta_j&space;(\theta_j&space;&plus;&space;\delta_j&space;-&space;\mu_j)^2&space;\newline&space;\newline&space;\nabla_{\underline{\delta}}&space;\phi(\underline{\delta})_j&space;\text{:&space;}&space;\text{&space;}&space;\beta_j&space;\delta_j&space;&plus;\beta_j(\theta_j&space;-&space;\mu_j)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\phi(\underline{\delta})&space;\text{:&space;}&space;\text{&space;}&space;\frac{1}{2}&space;\cdot&space;\beta_j&space;(\theta_j&space;&plus;&space;\delta_j&space;-&space;\mu_j)^2&space;\newline&space;\newline&space;\nabla_{\underline{\delta}}&space;\phi(\underline{\delta})_j&space;\text{:&space;}&space;\text{&space;}&space;\beta_j&space;\delta_j&space;&plus;\beta_j(\theta_j&space;-&space;\mu_j)" title="\phi(\underline{\delta}) \text{: } \text{ } \frac{1}{2} \cdot \beta_j (\theta_j + \delta_j - \mu_j)^2 \newline \newline \nabla_{\underline{\delta}} \phi(\underline{\delta})_j \text{: } \text{ } \beta_j \delta_j +\beta_j(\theta_j - \mu_j)" /></a>

Lognormal prior:

<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{100}&space;\begin{matrix}&space;\phi(\underline{\delta})&space;\text{:&space;}&space;&&space;\frac{1}{2}&space;\beta_j(\log(\theta_j&space;&plus;&space;\delta_j)&space;-&space;\log\mu_j)^2&space;&plus;&space;\log(\theta_j&space;&plus;&space;\delta_j)&space;\sim&space;\\&space;\\&space;&&space;\frac{1}{2}&space;\beta_j(\log\theta_j&space;&plus;\frac{\delta_j}{\theta_j}&space;-&space;\log\mu_j)^2&space;&plus;\log\theta_j&space;&plus;\frac{\delta_j}{\theta_j}&space;\text{&space;}&space;\text{&space;}&space;\text{&space;}&space;\text{&space;}&space;\text{&space;}&space;\text{&space;}&space;\text{&space;}&space;\end{matrix}&space;\newline&space;\newline&space;\newline&space;\nabla_{\underline{\delta}}&space;\phi(\underline{\delta})_j&space;\text{:&space;}&space;\text{&space;}&space;\text{&space;}&space;\frac{\beta_j}{\theta_j^2}&space;\cdot&space;\delta_j&space;&plus;&space;\{\frac{1}{\theta_j}&space;&plus;\frac{\beta_j}{\theta_j}\log\frac{\theta_j}{\mu_j}&space;\}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\dpi{100}&space;\begin{matrix}&space;\phi(\underline{\delta})&space;\text{:&space;}&space;&&space;\frac{1}{2}&space;\beta_j(\log(\theta_j&space;&plus;&space;\delta_j)&space;-&space;\log\mu_j)^2&space;&plus;&space;\log(\theta_j&space;&plus;&space;\delta_j)&space;\sim&space;\\&space;\\&space;&&space;\frac{1}{2}&space;\beta_j(\log\theta_j&space;&plus;\frac{\delta_j}{\theta_j}&space;-&space;\log\mu_j)^2&space;&plus;\log\theta_j&space;&plus;\frac{\delta_j}{\theta_j}&space;\text{&space;}&space;\text{&space;}&space;\text{&space;}&space;\text{&space;}&space;\text{&space;}&space;\text{&space;}&space;\text{&space;}&space;\end{matrix}&space;\newline&space;\newline&space;\newline&space;\nabla_{\underline{\delta}}&space;\phi(\underline{\delta})_j&space;\text{:&space;}&space;\text{&space;}&space;\text{&space;}&space;\frac{\beta_j}{\theta_j^2}&space;\cdot&space;\delta_j&space;&plus;&space;\{\frac{1}{\theta_j}&space;&plus;\frac{\beta_j}{\theta_j}\log\frac{\theta_j}{\mu_j}&space;\}" title="\begin{matrix} \phi(\underline{\delta}) \text{: } & \frac{1}{2} \beta_j(\log(\theta_j + \delta_j) - \log\mu_j)^2 + \log(\theta_j + \delta_j) \sim \\ \\ & \frac{1}{2} \beta_j(\log\theta_j +\frac{\delta_j}{\theta_j} - \log\mu_j)^2 +\log\theta_j +\frac{\delta_j}{\theta_j} \text{ } \text{ } \text{ } \text{ } \text{ } \text{ } \text{ } \end{matrix} \newline \newline \newline \nabla_{\underline{\delta}} \phi(\underline{\delta})_j \text{: } \text{ } \text{ } \frac{\beta_j}{\theta_j^2} \cdot \delta_j + \{\frac{1}{\theta_j} +\frac{\beta_j}{\theta_j}\log\frac{\theta_j}{\mu_j} \}" /></a>

(We have used the first-order expansion of log in the lognormal case).

The linear system is now

<a href="https://www.codecogs.com/eqnedit.php?latex=\left(&space;\frac{J^T&space;W^T&space;W&space;J}{\sigma^2}&space;&plus;&space;R&space;\right)&space;\cdot&space;\underline{\delta}&space;=&space;\frac{1}{\sigma}&space;J^T&space;W^T&space;\underline{\epsilon}&space;&plus;&space;\underline{r}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\left(&space;\frac{J^T&space;W^T&space;W&space;J}{\sigma^2}&space;&plus;&space;R&space;\right)&space;\cdot&space;\underline{\delta}&space;=&space;\frac{1}{\sigma}&space;J^T&space;W^T&space;\underline{\epsilon}&space;&plus;&space;\underline{r}" title="\left( \frac{J^T W^T W J}{\sigma^2} + R \right) \cdot \underline{\delta} = \frac{1}{\sigma} J^T W^T \underline{\epsilon} + \underline{r}" /></a>

where

<a href="https://www.codecogs.com/eqnedit.php?latex=R_{jj}&space;=&space;\left\{\begin{matrix}&space;\beta_j&space;&&space;&&space;\text{if&space;}&space;p_j(\theta_j)&space;=&space;N(\mu_j,&space;\frac{1}{\beta_j^2})&space;\\&space;\newline&space;\\&space;\frac{\beta_j}{\theta_j^2}&space;&&space;&&space;\text{if&space;}&space;p_j(\theta_j)&space;=&space;\text{logN}(\log\mu_j,&space;\frac{1}{\beta_j})&space;\end{matrix}\right.,&space;\text{&space;}&space;\text{&space;}&space;\text{&space;}&space;R_{ij}&space;=&space;0&space;\text{&space;if&space;}&space;i&space;\neq&space;j" target="_blank"><img src="https://latex.codecogs.com/svg.latex?R_{jj}&space;=&space;\left\{\begin{matrix}&space;\beta_j&space;&&space;&&space;\text{if&space;}&space;p_j(\theta_j)&space;=&space;N(\mu_j,&space;\frac{1}{\beta_j^2})&space;\\&space;\newline&space;\\&space;\frac{\beta_j}{\theta_j^2}&space;&&space;&&space;\text{if&space;}&space;p_j(\theta_j)&space;=&space;\text{logN}(\log\mu_j,&space;\frac{1}{\beta_j})&space;\end{matrix}\right.,&space;\text{&space;}&space;\text{&space;}&space;\text{&space;}&space;R_{ij}&space;=&space;0&space;\text{&space;if&space;}&space;i&space;\neq&space;j" title="R_{jj} = \left\{\begin{matrix} \beta_j & & \text{if } p_j(\theta_j) = N(\mu_j, \frac{1}{\beta_j^2}) \\ \newline \\ \frac{\beta_j}{\theta_j^2} & & \text{if } p_j(\theta_j) = \text{logN}(\log\mu_j, \frac{1}{\beta_j}) \end{matrix}\right., \text{ } \text{ } \text{ } R_{ij} = 0 \text{ if } i \neq j" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=r_j&space;=&space;\left\{\begin{matrix}&space;-\beta_j&space;(\theta_j&space;-&space;\mu_j)&&space;&&space;\text{if&space;}&space;p_j(\theta_j)&space;=&space;N(\mu_j,&space;\frac{1}{\beta_j})&space;\\&space;\newline&space;\\&space;-\frac{1}{\theta_j}&space;-&space;\frac{\beta_j}{\theta_j}&space;\log&space;\frac{\theta_j}{\mu_j}&space;&&space;&&space;\text{if&space;}&space;p_j(\theta_j)&space;=&space;\text{logN}(\log\mu_j,&space;\frac{1}{\beta_j})&space;\end{matrix}\right." target="_blank"><img src="https://latex.codecogs.com/svg.latex?r_j&space;=&space;\left\{\begin{matrix}&space;-\beta_j&space;(\theta_j&space;-&space;\mu_j)&&space;&&space;\text{if&space;}&space;p_j(\theta_j)&space;=&space;N(\mu_j,&space;\frac{1}{\beta_j})&space;\\&space;\newline&space;\\&space;-\frac{1}{\theta_j}&space;-&space;\frac{\beta_j}{\theta_j}&space;\log&space;\frac{\theta_j}{\mu_j}&space;&&space;&&space;\text{if&space;}&space;p_j(\theta_j)&space;=&space;\text{logN}(\log\mu_j,&space;\frac{1}{\beta_j})&space;\end{matrix}\right." title="r_j = \left\{\begin{matrix} -\beta_j (\theta_j - \mu_j)& & \text{if } p_j(\theta_j) = N(\mu_j, \frac{1}{\beta_j}) \\ \newline \\ -\frac{1}{\theta_j} - \frac{\beta_j}{\theta_j} \log \frac{\theta_j}{\mu_j} & & \text{if } p_j(\theta_j) = \text{logN}(\log\mu_j, \frac{1}{\beta_j}) \end{matrix}\right." /></a>

and &sigma; is replaced by its maximum likelihood estimate (see above).

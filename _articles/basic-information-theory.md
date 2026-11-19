---
title: "basic Information theory"
category: "Machine Learning"
order: 2
description: "从 Shannon 熵、编码与 KL 散度推到 Fisher 度量和自然梯度。"
---

* TOC
{:toc}

默认用 $$\ln$$，单位为 nat。下标 $$2$$ 表示以 bit 计量，$$H_2=H/\ln 2$$。离散部分取有限字母表，约定 $$0\ln0=0$$。

## 自信息和熵

独立事件的信息可加。若 $$j$$ 随概率严格递减，令 $$g(t)=j(e^{-t})$$。

$$
\begin{aligned}
j(ab)&=j(a)+j(b),\qquad j(1)=0,\\
g(t+s)&=g(t)+g(s),\qquad g(0)=0,\\
g(m/n)&=(m/n)g(1),\qquad m,n\in\mathbb N,\ n>0,\\
g(t)&=t g(1)\quad\text{（用单调性从有理数推广）},\\
j(p)&=-\ln p \quad\text{（取 }g(1)=1\text{）},\\
H(X)&=\mathbb E[j(p_X(X))]=-\sum_xp_X(x)\ln p_X(x).
\end{aligned}
$$

二元熵函数

$$
\begin{aligned}
h_b(u)&=-u\ln u-(1-u)\ln(1-u),\\
h_b'(u)&=\ln\frac{1-u}{u},\qquad
h_b''(u)=-\frac1{u(1-u)},\\
h_b(0)&=h_b(1)=0,\qquad \max_{u\in[0,1]} h_b(u)=h_b(\tfrac12)=\ln2.
\end{aligned}
$$

## 交叉熵和 KL 散度

$$
\begin{aligned}
H(p,q)&=-\sum_x p(x)\ln q(x),\\
D(p\|q)&=\sum_{x:p(x)>0}p(x)\ln\frac{p(x)}{q(x)}
            =H(p,q)-H(p).
\end{aligned}
$$

若存在 $$p(x)>0=q(x)$$，则 $$D(p\|q)=+\infty$$。其余情形由 Jensen 不等式，

$$
D(p\|q)
\ \ge\ -\ln\sum_{x:p(x)>0} q(x)
\ \ge\ 0,\qquad
D(p\|q)=0 \iff p=q .
$$

有限 $$K$$ 个结果，令 $$u_K$$ 为均匀分布。

$$
0\le H(p)=\ln K-D(p\|u_K)\le\ln K .
$$

令 $$d(a\|b)=D(\mathrm{Bern}(a)\|\mathrm{Bern}(b))$$。

$$
\begin{aligned}
d(0.2\|0.5)&\approx0.193,\qquad d(0.5\|0.2)\approx0.223,\\
d(0.2\|0.8)&\approx0.832
   > d(0.2\|0.5)+d(0.5\|0.8)\approx0.416.
\end{aligned}
$$

对互不相交的 $$p_1,p_2$$，若 $$p=(p_1+p_2)/2$$，$$q=p_1$$，则 $$D(p\|q)=+\infty$$，$$D(q\|p)=\ln2$$。

## 联合熵、条件熵和互信息

$$
\begin{aligned}
H(X,Y)
 &=-\sum_{x,y}p(x,y)\ln p(x,y)\\
 &=-\sum_{x,y}p(x,y)\ln p(x)
   -\sum_{x,y}p(x,y)\ln p(y\mid x)\\
 &=H(X)+H(Y\mid X),\\
H(X_1,\ldots,X_n)&=\sum_{i=1}^n H(X_i\mid X_{<i}).
\end{aligned}
$$

$$
\begin{aligned}
I(X;Y)
 &=H(X)+H(Y)-H(X,Y)\\
 &=H(Y)-H(Y\mid X)\\
 &=\sum_{x,y}p(x,y)\ln\frac{p(x,y)}{p(x)p(y)}
   =D(p_{XY}\|p_Xp_Y),\\
I(X;Y\mid Z)
 &=\mathbb E_Z D(p_{XY\mid Z}\|p_{X\mid Z}p_{Y\mid Z})
   =H(X\mid Z)-H(X\mid Y,Z),\\
I(X;Y,Z)&=I(X;Y)+I(X;Z\mid Y).
\end{aligned}
$$

于是 $$I(X;Y)=0\iff X\perp Y$$，且 $$H(X\mid Y,Z)\le H(X\mid Z)$$。

### 数据处理不等式

$$
\begin{aligned}
X\to Y\to Z &\implies I(X;Z\mid Y)=0,\\
I(X;Y)&=I(X;Y,Z)=I(X;Z)+I(X;Y\mid Z)\ge I(X;Z).
\end{aligned}
$$

### Jensen–Shannon、全变差和 Pinsker 不等式

令 $$S\sim\mathrm{Bern}(\tfrac12)$$，$$X\mid S=0\sim p$$，$$X\mid S=1\sim q$$，故 $$m=(p+q)/2$$。

$$
\begin{aligned}
\operatorname{JS}(p,q)
 &=\tfrac12D(p\|m)+\tfrac12D(q\|m)\\
 &=H(m)-\tfrac12H(p)-\tfrac12H(q)=I(S;X)\in[0,\ln2].
\end{aligned}
$$

令 $$A=\{x:p(x)\ge q(x)\}$$，$$a=p(A)$$，$$b=q(A)$$。由对数和不等式将字母表缩为 $$A,A^c$$。

$$
\begin{aligned}
\sum_i u_i\ln\frac{u_i}{v_i}
 &= U\ln\frac UV+U D(u/U\|v/V)\ge U\ln\frac UV,\\
\operatorname{TV}(p,q)&=\tfrac12\sum_x|p(x)-q(x)|=a-b,\\
D(p\|q)&\ge d(a\|b),\qquad
\partial_a^2 d(a\|b)=\frac1{a(1-a)}\ge4,\\
d(b\|b)&=0,\quad \left.\partial_a d(a\|b)\right|_{a=b}=0
 \implies D(p\|q)\ge2\operatorname{TV}(p,q)^2 .
\end{aligned}
$$

这里 $$U=\sum_i u_i$$，$$V=\sum_i v_i$$。边界由极限或无穷 KL 处理。

### Fano 不等式

设 $$X$$ 取 $$M\ge2$$ 个值，$$\hat X$$ 由 $$Y$$ 解码。令 $$E=\mathbf1_{\{X\ne\hat X\}}$$，$$P_e=\Pr(E=1)$$。

$$
\begin{aligned}
H(X\mid Y)&\le H(X\mid\hat X)=H(E,X\mid\hat X)\\
 &=H(E\mid\hat X)+H(X\mid E,\hat X)\\
 &\le h_b(P_e)+P_e\ln(M-1),\\
X\sim{\rm Unif}([M]) &\implies
P_e\ge1-\frac{I(X;Y)+\ln2}{\ln M}.
\end{aligned}
$$

## 无损信源编码与熵率

### Kraft 不等式与平均码长

二元前缀码在二叉树上对应互不包含的结点，叶区间总长度 $$K=\sum_x2^{-\ell(x)}\le1$$。取 $$q(x)=2^{-\ell(x)}/K$$。

$$
\mathbb E_p[\ell(X)]
=H_2(p,q)-\log_2K
=H_2(p)+\frac{D(p\|q)}{\ln2}-\log_2K\ge H_2(p).
$$

反过来，满足 Kraft 条件的整数码长可按从短到长分配互不重叠的二进制区间。Shannon 码的长度为

$$
\ell(x)=\lceil-\log_2p(x)\rceil,\qquad
\sum_x2^{-\ell(x)}\le1,\qquad
H_2(p)\le\mathbb E\ell(X)<H_2(p)+1 .
$$

独立同分布的 $$n$$ 符号块用同一构造，每符号的平均码长在 $$[H_2(p),H_2(p)+1/n)$$。 [Shannon, A Mathematical Theory of Communication](https://web.mit.edu/6.441/spring05/reading/shannon1948.pdf)

### AEP

$$
-\frac1n\ln p(X^n)=\frac1n\sum_{i=1}^n[-\ln p(X_i)]
 \xrightarrow{\mathrm{a.s.}}H(X).
$$

令 $$T_{n,\epsilon}=\{x^n:e^{-n(H+\epsilon)}\le p(x^n)\le e^{-n(H-\epsilon)}\}$$，$$\delta_n=1-\Pr(X^n\in T_{n,\epsilon})\to0$$。

$$
(1-\delta_n)e^{n(H-\epsilon)}\le |T_{n,\epsilon}|\le e^{n(H+\epsilon)}.
$$

典型集内用至多 $$\lceil n(H+\epsilon)/\ln2\rceil$$ bit 编号，集外允许错误。

### 平稳过程的熵率

$$
\begin{aligned}
a_n&=H(X_n\mid X_1,\ldots,X_{n-1}),\\
a_{n+1}&\le H(X_{n+1}\mid X_2,\ldots,X_n)=a_n,\\
\overline H&=\lim_{n\to\infty}\frac1n H(X^n)
 =\lim_{n\to\infty}\frac1n\sum_{i=1}^na_i=\lim_{n\to\infty}a_n .
\end{aligned}
$$

一阶平稳 Markov 链满足 $$\overline H=H(X_2\mid X_1)$$。

## 无记忆信道容量与率失真

### 信道容量

离散无记忆信道 $$W(y\mid x)$$。均匀消息 $$M$$ 含 $$N$$ 个值，编码 $$M\to X^n\to Y^n\to\widehat M$$。

$$
\begin{aligned}
(1-P_e)\ln N-\ln2 &\le I(M;Y^n)\\
 &\le I(X^n;Y^n)\\
 &=H(Y^n)-\sum_i H(Y_i\mid X_i)\\
 &\le\sum_i I(X_i;Y_i)\le n\max_{p_X}I(X;Y).
\end{aligned}
$$

可达性。独立生成 $$N$$ 个 $$p_X^n$$ 码字，解码选择唯一满足 $$\imath(x^n;y^n)>n(I-\epsilon)$$ 的码字，其中

$$
\begin{aligned}
\imath(x^n;y^n)&=\ln\frac{W^n(y^n\mid x^n)}{p_Y^n(y^n)},\\
\frac{\imath(X^n;Y^n)}n&\xrightarrow{\Pr} I(X;Y)
       \quad((X_i,Y_i)\sim p_X W\text{ 独立同分布}),\\
\mathbb E_{p_X^n p_Y^n}[e^{\imath(X^n;Y^n)}]&=1,\\
\mathbb E_{\rm code}[P_e]&\le
    \Pr_{p_X^n W^n}(\imath\le n(I-\epsilon))+(N-1)e^{-n(I-\epsilon)}.
\end{aligned}
$$

若 $$\ln N/n<I-\epsilon$$ 且有固定余量，右侧趋零。故 $$C=\max_{p_X}I(X;Y)$$。 [Shannon, A Mathematical Theory of Communication](https://web.mit.edu/6.441/spring05/reading/shannon1948.pdf)

二元对称信道 $$Y=X\oplus Z$$，$$Z\sim{\rm Bern}(\epsilon)$$ 独立。

$$
I(X;Y)=H(Y)-h_b(\epsilon)\le\ln2-h_b(\epsilon).
$$

均匀输入取等号，所以 $$C_2=1-h_b(\epsilon)/\ln 2$$。

### 率失真

有限字母表，逐符号失真 $$d(x,\hat x)$$ 有界。令

$$
R(D)=\min_{q(\hat x\mid x):\, \mathbb E d(X,\hat X)\le D}
                   I(X;\hat X).
$$

固定 $$p_X$$ 时，$$I=D(p_Xq\,\|\,p_Xp_{\hat X})$$ 对 $$q$$ 凸，由逐项对数和不等式得到。混合两个重建核同时混合其平均失真，故 $$R(D)$$ 凸。对独立同分布信源的 $$N$$ 码字量化，

$$
\begin{aligned}
\ln N&\ge I(X^n;\hat X^n)\\
 &=\sum_iH(X_i)-H(X^n\mid\hat X^n)\\
 &\ge\sum_i[H(X_i)-H(X_i\mid\hat X_i)]\\
 &\ge\sum_i R(D_i)\ge nR(\bar D),
 \quad D_i=\mathbb E d(X_i,\hat X_i),\quad \bar D=\frac1n\sum_iD_i.
\end{aligned}
$$

可达性。固定一个满足失真约束的 $$q$$，记 $$I=I(X;\hat X)$$，令 $$A_n(x^n)$$ 包含平均失真至多 $$D+\epsilon$$、信息密度 $$\ln(q^n/p_{\hat X}^n)$$ 至多 $$n(I+\epsilon)$$ 的重建序列。

$$
\begin{aligned}
\Pr_{p_X^n q^n}(A_n)&\to1,\\
\Pr_{p_X^n}\big[q^n(A_n(X^n)\mid X^n)\ge\tfrac12\big]&\to1,\\
p_{\hat X}^n(A_n(x^n))&\ge
 e^{-n(I+\epsilon)}q^n(A_n(x^n)\mid x^n).
\end{aligned}
$$

独立从 $$p_{\hat X}^n$$ 抽取 $$N$$ 个重建码字。对上述高概率的 $$x^n$$，无可用重建的概率至多 $$\exp[-Ne^{-n(I+\epsilon)}/2]$$。若 $$\ln N/n>I+\epsilon$$ 且有固定余量，平均失真至多 $$D+\epsilon+o(1)$$。

均匀二元信源，Hamming 失真。令 $$E=X\oplus\hat X$$。

$$
\begin{aligned}
I(X;\hat X)&=\ln2-H(E\mid\hat X)\\
&\ge\ln2-h_b(\Pr(E=1))\ge \ln2-h_b(D),
  \qquad 0\le D\le\tfrac12,\\
\hat X&=X\oplus Z,\quad Z\sim{\rm Bern}(D),\quad Z\perp X\\
&\implies Z\perp \hat X,\qquad
R(D)=
\begin{cases}
\ln2-h_b(D),&0\le D\le\tfrac12,\\
0,&D\ge\tfrac12 .
\end{cases}
\end{aligned}
$$

## 分类交叉熵与 Perplexity

$$
\begin{aligned}
\mathcal L(\theta)
 &=\mathbb E_{p(x,y)}[-\ln q_\theta(y\mid x)]\\
 &=H_p(Y\mid X)+\mathbb E_{p_X}D(p(\cdot\mid X)\|q_\theta(\cdot\mid X)),\\
\widehat{\mathcal L}(\theta)
 &=-\frac1n\sum_i\ln q_\theta(y_i\mid x_i)
   =\frac1n\sum_i H(\delta_{y_i},q_\theta(\cdot\mid x_i)).
\end{aligned}
$$

令 $$q_j=e^{z_j}/\sum_k e^{z_k}$$，目标分布 $$t$$ 满足 $$\sum_jt_j=1$$。

$$
\begin{aligned}
H(t,q)
 &= -\sum_j t_j z_j+\ln\sum_j e^{z_j},\\
\partial_{z_j}H(t,q)&=q_j-t_j,\qquad
\nabla_z^2 H(t,q)=\operatorname{diag}(q)-qq^\top.
\end{aligned}
$$

自回归因子分解 $$q(x^T)=\prod_{i=1}^Tq(x_i\mid x_{<i})$$ 给出

$$
\mathrm{PPL}
=q(x^T)^{-1/T}
=\exp\!\left(-\frac1T\sum_{i=1}^T\ln q(x_i\mid x_{<i})\right).
$$

## 微分熵与最大熵

若 $$Y=g(X)$$ 是同维可逆变换，

$$
\begin{aligned}
h(X)&=-\int p_X(x)\ln p_X(x)\,dx,\\
p_Y(g(x))&=p_X(x)/|\det J_g(x)|,\\
h(Y)&=h(X)+\mathbb E[\ln|\det J_g(X)|],\\
D(p_Y\|q_Y)&=D(p_X\|q_X).
\end{aligned}
$$

例如 $$X\sim{\rm Unif}[0,a]$$ 时 $$h(X)=\ln a$$。若 $$g=\mathcal N(\mu,\Sigma)$$，$$\Sigma\succ0$$，

$$
\begin{aligned}
h(g)&=\tfrac12\ln((2\pi)^d\det\Sigma)
         +\tfrac12\mathbb E_g[(X-\mu)^\top\Sigma^{-1}(X-\mu)]\\
 &=\tfrac12\ln((2\pi e)^d\det\Sigma),\\
\mathbb E_p[-\ln g(X)]&=\mathbb E_g[-\ln g(X)]\quad
  \text{（}p\text{ 与 }g\text{ 的均值和协方差相同）},\\
h(g)-h(p)&=D(p\|g)\ge0 .
\end{aligned}
$$

有限字母表上给定矩约束 $$\mathbb E_p[T]=m$$。若存在 $$\theta$$ 使 $$p_\theta$$ 满足该约束，

$$
\begin{aligned}
A(\theta)&=\ln\sum_x e^{\theta^\top T(x)},\qquad
p_\theta(x)=e^{\theta^\top T(x)-A(\theta)},\\
D(p\|p_\theta)&=A(\theta)-\theta^\top m-H(p),\\
H(p)&\le A(\theta)-\theta^\top m=H(p_\theta).
\end{aligned}
$$

令 $$q^*(x)=e^{r(x)/\tau}/Z$$，$$\tau>0$$。

$$
\mathbb E_q[r(X)]+\tau H(q)=\tau\ln Z-\tau D(q\|q^*)
\quad\Longrightarrow\quad
\arg\max_q\{\mathbb E_q[r(X)]+\tau H(q)\}=q^* .
$$

## 变分推断与 ELBO

固定观测 $$x$$ 与辅助分布 $$q(z\mid x)$$。

$$
\begin{aligned}
D(q\|p_\theta(\cdot\mid x))
 &=\mathbb E_q[\ln q(z\mid x)-\ln p_\theta(x,z)]+\ln p_\theta(x),\\
\ln p_\theta(x)&=\operatorname{ELBO}(\theta,q)+D(q\|p_\theta(\cdot\mid x)),\\
\operatorname{ELBO}(\theta,q)
 &=\mathbb E_q[\ln p_\theta(x,z)-\ln q(z\mid x)]\\
 &=\mathbb E_q[\ln p_\theta(x\mid z)]-D(q(z\mid x)\|p_\theta(z)).
\end{aligned}
$$

## 信息几何与自然梯度

### 从局部 KL 到 Fisher 度量

设 $$p_\theta$$ 有共同支集，可交换求导与积分。记 $$s_\theta(x)=\nabla_\theta\ln p_\theta(x)$$。

$$
\begin{aligned}
\mathbb E_\theta[s_\theta]&=\int\nabla_\theta p_\theta(x)\,dx=0,\\
\mathbb E_\theta[\nabla_\theta^2\ln p_\theta]
 +\mathbb E_\theta[s_\theta s_\theta^\top]
 &=\int\nabla_\theta^2p_\theta(x)\,dx=0,\\
F(\theta)&=\mathbb E_\theta[s_\theta s_\theta^\top],\\
D(p_\theta\|p_{\theta+\Delta})
 &= -\mathbb E_\theta[s_\theta]^\top\Delta
    -\tfrac12\Delta^\top\mathbb E_\theta[\nabla_\theta^2\ln p_\theta]\Delta
    +o(\|\Delta\|^2)\\
 &=\tfrac12\Delta^\top F(\theta)\Delta+o(\|\Delta\|^2).
\end{aligned}
$$

指数族 $$p_\theta(x)\propto e^{\theta^\top T(x)}$$ 具有

$$
\begin{aligned}
F(\theta)&=\operatorname{Cov}_\theta(T)=\nabla^2 A(\theta),\\
D(p_\theta\|p_\eta)&=A(\eta)-A(\theta)-\nabla A(\theta)^\top(\eta-\theta).
\end{aligned}
$$

### 局部 KL 约束下的最陡方向

取 $$g=\nabla_\theta L\ne0$$，$$F\succ0$$，$$\epsilon>0$$。

$$
\begin{aligned}
\min_\Delta g^\top\Delta &\quad\text{s.t.}\quad
                          \tfrac12\Delta^\top F\Delta\le\epsilon,\\
g+\lambda F\Delta&=0,\qquad
\tfrac12\Delta^\top F\Delta=\epsilon,\\
\Delta^*&=-\sqrt{\frac{2\epsilon}{g^\top F^{-1}g}}\,F^{-1}g,\qquad
\widetilde\nabla_\theta L=F^{-1}g.
\end{aligned}
$$

换坐标 $$\theta=\theta(\phi)$$，$$J=\partial\theta/\partial\phi$$ 可逆。

$$
g_\phi=J^\top g_\theta,\qquad F_\phi=J^\top F_\theta J
\quad\Longrightarrow\quad
J F_\phi^{-1}g_\phi=F_\theta^{-1}g_\theta.
$$

坐标不变性作用于切向量。 [Amari, Natural Gradient Works Efficiently in Learning](https://doi.org/10.1162/089976698300017746)

### Bernoulli 算例与策略推广

$$
\begin{aligned}
X&\sim\mathrm{Bern}(p),\quad 0<p<1,\qquad
s_p(X)=\frac{X-p}{p(1-p)},\\
F_p&=\frac1{p(1-p)},\qquad
ds^2=\frac{dp^2}{p(1-p)}=\big[d(2\arcsin\sqrt p)\big]^2,\\
L(p)&=-r\ln p-(1-r)\ln(1-p),\\
\partial_pL&=\frac{p-r}{p(1-p)},\qquad
\widetilde\nabla_p L=p-r .
\end{aligned}
$$

策略情形，固定旧策略的状态分布 $$d_0$$，在 $$\theta_0$$ 对策略的代理目标取梯度 $$g$$。

$$
\begin{aligned}
s_0(s,a)&=\left.\nabla_\theta\ln\pi_\theta(a\mid s)\right|_{\theta_0},\\
F_\pi&=\mathbb E_{s\sim d_0,\,a\sim\pi_{\theta_0}(\cdot\mid s)}[s_0s_0^\top],\\
\mathbb E_{d_0}D(\pi_{\theta_0}\|\pi_{\theta_0+\Delta})
 &=\tfrac12\Delta^\top F_\pi\Delta+o(\|\Delta\|^2),\\
\max_{\frac12\Delta^\top F_\pi\Delta\le\epsilon} g^\top\Delta
 &\implies\Delta^*=
   \sqrt{\frac{2\epsilon}{g^\top F_\pi^{-1}g}}\,F_\pi^{-1}g .
\end{aligned}
$$

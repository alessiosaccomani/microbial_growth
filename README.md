# Microbial Growht :microbe: Final project of Laboratory of Computational Physics - Mod. B

**Group members:** Giovanni Zago, Emanuele Sarte, Alessio Saccomani, Fateme <br>
**Supervisor:** Prof. Carlo Albert <br>
**Academic year:** 2022/2023

This project has been carried out by a group of Physics of Data student as final project of the course Laboratory of Computational Physics - Mod. B, held by Prof. Marco Baiesi. The subject of the project and the supervision of the work has been provided by Prof. Carlo Albert. 

## Table of contents
1. [Introduction](#introduction)
    1. [Formalism](#formalism)
    2. [Model 1](#model1)
    3. [Model 1.2](#model1.2)
    4. [Model 2](#model2)
    4. [Model 3](#model3)

## Introduction <a name="introduction"></a>

The aim of this project is analyzing data coming from microbial coltures by performing Bayesian inference of parameters belonging to stochastic models that describe the growth and the lifetime of the microbes themselves. The datasets used in this work are *Tanouchi25c*, *Tanouchi37c* and *Susman18*. 

### Formalism <a name="formalism"></a>
The underlying idea that is shared by all the models considered for the analysis of the data is that a microbe can be described as a *dynamical system* characterized by a few dynamical variables $\vec{x}$ that obey a *"motion" equation*:

$$\begin{equation}\dot{\vec{x}} = F(\vec{x}) \tag{1}\end{equation}$$

Clearly, both $\vec{x}$ and $F(\vec{x})$ are specified by the considered model, and thus can vary a lot. For example, a dynamical variable that is common to all the models is the *microbe size* $m$ (with $[m] = \Bbb{L}$), thus by solving eq. (1) one gets how the size of a microbe grows over time. Another importat aspect that must be embedded in a model is the lifetime of a microbe. As a matter of fact, it is evident that $m$ can not become arbitrairly large, since the microbe would divide at a certain moment, generating a daughter cell. However, it is reasonable to think that the time at which the microbe divides is not deterministic, and so its lifespan should be drawn from a *probability distribution*, that, again, can vary according to the considered model. The best way to formalize this is to define a function $S(t)$ called *survival probability* that represents the probability of the microbe cell to survive (i.e. not divide) for $\hat{t} \leq t$. In general the survival probability obeys

$$\begin{equation}\frac{\dot{S}(t)}{S(t)} = - h(\vec{x}(t)) \tag{2}\end{equation}$$

where $h(\vec{x})$ is a generic function of the dynamic variables. Since $S(t)$ is a (sort of) cumulative distribudion function then $\dot{S}(t)$ is a probability distribution, and so $t \sim \dot{S}(t)$, with $t \geq 0$, represents, in theory, a possibile lifespan of the cell. This is a key concept because, for example, by collecting data samples of the microbes lifespans we can infer the parameters that describe $\dot{S}(t)$, which are also the parameters that describe $\vec{x}(t)$, allowing us to validate our models. <br>
Next we are going to showcase the models considered for the Bayesian inference through the project. 

### Model 1 <a name="model1"></a>
This first model is very basic and aims to get the essential features that a good model shoud embed to correctly describe the phenomenon we are facing. This model is single-traited, namely

$$\vec{x} \equiv x = m$$

with the microbe size $m$ obeying 

$$\dot{m} = \omega_1 (m + u)$$

In this case eq. (2) holds with

$$h(m) = \omega_2 \Big(1 + \frac{m}{v}\Big)$$

In these equations $\omega_1$ and $\omega_2$ represent two rates, so $[\omega_1] = [\omega_2] = \Bbb{T} ^ {-1}$ while $u, \, v$, with $[u] = [v] = \Bbb{L}$ represent two parameters that help the model to be more realistic. Another important feature that a model must provide is the conditional distibution that specifies how the microbe traits change after a division. In this model the distribution is

$$\mathcal{J}(m|m') = \delta \Big(m - \frac{m'}{2}\Big)$$

thus

$$m_{after} = \Braket{\mathcal{J}(m|m'), m} = \int m \, \delta \Big(m - \frac{m'}{2}\Big) dm = \frac{m'}{2}$$

By integrating eq. () we obtain

$$m(t) = e ^ {\omega_1 t} (m_0 + u) - u$$

that substituted in () returns

$$S(t) = \exp \Bigg(\omega_2 t \Big(\frac{u}{v} - 1\Big) - \frac{\omega_2 (m_0 + u)}{w_1 u} \Big(e^{\omega_1 t} - 1\Big)\Bigg)$$

### Model 1.2 <a name="model1.2"></a>
This model is a more accurate version of Model 1, since it embeds the fact that a microbe cell can divide only if a minimum size has been reached. Nevertheless, it is still a single trait model:

$$\vec{x} \equiv x = m$$

with the microbe size $m$ obeying

$$\dot{m} = \omega_1 m$$

Eq. () holds also in this case, but now

$$h(m) = \begin{cases} 0 \qquad \qquad m \lt u \\
\omega_2 \frac{m + v}{u + v} \qquad m \geq u\end{cases}$$

and also the expression for $\mathcal{J}(m|m')$ remains unchanged. Eq. () leads to 

$$m(t) = m_0 e ^ {\omega_1 t}$$

and thus we can define the following quantity

$$t^* \equiv \frac{1}{w_1} \log \Big(\frac{u}{m_0}\Big)$$

which represents the amount of time elapsed since the division of the mother cell in order for $h(m)$ to become non-zero. Now we can integrate eq. (), obtaining 

$$S(t) = 0 \qquad t \lt t^*$$

$$S(t) = \exp \Big(- \frac{\omega_2}{\omega_1(u+v)}[e ^ {\omega_1 t} - \omega_1 v t  - 1]\Big) \qquad t \geq t^*$$


Since $m_0$ could be smaller than $u$, then, in those cases, $t^* <0$, meaning that $h(m)$ is non-zero since the start of the daughter microbe's life and $S(t)$ has only the expression reported in eq. (). This means that $t=0$ is a valid sample from $\dot{S}(t)%$ and thus istantaneous division is allowed by this model. However, this feature is not realistic, because no obervation of this behaviours has been made on microbes. 

### Model 2 <a name="model2"></a>
Model 2 aims at correcting the istantaneous division problem that arises in Model 1.2. To do so we need to introduce another dynamic variable, $p$, which represents the amount of a fictious protein that needs to be accumulated over a certain threshold quantity in order to allow the division process. Thus in this case we have that

$$\vec{x} = \begin{pmatrix} m \\ 
p \end{pmatrix}$$

and

$$\dot{\vec{x}} = \begin{pmatrix} \omega_1 m \\ 
\omega_2 c m \end{pmatrix}\tag{3}$$

where $\omega_1$ and $\omega_2$ are the growth and the protein accumulation rate respectively and $c$ represents the protein concentration, which we can set equal to 1 without losing generality. The survival probability equation becomes

$$\frac{\dot{S}(t)}{S(t)} = - h(m,p) \tag{4}$$

with

$$h(m,p) = \begin{cases}0 \qquad \qquad \text{for} \quad p \lt u \\ 
\omega_2 \frac{p + v}{u + v} \qquad \text{for} \quad p \geq u\end{cases} \tag{5}$$

and the division equation becomes

$$\mathcal{J}(m,p | m',p') = \delta(p)\delta\Big(m - \frac{m'}{2}\Big)$$

meaning that the protein amount drops to zero at each cell division, while the microbe size after the division becomes half the one reached at the end of the previous life cycle. By integrating (3) we get

$$m(t) = m_0 e ^ {w_1 t} \tag{7}$$

$$p(t) = \frac{\omega_2}{\omega_1} m_0 (e ^ {w_1 t} - 1) \tag{8}$$

Now it is clear that the condition on the value of $h(m,p)$ reporten in eq. (5) is equivalent to 

$$t \geq \frac{1}{\omega_1} \log \Big(\frac{\omega_1 u}{\omega_2 m_0} + 1\Big) \equiv t^* \tag{9}$$

which is always a positive quantity. Thus by integrating (5) we get

$$S(t) = 1 \qquad t \lt t^* \tag{10a}$$

$$S(t) = \exp \Bigg(- \frac{\omega_2^2 m_0}{\omega_1^2 (u + v)} [e ^ {\omega_1 t} + \omega_1 t \Big(\frac{\omega_1 v}{\omega_2 m_0} - 1\Big) - 1]\Bigg) \qquad t \geq t^* \tag{10b}$$

from which it is evident that istantaneous division is not allowed. 


### Model 3 <a name="model3"></a>

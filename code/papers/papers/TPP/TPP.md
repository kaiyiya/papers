# <span id="page-0-1"></span>TPP-Gaze: Modelling Gaze Dynamics in Space and Time with Neural Temporal Point Processes

<sup>1</sup>Alessandro D'Amelio, <sup>2</sup>Giuseppe Cartella, <sup>2</sup>Vittorio Cuculo, <sup>1</sup>Manuele Lucchi, <sup>2</sup>Marcella Cornia, <sup>2</sup>Rita Cucchiara, <sup>1</sup>Giuseppe Boccignone <sup>1</sup>University of Milan, Italy <sup>2</sup>University of Modena and Reggio Emilia, Italy

> 1 {name.surname}@unimi.it <sup>2</sup> {name.surname}@unimore.it

# Abstract

*Attention guides our gaze to fixate the proper location of the scene and holds it in that location for the deserved amount of time given current processing demands, before shifting to the next one. As such, gaze deployment crucially is a temporal process. Existing computational models have made significant strides in predicting spatial aspects of observer's visual scanpaths (*where *to look), while often putting on the background the temporal facet of attention dynamics (*when*). In this paper we present* TPP-Gaze*, a novel and principled approach to model scanpath dynamics based on Neural Temporal Point Process (TPP), that jointly learns the temporal dynamics of fixations position and duration, integrating deep learning methodologies with point process theory. We conduct extensive experiments across five publicly available datasets. Our results show the overall superior performance of the proposed model compared to state-of-the-art approaches. Source code and trained models are publicly available at:* <https://github.com/phuselab/tppgaze>*.*

# 1. Introduction

Gaze, the act of directing the eyes toward a location in the visual world, is considered a good measure of overt attention and, more generally, a window to the observer's thoughts, intentions, and emotions. It is no surprise that research spanning decades has struggled to produce several computational models aiming at effectively predicting attention towards regions or events within the landscape of visual and multimodal stimuli. With roots in psychology and neuroscience, these approaches have gained traction in the computer vision and pattern recognition fields since the seminal Itti *et al.* [\[32\]](#page-9-0) model; more recently, state-of-the-art approaches rely on machine learning advancements, typically employing deep neural architectures to the purpose (but see [\[37\]](#page-9-1) or [\[13\]](#page-8-0) for an in-depth review). As a matter

<span id="page-0-0"></span>![](_page_0_Figure_8.jpeg)

Figure 1. Scanpath dynamics as a marked TPP. Time is represented on the horizontal axis, and different scanpath fixations occurs at time t1, t2, t<sup>3</sup> and t4.

of fact, the vast majority of works in the field has focused on the computational modelling of spatial saliency in the shape of saliency maps, namely, a topographic map representing the likelihood of fixating a given location of the scrutinised stimulus, a fixation being defined as the period of time during which a part of the visual stimulus (the patch) on the screen is gazed at. Nevertheless, a growing number of models (*i.e.*, scanpath models) are addressing the prediction of a sequence of fixations – namely, the scanpath – where the gaze shift from one fixation to the next represents a saccade. Beyond the salience representation, these models explicitly unfold the dynamics of overt attention allocation over a stimulus [\[38\]](#page-9-2). It is worth remarking, though, that barely predicting the spatial sequence of fixations, does not entail proper modelling of the temporal evolution of attention. By and large, most scanpath models predict an ordered sequence of events while neglecting their continuous timestamp information. As a result, these models are able to tell *where* to look and in what order, but fail in answering *when*. In many respects, this is not an innocent flaw: human actions often rely on visual information, therefore it is important to direct attention to the right place at the right time [\[52\]](#page-9-3). Practically, modelling *when* to perform a saccade translates to devising scanpath models able to predict the sequence of both fixations position and corresponding <span id="page-1-1"></span>duration. Albeit recently few approaches have successfully dealt with such problem [15, 16, 18, 45, 51] via fully engineered approaches, only a marginal subset of them has tackled it in a mathematically principled way [6, 21, 52]. This has typically resulted in a weaker generality of the methods which are tailored to specific contexts or applications. Under such circumstances, the chief concern of the present work is to introduce a fresh, general and simple view on the problem of scanpath modelling: in brief, we consider a scanpath as the realisation of a point process in space and time, precisely that of a Neural Temporal Point Process.

Temporal Point Processes (TPPs) are probabilistic generative models designed for continuous-time event sequences. Neural TPPs [44, 48, 64, 65] integrate key concepts from point process literature with deep learning methodologies, facilitating the creation of adaptable and effective models. Notably, the modelling assumptions of Neural TPPs align perfectly with the structure of scanpath data. A scanpath consists of a series of events (saccades) occurring at irregular intervals (fixation durations), which is exactly what Neural TPPs are designed to model. While the psychological and neuroscience literature has used traditional point processes for eye movement analysis [5, 25, 60], these tools are not well-suited for scanpath prediction due to their inability to handle stimuli. In other words, traditional TPPs are effective for studying the observer but fall short when addressing Computer Vision tasks related to attention allocation prediction. In contrast, Neural TPP-based models offer the best of both worlds: they combine the robust theoretical framework of TPPs with the flexibility and power of modern neural networks. Nevertheless, this is the first attempt to adopt them for the scanpath modelling problem.

Our key contributions can be summarised as follows: 1) We propose a novel scanpath model able to jointly learn the temporal dynamics of both fixations position and duration. 2) We extend recent Neural TPP models to deal with visual data (*i.e.*, images) and connect scanpath modelling and prediction to point process theory. To assess our proposal, which can be appreciated at a glance in Fig. 1, we conduct experiments on five publicly available datasets, showing an overall superior performance of the proposed model when compared to state-of-the-art approaches.

### 2. Background and Related Work

#### 2.1. Neural Temporal Point Processes (TPPs)

Consider a sequence of generic events happening irregularly over time, TPPs model the next arrival time of an event by conditioning on the past events. Specifically, denote  $\mathcal{H}_t = \{t_n \in \mathcal{T} : t_n < t\}$  (with  $\mathcal{T}$  representing the sequence of strictly increasing arrival times of events) the history of arrival times of all events up to time t, the relation between the current arrival time t and the history, is

typically determined by the conditional intensity function  $\lambda^*(t) = \lambda(t|\mathcal{H}_t)$ , whose functional form determines the properties of the TPP. Equivalently, the sequence of strictly positive inter-event times  $\tau_n = t_n - t_{n-1}$  can be considered. Knowing the conditional intensity function allows to recover the conditional probability of the inter-arrival time of an event:

<span id="page-1-0"></span>
$$p^*(\tau_n) = p(\tau_n | \mathcal{H}_{t_n})$$

$$= \lambda^*(t_{n-1} + \tau_n) \exp\left(-\int_0^{\tau_n} \lambda^*(t_{n-1} + s) ds\right).$$
(1)

For instance, under the the assumptions of no dependence on the history and constancy over time (i.e.,  $\lambda^*(t) =$ k, with  $k \geq 0$ ), the homogeneous Poisson process is recovered, with inter-event times distributed according to the exponential distribution. Choosing more complex functional forms for  $\lambda^*(t)$  allows to recover many well known TPPs such as Hawkes or self-correcting processes [27, 31]. Clearly, restricting  $\lambda^*(t)$  to a specific parameterisation limits the general applicability of TPPs. For this reason, most recent solutions resorted to neural approaches (Neural TPPs) implementing learnable parametric forms of the intensity function,  $\lambda_{\theta}^{*}(t)$  [24, 30]. As an example, early Neural TPPs, such as the Neural Hawkes Process [44], used RNNs to model the intensity function of the process. More recently, self-attention mechanisms have been employed to the same purpose [64, 65]. The choice of the parametric form for the intensity function has to take into account the necessity of a closed form solution of the integral in Eq. (1), thus practically restricting the expressiveness of the model. More complex parametric forms would require Monte Carlo approximation of the integral [44]. To overcome such limitations, Shchur et al. [48] recently proposed to directly learn the parametric conditional distribution  $p_{\theta}^*(\tau)$  of the interarrival times rather than the conditional intensity function  $\lambda_{\theta}^{*}(t)$ , thus recasting learning Neural TPPs as a density estimation problem.

Marked TPPs. The basic mathematical formalism of TPPs allows to naturally handle the dynamics of arrival times of events. However, the distribution of time until the next event might depend on factors other than the history. Event data is often accompanied with some kind of covariate indicating the nature of the specific event being predicted. In the realm of TPPs, such covariate are called *marks*. More formally, a marked TPP is a random process whose realisations consists of a sequence of discrete events localised in time,  $\{\mathbf{r}_{F_n}, t_n\}$ , with the timing  $t_n \in \mathbb{R}^+$  and the mark  $\mathbf{r}_{F_n} \in \mathcal{M}$ . The mark  $\mathbf{r}_{F_n}$  is typically modelled as an integer representing the type of event, however other kinds of marks  $(e.g., \mathcal{M} = \mathbb{R}^2)$  can be eventually adopted. Specifying a marked-TPP involves the definition of the joint conditional density function of the next event, with inter-event time  $\tau_n$  and mark  $\mathbf{r}_{F_n}$ , given

<span id="page-2-4"></span>the history of past events:  $p^*(\mathbf{r}_{F_n}, \tau_n) = p(\mathbf{r}_{F_n}, \tau_n | \mathcal{H}_{t_n})$ . By assuming a conditional distribution parameterised by the weights of a neural model,  $p^*_{\theta}(\mathbf{r}_{F_n}, \tau_n)$ , inference can be performed by maximising the joint likelihood of the N observed events in a sequence:

$$\theta^* = \arg\max_{\theta} \prod_{n=0}^{N} p_{\theta}(\mathbf{r}_{F_n}, \tau_n | \mathcal{H}_t) = \prod_{n=0}^{N} p_{\theta}^*(\mathbf{r}_{F_n}, \tau_n).$$
(2)

Applications of Neural TPPs span a variety of fields of research [49], such as healthcare [26], finance [4], social network analysis [53], earthquake forecasting [10], and recommender systems [36]. Here we adopt the Neural TPP framework to model the dynamics of attention allocation on visual data.

#### 2.2. Scanpath Modelling

Modelling scanpaths involves defining a mapping from visual data, I (raw image data representing either a static picture or a stream of images), to a sequence of time-stamped gaze locations  $\mathcal{S}=\{(\mathbf{r}_{F_1},t_1),(\mathbf{r}_{F_2},t_2),\dots(\mathbf{r}_{F_N},t_N)\}$ . Here  $\mathbf{r}_{F_n}\in\mathbb{R}^2$  represents the two-dimensional vector of spatial coordinates of the n-th fixation on the stimulus I, while  $t_n\in\mathbb{R}^+$  represents its arrival time. Eventually, a perceptual representation of the input stimuli,  $\mathcal{Z}$ , is computed, with the aim of locating the relevant objects inside the scene:

<span id="page-2-0"></span>
$$\mathbf{I} \to \mathcal{Z} \to \{(\mathbf{r}_{F_1}, t_1), (\mathbf{r}_{F_2}, t_2), \dots (\mathbf{r}_{F_N}, t_N)\}.$$
 (3)

Here we assume that no specific external task or goal is given to the observer (*i.e.*, free-viewing condition). Notably, the dynamics of the attentive process, which unrolls as a sequence of fixations location with corresponding duration/arrival time, is characterised by an inherent randomness which likely stems from internal stochastic fluctuations affecting sensory and information processing, movement planning, and execution [54], in both fixations location and corresponding duration. Notably, many scanpath models proposed in the recent literature [2, 3, 39, 50] get rid of fixations' timestamp information by rearranging the sequence  $\{(\mathbf{r}_{F_1}, t_1), (\mathbf{r}_{F_2}, t_2), \dots\}$  as  $\{\mathbf{r}_F(1), \mathbf{r}_F(2), \dots\}$ , thus assuming  $(\mathbf{r}_{F_n}, t_n) = \mathbf{r}_F(n)$ .

A handful of solutions [6, 21, 52] have dealt with this problem in its entirety by starting from specific theoretical frameworks. In [52] Tatler *et al.* modelled saccades timings as an evidence accumulation process with clear neurobiological significance. In a similar vein, in [21] a Langevintype SDE race model [8] was adopted to predict fixations and their duration in socially relevant contexts, while in [6] fixation duration was equated to the patch residence time of a forager searching for nourishment. Conversely, the vast majority of recent methods [15, 16, 18, 45, 51] simply model fixation duration by employing specific neural architectural

<span id="page-2-1"></span>![](_page_2_Picture_8.jpeg)

Figure 2. Overview of TPP-Gaze model architecture. Given a semantic representation of the image  $(z_j)$  and the history of past events  $(h_n)$ , the next fixation position and duration are simulated.

choices that aim at associating each fixation to its corresponding duration.

In a different vein, this work recasts the whole visual attention allocation process in the mathematical framework of point process theory [20]. This emphasises the central role of visual attention's spatio-temporal dynamics by explicitly modelling scanpaths as sequences of discrete events happening at irregular intervals. Specifically, we conceive a scanpath as a realisation of a random process whose events happen at strictly increasing arrival times  $\mathcal{T} = \{t_1, \ldots, t_N\}$ . Fixations duration can be recovered by resorting to inter-event times  $\tau_n = t_n - t_{n-1}$ , while their locations can be represented as the two-dimensional continuous mark associated to the n-th event. Under this assumption, (Neural) Temporal Point Processes (TPPs) represent the natural choice for modelling this kind of data.

#### 3. Proposed Method

Given a stimulus (image)  $\mathbf{I}_j$ , an ensemble of  $N_{obs}$  observers performs a sequence of fixations and saccades (scanpath) on it, thus obtaining a set of sequences  $\mathcal{C}_j = \{S^1,\ldots,S^{N_{obs}}\}$ . Each scanpath  $S^i$  is a sequence of pairs (events)  $S^i_n = (\mathbf{r}_{F_n},t_n)$  each composed by a fixation position (marker)  $\mathbf{r}_{F_n} \in \mathbb{R}^2$ , and a corresponding arrival time  $t_n \in \mathbb{R}^+$ . At the most general level, we are interested in modelling the stochastic generative process that given a semantic representation of the image  $\mathcal{Z}_j$  and the history of past events  $\mathcal{H}_t$ , simulates the next fixation position and duration. More formally:

<span id="page-2-2"></span>
$$S_{n+1}^i \sim p_{\theta}(\mathbf{r}_{F_{n+1}}, t_{n+1} | \mathcal{H}_t, \mathcal{Z}_i), \tag{4}$$

where  $p_{\theta}(\cdot)$  represents the parametric joint conditional distribution of a Neural TPP [48].

#### <span id="page-2-3"></span>3.1. Architecture

In the following, we present the architecture of TPP-Gaze, implementing a scanpath model on an image as a Neural TPP.

<span id="page-3-1"></span>**Representing Scene Semantics.** As outlined in Eq. (3), the sequence of events composing a scanpath depends not only on the history of past events, but on a perceptual representation of the input stimulus  $I_i$ , encoding scene semantics and relevant objects location. We extract the perceptual representation of the input image via a CNN architecture inspired by [39]. Specifically, the input image is first processed by a pre-trained DenseNet201 CNN [29]. Activation maps from various convolutional layers (as reported in [39]) are extracted, thus obtaining a 2,048 channels volume, each representing the location of semantic features inside the scene. It is worth noticing that learning to predict fixations location (i.e., marks) involves a mapping between coordinates in Cartesian space, a task in which standard convolutions have been reported to fail [41]. In the vein of [43, 50], we adopt a CoordConv layer [41] to give convolutions access to their own input coordinates. This results in a 2,051 channels volume which is fed as input to 3 layers of  $1 \times 1$  convolutions, followed by a linear layer mapping to  $\mathbf{z}_j$  acting as our semantic representation.

Representing History. Neural TPPs employ either Recurrent Neural Networks (RNNs) and their variants (e.g., LSTM, GRU) [24,48,55] or Transformer encoders [64,65] to model the nonlinear dependency over both the markers and the timings from past events [49]. As shown in Fig. 2, the pair  $(\mathbf{r}_{F_n}, \tau_n)$  representing the event occurring at the time  $t_n$  with fixation position  $\mathbf{r}_{F_n}$  and duration  $\tau_n = t_n - t_{n-1}$ , is fed as the input into either a GRU or a Transformer encoder as described in [65]. The Transformer/GRU state embedding  $\mathbf{h}_n$  represents the influence of the history up to the n-th fixation. Hence, can be employed as a vector space representation of  $\mathcal{H}_{t_n}$ . Taking into account the semantic representation  $\mathbf{z}_j$  and the history embedding  $\mathbf{h}_n$ , Eq. (4) can be rewritten as:

$$S_{n+1}^i \sim p_{\theta}(\mathbf{r}_{F_{n+1}}, t_{n+1} | \mathbf{h}_n, \mathbf{z}_j). \tag{5}$$

**Fixation Duration Generation.** We model the conditional dependence of the distribution  $p_{\theta}(\tau_{n+1}|\mathbf{h}_n,\mathbf{z}_j)$  on both past events and stimulus by concatenating the history embedding and semantic vectors into a context vector  $\mathbf{c}_{j,n} = [\mathbf{h}_n||\mathbf{z}_j]$ . In the vein of [48], the latter is employed to learn the parameters of a Log-Gaussian Mixture Model (LGMM) via an affine transform:

$$\mathbf{w} = softmax(\mathbf{V_w}\mathbf{c}_{j,n}) \quad \mathbf{s} = \exp(\mathbf{V_s}\mathbf{c}_{j,n})$$
$$m = \mathbf{V_m}\mathbf{c}_{j,n}$$
(6)

where  $\boldsymbol{w} \in \mathbb{R}_+^K$  are the mixture weights,  $\boldsymbol{m} \in \mathbb{R}^K$  are the mixture means, and  $\boldsymbol{s} \in \mathbb{R}_+^K$  are the standard deviations. K represents the number of mixture components. The fixation duration for the n-th event can be generated by sampling

from the LGMM defined by:

$$p_{\theta}^*(\tau_n|\mathbf{c}_{j,n}) = p(\tau_n|\mathbf{w}, \mathbf{m}, \mathbf{s})$$

$$= \sum_{k=1}^K w_k \frac{1}{\tau_n s_k \sqrt{2\pi}} \exp\left(-\frac{(\log \tau_n - m_k)^2}{2s_k^2}\right).$$
(7)

**Fixation Position (Mark) Generation.** Similarly, given the context vector  $\mathbf{c}_{j,n}$ , we define the conditional probability of the next mark (fixation position),  $p_{\theta}(\mathbf{r}_{F_{n+1}}|\mathbf{h}_n,\mathbf{z}_j)$ , as a 2D Gaussian Mixture Model (GMM) whose parameters are obtained via another affine projection:

$$\omega_{g} = softmax(\mathbf{R}_{\omega}^{g} \mathbf{c}_{j,n}) \quad \Sigma_{g} = diag(\exp(\mathbf{R}_{\Sigma}^{g} \mathbf{c}_{j,n}))$$
$$\mu_{g} = \mathbf{R}_{\mu}^{g} \mathbf{c}_{j,n}$$
(8)

where  $\omega_g \in \mathbb{R}^2_+$  are the mixture weights,  $\mu_g \in \mathbb{R}^2$  are the mixture means, and  $\Sigma_g \in \mathbb{R}^{2 \times 2}$  are the diagonal covariance matrices of G bi-variate Gaussian distributions. The x and y coordinates of the n-th fixation can be generated by sampling from the GMM defined by:

$$p_{\theta}^{*}(\mathbf{r}_{F_{n}}|\mathbf{c}_{j,n}) = p(\mathbf{r}_{F_{n}}|\boldsymbol{\omega},\boldsymbol{\mu},\boldsymbol{\Sigma})$$

$$= \sum_{g=1}^{G} \boldsymbol{\omega}_{g} \frac{\exp\left(-\frac{1}{2}(\mathbf{r}_{F_{n}} - \boldsymbol{\mu}_{g})^{\mathrm{T}}\boldsymbol{\Sigma}^{-1}\left(\mathbf{r}_{F_{n}} - \boldsymbol{\mu}_{g}\right)\right)}{\sqrt{(2\pi)^{2}|\boldsymbol{\Sigma}_{g}|}}.$$
(9)

### 3.2. Model Inference

Consider a set of stimuli  $\mathcal{I} = \{\mathbf{I}_1, \dots, \mathbf{I}_j, \dots, \mathbf{I}_J\}$  each gazed by  $N_{obs}$  human observers. Each observer produces an ensemble of scanpaths  $\mathcal{C}_j = \{S^1, \dots, S^{N_{obs}}\}$  with  $S^i_n = (\mathbf{r}^i_{F_n}, \tau^i_n)$  representing an event (i.e., fixation position and duration). Model inference is performed by minimising a negative log-likelihood loss with respect to the parameters of the semantic network, the GRU/Transformer encoding history of events, and the affine transforms of the LGMM and GMM. Formally, the loss function is defined as follows:

$$\mathcal{L}(\boldsymbol{\theta}) = -\sum_{j} \sum_{i} \sum_{n} \left[ \log p_{\boldsymbol{\theta}}^*(\tau_n^i | \boldsymbol{c}_{j,n}) + \log p_{\boldsymbol{\theta}}^*(\mathbf{r}_{F_n}^i | \boldsymbol{c}_{j,n}) \right].$$
(10)

# 4. Experiments

### <span id="page-3-0"></span>4.1. Experimental Setup

**Datasets.** Regarding the stimuli and eye tracking data, we select five publicly available datasets of human recorded scanpaths comprising both fixation positions and durations: COCO-FreeView, MIT1003, OSIE, NUSEF, and FiFa.

COCO-FreeView [59] is a high-quality dataset capturing free viewing behaviour, featuring the same natural images

<span id="page-4-1"></span>adopted in COCO-Search18, annotated with 822, 602 eye fixations from a free-viewing task. Only train and validation splits are publicly released. Each image was presented for 5 seconds. The MIT1003 dataset [\[35\]](#page-9-21) comprises 1, 003 images primarily featuring natural scenes. It provides eye movement data from 15 subjects, observing stimuli for 3 seconds. The OSIE dataset [\[56\]](#page-9-22) comprises 700 images with eye-tracking data of 15 viewers. The dataset was explicitly devised to incorporate high-level semantic attributes. The NUSEF (NUS Eye Fixation) dataset [\[46\]](#page-9-23) features a diverse collection of images, representing a range of semantic concepts and capturing objects with varying scale, illumination, and orientation. Each free-view experiment lasted 5 seconds. The Fixations In Faces (FiFa) database [\[14\]](#page-8-18) shares data related to observers' viewing of faces in natural settings.Each image was presented for 2 seconds.

Implementation and Training Details. COCO-FreeView and MIT1003 datasets are used for model training. To this end, 70% of the images from both datasets are used for training, while the remaining 30% is equally partitioned between validation and test sets. We use AdamW as optimizer, with weight decay set to 10<sup>−</sup><sup>1</sup> , and the learning rate set to 10<sup>−</sup><sup>3</sup> . Batch size is equal to 128. We employ early stopping after 20 epochs with no improvement on the validation set. Following previous literature [\[18,](#page-8-3) [39\]](#page-9-15), during training and evaluation, we discard the first fixation and removed all scanpaths containing less than four fixations.

Scanpath Evaluation Metrics. A variety of scanpath evaluation metrics have been proposed to quantitatively assess the similarity between real and simulated eyemovements [\[1,](#page-8-19)[37\]](#page-9-1). Here we employ the MultiMatch, Scan-Match, and Sequence Score evaluation metrics since they explicitly consider fixation duration in the evaluation process. Moreover the String Edit Distance is adopted to further evaluate predicted scanpaths.

MultiMatch (MM) [\[22,](#page-8-20) [33\]](#page-9-24) assesses scanpaths based on five features: shape (Sh), length (Len), direction (Dir), position (Pos), and duration (Dur). Scanpaths are temporally aligned and compared using the Dijkstra algorithm. Similarity is determined by applying vector arithmetic to the aligned saccade pairs. ScanMatch (SM) [\[19\]](#page-8-21) encodes scanpaths as letter sequences by segmenting them into spatial and temporal bins. In our experiments, the longest dimension of the stimuli is divided into 14 bins, while the shortest is split into 8 bins. The temporal bin size is set to 50 ms for scanpath models delivering fixation duration estimates. The encoded scanpaths are then aligned and compared, with higher scores reflecting greater spatial, temporal, and sequential similarity. Sequence Score (SS) [\[57\]](#page-9-25) transforms the human and predicted scanpaths into sequences of fixation cluster IDs and compares them using a string-matching algorithm. String Edit Distance (SED) [\[9\]](#page-8-22), first partitions the input stimulus into an n × n grid. Scanpaths are then

<span id="page-4-0"></span>

|                                | Dim             |      |       | GMM MM (KL-Div) ↓ | SM (KL-Div) ↓ | SED ↓          |        |
|--------------------------------|-----------------|------|-------|-------------------|---------------|----------------|--------|
|                                | CNN Img TPP K G |      | Dur   | Avg               |               | w/ Dur w/o Dur | Avg    |
| Image Backbone                 |                 |      |       |                   |               |                |        |
| RN                             | 256 256         | 4 16 | 0.011 | 0.037             | 0.113         | 0.101          | 17.575 |
| DN                             | 256 256         | 4 16 | 0.012 | 0.028             | 0.078         | 0.060          | 17.032 |
| Image and TPP Dimensionalities |                 |      |       |                   |               |                |        |
| DN                             | 128 128         | 4 16 | 0.010 | 0.031             | 0.094         | 0.069          | 16.959 |
| DN                             | 128 256         | 4 16 | 0.012 | 0.030             | 0.084         | 0.063          | 16.887 |
| DN                             | 256 128         | 4 16 | 0.009 | 0.037             | 0.105         | 0.082          | 17.413 |
| DN                             | 256 256         | 4 16 | 0.012 | 0.028             | 0.078         | 0.060          | 17.032 |
| DN                             | 256 512         | 4 16 | 0.010 | 0.031             | 0.101         | 0.095          | 17.462 |
| DN                             | 512 256         | 4 16 | 0.008 | 0.032             | 0.110         | 0.093          | 17.497 |
| DN                             | 512 512         | 4 16 | 0.009 | 0.027             | 0.104         | 0.077          | 17.154 |
| Mixture Components             |                 |      |       |                   |               |                |        |
| DN                             | 256 256         | 2 16 | 0.014 | 0.027             | 0.092         | 0.071          | 16.944 |
| DN                             | 256 256         | 4 16 | 0.012 | 0.028             | 0.078         | 0.060          | 17.032 |
| DN                             | 256 256         | 2 32 | 0.009 | 0.030             | 0.109         | 0.098          | 17.216 |
| DN                             | 256 256         | 4 32 | 0.009 | 0.031             | 0.103         | 0.076          | 17.252 |

Table 1. Ablation study results comparing different model configurations and hyperparameters. We report the results for ResNet50 (RN) and DenseNet201 (DN) visual backbones, various embedding vector dimensions for the image representation and the TPP history, and different numbers of Gaussian mixture components.

transformed into strings and the string-edit algorithm calculates the distance between them.

Evaluation Protocol. We compare the scanpaths synthesised from various models with those recorded from human observers. The objective is to evaluate whether the simulated behaviours exhibited statistical properties closely resembling those exhibited by human subjects who are eyetracked while viewing a given stimulus. The evaluation protocol unfolds as follows. Suppose there are Nobs human observers. For each stimulus, we first compute the evaluation scores for every possible pair of the Nobs observers (Real vs. Real). Then, for each model, (i) we generate gaze trajectories from artificial observers and (ii) calculate the evaluation scores for every possible pair of real and artificial scanpaths (Real vs. Simulated).

For a given metric this procedure yields a target distribution P of similarity scores between observers (Real vs. Real) and a distribution Q of similarity scores for the given model w.r.t. humans (Real vs. Simulated). As reported in [\[38\]](#page-9-2), MM, SM, and SS average values may deliver inconsistent results: models exhibiting less variability w.r.t. humans, can score systematically better than the ground truth model. This issue can be tackled by considering a good model as the one that minimises the discrepancy between the target and model-derived score distributions. We quantified such discrepancy using the P Kullback-Leibler Divergence (KL-Div): DKL(P ∥ Q) = <sup>x</sup>∈X P(x) log(P(x)/Q(x)). Conversely, as SED is an evaluation metric not requiring alignment, it is not susceptible to the inconsistency issues associated with MM, SM, and SS. Consequently, its values are directly reported without any further processing.

<span id="page-5-3"></span><span id="page-5-0"></span>

|                    | COCO-FreeView                |          |         |                          |         |                          |         |              | MIT1003 |      |      |      |      |               |        |                          |        |         |      |
|--------------------|------------------------------|----------|---------|--------------------------|---------|--------------------------|---------|--------------|---------|------|------|------|------|---------------|--------|--------------------------|--------|---------|------|
|                    | MM (KL-Div) ↓                |          | SM (K   | SM (KL-Div) $\downarrow$ |         | SS (KL-Div) $\downarrow$ |         | MM (KL-Div)↓ |         |      |      |      |      | SM (KL-Div) ↓ |        | SS (KL-Div) $\downarrow$ |        | SED↓    |      |
|                    | Sh Len Dir                   | Pos Du   | ır Avg  | w/ Dur                   | w/o Dur | w/ Dur                   | w/o Dur | Avg          | Sh      | Len  | Dir  | Pos  | Dur  | Avg           | w/ Dur | w/o Dur                  | w/ Dur | w/o Dur | Avg  |
| Itti-Koch [32]     | 0.42 0.40 0.21               | 1.02 -   | 0.51    | -                        | 2.54    | -                        | 1.01    | 14.00        | 0.91    | 0.64 | 0.71 | 1.53 | -    | 0.95          | -      | 2.27                     | -      | 6.30    | 8.86 |
| CLE (Itti) [7, 32] | 0.07 0.30 0.35               | 1.43 -   | 0.54    | -                        | 2.50    | -                        | 1.27    | 14.37        | 0.10    | 0.10 | 0.32 | 1.04 | -    | 0.39          | -      | 2.16                     | -      | 6.25    | 9.09 |
| CLE (DG) [7,40]    | 0.06 0.18 0.23               | 1.31 -   | 0.44    | -                        | 2.37    | -                        | 1.22    | 14.31        | -       | -    | -    | -    | -    | -             | -      | -                        | -      | -       | -    |
| G-Eymol [61]       | 0.37 0.73 0.93               | 1.22 1.9 | 9 1.05  | 9.00                     | 6.67    | 8.75                     | 6.30    | 14.20        | 0.68    | 0.68 | 0.46 | 1.54 | 1.03 | 0.88          | 15.90  | 4.89                     | 3.32   | 6.96    | 6.96 |
| IOR-ROI-LSTM [18]  | 1.15 0.47 0.03               | 0.19 0.0 | 0.38    | 1.54                     | 0.76    | 0.56                     | 0.64    | 13.55        | 0.59    | 0.27 | 0.07 | 0.57 | 0.05 | 0.31          | 0.69   | 0.45                     | 5.08   | 8.61    | 8.61 |
| DeepGazeIII [39]   | 0.04 0.02 0.03               | 0.03 -   | 0.03    | -                        | 0.33    | -                        | 0.33    | 13.15        | -       | -    | -    | -    | -    | -             | -      | -                        | -      | -       | -    |
| Scanpath-VQA [15]  | 0.05 0.16 0.10               | 0.06 0.2 | 25 0.12 | 1.07                     | 0.34    | 0.43                     | 0.28    | 12.76        | 0.04    | 0.05 | 0.08 | 0.05 | 0.14 | 0.07          | 0.06   | 0.05                     | 0.05   | 0.11    | 7.26 |
| DeepGazeIII [39]   | <b>0.01</b> 0.03 0.05        | 0.05 -   | 0.04    | -                        | 0.34    | -                        | 0.36    | 13.15        | 0.05    | 0.01 | 0.20 | 0.05 | -    | 0.08          | -      | 0.19                     | -      | 5.06    | 8.28 |
| Scanpath-VQA [15]  | 0.62 0.41 <b>0.02</b>        | 0.05 0.0 | 3 0.23  | 0.08                     | 0.03    | 0.03                     | 0.31    | 14.34        | 0.20    | 0.14 | 0.15 | 0.08 | 0.02 | 0.12          | 0.23   | 0.19                     | 0.14   | 0.25    | 9.27 |
| TPP-Gaze (GRU)     | 0.06 0.02 <b>0.02</b>        | 0.03 0.0 | 0.03    | 0.08                     | 0.06    | 0.05                     | 0.11    | 17.03        | 0.01    | 0.03 | 0.09 | 0.04 | 0.01 | 0.04          | 0.15   | 0.11                     | 0.12   | 0.11    | 7.21 |
| TPP-Gaze (Trans.)  | 0.05 <u>0.01</u> <u>0.02</u> | 0.03 0.0 | 0.03    | 0.10                     | 0.07    | 0.06                     | 0.12    | 16.93        | 0.01    | 0.02 | 0.09 | 0.07 | 0.02 | 0.04          | 0.22   | 0.16                     | 0.14   | 0.14    | 7.33 |

Table 2. Comparison of various models on COCO-FreeView and MIT1003. **Gray color** indicates models trained under the same settings and datasets. Within this group, **bold** values represent the best performance for each metric. <u>Underline</u> values indicate the overall best performance across all models and metrics.

<span id="page-5-1"></span>

|                    | OSIE                                             |                         | NUSEF                                                            |                  | FiFa                                               |                         |  |  |  |  |
|--------------------|--------------------------------------------------|-------------------------|------------------------------------------------------------------|------------------|----------------------------------------------------|-------------------------|--|--|--|--|
|                    | MM (KL-Div) ↓                                    | SM (KL-Div)↓            | MM (KL-Div) ↓                                                    | SM (KL-Div) ↓    | MM (KL-Div) ↓                                      | SM (KL-Div)↓            |  |  |  |  |
|                    | Sh Len Dir Pos Dur Avg                           | w/ Dur w/o Dur          | Sh Len Dir Pos Dur Avg                                           | w/ Dur w/o Dur   | Sh Len Dir Pos Dur Avg                             | w/ Dur w/o Dur          |  |  |  |  |
| Itti-Koch [32]     | 1.62 0.89 0.45 3.69 - 1.66                       | - 2.22                  | 0.63 0.44 0.17 0.56 - 0.45                                       | - 0.61           | 1.51 0.51 1.08 3.46 - 1.64                         | - 6.08                  |  |  |  |  |
| CLE (Itti) [7, 32] | 0.13 <u>0.03</u> 0.20 0.75 - 0.28                | - 1.98                  | 0.26 0.03 0.09 0.42 - 0.20                                       | - 0.79           | 0.38 0.10 0.29 1.14 - 0.48                         | - 3.97                  |  |  |  |  |
| CLE (DG) [7,40]    | 0.17 0.03 0.15 0.60 - 0.24                       | - 1.43                  | 0.28 0.06 0.06 0.18 - 0.15                                       | - 0.50           | 0.40 0.14 0.36 0.97 - 0.46                         | - 3.10                  |  |  |  |  |
| G-Eymol [61]       | 1.18 1.08 0.25 2.12 1.18 1.16                    | 16.17 7.29              | 0.38 0.30 0.05 0.29 3.02 0.81                                    | 1.76 0.55        | 0.34 0.57 0.59 2.48 2.40 1.28                      | 17.36 11.71             |  |  |  |  |
| IOR-ROI-LSTM [18]  | 1.72 0.73 <u>0.03</u> 0.96 0.03 0.69             | 0.75 0.76               | 0.90 0.36 0.12 0.23 0.17 0.36                                    | 0.11 0.13        | 1.24 0.51 <u>0.10</u> 1.71 <u>0.05</u> 0.72        | 1.25 1.56               |  |  |  |  |
| DeepGazeIII [39]   | 0.14 0.08 0.06 0.15 - 0.11                       | - 0.12                  | 0.10 0.06 0.08 0.05 - 0.07                                       | - 0.07           | 0.28 0.12 0.21 0.34 - 0.24                         | - 0.60                  |  |  |  |  |
| Scanpath-VQA [15]  | $0.07\ 0.07\ 0.04\ \underline{0.04}\ 0.16\ 0.08$ | <u>0.03</u> <u>0.03</u> | 0.11 0.04 0.02 0.05 <u>0.08</u> 0.06                             | <u>0.02</u> 0.03 | 0.14 <u>0.04</u> 0.13 <u>0.07</u> 0.12 <u>0.10</u> | <u>0.03</u> <u>0.13</u> |  |  |  |  |
| DeepGazeIII [39]   | 0.04 0.03 0.09 0.14 - 0.08                       | - 0.22                  | 0.11 0.07 0.09 0.04 - 0.08                                       | - 0.06           | 0.25 0.13 0.40 <b>0.18</b> - 0.24                  | - 0.69                  |  |  |  |  |
| Scanpath-VQA [15]  | 0.49 0.35 0.09 0.20 <b>0.02</b> 0.23             | 0.40 0.28               | 0.11 0.07 0.06 0.03 0.16 0.09                                    | 0.06 0.06        | 0.44 0.26 0.33 0.31 0.08 0.28                      | 0.47 0.79               |  |  |  |  |
| TPP-Gaze (GRU)     | 0.03 0.04 <b>0.05 0.12</b> 0.03 <b>0.05</b>      | <b>0.20</b> 0.30        | <u>0.03</u> 0.02 <u>0.01</u> 0.02 <b>0.10</b> <u>0.04</u>        | 0.04 0.04        | <u>0.05</u> 0.05 0.12 0.25 <u>0.05</u> <u>0.10</u> | 0.23 0.47               |  |  |  |  |
| TPP-Gaze (Trans.)  | <b>0.02</b> 0.04 0.06 0.14 0.05 0.06             | 0.25 0.44               | <u>0.03</u> <u>0.01</u> <u>0.02</u> <u>0.01</u> 0.13 <u>0.04</u> | 0.04 <u>0.01</u> | 0.06 <b>0.05 0.12</b> 0.30 <b>0.05</b> 0.12        | 0.32 0.52               |  |  |  |  |

Table 3. Comparison of various models on OSIE, NUSEF, and FiFa datasets. **Gray color** indicates models trained under the same settings and datasets. Within this group, **bold** values represent the best performance for each metric. <u>Underline</u> values indicate the overall best performance across all models and metrics.

#### 4.2. Scanpath Prediction

Ablation Studies. The TPP-Gaze architecture consists of three main blocks: image encoding (CNN backbone), history encoding (RNN/Transformer), and fixation/inter-time prediction (GMM/LGMM). To break down these components and make the adopted design choices explicit, we perform extensive ablation studies. Specifically, we evaluate two different CNN backbones for image encoding (i.e., a ResNet50 [28] and a DenseNet201 [29]) as well as three embedding vector dimensions for the image semantic representation  $(\mathbf{z}_i)$  and the history embedding  $(\mathbf{h}_n, \text{TPP di-}$ mensionality). Moreover, different numbers of components for the GMM/LGMM are considered. Table 1 reports the results of the ablation studies conducted on the COCO-FreeView dataset. In our experiments, we select the hyperparameters yielding the best trade-off according to the considered evaluation metrics, resulting in a DesNet201 backbone and a dimensionality equal to 256 for all embedding vectors. Moreover, the parameters K and G representing mixture components are respectively set to 4 and 16.

Comparison with the State of the Art. To compare the proposed approach with others, we include state-of-the-art

approaches that either reach high performance in recent scanpath benchmarks [37], offer source code availability, and are representative of different approaches and architectures. As to the latter criteria, following the taxonomy proposed in [37], scanpath models can be aggregated into the following categories: biologically inspired (*e.g.* Itti-Koch model [32] and G-Eymol [61]); statistically inspired (*e.g.* CLE model [7]); cognitively inspired (*e.g.* IOR-ROI-LSTM [18]); engineered models (*e.g.* DeepGazeIII [39] and Scanpath-VQA [15]); but see [37–39] for an in-depth review. Under such circumstances, we assess the performance of TPP-Gaze against the aforementioned models.

Table 2 reports quantitative results on the COCO-FreeView and MIT1003 datasets in terms of all considered metrics, while model performance on OSIE, NUSEF, and FiFa are shown in Table 3 in terms of MM and SM<sup>1</sup>. In all experiments, we compare the aforementioned approaches using the pre-trained model weights released by the authors. As DeepGaze models were trained on the entire MIT1003 dataset, the results from DeepGazeIII and CLE

<span id="page-5-2"></span><sup>&</sup>lt;sup>1</sup>We refer to the supplementary material for the results in terms of SS and SED on OSIE, NUSEF, and FiFa datasets.

<span id="page-6-4"></span><span id="page-6-0"></span>![](_page_6_Figure_0.jpeg)

Figure 3. Comparison of simulated and human scanpaths. Each circle represents a fixation point, with its diameter proportional to the fixation duration. For methods that do not model fixation duration, circles are shown with a uniform size.

<span id="page-6-1"></span>![](_page_6_Figure_2.jpeg)

Figure 4. Statistical properties exhibited by TPP-Gaze and other methods relative to those of human observers, in terms of empirical fixation durations and saccade amplitudes on the COCO-FreeView (top row) and OSIE (bottom row) datasets.

(DG) have not been included in this comparison. Additionally, to explicitly measure the effect of the proposed architecture and mathematical framework, we retrain and test the two most recent models (DeepGazeIII and Scanpath-VQA) under the same conditions adopted for TPP-Gaze (see Sec. 4.1). Specifically, beyond training on the same data, the large-scale pre-training of DeepGazeIII as well as the fine-tuning stage of Scanpath-VQA based on reinforcement learning [47] have been inhibited. These results are reported in gray color at the bottom of the tables.

As can be observed, when trained under the same settings and datasets, TPP-Gaze (with either GRU or Transformer-based history encoding) outperforms all the other approaches on most of the adopted metrics. Interestingly, in many cases the proposed approach offers the best overall performance, even when considering the pretrained models released by the authors, except for Scan-Match where Scanpath-VQA, which is directly optimized via reinforcement learning on this metric, understandably

<span id="page-6-2"></span>![](_page_6_Figure_6.jpeg)

Figure 5. Return fixations analysis comparing TPP-Gaze with other methods and human observers. Results are shown on COCO-FreeView (left plot) and OSIE (right plot) datasets.

proves to be the best. Some qualitative results are shown in Fig. 3, where we report sampled scanpaths from five models alongside those from humans. Notably, TPP-Gaze can predict fixations that better align with those recorded from human subjects, confirming the advantages of the proposed approach for predicting scanpaths during free-viewing.

Additional analyses are reported in Fig. 4 that shows empirical distributions summarizing TPP-Gaze's scanpath statistics compared to those yielded by human observers and other methods. Beyond common scanpath statistics, we further evaluate the proposed approach using a return fixations (RF) analysis [62]. RF analysis describes the tendency of observers (either human or simulated) to revisit previously foveated locations. The frequency of RFs and the temporal offset (i.e., the number of intervening fixations before returning to a location) at which they occur, provide a more nuanced description of the cognitive processes underlying attention allocation [62]. Fig. 5 reports the results of this analysis in comparison with existing methods across two datasets. Notably, although TPP-Gaze was not explicitly trained for this objective, it produces the most accurate RF patterns with respect to human behavior when compared to state-of-the-art approaches<sup>2</sup>.

### 4.3. Applications

**Saliency Prediction.** The performance of TPP-Gaze are further evaluated by comparing the saliency maps "back-

<span id="page-6-3"></span> $<sup>^2\</sup>mbox{Results}$  of the RF analysis on OSIE, NUSEF and FiFa are reported in the supplementary material.

<span id="page-7-4"></span><span id="page-7-0"></span>

|                        | COCO-FreeView |       |       | MIT1003 |         |       | OSIE    |        |       | NUSEF   |        |       | FiFa    |             |       |
|------------------------|---------------|-------|-------|---------|---------|-------|---------|--------|-------|---------|--------|-------|---------|-------------|-------|
|                        | KL-Div↓       | AUC ↑ | NSS ↑ | KL-Div↓ | . AUC ↑ | NSS ↑ | KL-Div↓ | . AUC↑ | NSS ↑ | KL-Div↓ | . AUC↑ | NSS ↑ | KL-Div↓ | AUC ↑       | NSS ↑ |
| Saliency-based         |               |       |       |         |         |       |         |        |       |         |        |       |         |             |       |
| CLE (DG) [7,40]        | 8.65          | 0.55  | 0.09  | -       | -       | -     | 5.08    | 0.59   | 0.28  | 4.99    | 0.63   | 0.38  | 6.39    | 0.59        | 0.25  |
| DeepGazeIII [39]       | 0.85          | 0.84  | 1.75  | -       | -       | -     | 0.32    | 0.87   | 2.01  | 0.49    | 0.85   | 1.89  | 0.62    | 0.88        | 2.52  |
| Saliency-free          |               |       |       |         |         |       |         |        |       |         |        |       |         |             |       |
| Itti-Koch [32]         | 8.94          | 0.56  | 0.24  | 5.01    | 0.64    | 0.47  | 3.35    | 0.65   | 0.51  | 4.84    | 0.63   | 0.40  | 5.47    | 0.64        | 0.42  |
| CLE (Itti) [7,32]      | 7.45          | 0.54  | 0.07  | 4.15    | 0.61    | 0.23  | 3.45    | 0.61   | 0.23  | 3.36    | 0.63   | 0.31  | 4.84    | 0.60        | 0.23  |
| G-Eymol [61]           | 10.98         | 0.56  | 0.26  | 7.64    | 0.62    | 0.35  | 4.58    | 0.67   | 0.60  | 5.09    | 0.66   | 0.55  | 9.04    | 0.62        | 0.47  |
| IOR-ROI-LSTM [18]      | 1.30          | 0.77  | 0.99  | 0.78    | 0.81    | 1.40  | 0.50    | 0.83   | 1.46  | 0.74    | 0.80   | 1.32  | 0.83    | 0.85        | 1.72  |
| Scanpath-VQA [15]      | 3.53          | 0.77  | 1.56  | 2.12    | 0.82    | 2.01  | 1.26    | 0.84   | 2.12  | 2.45    | 0.80   | 1.76  | 1.88    | 0.86        | 2.89  |
| TPP-Gaze (GRU)         | 1.01          | 0.84  | 1.65  | 0.78    | 0.86    | 2.06  | 0.67    | 0.84   | 1.72  | 0.84    | 0.84   | 1.71  | 1.07    | 0.86        | 2.06  |
| TPP-Gaze (Transformer) | <u>1.11</u>   | 0.83  | 1.54  | 0.83    | 0.85    | 1.93  | 0.68    | 0.84   | 1.68  | 0.79    | 0.84   | 1.70  | 1.11    | <u>0.85</u> | 1.91  |

Table 4. Saliency prediction results on COCO-FreeView, MIT1003, OSIE, NUSEF, and FiFa datasets. Models are grouped into saliency-based and saliency-free methods, where the former (*i.e.*, CLE (DG) and DeepGazeIII) incorporate components trained to predict saliency maps. **Bold** values represent the best performance within each metric, while underline values indicate the second-best results.

<span id="page-7-2"></span>![](_page_7_Figure_2.jpeg)

Figure 6. Empirical distributions of the adopted metrics quantifying inter-humans (top) and human vs. TPP-Gaze (bottom) scan-path similarity for the visual search task on COCO-Search18.

ward" generated from fixations with those of human observers across all evaluated scanpath models. The results, presented in Table 4, are measured using three commonly adopted saliency metrics [11, 12]: Kullback-Leibler Divergence (KL-Div), Judd's Area Under the Curve (AUC), and Normalised Scanpath Saliency (NSS). DeepGazeIII and CLE (DG) are reported here only as references for the performance of a saliency prediction model, given their adoption of an extensive pre-training phase designed expressly for saliency generation (DeepGazeIII), or explicit adoption of a saliency model (CLE (DG)). Overall, TPP-Gaze obtains the best or second-best performance across all metrics and datasets. It yields results that are comparable to or surpass those of IOR-ROI-LSTM [18] and Scanpath-VQA [15], which are significantly better than all other approaches. This further demonstrates the effectiveness of our approach in predicting fixation points that better resemble human scanpaths than those predicted by existing methods.

**Extending the Model to Visual Search Tasks.** Recently, several works [15,23,45,58,63] have focused on predicting

<span id="page-7-3"></span>![](_page_7_Picture_6.jpeg)

Figure 7. Human (left) and simulated (right) scanpaths for the visual search task. Search objective is "Sink".

attention allocation on specific targets (visual search tasks). Although TPP-Gaze was originally devised and evaluated for the free-viewing scenario, it can be extended to tackle the visual search problem in various ways. Here, we propose a proof-of-concept model featuring a simple architectural variation that enables goal-directed attention prediction with TPP-Gaze. In a nutshell, we use RoBERTa [42] to perform a linguistic embedding of the search target and learn a target-oriented image semantic representation<sup>3</sup>. (*cf.* Sec. 3.1). Preliminary results show encouraging trends on the COCO-Search18 dataset [17,57], as illustrated in Fig. 6, where visual search patterns produced by TPP-Gaze are compared to human patterns using the MM and SM metrics. A qualitative example is depicted in Fig. 7.

#### 5. Conclusion

We presented TPP-Gaze, a novel approach that explicitly models via Neural Temporal Point Processes the temporal evolution of visual attention as instantiated through a scanpath. Cogently, TPP-Gaze allows for a principled modelling of both fixations position and corresponding duration. Extensive experiments on five publicly available datasets have proven the effectiveness of the proposed approach in modelling gaze spatio-temporal dynamics, as witnessed by the overall best performances in scanpath similarity and fixation duration prediction. Also, it exhibits competitive results in terms of saliency prediction.

<span id="page-7-1"></span><sup>&</sup>lt;sup>3</sup>More details and simulations are shown in the supplementary material

# Acknowledgments

This work was supported by a grant from Universita degli Studi di Milano (Bando Linea 3 My First ` SEED – DM 737/2021 MUR) and by the PNRR project "Italian Strengthening of Esfri RI Resilience (ITSERR)" funded by the European Union - NextGenerationEU (CUP B53C22001770006).

# References

- <span id="page-8-19"></span>[1] Nicola C Anderson, Fraser Anderson, Alan Kingstone, and Walter F Bischof. A comparison of scanpath comparison methods. *Behavior Research Methods*, 47(4):1377–1392, 2015. [5](#page-4-1)
- <span id="page-8-13"></span>[2] Marc Assens, Xavier Giro i Nieto, Kevin McGuinness, and Noel E. O'Connor. Pathgan: Visual scanpath prediction with generative adversarial networks. In *ECCV Workshops*, 2018. [3](#page-2-4)
- <span id="page-8-14"></span>[3] Marc Assens Reina, Xavier Giro-i Nieto, Kevin McGuinness, and Noel E O'Connor. SaltiNet: Scan-Path Prediction on 360 Degree Images Using Saliency Volumes. In *ICCV Workshops*, 2017. [3](#page-2-4)
- <span id="page-8-11"></span>[4] Emmanuel Bacry, Iacopo Mastromatteo, and Jean-Franc¸ois Muzy. Hawkes processes in finance. *Market Microstructure and Liquidity*, 1(01):1550005, 2015. [3](#page-2-4)
- <span id="page-8-6"></span>[5] Simon Barthelme, Hans Trukenbrod, Ralf Engbert, and Felix ´ Wichmann. Modeling fixation locations using spatial point processes. *Journal of Vision*, 13(12):1–1, 2013. [2](#page-1-1)
- <span id="page-8-4"></span>[6] Giuseppe Boccignone, Vittorio Cuculo, Alessandro D'Amelio, Giuliano Grossi, and Raffaella Lanzarotti. On gaze deployment to audio-visual cues of social interactions. *IEEE Access*, 8:161630–161654, 2020. [2,](#page-1-1) [3](#page-2-4)
- <span id="page-8-23"></span>[7] G. Boccignone and M. Ferraro. Modelling gaze shift as a constrained random walk. *Physica A: Statistical Mechanics and its Applications*, 331(1-2):207–218, 2004. [6,](#page-5-3) [8,](#page-7-4) [11](#page-10-2)
- <span id="page-8-15"></span>[8] Rafal Bogacz, Eric Brown, Jeff Moehlis, Philip Holmes, and Jonathan D Cohen. The physics of optimal decision making: a formal analysis of models of performance in two-alternative forced-choice tasks. *Psychological Review*, 113(4):700, 2006. [3](#page-2-4)
- <span id="page-8-22"></span>[9] Stephan A Brandt and Lawrence W Stark. Spontaneous eye movements during visual imagery reflect the content of the visual scene. *Journal of Cognitive Neuroscience*, 9(1):27– 38, 1997. [5](#page-4-1)
- <span id="page-8-12"></span>[10] Andrew Bray and Frederic Paik Schoenberg. Assessment of point process models for earthquake forecasting. *Statistical Science*, 28(4):510–520, 2013. [3](#page-2-4)
- <span id="page-8-25"></span>[11] Zoya Bylinskii, Tilke Judd, Ali Borji, Laurent Itti, Fredo ´ Durand, Aude Oliva, and Antonio Torralba. MIT Saliency Benchmark. http://saliency.mit.edu/. [8](#page-7-4)
- <span id="page-8-26"></span>[12] Zoya Bylinskii, Tilke Judd, Aude Oliva, Antonio Torralba, and Fredo Durand. What Do Different Evaluation Met- ´ rics Tell Us About Saliency Models? *IEEE Trans. PAMI*, 41(3):740–757, 2019. [8](#page-7-4)
- <span id="page-8-0"></span>[13] Giuseppe Cartella, Marcella Cornia, Vittorio Cuculo, Alessandro D'Amelio, Dario Zanca, Giuseppe Boccignone,

- and Rita Cucchiara. Trends, Applications, and Challenges in Human Attention Modelling. In *IJCAI*, 2024. [1](#page-0-1)
- <span id="page-8-18"></span>[14] M. Cerf, J. Harel, W. Einhauser, and C. Koch. Predicting ¨ human gaze using low-level saliency combined with face detection. In *NeurIPS*, 2008. [5](#page-4-1)
- <span id="page-8-1"></span>[15] Xianyu Chen, Ming Jiang, and Qi Zhao. Predicting Human Scanpaths in Visual Question Answering. In *CVPR*, 2021. [2,](#page-1-1) [3,](#page-2-4) [6,](#page-5-3) [7,](#page-6-4) [8,](#page-7-4) [11,](#page-10-2) [13,](#page-12-0) [14,](#page-13-0) [15,](#page-14-0) [16,](#page-15-0) [17,](#page-16-0) [18,](#page-17-0) [19,](#page-18-0) [20,](#page-19-0) [21,](#page-20-0) [22](#page-21-0)
- <span id="page-8-2"></span>[16] Xianyu Chen, Ming Jiang, and Qi Zhao. Beyond average: Individualized visual scanpath prediction. In *CVPR*, 2024. [2,](#page-1-1) [3](#page-2-4)
- <span id="page-8-28"></span>[17] Yupei Chen, Zhibo Yang, Seoyoung Ahn, Dimitris Samaras, Minh Hoai, and Gregory Zelinsky. Coco-search18 fixation dataset for predicting goal-directed attention control. *Scientific reports*, 11(1):1–11, 2021. [8](#page-7-4)
- <span id="page-8-3"></span>[18] Zhenzhong Chen and Wanjie Sun. Scanpath Prediction for Visual Attention using IOR-ROI LSTM. In *IJCAI*, 2018. [2,](#page-1-1) [3,](#page-2-4) [5,](#page-4-1) [6,](#page-5-3) [7,](#page-6-4) [8,](#page-7-4) [11,](#page-10-2) [13,](#page-12-0) [14,](#page-13-0) [15,](#page-14-0) [16,](#page-15-0) [17,](#page-16-0) [18,](#page-17-0) [19,](#page-18-0) [20,](#page-19-0) [21,](#page-20-0) [22](#page-21-0)
- <span id="page-8-21"></span>[19] Filipe Cristino, Sebastiaan Mathot, Jan Theeuwes, and ˆ Iain D Gilchrist. Scanmatch: A novel method for comparing fixation sequences. *Behavior Research Methods*, 42(3):692– 700, 2010. [5](#page-4-1)
- <span id="page-8-16"></span>[20] Daryl J Daley and David Vere-Jones. *An introduction to the theory of point processes: volume II: general theory and structure*. Springer Science & Business Media, 2007. [3](#page-2-4)
- <span id="page-8-5"></span>[21] Alessandro D'Amelio and Giuseppe Boccignone. Gazing at social interactions between foraging and decision theory. *Frontiers in Neurorobotics*, 15:639999, 2021. [2,](#page-1-1) [3](#page-2-4)
- <span id="page-8-20"></span>[22] Richard Dewhurst, Marcus Nystrom, Halszka Jarodzka, Tom ¨ Foulsham, Roger Johansson, and Kenneth Holmqvist. It depends on how you look at it: Scanpath comparison in multiple dimensions with multimatch, a vector-based approach. *Behavior Research Methods*, 44(4):1079–1100, 2012. [5](#page-4-1)
- <span id="page-8-27"></span>[23] Zhiwei Ding, Xuezhe Ren, Erwan David, Melissa Vo, Gabriel Kreiman, and Mengmi Zhang. Efficient Zero-shot Visual Search via Target and Context-aware Transformer. *arXiv preprint arXiv:2211.13470*, 2022. [8](#page-7-4)
- <span id="page-8-9"></span>[24] Nan Du, Hanjun Dai, Rakshit Trivedi, Utkarsh Upadhyay, Manuel Gomez-Rodriguez, and Le Song. Recurrent marked temporal point processes: Embedding event history to vector. In *ACM SIGKDD*, 2016. [2,](#page-1-1) [4](#page-3-1)
- <span id="page-8-7"></span>[25] Ralf Engbert, Hans A Trukenbrod, Simon Barthelme, and ´ Felix A Wichmann. Spatial Statistics and Attentional Dynamics in Scene Viewing. *Journal of Vision*, 15(1):14–14, 2015. [2](#page-1-1)
- <span id="page-8-10"></span>[26] Joseph Enguehard, Dan Busbridge, Adam Bozson, Claire Woodcock, and Nils Hammerla. Neural temporal point processes for modelling electronic health records. In *Machine Learning for Health*, 2020. [3](#page-2-4)
- <span id="page-8-8"></span>[27] Alan G Hawkes. Spectra of some self-exciting and mutually exciting point processes. *Biometrika*, 58(1):83–90, 1971. [2](#page-1-1)
- <span id="page-8-24"></span>[28] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In *CVPR*, 2016. [6](#page-5-3)
- <span id="page-8-17"></span>[29] Gao Huang, Zhuang Liu, Laurens Van Der Maaten, and Kilian Q Weinberger. Densely connected convolutional networks. In *CVPR*, 2017. [4,](#page-3-1) [6](#page-5-3)

- <span id="page-9-10"></span>[30] Hengguan Huang, Hao Wang, and Brian Mak. Recurrent poisson process unit for speech recognition. In *AAAI*, 2019. [2](#page-1-1)
- <span id="page-9-9"></span>[31] Valerie Isham and Mark Westcott. A self-correcting point process. *Stochastic Processes and Their Applications*, 8(3):335–347, 1979. [2](#page-1-1)
- <span id="page-9-0"></span>[32] L. Itti, C. Koch, and E. Niebur. A model of saliency-based visual attention for rapid scene analysis. *IEEE Trans. PAMI*, 20:1254–1259, 1998. [1,](#page-0-1) [6,](#page-5-3) [8,](#page-7-4) [11](#page-10-2)
- <span id="page-9-24"></span>[33] Halszka Jarodzka, Kenneth Holmqvist, and Marcus Nystrom. A Vector-based, Multidimensional Scanpath Sim- ¨ ilarity Measure. In *ETRA*, 2010. [5](#page-4-1)
- <span id="page-9-33"></span>[34] Ming Jiang, Shengsheng Huang, Juanyong Duan, and Qi Zhao. Salicon: Saliency in context. In *CVPR*, 2015. [12](#page-11-0)
- <span id="page-9-21"></span>[35] Tilke Judd, Krista Ehinger, Fredo Durand, and Antonio Tor- ´ ralba. Learning to predict where humans look. In *ICCV*, 2009. [5,](#page-4-1) [12](#page-11-0)
- <span id="page-9-13"></span>[36] Srijan Kumar, Xikun Zhang, and Jure Leskovec. Predicting dynamic embedding trajectory in temporal interaction networks. In *ACM SIGKDD*, 2019. [3](#page-2-4)
- <span id="page-9-1"></span>[37] Matthias Kummerer and Matthias Bethge. State-of- ¨ the-art in human scanpath prediction. *arXiv preprint arXiv:2102.12239*, 2021. [1,](#page-0-1) [5,](#page-4-1) [6](#page-5-3)
- <span id="page-9-2"></span>[38] Matthias Kummerer and Matthias Bethge. Predicting visual ¨ fixations. *Annual Review of Vision Science*, 9, 2023. [1,](#page-0-1) [5,](#page-4-1) [6](#page-5-3)
- <span id="page-9-15"></span>[39] Matthias Kummerer, Matthias Bethge, and Thomas SA Wal- ¨ lis. DeepGaze III: Modeling free-viewing human scanpaths with deep learning. *Journal of Vision*, 22(5):7–7, 2022. [3,](#page-2-4) [4,](#page-3-1) [5,](#page-4-1) [6,](#page-5-3) [7,](#page-6-4) [8,](#page-7-4) [11,](#page-10-2) [13,](#page-12-0) [15,](#page-14-0) [16,](#page-15-0) [17,](#page-16-0) [18,](#page-17-0) [20,](#page-19-0) [21,](#page-20-0) [22](#page-21-0)
- <span id="page-9-26"></span>[40] Matthias Kummerer, Lucas Theis, and Matthias Bethge. ¨ Deep Gaze I: Boosting saliency prediction with feature maps trained on ImageNet. *arXiv preprint arXiv:1411.1045*, 2014. [6,](#page-5-3) [8,](#page-7-4) [11](#page-10-2)
- <span id="page-9-17"></span>[41] Rosanne Liu, Joel Lehman, Piero Molino, Felipe Petroski Such, Eric Frank, Alex Sergeev, and Jason Yosinski. An intriguing failing of convolutional neural networks and the coordconv solution. In *NeurIPS*, 2018. [4](#page-3-1)
- <span id="page-9-32"></span>[42] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. RoBERTa: A Robustly Optimized BERT Pretraining Approach. *arXiv preprint arXiv:1907.11692*, 2019. [8](#page-7-4)
- <span id="page-9-18"></span>[43] Daniel Martin, Ana Serrano, Alexander W Bergman, Gordon Wetzstein, and Belen Masia. ScanGAN360: A Generative Model of Realistic Scanpaths for 360° Images. *IEEE Trans. VCG*, 28(5):2003–2013, 2022. [4](#page-3-1)
- <span id="page-9-6"></span>[44] Hongyuan Mei and Jason M Eisner. The neural hawkes process: A neurally self-modulating multivariate point process. In *NeurIPS*, 2017. [2](#page-1-1)
- <span id="page-9-4"></span>[45] Sounak Mondal, Zhibo Yang, Seoyoung Ahn, Dimitris Samaras, Gregory Zelinsky, and Minh Hoai. GazeFormer: Scalable, Effective and Fast Prediction of Goal-Directed Human Attention. In *CVPR*, 2023. [2,](#page-1-1) [3,](#page-2-4) [8](#page-7-4)
- <span id="page-9-23"></span>[46] Subramanian Ramanathan, Harish Katti, Nicu Sebe, Mohan Kankanhalli, and Tat-Seng Chua. An eye fixation database for saliency detection in images. In *ECCV*, 2010. [5](#page-4-1)

- <span id="page-9-28"></span>[47] Steven J Rennie, Etienne Marcheret, Youssef Mroueh, Jerret Ross, and Vaibhava Goel. Self-Critical Sequence Training for Image Captioning. In *CVPR*, 2017. [7](#page-6-4)
- <span id="page-9-7"></span>[48] Oleksandr Shchur, Marin Bilos, and Stephan G ˇ unnemann. ¨ Intensity-free learning of temporal point processes. *arXiv preprint arXiv:1909.12127*, 2019. [2,](#page-1-1) [3,](#page-2-4) [4](#page-3-1)
- <span id="page-9-11"></span>[49] Oleksandr Shchur, Ali Caner Turkmen, Tim Januschowski, ¨ and Stephan Gunnemann. Neural temporal point processes: ¨ A review. *arXiv preprint arXiv:2104.03528*, 2021. [3,](#page-2-4) [4](#page-3-1)
- <span id="page-9-16"></span>[50] Xiangjie Sui, Yuming Fang, Hanwei Zhu, Shiqi Wang, and Zhou Wang. ScanDMM: A Deep Markov Model of Scanpath Prediction for 360◦ Images. In *CVPR*, 2023. [3,](#page-2-4) [4](#page-3-1)
- <span id="page-9-5"></span>[51] Wanjie Sun, Zhenzhong Chen, and Feng Wu. Visual Scanpath Prediction Using IOR-ROI Recurrent Mixture Density Network. *IEEE Trans. PAMI*, 43(6):2101–2118, 2019. [2,](#page-1-1) [3](#page-2-4)
- <span id="page-9-3"></span>[52] Benjamin W Tatler, James R Brockmole, and Roger HS Carpenter. Latest: A model of saccadic decisions in space and time. *Psychological Review*, 124(3):267, 2017. [1,](#page-0-1) [2,](#page-1-1) [3](#page-2-4)
- <span id="page-9-12"></span>[53] Rakshit Trivedi, Mehrdad Farajtabar, Prasenjeet Biswal, and Hongyuan Zha. Dyrep: Learning representations over dynamic graphs. In *ICLR*, 2019. [3](#page-2-4)
- <span id="page-9-14"></span>[54] R.J. van Beers. The sources of variability in saccadic eye movements. *The Journal of Neuroscience*, 27(33):8757– 8770, 2007. [3](#page-2-4)
- <span id="page-9-19"></span>[55] Shuai Xiao, Junchi Yan, Xiaokang Yang, Hongyuan Zha, and Stephen Chu. Modeling the intensity function of point process via recurrent neural networks. In *AAAI*, 2017. [4](#page-3-1)
- <span id="page-9-22"></span>[56] Juan Xu, Ming Jiang, Shuo Wang, Mohan S Kankanhalli, and Qi Zhao. Predicting human gaze beyond pixels. *Journal of Vision*, 14(1):28–28, 2014. [5](#page-4-1)
- <span id="page-9-25"></span>[57] Zhibo Yang, Lihan Huang, Yupei Chen, Zijun Wei, Seoyoung Ahn, Gregory Zelinsky, Dimitris Samaras, and Minh Hoai. Predicting goal-directed human attention using inverse reinforcement learning. In *CVPR*, June 2020. [5,](#page-4-1) [8](#page-7-4)
- <span id="page-9-30"></span>[58] Zhibo Yang, Sounak Mondal, Seoyoung Ahn, Gregory Zelinsky, Minh Hoai, and Dimitris Samaras. Target-absent Human Attention. In *ECCV*, 2022. [8](#page-7-4)
- <span id="page-9-20"></span>[59] Zhibo Yang, Sounak Mondal, Seoyoung Ahn, Gregory Zelinsky, Minh Hoai, and Dimitris Samaras. Predicting human attention using computational attention. *arXiv preprint arXiv:2303.09383*, 2023. [4](#page-3-1)
- <span id="page-9-8"></span>[60] Anna-Kaisa Ylitalo. *Statistical inference for eye movement sequences using spatial and spatio-temporal point processes*. PhD thesis, University of Jyvaskyl ¨ a, 2017. ¨ [2](#page-1-1)
- <span id="page-9-27"></span>[61] Dario Zanca, Stefano Melacci, and Marco Gori. Gravitational laws of focus of attention. *IEEE Trans. PAMI*, 42(12):2983–2995, 2020. [6,](#page-5-3) [7,](#page-6-4) [8,](#page-7-4) [11,](#page-10-2) [13,](#page-12-0) [14,](#page-13-0) [15,](#page-14-0) [16,](#page-15-0) [17,](#page-16-0) [18,](#page-17-0) [19,](#page-18-0) [20,](#page-19-0) [21,](#page-20-0) [22](#page-21-0)
- <span id="page-9-29"></span>[62] Mengmi Zhang, Marcelo Armendariz, Will Xiao, Olivia Rose, Katarina Bendtz, Margaret Livingstone, Carlos Ponce, and Gabriel Kreiman. Look twice: A generalist computational model predicts return fixations across tasks and species. *PLoS Computational Biology*, 18(11):1–38, 2022. [7](#page-6-4)
- <span id="page-9-31"></span>[63] Mengmi Zhang, Jiashi Feng, Keng Teck Ma, Joo Hwee Lim, Qi Zhao, and Gabriel Kreiman. Finding any Waldo with zero-shot invariant and efficient visual search. *Nature Communications*, 9(1), 2018. [8](#page-7-4)

- <span id="page-10-2"></span><span id="page-10-0"></span>[64] Qiang Zhang, Aldo Lipani, Omer Kirnap, and Emine Yilmaz. Self-attentive hawkes process. In ICML, 2020. 2, 4
- <span id="page-10-1"></span>[65] Simiao Zuo, Haoming Jiang, Zichong Li, Tuo Zhao, and Hongyuan Zha. Transformer hawkes process. In *ICML*, 2020. 2, 4

## **Supplementary Material**

We introduced TPP-Gaze, a scanpath prediction method that models gaze dynamics as a neural temporal point process. In the following sections, we provide additional results showing evidence of the superiority of our proposed approach compared to the state-of-the-art. Additionally, we describe how the proposed approach can be extended for the visual search task.

#### A. Additional Quantitative Results

Additional Metrics on OSIE, NUSEF, and FiFa. As a complement of Table 3 of the main paper, we report in Table 5 the results on OSIE, NUSEF, and FiFa datasets in terms of SS and SED. Also for these metrics, TPP-Gaze achieves the best results when compared with models trained under the same settings and datasets. It is also worth noting that, especially for the NUSEF and FiFa datasets, our approach can achieve the best overall results in terms of SS with and without duration.

Scanpath Statistics on MIT1003, NUSEF, and FiFa. As discussed in the main paper, TPP-Gaze features scanpath statistics that better align with human behavior when compared with IOR-ROI-LSTM [18], DeepGazeIII [39] and Scanpath-VQA [15]. The same trend is appreciable from Fig. 8. Notably, even when tested on MIT1003, NUSEF, and FiFa, TPP-Gaze effectively models the long-tail distribution of both fixation durations and saccade amplitudes. In contrast, other methods tend to capture only the average human gaze dynamics. An exception is DeepGazeIII on NUSEF, which achieves comparable results for saccade amplitudes but does not model fixation duration.

Return Fixations Analysis on MIT1003, NUSEF, and FiFa. Fig. 9 complements the analysis reported in the main paper by showing the distribution of return fixations (RFs) for the MIT1003, NUSEF, and FiFa datasets. In these settings as well, TPP-Gaze demonstrates its ability to model RF patterns effectively, generally presenting an RF distribution that aligns better with human observers compared to other methods.

### **B.** Extending the Model to Visual Search

We extend the TPP-Gaze architecture to handle the visual search task by forcing the model to learn a task-specific semantic representation of the input image (see Fig. 10). Recall that the TPP-Gaze's semantic representation mod-

<span id="page-10-3"></span>

|                    |               | OSIE    |                                      |        | NUSEF   |       |        |         |      |
|--------------------|---------------|---------|--------------------------------------|--------|---------|-------|--------|---------|------|
|                    | SS (KL-Div) ↓ |         | $(KL-Div) \downarrow SED \downarrow$ |        | Div)↓   | SED↓  | SS (KI | SED↓    |      |
|                    | w/ Dur        | w/o Dur | Avg                                  | w/ Dur | w/o Dur | Avg   | w/ Dur | w/o Dur | Avg  |
| Itti-Koch [32]     | -             | 3.93    | 9.07                                 | -      | 1.89    | 9.97  | -      | 14.70   | 8.65 |
| CLE (Itti) [7, 32] | -             | 3.24    | 9.29                                 | -      | 1.40    | 10.16 | -      | 12.62   | 8.86 |
| CLE (DG) [7,40]    | -             | 3.65    | 9.23                                 | -      | 1.35    | 10.07 | -      | 14.38   | 8.83 |
| G-Eymol [61]       | 12.28         | 2.95    | 8.00                                 | 1.99   | 0.53    | 8.02  | 13.17  | 5.00    | 6.13 |
| IOR-ROI-LSTM [18]  | 0.20          | 2.84    | 8.82                                 | 0.06   | 1.10    | 9.69  | 0.30   | 12.44   | 8.27 |
| DeepGazeIII [39]   | -             | 2.51    | 8.47                                 | -      | 1.04    | 9.38  | -      | 12.08   | 7.97 |
| Scanpath-VQA [15]  | 0.02          | 0.09    | 7.55                                 | 0.02   | 0.10    | 8.39  | 0.03   | 0.44    | 6.81 |
| DeepGazeIII [39]   | -             | 2.52    | 8.57                                 | -      | 1.04    | 9.42  | -      | 12.25   | 8.00 |
| Scanpath-VQA [15]  | 0.29          | 0.31    | 9.70                                 | 0.06   | 0.18    | 10.61 | 0.35   | 0.90    | 9.73 |
| TPP-Gaze (GRU)     | 0.25          | 0.30    | 8.05                                 | 0.02   | 0.03    | 8.41  | 0.15   | 0.24    | 7.00 |
| TPP-Gaze (Transf.) | 0.29          | 0.35    | 8.10                                 | 0.02   | 0.04    | 8.40  | 0.21   | 0.31    | 7.05 |

Table 5. Additional results on OSIE, NUSEF, and FiFa datasets. **Gray color** indicates models trained under the same settings and datasets. Within this group, **bold** values represent the best performance for each metric. <u>Underline</u> values indicate the overall best performance across all models and metrics.

<span id="page-10-4"></span>![](_page_10_Figure_12.jpeg)

Figure 8. Statistical properties exhibited by TPP-Gaze and other methods relative to those of human observers, in terms of empirical fixation durations and saccade amplitudes on MIT1003 (top row), NUSEF (middle row) and FiFa (bottom row) datasets. For consistency with the main paper, comparison against DeepGazeIII on MIT1003 is omitted.

ule consists of a DenseNet201 CNN backbone and a learnable readout network composed of three  $1 \times 1$  convolutional layers with 8, 16, and 1 channels, respectively. The obtained spatial priority map is then projected to a fixed-dimensional vector,  $\mathbf{z}_j$ , to obtain the j-th image semantic representation. Specifically, the last layer performing a  $1 \times 1$  convolution is responsible for learning a (non-linear) combination of the

<span id="page-11-1"></span><span id="page-11-0"></span>![](_page_11_Figure_0.jpeg)

Figure 9. Return fixations analysis comparing TPP-Gaze with other methods and human observers. Results are shown on MIT1003 (top-left plot), NUSEF (top-right plot), and FiFa (bottom plot) datasets.

<span id="page-11-2"></span>![](_page_11_Figure_2.jpeg)

Figure 10. Overview of TPP-Gaze model architecture extended to handle the visual search task. A linguistic embedding (RoBERTa) of the search target is employed to learn a task-drive semantic representation  $(z_j)$ . The latter, together with the history of past events  $(h_n)$ , is used to simulated the next fixation position and duration.

feature maps from the previous layers.

To guide the model toward a specific search objective, we redefine the architecture to enable TPP-Gaze to learn such a combination conditioned on a given text string. To this end, we first obtain a linguistic embedding of the search target using the RoBERTa language model. Let  $\mathbf{F}_{target}$  be the embedding vector representing the search objective. The readout network for the visual search model consists of three  $1\times 1$  convolutional layers with 16, 64, and 256 channels, respectively. Thus, it is modified to output M=256 feature maps. Let  $\mathbf{X}=[\mathbf{x}_0;\cdots;\mathbf{x}_M]\in\mathbb{R}^{M\times d}$  represent the matrix of flattened image features. The task-specific semantic representation for the j-th image,  $\mathbf{z}_{j,target}$ , is then obtained as follows:

$$w = \text{softplus}(\text{MLP}(\mathbf{F}_{target}))$$

$$\mathbf{z}_{j,target} = \sum_{i=1}^{M} w_i \mathbf{x}_i.$$
(11)

## C. Additional Qualitative Results

Additional qualitative results are depicted from Fig. 11 to Fig. 15 on COCO-FreeView, MIT1003, OSIE, NUSEF, and FiFa datasets, respectively. Each fixation is represented by a circle, with its diameter proportional to the fixation duration. For methods that do not model fixation duration, circles are shown with a uniform size. The first fixation of each scanpath is omitted. The qualitative results support the findings of the main paper, highlighting the accuracy of TPP-Gaze in predicting human-like scanpaths. Other methods, instead, either overfit on a few salient objects, especially people and faces in the case of Scanpath-VQA, or predict scanpath trajectories containing fixations on unlikely locations (see the bottom sample in Fig. 13 or the top sample in Fig. 14).

In the main paper, we also quantitatively assess the performance of the scanpath models on the saliency prediction task. In particular, given a sample image, we construct the aggregated saliency map by convolving a Gaussian kernel over all the locations of predicted fixations [35]. To support our quantitative analysis, we present the saliency prediction of our model against the competitors from Fig. 16 to Fig. 20 on COCO-FreeView, MIT1003, OSIE, NUSEF, and FiFa datasets, respectively. Note that we include DeepGazeIII in the comparison for reference even though its results are not directly comparable. Indeed, DeepGazeIII was specifically trained on a large scale dataset [34] to predict saliency maps along with scanpaths. Nevertheless, TPP-Gaze outperforms DeepGazeIII and the other models in many cases, demonstrating better alignment with humans.

Finally, in Fig. 21 we show additional qualitative results on sample images from COCO-Search18 for the visual search task. As can be observed, TPP-Gaze can effectively simulate human-like goal-directed visual attention patterns for various target objects. The model demonstrates its ability to adapt, with a simple architectural variation, from a free-viewing setting to a task-specific visual search scenario. The results illustrate how TPP-Gaze generates plausible attention trajectories that focus on regions likely to contain the target object, mimicking the efficient search strategies employed by humans when looking for specific items in complex scenes.

<span id="page-12-1"></span><span id="page-12-0"></span>![](_page_12_Figure_0.jpeg)

Figure 11. Qualitative comparison of simulated and human scanpaths on the COCO-FreeView dataset.

<span id="page-13-0"></span>![](_page_13_Figure_0.jpeg)

Figure 12. Qualitative comparison of simulated and human scanpaths on the MIT1003 dataset. We omit DeepGazeIII for consistency with the experimental settings described in the main paper.

<span id="page-14-1"></span><span id="page-14-0"></span>![](_page_14_Figure_0.jpeg)

Figure 13. Qualitative comparison of simulated and human scanpaths on the OSIE dataset.

<span id="page-15-1"></span><span id="page-15-0"></span>![](_page_15_Figure_0.jpeg)

![](_page_15_Figure_1.jpeg)

Figure 14. Qualitative comparison of simulated and human scanpaths on the NUSEF dataset.

<span id="page-16-1"></span><span id="page-16-0"></span>![](_page_16_Figure_0.jpeg)

Figure 15. Qualitative comparison of simulated and human scanpaths on the FiFa dataset.

<span id="page-17-1"></span><span id="page-17-0"></span>![](_page_17_Figure_0.jpeg)

Figure 16. Saliency maps of sample images from COCO-FreeView dataset computed from the fixations generated by the considered scanpath models. For completeness, we include DeepGazeIII, but note that its training procedure also involves saliency prediction.

<span id="page-18-0"></span>![](_page_18_Figure_0.jpeg)

Figure 17. Saliency maps of sample images from MIT1003 dataset computed from the fixations generated by the considered scanpath models. We omit DeepGazeIII for consistency with the experimental settings described in the main paper.

<span id="page-19-0"></span>![](_page_19_Figure_0.jpeg)

Figure 18. Saliency maps of sample images from OSIE dataset computed from the fixations generated by the considered scanpath models. For completeness, we include DeepGazeIII, but note that its training procedure also involves saliency prediction.

<span id="page-20-0"></span>![](_page_20_Figure_0.jpeg)

Figure 19. Saliency maps of sample images from NUSEF dataset computed from the fixations generated by the considered scanpath models. For completeness, we include DeepGazeIII, but note that its training procedure also involves saliency prediction.

<span id="page-21-1"></span><span id="page-21-0"></span>![](_page_21_Figure_0.jpeg)

Figure 20. Saliency maps of sample images from FiFa dataset computed from the fixations generated by the considered scanpath models. For completeness, we include DeepGazeIII, but note that its training procedure also involves saliency prediction.

<span id="page-22-0"></span>![](_page_22_Figure_0.jpeg)

Figure 21. Qualitative comparison of simulated and human scanpaths on the COCO-Search18 dataset for the visual search task.
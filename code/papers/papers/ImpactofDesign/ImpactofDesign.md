# Impact of Design Decisions in Scanpath Modeling

PARVIN EMAMI, University of Luxembourg, Luxembourg YUE JIANG and ZIXIN GUO, Aalto University, Finland LUIS A. LEIVA, University of Luxembourg, Luxembourg

Modeling visual saliency in graphical user interfaces (GUIs) allows to understand how people perceive GUI designs and what elements attract their attention. One aspect that is often overlooked is the fact that computational models depend on a series of design parameters that are not straightforward to decide. We systematically analyze how different design parameters affect scanpath evaluation metrics using a state-ofthe-art computational model (DeepGaze++). We particularly focus on three design parameters: input image size, inhibition-of-return decay, and masking radius. We show that even small variations of these design parameters have a noticeable impact on standard evaluation metrics such as DTW or Eyenalysis. These effects also occur in other scanpath models, such as UMSS and ScanGAN, and in other datasets such as MASSVIS. Taken together, our results put forward the impact of design decisions for predicting users' viewing behavior on GUIs.

CCS Concepts: • Human-centered computing → Empirical studies in ubiquitous and mobile computing; • Computing methodologies → Computer vision.

Additional Key Words and Phrases: Visual Saliency; Interaction Design; Computer Vision; Deep Learning; Eye Tracking

#### ACM Reference Format:

Parvin Emami, Yue Jiang, Zixin Guo, and Luis A. Leiva. 2024. Impact of Design Decisions in Scanpath Modeling. Proc. ACM Hum.-Comput. Interact. 8, ETRA, Article 228 (May 2024), [16](#page-15-0) pages. <https://doi.org/10.1145/3655602>

#### 1 INTRODUCTION

Understanding how user attention is allocated in graphical user interfaces (GUIs) is an important research challenge, considering that many different GUI elements (e.g. buttons, headers, cards, etc.) may stand out and engage users effectively [\[33\]](#page-15-1). By modeling eye movement patterns of visual saliency, we can gain invaluable insights into how users perceive and interact with GUIs, without having to recruit users early in the GUI design process. When presented with a GUI screenshot, a saliency model can predict how users would spend their attention, typically over short periods of time during free-viewing scenarios (bottom-up saliency) or over longer periods in task-based scenarios (top-down saliency). We can use these predictions to quantify the impact of a visual change in the GUI (e.g. after rescaling an element or changing its position), optimize some design components so that it can grab less or more user attention, or understand whether users notice some element after some time of exposure.

Saliency models can predict either (static) saliency maps [\[41\]](#page-15-2) or (dynamic) scanpaths [\[19\]](#page-15-3). Most research has focused on predicting saliency maps [\[2,](#page-14-0) [9,](#page-14-1) [15,](#page-14-2) [17,](#page-14-3) [20\]](#page-15-4), overlooking key temporal aspects

Authors' addresses: Parvin Emami, parvin@emami@uni.lu, University of Luxembourg, Luxembourg; Yue Jiang, yue.jiang@ aalto.fi; Zixin Guo, zixin.guo@aalto.fi, Aalto University, Finland; Luis A. Leiva, name.surname@uni.lu, University of Luxembourg, Luxembourg.

![](_page_0_Picture_12.jpeg)

[This work is licensed under a Creative Commons Attribution International 4.0 License.](https://creativecommons.org/licenses/by/4.0/)

© 2024 Copyright held by the owner/author(s). ACM 2573-0142/2024/5-ART228 <https://doi.org/10.1145/3655602>

like fixation timing and duration. Saliency maps show aggregated fixation locations, i.e., areas that users will pay attention to, where the eye remains relatively static. In contrast, scanpaths contain sequential information on individual fixations and sometimes also saccades, i.e., eye movements between those points, thus retaining information about fixation order and their temporal dynamics. In other words, scanpaths comprise rich data from which second-order representations such as saliency maps can be computed. Critically, scanpaths can inform about the users' visual flows, through which one can better assess how attention deploys over time. This is vital for understanding how individual users, instead of a group of users, would perceive the GUI and for design adjustments that encourage viewing GUI elements in a desired order for different people [\[12\]](#page-14-4). For these reasons, in this paper we focus on scanpath models of visual saliency.

One problem that is often overlooked is that many computational models of visual saliency rely on a set of design parameters that must be defined beforehand. Some of them can be inferred from the collected data, such as deciding the number of fixations to predict on average. However other parameters must be established by the researcher, such as deciding the resolution of the GUI screenshots for model input. These design parameters cannot be learned e.g. through backpropagation, so researchers have to rely on their own expertise, trial and error, previous work, or best practices. To the best of our knowledge, their impact on downstream performance has not been systematically analyzed. We believe this kind of analysis is very much needed because any evaluation depends on the quality of the model predictions, so it may be the case that small variations on some parameters produce different performance results. In this context, we pose the following research question: To what extent do saliency model predictions depend on the choices made in their design parameters?

We use DeepGaze++ [\[17\]](#page-14-3) as a reference model to investigate the potential impact that different design parameters may have in scanpath prediction. DeepGaze++ is a state-of-the-art scanpath model for visual saliency prediction that has shown promising results. However, like other models, it relies on "hardcoded" design parameters such as the aforementioned input screenshot size or, more interestingly, the masking radius used for inhibition of return (IOR) mechanisms. IOR, which refers to the phenomenon where attention is less likely to return to previously attended locations, plays a crucial role in visual perception. If the masking radius is not appropriately calibrated, there is a possibility of omitting potentially salient areas within a GUI [\[18\]](#page-15-5). Also, as explained later, DeepGaze++ relies on a sub-optimal IOR weighting mechanism limited to 12 fixation points, so we propose a new weighting scheme to overcome this limitation as an aside research contribution.

While using hardcoded design parameters may be the most straightforward approach, determining their optimal values for each type of GUI remains an open question. For example, if we were to have a masking radius equivalent to the whole image size, the model could only predict one fixation, as the whole GUI would have been masked out (i.e. no other GUI parts could be fixated on because, by definition, there is nothing left to fixate on if everything is masked out). In this paper, we focus on three key design parameters common to every scanpath model:

Input image size, which determines the granularity of the predicted fixations (higher resolution gives more room to fixate on more GUI parts).

IOR decay, which implements the importance that previous fixations have in successive fixation predictions.

Masking radius, which allows to prevent that previously fixated GUI parts are fixated on again.

We study the impact of these parameters on different GUI types, including web-based, mobile, and desktop applications. Then, we show that the optimized parameters we discovered for DeepGaze++ [\[17\]](#page-14-3) help improve the performance of other scanpath models and also generalize to other datasets. In sum, this paper makes the following contributions:

- (1) A comparative study on model design parameters on scanpath prediction performance, across different types of GUIs.
- (2) An optimized set of design parameters for scanpath models, evaluated across multiple models and datasets.
- (3) A new IOR decay parameter, designed to work with an arbitrary number of fixation points.

## 2 RELATED WORK

Visual saliency prediction in GUIs has witnessed substantial growth in recent years [\[39,](#page-15-6) [40,](#page-15-7) [42\]](#page-15-8), driven by the increasing demand for accurate models that can anticipate where human attention is likely to be directed within digital displays. As previously hinted, existing research has predominantly focused on the development and refinement of visual saliency prediction models, often overlooking the crucial impact that different design parameters may have. For example, de Belen et al. [\[11\]](#page-14-5) proposed ScanpathNet, a deep learning model inspired by neuroscience, and noted that model performance was influenced by the choice of the number of Gaussian components. This highlights the significant impact that the choice of design parameters may have.

Previous research considered different image sizes to predict visual saliency, informed by the datasets they used for training the models [\[3,](#page-14-6) [8,](#page-14-7) [19\]](#page-15-3). It became apparent that higher image resolutions are not desired, as drifting errors tend to increase [\[24\]](#page-15-9), but no systematic examination was provided in this regard. In addition, Parmar et al. [\[29\]](#page-15-10) demonstrated that generative models are particularly sensible to image resizing artifacts such as quantization and compression. Therefore, we decided to examine the impact of image resizing in visual saliency prediction.

Generating fixation sequences accurately, while promoting a coherent and natural order, remains as the main challenge in scanpath modeling. Itti et al. [\[16\]](#page-14-8) introduced the Inhibition of Return (IOR) mechanism as a way to ensure that predicted fixations do not bounce back and forth around previously visited areas. This was later exploited in scanpath modeling [\[3,](#page-14-6) [32\]](#page-15-11), although all scanpath models that incorporate IOR employ a fixed masking radius [\[7,](#page-14-9) [10\]](#page-14-10). Furthermore, there is no discussion about how this radius affects model predictions. As mentioned in the previous section, when this radius is too large, the model's ability to predict multiple fixation points will be severely limited. Therefore, we decided to examine the impact of masking radius and IOR decay in visual saliency prediction.

Several methods, including deep neural networks and first-principle models, have been proposed to predict scanpaths in natural images [\[9\]](#page-14-1), videos [\[23\]](#page-15-12), and, more recently, GUIs [\[17\]](#page-14-3). Ngo and Manjunath [\[28\]](#page-15-13) developed a recurrent neural network to predict sequences of saccadic eye movements and Wloka et al. [\[36\]](#page-15-14) predicted fixation sequences by relying on a "history map" of previously observed fixations. These works were evaluated on small datasets and using a limited set of metrics, therefore it remains unclear whether these models can compare favorably in GUIs.

Later on, Xia et al. [\[37\]](#page-15-15) introduced an iterative representation learning framework to predict saccadic movements. More recently, Jiang et al. [\[17\]](#page-14-3) developed DeepGaze++ based on DeepGaze III [\[20\]](#page-15-4). DeepGaze III takes both an input image and the positions of the previous fixation points to predict a probabilistic density map for the next fixation point. It frequently tends to predict clusters of nearby points, potentially leading to stagnation within those clusters. To address this problem, DeepGaze++ recurrently chooses the position with the highest probability from the density map, concurrently implementing a custom IOR decay to suppress the selected position in the saliency map. (As explained in the next section, this decay only works for a relatively small number of

fixation points, therefore we propose a new IOR decay to address this limitation.) Nevertheless, DeepGaze++ is a state-of-the-art scanpath model so we use it in our investigation.

## 3 METHODOLOGY

The goal of scanpath prediction is to generate a plausible sequence of fixations, where fixations refer to distinct focal points during visual exploration. As previously mentioned, our study leverages the advanced capabilities of DeepGaze++ to answer our research question.

#### 3.1 Dataset

In our study, we use the UEyes dataset [\[17\]](#page-14-3), a collection of eye-tracking data over 1,980 screenshots covering four GUI types (495 screenshots per type): posters, desktop UIs, mobile UIs, and webpages. This dataset was collected from 66 participants (23 male, 43 female) aged 27.25 years (SD=7.26) via a high-fidelity in-lab eye tracker Gazepoint GP3. Participants had normal vision (43) or wore either glasses (18) or contact lenses (5). No participant was colorblind. Eye-tracking data were recorded after participant-specific calibration, a step that accounts for variables such as eye-display distance and visual angle, to ensure accurate recording of eye data. Participants were given 7 seconds to freely view each GUI screenshot in a 1920x1200 px monitor. For our study, we considered the same data partitions as in the UEyes dataset: 1,872 screenshots for training and 108 for testing. Our experiments are performed over the training partition of the UEyes dataset. The testing partition simulates unseen data, therefore it is used for final model evaluation.

### 3.2 Design parameters

In the following, we describe the parameters we have considered for our study. Note that they are all common to every scanpath model, they cannot be inferred from data, and they cannot be learned automatically. In our experiments, whenever we modify the values of each parameter, everything else remains constant. This way, it is easy to understand the concrete influence of each parameter in model performance.

- 3.2.1 Image size. Each GUI type has a preferred size (e.g. desktop applications are usually designed for FullHD monitors) or proportion (e.g. mobile apps have around 9:16 aspect ratio). When GUI images are resized (downsampled), to speed up computations, the models may perform differently. Therefore, it is unclear which image resolution should be used as model input. To shed light in this regard, we tested different input sizes and aspect ratios.
- 3.2.2 IOR decay. Initially introduced by Posner et al. [\[30\]](#page-15-16), IOR is a neural mechanism that suppresses visual processing within recently attended locations. In the context of scanpath modeling, DeepGaze++ uses an IOR decay of 1 − 0.1( − − 1), for the -th fixation point when predicting fixation points, to prevent that older fixation points are likely to be revisited. As can be noticed, this is limited to a maximum number of 12 fixation points, after which the decay values may become negative; e.g., for = 13 the first fixation point = 1 gets an IOR of −0.1. Consequently, we have developed a new IOR decay designed to accommodate any number of fixation points. We propose (−−1) , in which is a design parameter, between 0 and 1, that we also analyze systematically.
- 3.2.3 Masking radius. To implement any IOR mechanism, we need to mask some areas around the previous fixation points. However, determining the optimal size of the masked areas is unclear. Therefore, we consider the masking radius as a third design parameter and, in line with the previous discussions, examine how various masking radii impact the scanpath prediction results.

#### 3.3 Evaluation metrics

<span id="page-4-0"></span>We employ a set of four metrics that, together, provide a holistic assessment about the predictive performance of scanpath models [\[1,](#page-14-11) [13,](#page-14-12) [26\]](#page-15-17): Dynamic Time Warping (DTW), Eyenalysis, Determinism, and Laminarity. These metrics are well-established in the research literature [\[14\]](#page-14-13). While DTW measures the location and sequence of fixations in temporal order, Eyenalysis measures only locations and Determinism measures only the order of fixation points. Conversely, Laminarity is a measure of repeated fixations on a particular region, without considering their location or order. [Table 1](#page-4-0) provides an overview of these metrics.

| Metric      | Location | Order |
|-------------|----------|-------|
| DTW         | Yes      | Yes   |
| Determinism | No       | Yes   |
| Eyeanalysis | Yes      | No    |
| Laminarity  | No       | No    |
|             |          |       |

Table 1. Overview of the chosen scanpath evaluation metrics [\[14\]](#page-14-13).

- 3.3.1 Dynamic Time Warping (DTW). First introduced by Berndt and Clifford [\[4\]](#page-14-14), DTW is a method for comparing time series with varying lengths. It involves creating a distance matrix between two sequences and finding the optimal path that respects boundary, continuity, and monotonicity conditions. The optimal solution is the minimum path from the starting point to the endpoint of the matrix. DTW identifies such an optimal match between two scanpaths in an iteratively manner, ensuring the inclusion of critical features [\[27,](#page-15-18) [34\]](#page-15-19).
- 3.3.2 Eyenalysis. This is a technique that involves double mapping of fixations between two scanpaths, aiming to reduce positional variability [\[26\]](#page-15-17). Like in DTW, this approach may result in multiple points from one scanpath being assigned to a single point in the other. Eyeanalysis performs dual mapping by finding spatially closest fixation points between two scanpaths, measuring average distances for these corresponding pairs.
- 3.3.3 Determinism. This metric gauges diagonal alignments among cross-recurrent points, representing shared fixation trajectories [\[14\]](#page-14-13). With a minimum line length of = 2 for diagonal elements, Determinism measures the congruence of fixation sequences. Computed as the percentage of recurrent fixation points in sub-scanpaths, Determinism considers pairs of distinct fixation points from two scanpaths, enhancing the original (unweighted) Determinism metric for subscanpath evaluation.
- 3.3.4 Laminarity. It measures the percentage of fixation points on sub-scanpaths in which all the pairs of corresponding fixation points are recurrences but all such recurrent fixation point pairs contain the same fixation point from one of the scanpaths [\[1,](#page-14-11) [14\]](#page-14-13). In sum, Laminarity indicates the tendency of scanpath fixations to cluster on one or a few specific locations.

#### 4 EXPERIMENTS

In the following, we report the experiments aimed at finding the optimal set of design parameters. For the sake of conciseness, we consider DTW for determining the best result for each design parameter, as this metric accounts for both location and order of fixations [\(Table 1\)](#page-4-0).

<span id="page-5-0"></span>![](_page_5_Figure_2.jpeg)

Fig. 1. Impact of resizing to square or non-square image on different GUI types. The height is always fixed to 225 px. We consider widths of 128, 225, and 512 px. The best results are observed for widths of 225 px (resulting in a square aspect ratio).

<span id="page-5-1"></span>![](_page_5_Figure_4.jpeg)

Fig. 2. Impact of resizing to different square image sizes on different GUI types. We consider sizes of 128, 225, and 512 px. The best results are usually observed for the 128 px cases.

#### 4.1 Sensitivity to input image size

We analyzed the impact of resizing under different aspect ratios (square and non-square images). In the first experiment [\(Figure 1\)](#page-5-0), the height of the resized images remained constant at 225 px, as suggested in previous studies [\[17\]](#page-14-3), while we modified their width. The other width values were chosen as the closest powers of two around this baseline value of 225, for convenience.

The results, presented in [Figure 1,](#page-5-0) indicate that resizing any input image to a square aspect ratio consistently yields superior performance across all GUI types. An intriguing observation is that mobile GUIs are particularly sensible to this parameter as compared to other GUI types. We attribute this effect to the fact that mobile GUIs, despite having the largest aspect ratio, make heavy use of icons and usually icons have a square aspect ratio.

In the second experiment [\(Figure 2\)](#page-5-1), we resized images down to various dimensions, while ensuring a square aspect ratio, as per the results of our previous experiment. The results in this case indicate that resizing images to smaller dimensions has a positive impact on the prediction of both scanpaths and fixation points in mobile UIs. However, the opposite holds true for desktop UIs, as they typically have smaller elements as compared to mobile UIs.

#### 4.2 Sensitivity to IOR decay

In this experiment, we varied the parameter of our proposed IOR decay to assess its impact on scanpath prediction. As a reminder, a larger indicates a higher probability of revisiting previously observed fixation points. [Figure 3](#page-6-0) illustrates the findings of this experiment, indicating that smaller values lead to improved scanpath prediction performance. This suggests that when the likelihood of revisiting a previously observed fixation point is low, the model performs better in predicting subsequent fixation points. Conversely, when the likelihood of revisiting a fixation point is high, the model excels in predicting individual fixation points.

<span id="page-6-0"></span>![](_page_6_Figure_2.jpeg)

Fig. 3. Impact of different  $\gamma$  values on different GUI types. Lower  $\gamma$  means a high probability of revisiting fixation points. The best results are observed when  $\gamma = 0.1$ .

#### 4.3 Sensitivity to masking radius

In this experiment, we examined how altering the masking radius impacts scanpath prediction performance. The results are provided in Figure 4. We observed a negative correlation between the masking radius and the quality of the scanpath predictions, indicating that, as the radius increases, scanpath prediction quality decreases. However, we observed a sweet spot when the radius is set between 0.1 and 0.2, as better results are obtained according to the three non-DTW metrics.

<span id="page-6-1"></span>![](_page_6_Figure_6.jpeg)

Fig. 4. Impact of different masking radius on different GUI types. Masking radii are relative to the input image size (e.g. 0.2 means 20% of the size). The best results are observed when the radius is set to 0.05.

#### 4.4 Putting it all together

With the optimal parameters in place, we conducted an additional experiment on the test partition of UEyes to understand the impact of an improved scanpath model. Figure 5 illustrates the results. The "DeepGaze++" cases represent the baseline model implementation [17]. The "Baseline IOR" cases represent DeepGaze++ using the original IOR decay and the optimal parameters derived from our experiments, whereas the "Improved IOR" cases represent DeepGaze++ with our proposed IOR decay and the optimal parameters derived from our experiments. The results highlight that adopting the new IOR decay addresses the challenge of the limited number of fixation points and contributes to enhanced prediction performance as compared with the baseline DeepGaze++ model, although the baseline IOR with optimal parameters is comparable in predicting fixation points.

The results indicate significant improvements when using these optimal parameters, underscoring their substantial impact on prediction performance. Figure 6 provides additional evidence by showing the ratios of visited-revisited elements for three types of GUI elements (image, text, face) following previous work [17, 22].

Table 2 shows that, by setting all the optimized parameters, the results of DeepGaze++ improve in all the metrics except Eyenalysis. The table presents the results of DeepGaze++ with baseline parameters, as described in [17], and with the set of optimized parameters. According to the two-sample paired t-test, differences are statistically significant for all metrics except Eyenalysis: **DTW**: t(107) = 5.36, p < .0001, d = 0.367; **Eyenalysis**: t(107) = 5.36, p = .4503 (n.s.), d = 0.074;

<span id="page-7-0"></span>![](_page_7_Figure_2.jpeg)

Fig. 5. Impact of different IOR mechanisms, using optimal parameters, on different GUI types. "'Baseline IOR" uses DeepGaze++ with the original IOR decay and the optimal parameters. "'Improved IOR" uses DeepGaze++ with our proposed IOR decay and the optimal parameters. The best results are usually observed with the baseline IOR.

<span id="page-7-1"></span>

| DeepGaze++ | DTW↓  | Eyenalysis↓ | Determinism↑ | Laminarity↑ |
|------------|-------|-------------|--------------|-------------|
| Baseline   | 5.118 | 0.040       | 1.101        | 16.908      |
|            | 0.482 | 0.004       | 0.536        | 5.900       |
|            | ±     | ±           | ±            | ±           |
| Improved   | 4.669 | 0.044       | 2.529        | 24.557      |
|            | 0.667 | 0.010       | 2.059        | 4.705       |
|            | ±     | ±           | ±            | ±           |

Table 2. Evaluation of baseline (original) and improved DeepGaze++ model (using the optimized parameters), showing Mean ± SD results for each metric. Arrows denote the direction of the importance; e.g., ↓ means "lower is better." Each column's best result is highlighted in boldface.

Determinism: (107) = 3.60, < .001, = 0.432; Laminarity: (107) = 8.98, < .0001, = 0.580. Effect sizes (Cohen's ) suggest a moderate practical importance of the results [\[21\]](#page-15-21).

#### 4.5 Analysis of visited and revisited patterns

In line with previous research that quantified the impact of scanpath models in GUI elements [\[17\]](#page-14-3), we categorized the GUI elements in UEyes into three categories (image, text, and face) using an enhanced version of the UIED model [\[38\]](#page-15-22), which is designed to detect images and text on GUIs. We then analyzed the number of elements in each category that were initially visited and subsequently revisited. An element is considered revisited if the element gets a fixation again after at least three fixations on other elements. The findings are presented in [Figure 6.](#page-8-0) We observed that text elements have a higher fixation probability than images in our improved model, which is better aligned with the ground-truth cases. The improved model is also more aligned with the ground-truth cases in terms revisited fixations. For visited fixations, no differences between models were observed.

#### 4.6 Example gallery

[Figure 7](#page-9-0) illustrates our qualitative comparison of different scanpath models across various GUI types. The baseline DeepGaze++ model can predict fixation points but the resulting scanpaths are not very realistic. The improved DeepGaze++ model is able to predict realistic trajectories, with more accurate fixation points overall. It is worth noting that both models tend to have a center bias and tend to generate clusters of fixation points. The scanpaths shown in [Figure 7](#page-9-0) follow a color gradient from red (beginning of trajectory) to blue (end of trajectory).

#### 4.7 Comparison against other scanpath models

To show the generalizability of our findings, we further conducted evaluations on a more diverse set of scanpath prediction models: Itti-Koch model [\[16\]](#page-14-8), UMSS [\[35\]](#page-15-23), ScanGAN [\[25\]](#page-15-24), ScanDMM [\[31\]](#page-15-25), and the model by Chen et al. [\[6\]](#page-14-15). We applied the same set of optimized parameters obtained from

<span id="page-8-0"></span>![](_page_8_Figure_2.jpeg)

![](_page_8_Figure_3.jpeg)

![](_page_8_Figure_4.jpeg)

- (a) Ground-truth (b) Baseline DeepGaze++ (c) Improved DeepGaze++

Fig. 6. Visit vs. revisit bias analysis, showing the ratios of visited-revisited elements for three element categories. According to the ground-truth data, text elements are more likely to be visited and revisited than images. The improved model is better aligned with this observation.

<span id="page-8-1"></span>

| Model       |                      | DTW↓                                       | Eyenalysis↓                                | Determinism↑                               | Laminarity↑                                  |
|-------------|----------------------|--------------------------------------------|--------------------------------------------|--------------------------------------------|----------------------------------------------|
| Itti-Koch   | Baseline<br>Improved | 7.023<br>0.261<br>±<br>5.824<br>0.219<br>± | 0.075<br>0.014<br>±<br>0.053<br>0.012<br>± | 0.363<br>0.154<br>±<br>0.378<br>0.141<br>± | 5.823<br>1.169<br>±<br>4.943<br>1.034<br>±   |
| Chen et al. | Baseline<br>Improved | 4.298<br>0.225<br>±<br>4.111<br>0.187<br>± | 0.028<br>0.003<br>±<br>0.025<br>0.001<br>± | 1.597<br>1.556<br>±<br>1.483<br>0.188<br>± | 7.028<br>0.344<br>±<br>7.724<br>1.259<br>±   |
| UMSS        | Baseline<br>Improved | 4.567<br>0.394<br>±<br>4.537<br>0.432<br>± | 0.031<br>0.006<br>±<br>0.033<br>0.003<br>± | 2.390<br>1.044<br>±<br>3.123<br>1.028<br>± | 10.541<br>1.764<br>±<br>12.302<br>1.815<br>± |
| ScanGAN     | Baseline<br>Improved | 4.001<br>0.379<br>±<br>3.973<br>0.204<br>± | 0.026<br>0.003<br>±<br>0.027<br>0.002<br>± | 1.306<br>1.025<br>±<br>1.331<br>0.798<br>± | 7.311<br>1.817<br>±<br>7.900<br>1.423<br>±   |
| ScanDMM     | Baseline<br>Improved | 4.584<br>0.336<br>±<br>4.452<br>0.378<br>± | 0.033<br>0.002<br>±<br>0.029<br>0.002<br>± | 0.597<br>0.499<br>±<br>0.472<br>0.546<br>± | 5.123<br>1.042<br>±<br>5.596<br>0.996<br>±   |

Table 3. Evaluation of baseline and improved models, showing Mean ± SD for each metric. Arrows denote the direction of the importance; e.g., ↓ means "lower is better." Each column's best result is highlighted in boldface.

DeepGaze++ to these other models. [Figure 8](#page-10-0) and [Table 3](#page-8-1) show the results. This approach allowed us to assess the performance of the parameters across different scanpath models, providing valuable insights into their applicability beyond a single model.

All the models show improved results on most metrics after employing the optimized set of parameters on most types of GUIs. Similar to DeepGaze++, the Itti-Koch model also incorporates the IOR mechanism. Therefore, adjusting the masking radius to its optimal value has a notable impact on prediction performance. According to our findings, it is clear that there is potential to enhance the performance of scanpath prediction models by utilizing a set of optimal design parameters that cannot be learned from the data. This highlights the importance of considering and optimizing these design parameters to achieve improved performance in scanpath models.

#### 4.8 Comparison against other datasets

To further demonstrate the generalization of our optimal parameters and the improved DeepGaze++ model, we evaluate it on MASSVIS [\[5\]](#page-14-16), one of the largest real-world visualization databases, scraped

<span id="page-9-0"></span>![](_page_9_Figure_2.jpeg)

Fig. 7. Qualitative comparison between scanpaths. The scanpaths follow a color gradient from red (beginning of trajectory) to blue (end of trajectory).

from various online sources including government reports, infographic blogs, news media websites, and scientific journals. MASSVIS includes scanpaths from 393 screenshots observed by 33 viewers, with an average of 16 viewers per visualization. Each viewer spent 10 seconds examining each visualization, resulting in an average of 37 fixation points. To accommodate the limitation of the baseline DeepGaze++ model of 12 fixation points, we considered the first 15 fixation points in each scanpath.

[Table 4](#page-11-0) shows that the improved DeepGaze++ model consistently outperforms the baseline model on the four MASSVIS datasets across all scanpath metrics except Laminarity. When Laminarity is high but Determinism is low, it means that the scanpath model quantifies the number of locations that were fixated in detail in the ground-truth scanpath, but were only fixated briefly in the predicted scanpath [\[1\]](#page-14-11). In this regard, we can see that the improved model has always a smaller difference between these two metrics, suggesting thus a better alignment with the ground-truth scanpaths.

<span id="page-10-0"></span>![](_page_10_Figure_2.jpeg)

Fig. 8. Comparison against other scanpath models across GUI types. Each model (baseline, light blue bars) is re-trained with the optimized parameters (improved, dark blue bars) and evaluated on the testing partition. Error bars denote standard deviations

.

Notably, the best improvements were observed on the InfoVis dataset and the best performance overall was observed on the Science dataset.

#### 4.9 Understanding the role of the number of fixation points

All scanpath models are ultimately used to produce a number of fixation points. While we do not consider this to be a design parameter, since it is actually a model outcome, we do find it interesting to study their role in downstream performance. Therefore, we conducted an additional analysis across all the scanpath models considered in our previous experiment. We systematically varied the

<span id="page-11-0"></span>

| Dataset    |                      | DTW↓                                       | Eyenalysis↓                                | Determinism↑                                 | Laminarity↑                                    |
|------------|----------------------|--------------------------------------------|--------------------------------------------|----------------------------------------------|------------------------------------------------|
| Government | Baseline<br>Improved | 8.073<br>1.932<br>±<br>6.674<br>2.294<br>± | 0.171<br>0.102<br>±<br>0.125<br>0.111<br>± | 0.253<br>4.057<br>±<br>1.680<br>10.192<br>±  | 39.555<br>24.446<br>±<br>33.980<br>26.814<br>± |
| InfoVis    | Baseline<br>Improved | 7.318<br>1.782<br>±<br>5.726<br>1.558<br>± | 0.147<br>0.094<br>±<br>0.088<br>0.066<br>± | 1.418<br>10.334<br>±<br>3.268<br>14.266<br>± | 49.564<br>25.672<br>±<br>40.855<br>27.360<br>± |
| Science    | Baseline<br>Improved | 5.844<br>1.425<br>±<br>5.323<br>1.475<br>± | 0.074<br>0.040<br>±<br>0.064<br>0.047<br>± | 4.555<br>17.859<br>±<br>5.611<br>18.521<br>± | 49.658<br>26.855<br>±<br>45.203<br>25.027<br>± |
| News       | Baseline<br>Improved | 8.103<br>2.214<br>±<br>7.648<br>2.776<br>± | 0.163<br>0.134<br>±<br>0.168<br>0.169<br>± | 0.110<br>2.926<br>±<br>0.875<br>7.100<br>±   | 28.426<br>25.567<br>±<br>29.744<br>24.629<br>± |
| Averaged   | Baseline<br>Improved | 7.334<br>1.058<br>±<br>6.343<br>1.038<br>± | 0.139<br>0.044<br>±<br>0.111<br>0.045<br>± | 1.584<br>2.065<br>±<br>2.859<br>2.087<br>±   | 41.801<br>10.098<br>±<br>37.445<br>6.907<br>±  |

Table 4. Evaluation of baseline and improved DeepGaze++ in the MASSVIS datasets, showing Mean ± SD for each metric. Arrows denote the direction of the importance; e.g., ↓ means "lower is better." The best result in each case is highlighted in boldface.

number of fixation points from 5 to 10 and evaluated model performance on the testing partition. The results are shown in [Figure 9.](#page-12-0)

We can observe that an increase in the number of fixation points correlates with improved Determinism and Laminarity values across all models. In addition, Eyenalysis exhibits enhancement in predictive accuracy for more fixation points except the Itti-Koch model. Thus, scanpaths with a larger number of fixation points are more likely to simulate the human's real scanpaths.

# 5 DISCUSSION

Despite the development of scanpath prediction models for GUIs, the extent to which the design parameter choices influence saliency predictions performance has remained underexplored. We have conducted comprehensive experiments in this regard, using a state-of-the-art scanpath model as a reference. By understanding the significance of these parameters, we contribute to the body of knowledge of how people look at GUIs and how to better develop models to predict it.

To what extent do saliency predictions depend on the choices made in design parameters? Our findings draw attention to the considerable influence of design parameters in determining the accuracy of predicting scanpaths in GUIs. Specifically, the role of input image size, masking radius, and IOR decay is significant in assessing user attention and eye movement patterns in GUIs. As shown in [Figure 8,](#page-10-0) optimizing these parameters can substantially enhance scanpath prediction performance. In summary, our research has led to the following findings:

- (1) Image size has a large impact on model predictions. Resizing images to smaller dimensions positively impacts prediction performance The best results were observed for images resized to 225 px.
- (2) Resizing any input image to a square aspect ratio consistently yields superior performance across all GUI types. Mobile GUIs are particularly sensible to the image aspect ratio.
- (3) IOR is essential to reduce the likelihood of a user revisiting earlier seen GUI points. Our proposed decay = 0.1 addresses an important limitation in DeepGaze++ and leads to improved prediction performance.

<span id="page-12-0"></span>![](_page_12_Figure_2.jpeg)

Fig. 9. Impact of different numbers of fixation points on different models on different GUI types.

(4) When the masking radius increases, prediction quality decreases. The masking radius should find a balance between repetition of viewed areas (small radius) and blocking out too large parts of the GUI (large radius). A sensible value is 0.1, i.e. 10% of the available image size.

(5) All the studied GUI types follow a similar trend in terms of optimal parameter settings, although some GUIs may be affected slightly differently, as reported by the four evaluation metrics considered.

Understanding the effect of design decisions on scanpath predictive models allows researchers to be aware of the fact that even small variations can lead to more accurate results. By examining how different design elements gauge users' attention, researchers can identify effective design strategies that promote user engagement and optimize information presentation. This knowledge can be applied to various domains, including website design, multimedia content creation, and advertising, enabling designers to create more visually appealing and user-friendly interfaces. Furthermore, the evaluation on multiple scanpath models shows the generalizability of our findings. We hope the insights presented in this paper could serve as a reference for future researchers working on saliency prediction in GUIs.

## 5.1 Limitations and future work

While our findings offer valuable new knowledge to optimize scanpath model performance, our experiments examined the impact of design parameters in isolation (i.e. we studied one design parameter at a time), therefore future work should consider a joint optimization procedure. It may be the case that an automatically optimized set (e.g. with Bayesian optimization) can lead to more accurate performance results.

In principle, the proposed values of the design parameters we studied are meant to be applicable to every scanpath model. We found that this is the case for the 5 models we evaluated, most of them offering state-of-the-art performance, but we also acknowledge that there might be other sets of values that could work better for a particular model.

Future work should also consider more fixations points as model output. In all our experiments, DeepGaze++ was used to predict trajectories of 10 fixations each, to facilitate comparisons against previous work [\[17,](#page-14-3) [20\]](#page-15-4). However, it may be the case that predicting more fixation points would result in more (or less) informed trajectories, which may in turn affect the performance evaluation metrics. For example, if far-distant points (in time) tend to be more dispersed, the DTW values will increase. Such exploration can help develop better computational scanpath models.

#### 5.2 Privacy and ethics statement

On the positive side, our research focuses on providing optimal parameters for scanpath models, enabling more accurate predictions. But this enhanced prediction goes beyond mere gaze direction, offering valuable insights into an individual's perceptual and cognitive processes. Our advancements open up opportunities for innovative applications, particularly in the realm of designing or adapting user interfaces. However, it is important to consider that the use of these optimal parameters and more accurate models can also be exploited for unforeseen purposes, such as optimizing advertisements placement on websites or enabling "dark patterns" such as making the user click on some content as a result of some GUI adaptation that optimized the interface elements for quick scannability. Overall, we should note that striking a balance between harnessing the technology's potential benefits and safeguarding individuals' rights is crucial for responsible development and deployment.

#### 6 CONCLUSION

Scanpath models rely on a series of design parameters that are usually taken for granted. We have shown that even small variations of these parameters have a noticeable impact on several evaluation metrics. As a result, we have found a set of optimal parameters that improve the state of the art in scanpath modeling. These parameters have resulted in an improved DeepGaze++ model that can better capture both the spatial and temporal characteristics of scanpaths. These parameters are replicable to other computational models and datasets, showing the generalizability of our findings. The community can use therefore this improved set of model parameters (or even the improved models themselves) to get a better understanding of how users are likely to perceive GUIs. Ultimately, this work provides invaluable insights for designers and researchers interested in predicting users' viewing strategies on GUIs. Our software and models are publicly available [https://github.com/prviin/scanpath-design-decisions.](https://github.com/prviin/scanpath-design-decisions)

## ACKNOWLEDGMENTS

Research supported by the Horizon 2020 FET program of the European Union (grant CHIST-ERA-20-BCI-001) and the European Innovation Council Pathfinder program (SYMBIOTIK project, grant 101071147).

#### REFERENCES

- <span id="page-14-11"></span>[1] Nicola C Anderson, Fraser Anderson, Alan Kingstone, and Walter F Bischof. 2015. A comparison of scanpath comparison methods. Behavior research methods 47 (2015), 1377–1392.
- <span id="page-14-0"></span>[2] Marc Assens, Xavier Giro-i Nieto, Kevin McGuinness, and Noel E O'Connor. 2018. PathGAN: Visual scanpath prediction with generative adversarial networks. In Proceedings of the European Conference on Computer Vision (ECCV) Workshops. 0–0.
- <span id="page-14-6"></span>[3] Wentao Bao and Zhenzhong Chen. 2020. Human scanpath prediction based on deep convolutional saccadic model. Neurocomputing 404 (2020), 154–164.
- <span id="page-14-14"></span>[4] Donald J Berndt and James Clifford. 1994. Using dynamic time warping to find patterns in time series. In Proceedings of the 3rd international conference on knowledge discovery and data mining. 359–370.
- <span id="page-14-16"></span>[5] Michelle A. Borkin, Zoya Bylinskii, Nam Wook Kim, Constance May Bainbridge, Chelsea S. Yeh, Daniel Borkin, Hanspeter Pfister, and Aude Oliva. 2016. Beyond Memorability: Visualization Recognition and Recall. IEEE Transactions on Visualization and Computer Graphics 22, 1 (2016).
- <span id="page-14-15"></span>[6] Xianyu Chen, Ming Jiang, and Qi Zhao. 2021. Predicting human scanpaths in visual question answering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 10876–10885.
- <span id="page-14-9"></span>[7] Zhenzhong Chen and Wanjie Sun. 2018. Scanpath Prediction for Visual Attention using IOR-ROI LSTM.. In IJCAI. 642–648.
- <span id="page-14-7"></span>[8] Aladine Chetouani and Leida Li. 2020. On the use of a scanpath predictor and convolutional neural network for blind image quality assessment. Signal Processing: Image Communication 89 (2020), 115963.
- <span id="page-14-1"></span>[9] Marcella Cornia, Lorenzo Baraldi, Giuseppe Serra, and Rita Cucchiara. 2018. SAM: Pushing the limits of saliency prediction models. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops. 1890–1892.
- <span id="page-14-10"></span>[10] Erwan Joel David, Pierre Lebranchu, Matthieu Perreira Da Silva, and Patrick Le Callet. 2019. Predicting artificial visual field losses: A gaze-based inference study. Journal of Vision 19, 14 (2019), 22–22.
- <span id="page-14-5"></span>[11] Ryan Anthony Jalova de Belen, Tomasz Bednarz, and Arcot Sowmya. 2022. Scanpathnet: A recurrent mixture density network for scanpath prediction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 5010–5020.
- <span id="page-14-4"></span>[12] Sukru Eraslan, Yeliz Yesilada, and Simon Harper. 2016. Eye tracking scanpath analysis on web pages: how many users?. In Proceedings of the ninth biennial ACM symposium on eye tracking research & applications. 103–110.
- <span id="page-14-12"></span>[13] Ramin Fahimi. 2018. Sequential selection, saliency and scanpaths. (2018).
- <span id="page-14-13"></span>[14] Ramin Fahimi and Neil DB Bruce. 2021. On metrics for measuring scanpath similarity. Behavior Research Methods 53 (2021), 609–628.
- <span id="page-14-2"></span>[15] Camilo Fosco, Vincent Casser, Amish Kumar Bedi, Peter O'Donovan, Aaron Hertzmann, and Zoya Bylinskii. 2020. Predicting visual importance across graphic design types. In Proceedings of the 33rd Annual ACM Symposium on User Interface Software and Technology. 249–260.
- <span id="page-14-8"></span>[16] Laurent Itti, Christof Koch, and Ernst Niebur. 1998. A model of saliency-based visual attention for rapid scene analysis. IEEE Transactions on pattern analysis and machine intelligence 20, 11 (1998), 1254–1259.
- <span id="page-14-3"></span>[17] Yue Jiang, Luis A Leiva, Hamed Rezazadegan Tavakoli, Paul RB Houssel, Julia Kylmälä, and Antti Oulasvirta. 2023. UEyes: Understanding Visual Saliency across User Interface Types. In Proceedings of the 2023 CHI Conference on Human Factors in Computing Systems. 1–21.

- <span id="page-15-5"></span><span id="page-15-0"></span>[18] Raymond M Klein. 2000. Inhibition of return. Trends in cognitive sciences 4, 4 (2000), 138–147.
- <span id="page-15-3"></span>[19] Matthias Kümmerer and Matthias Bethge. 2021. State-of-the-art in human scanpath prediction. arXiv preprint arXiv:2102.12239 (2021).
- <span id="page-15-4"></span>[20] Matthias Kümmerer, Matthias Bethge, and Thomas SA Wallis. 2022. DeepGaze III: Modeling free-viewing human scanpaths with deep learning. Journal of Vision 22, 5 (2022), 7–7.
- <span id="page-15-21"></span>[21] Daniël Lakens. 2013. Calculating and reporting effect sizes to facilitate cumulative science: a practical primer for t-tests and ANOVAs. Front. Psychol. 4, 863 (2013).
- <span id="page-15-20"></span>[22] Luis A. Leiva, Yunfei Xue, Avya Bansal, Hamed R. Tavakoli, Tuğçe Köroğlu, Jingzhou Du, Niraj R. Dayama, and Antti Oulasvirta. 2020. Understanding Visual Saliency in Mobile User Interfaces. In Proceedings of the Intl. Conf. on Human-computer interaction with mobile devices and services (MobileHCI).
- <span id="page-15-12"></span>[23] Mu Li, Kanglong Fan, and Kede Ma. 2023. Scanpath Prediction in Panoramic Videos via Expected Code Length Minimization. arXiv preprint arXiv:2305.02536 (2023).
- <span id="page-15-9"></span>[24] Yue Li, Dong Liu, Houqiang Li, Li Li, Zhu Li, and Feng Wu. 2018. Learning a convolutional neural network for image compact-resolution. IEEE Transactions on Image Processing 28, 3 (2018), 1092–1107.
- <span id="page-15-24"></span>[25] Daniel Martin, Ana Serrano, Alexander W Bergman, Gordon Wetzstein, and Belen Masia. 2022. ScanGAN360: A generative model of realistic scanpaths for 360 images. IEEE Transactions on Visualization and Computer Graphics 28, 5 (2022), 2003–2013.
- <span id="page-15-17"></span>[26] Sebastiaan Mathôt, Filipe Cristino, Iain D Gilchrist, and Jan Theeuwes. 2012. A simple way to estimate similarity between pairs of eye movement sequences. Journal of Eye Movement Research 5, 1 (2012), 1–15.
- <span id="page-15-18"></span>[27] Meinard Müller. 2007. Dynamic time warping. Information retrieval for music and motion (2007), 69–84.
- <span id="page-15-13"></span>[28] Thuyen Ngo and BS Manjunath. 2017. Saccade gaze prediction using a recurrent neural network. In 2017 IEEE International Conference on Image Processing (ICIP). IEEE, 3435–3439.
- <span id="page-15-10"></span>[29] Gaurav Parmar, Richard Zhang, and Jun-Yan Zhu. 2022. On aliased resizing and surprising subtleties in GAN evaluation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 11410–11420.
- <span id="page-15-16"></span>[30] Michael I Posner, Yoav Cohen, et al. 1984. Components of visual orienting. Attention and performance X: Control of language processes 32 (1984), 531–556.
- <span id="page-15-25"></span>[31] Xiangjie Sui, Yuming Fang, Hanwei Zhu, Shiqi Wang, and Zhou Wang. 2023. ScanDMM: A Deep Markov Model of Scanpath Prediction for 360° Images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 6989–6999.
- <span id="page-15-11"></span>[32] Wanjie Sun, Zhenzhong Chen, and Feng Wu. 2019. Visual scanpath prediction using IOR-ROI recurrent mixture density network. IEEE transactions on pattern analysis and machine intelligence 43, 6 (2019), 2101–2118.
- <span id="page-15-1"></span>[33] Nađa Terzimehić, Renate Häuslschmid, Heinrich Hussmann, and MC Schraefel. 2019. A review & analysis of mindfulness research in HCI: Framing current lines of research and future opportunities. In Proceedings of the 2019 CHI Conference on Human Factors in Computing Systems. 1–13.
- <span id="page-15-19"></span>[34] Xiaowei Wang, Xubo Li, Haiying Wang, Wenning Zhao, and Xia Liu. 2023. An Improved Dynamic Time Warping Method Combining Distance Density Clustering for Eye Movement Analysis. Journal of Mechanics in Medicine and Biology 23, 02 (2023), 2350031.
- <span id="page-15-23"></span>[35] Yao Wang, Andreas Bulling, et al. 2023. Scanpath prediction on information visualisations. IEEE Transactions on Visualization and Computer Graphics (2023).
- <span id="page-15-14"></span>[36] Calden Wloka, Iuliia Kotseruba, and John K Tsotsos. 2018. Active fixation control to predict saccade sequences. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 3184–3193.
- <span id="page-15-15"></span>[37] Chen Xia, Junwei Han, Fei Qi, and Guangming Shi. 2019. Predicting human saccadic scanpaths based on iterative representation learning. IEEE Transactions on Image Processing 28, 7 (2019), 3502–3515.
- <span id="page-15-22"></span>[38] Mulong Xie, Sidong Feng, Zhenchang Xing, Jieshan Chen, and Chunyang Chen. 2020. UIED: a hybrid tool for GUI element detection. In Proceedings of the 28th ACM Joint Meeting on European Software Engineering Conference and Symposium on the Foundations of Software Engineering. 1655–1659.
- <span id="page-15-6"></span>[39] Fei Yan, Cheng Chen, Peng Xiao, Siyu Qi, Zhiliang Wang, and Ruoxiu Xiao. 2021. Review of visual saliency prediction: Development process from neurobiological basis to deep models. Applied Sciences 12, 1 (2021), 309.
- <span id="page-15-7"></span>[40] Jiawei Yang, Guangtao Zhai, and Huiyu Duan. 2019. Predicting the visual saliency of the people with VIMS. In 2019 IEEE Visual Communications and Image Processing (VCIP). IEEE, 1–4.
- <span id="page-15-2"></span>[41] Yucheng Zhu, Guangtao Zhai, Xiongkuo Min, and Jiantao Zhou. 2019. The prediction of saliency map for head and eye movements in 360 degree images. IEEE Transactions on Multimedia 22, 9 (2019), 2331–2344.
- <span id="page-15-8"></span>[42] Yucheng Zhu, Guangtao Zhai, Yiwei Yang, Huiyu Duan, Xiongkuo Min, and Xiaokang Yang. 2021. Viewing behavior supported visual saliency predictor for 360 degree videos. IEEE Transactions on Circuits and Systems for Video Technology 32, 7 (2021), 4188–4201.

Received November 2023; revised January 2024; accepted March 2024
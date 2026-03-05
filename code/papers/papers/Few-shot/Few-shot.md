# Few-shot Personalized Scanpath Prediction

<span id="page-0-0"></span>Ruoyu Xue<sup>1</sup> , Jingyi Xu<sup>1</sup> , Sounak Mondal<sup>1</sup> , Hieu Le<sup>2</sup> , Gregory Zelinsky<sup>1</sup> , Minh Hoai<sup>3</sup> , Dimitris Samaras<sup>1</sup> <sup>1</sup>Stony Brook University, USA <sup>2</sup>EPFL, Switzerland <sup>3</sup>The University of Adelaide, Australia

# Abstract

*A personalized model for scanpath prediction provides insights into the visual preferences and attention patterns of individual subjects. However, existing methods for training scanpath prediction models are data-intensive and cannot be effectively personalized to new individuals with only a few available examples. In this paper, we propose fewshot personalized scanpath prediction task (FS-PSP) and a novel method to address it, which aims to predict scanpaths for an unseen subject using minimal support data of that subject's scanpath behavior. The key to our method's adaptability is the Subject-Embedding Network (SE-Net), specifically designed to capture unique, individualized representations for each subject's scanpaths. SE-Net generates subject embeddings that effectively distinguish between subjects while minimizing variability among scanpaths from the same individual. The personalized scanpath prediction model is then conditioned on these subject embeddings to produce accurate, personalized results. Experiments on multiple eye-tracking datasets demonstrate that our method excels in FS-PSP settings and does not require any fine-tuning steps at test time. Code is available at: <https://github.com/cvlab-stonybrook/few-shot-scanpath>*

## 1. Introduction

Recent models of scanpath prediction have excelled at predicting human attention [\[9,](#page-8-0) [10,](#page-8-1) [12,](#page-8-2) [32,](#page-9-0) [33,](#page-9-1) [54,](#page-9-2) [55,](#page-9-3) [57\]](#page-10-0), which is important for applications such as autonomous driving [\[13,](#page-8-3) [34\]](#page-9-4), virtual and augmented reality [\[23,](#page-8-4) [45\]](#page-9-5), healthcare diagnostics [\[5,](#page-8-5) [43\]](#page-9-6), and information visualization [\[48\]](#page-9-7). However, these methods are trained using attention data from multiple subjects and therefore learn population-level "average" attention patterns that will fail to reflect individual differences shaped by culture, memory, and experience [\[22\]](#page-8-6). To capture these individualized attention patterns, and to avoid biases that can result from group averaging, models of personalized scanpath prediction (PSP) attempt to learn individual subject embeddings that are used to predict an individual's scanpath [\[10,](#page-8-1) [21,](#page-8-7) [59\]](#page-10-1). PSP is particularly useful for applications such as recommendation sys-

![](_page_0_Figure_7.jpeg)

Figure 1. Few-shot Personalized Scanpath Prediction (FS-PSP). Given a new subject with only a few support examples of their gaze behavior, can we adapt a base scanpath prediction model to this subject? We propose a subject-embedding extracting network, SE-Net, to achieve this personalized adaption.

tems [\[7,](#page-8-8) [40,](#page-9-8) [42\]](#page-9-9) and advertisements [\[37\]](#page-9-10) because it allows subject personality to be decoded from an individual's attention.

A limitation of existing PSP models is that they require extensive data to accurately capture individual attention patterns. For example, to train models to predict the scanpaths made by people as they are searching for objects [\[54\]](#page-9-2), it was necessary to collect the COCO-Search18 dataset of 269,760 search fixations [\[9\]](#page-8-0), an effort requiring 10 subjects to each come to a laboratory and have their eye movements recorded for 10–12 hours. To be useful in practice, PSP models must be trainable on orders of magnitude less data. We therefore introduce the task of *few-shot personalized scanpath prediction* (FS-PSP) where an individual's attention must be predicted using fixations from only a few behavioral observations, defined here as the scanpaths of people viewing ≤ 10 images. We refer to the image-scanpath pairs collected for each subject as *support samples*.

FS-PSP is challenging because individuals' patterns of attention must be captured in just a few support samples. Existing PSP methods [\[10,](#page-8-1) [21,](#page-8-7) [59\]](#page-10-1) often overfit to limited image content due to their reliance on large training datasets and a lack of efficient adaptation mechanisms. Typically, the primary goal of these methods is to jointly learn a subject embedding along with scanpath patterns to improve <span id="page-1-0"></span>performance on "*seen*" subjects, *i.e.*, ones with sufficient training scanpath data. However, subject embeddings are treated as a byproduct of scanpath prediction, limiting these models' ability to leverage knowledge from previously seen embeddings to adapt to new subjects. Consequently, overfitting on a few scenes and insufficient representation of novel individual visual patterns both cause PSP model performance to drop substantially in few-shot settings, as in the case of ISP [\[10\]](#page-8-1) with ten support samples. EyeFormer[\[21\]](#page-8-7) also requires a support set with at least 50 scanpaths to achieve stable personalized embeddings.

In this paper, we propose a flexible scanpath model that can adapt to new subject without requiring retraining or fine-tuning. This is achieved by decoupling subject embedding learning from the scanpath prediction process. First, we learn a subject embedding space that encodes personalized attention traits, then condition the scanpath prediction model on these embeddings. This separation enables robust performance in few-shot scenarios by avoiding the complexities of joint learning: the subject embedding extractor focuses exclusively on capturing the unique features of each subject's scanpath, while the scanpath prediction model only needs to learn a conditional mapping based on the corresponding subject embedding.

More specifically, we propose a Subject Embedding Network (SE-Net) to extract subject embeddings from imagescanpath pairs. During training, we use a base dataset containing a large number of images and scanpaths collected from the seen subjects. SE-Net is trained with a classification loss to distinguish between these subjects, focusing on extracting distinctive, personalized traits that support effective personalization. Additionally, we apply a contrastive loss to ensure embeddings from different scanpaths of the same subject remain similar while being distinct across different subjects. This training strategy emphasizes extracting robust, unbiased, and representative embeddings for each subject, akin to previous representation-learning-based fewshot approaches [\[20,](#page-8-9) [49,](#page-9-11) [51,](#page-9-12) [56\]](#page-10-2). We then train a personalized scanpath prediction network [\[10\]](#page-8-1) on the base dataset using these learned subject embeddings, enhancing its capacity to infer scanpaths based on the given subject embeddings. At the inference stage, we extract embeddings for unseen subjects from a few image-scanpath exemplars (support set), average them to obtain a single subject embedding, and condition the scanpath prediction model on this embedding to enable generalization to new subjects.

To demonstrate the performance of our method on FS-PSP, we predict personalized scanpaths of unseen subjects under three n-shot settings (n = 1, 5, 10) and on three datasets: OSIE [\[50\]](#page-9-13), COCO-Freeview [\[12\]](#page-8-2), and COCO-Search18 [\[54\]](#page-9-2). Our method outperforms the second-best model on the ScanMatch metric[\[14\]](#page-8-10) on these datasets by 5.9%, 7.9% and 6.0%.

# 2. Related Works

Personalied Scanpath Prediction. Scanpath prediction aims to model both bottom-up and top-down human attention by predicting sequences of fixations as observers either freely explore a scene without specific instructions[\[2,](#page-7-0) [3,](#page-7-1) [24,](#page-8-11) [45,](#page-9-5) [46,](#page-9-14) [48\]](#page-9-7), or engage in task-specific activities, such as visual question answering[\[9\]](#page-8-0), webpage browsing, searching[\[11,](#page-8-12) [32,](#page-9-0) [54,](#page-9-2) [55,](#page-9-3) [57\]](#page-10-0), and object referral[\[33\]](#page-9-1). Personalized scanpath prediction (PSP) aims to predict the scanpath of an individual subject rather than the group average. Although there are multiple works on personalized saliency prediction[\[6,](#page-8-13) [8,](#page-8-14) [25,](#page-8-15) [44,](#page-9-15) [53\]](#page-9-16), PSP is a relatively new and underexplored task, as it requires building upon robust population-level scanpath prediction models. The key challenge in personalized scanpath prediction is enabling the model to recognize which subject's scanpath it is predicting. Current approaches [\[10,](#page-8-1) [21,](#page-8-7) [59\]](#page-10-1) incorporate subject embeddings to represent different individuals. ISP [\[10\]](#page-8-1) adds three modules to existing scanpath prediction models [\[9,](#page-8-0) [32\]](#page-9-0) to encode subject embeddings at different stages within the model. EyeFormer[\[21\]](#page-8-7) designs a new scanpath prediction model that leverages reinforcement learning and uses a viewer encoder to indicate different subjects. However, ISP suffers significant performance drops when working with very limited data (N ≤ 10) for a new subject, while EyeFormer has only demonstrated its effectiveness with no fewer than 50 samples. This issue arises because over-parameterized models tend to overfit on small datasets, and must relearn subject embedding for the new subject, limiting their ability to fully utilize prior experience with previously seen subjects. To address these issues, we propose SE-Net, designed to mitigate overfitting and effectively leverage model experience.

Few-shot Learning Leveraging User Emebedding. Fewshot learning (FSL) has been widely studied in various topics, and a commonly used approach in FSL is to learn a function that maps the input to an embedding to effectively represent the prototype of different classes[\[17,](#page-8-16) [26,](#page-8-17) [41,](#page-9-17) [47,](#page-9-18) [49,](#page-9-11) [51,](#page-9-12) [52,](#page-9-19) [58\]](#page-10-3). User embedding, *i.e*. embeddings to represent person-specific patterns, encodes a user's unique behavioral patterns by mapping their actions through a network, capturing both intra-personal and inter-personal similarities and differences. This not only enables downstream tasks to effectively distinguish between individuals but, more importantly, allows for an understanding of unique user traits from the user embedding, enabling predictions that align with each person's distinctive patterns. Many topics address the few-shot problem by learning user embeddings, including recommendation systems [\[28\]](#page-8-18) and gaze estimation models [\[18,](#page-8-19) [36\]](#page-9-20). Although several works utilize scanpath for classification[\[35,](#page-9-21) [60\]](#page-10-4), to the best of our knowledge, this is the first work that aims to extract user embeddings from scanpaths—a non-trivial task requiring the <span id="page-2-0"></span>disentanglement of visual patterns driven by both bottomup and top-down attention from fixation locations, temporal sequences, and durations.

#### 3. Proposed FS-PSP Framework

This section describes our framework for FS-PSP, which consists of two major components: (1) a personalized scanpath predictor, called ISP-SENet, that predicts the scanpath for a subject, conditioned on the subject's embedding, and (2) a subject-embedding network, called SE-Net, that computes the embedding for a subject based on a small set of their gaze behaviors. In this section, we first provide a formal definition of the task and then describe the different components of our framework. The overview is shown in Fig. 2.

#### 3.1. Problem Formulation

Let d represent a scanpath data instance; it is a tuple consisting of an image and a sequence of 2D gaze fixations. Let  $S(\mathbf{d})$  denote the identity of the subject from whom the scanpath was collected. PSP considers a scenario where we have a base training set of gaze behavior with size m:  $\mathcal{D}_{base} = \{\mathbf{d}_{base}^i\}_{i=1}^m$ , which can be used to train a personalized scanpath predictor, but only for the subjects in the base training set. FS-PSP goes beyond PSP, considering the task of personalized scanpath prediction for a subject s who was not seen during training ( $\mathbf{s} \neq S(\mathbf{d}_{base}^i) \ \forall i$ ) but for whom we have some gaze behavior data on a small subset of images from the base dataset, referred to as the support set. Let  $\mathcal{D}_{supp} = \{\mathbf{d}_{supp}^i\}_{i=1}^n$  represent this support set, where  $S(\mathbf{d}_{supp}^i) = \mathbf{s} \ \forall i$ , and n is small (at most 10). The goal of FS-PSP is, given a set of unseen subjects, to predict their scanpaths on a set of query images  $\mathbf{x}^i$  with size q:  $\mathcal{D}_{query} = \{\mathbf{x}_{query}^i\}_{i=1}^q.$ 

#### 3.2. ISP-SENet – Personalized Scanpath Predictor

ISP-SENet is one of the two core components of our framework. In this paper, we propose developing it based on Gazeformer-ISP [10]. This model accounts for the scanpath behaviors of individual subjects, with each subject in the training set associated with a separate embedding vector that is learnable but fixed after training. By assigning a unique embedding vector to each subject, the model achieves significant improvement over a generic model that does not account for subject identity. However, this approach only works for subjects seen during training, as the embedding vector can be retrieved from a lookup table. For an unseen subject, there is no way to compute the embedding vector, thereby preventing prediction for new subjects.

In this work, we propose replacing the fixed embedding vector with the output of a subject-embedding network, SE-Net. SE-Net can compute the subject embedding vector for any subject, as long as a support set of gaze behavior data

from that subject is available. SE-Net represents the main technical innovation of our work, which we describe in the following section.

#### 3.3. SE-Net – Subject Embedding Network

SE-Net computes a function f that takes a scanpath (both an image and a trajectory of eye fixations with durations) as input and outputs an embedding vector. This network calculates the subject embedding given a single scanpath behavior data point. For a support set contains more than one gaze behavior data point,  $\mathcal{D}_{supp} = \{\mathbf{d}_{supp}^i\}_{i=1}^n$ , we follow the approach of prototypical networks [41] and simply take the average as the overall subject's embedding, i.e.,  $\frac{1}{n} \sum_{i=1}^{n} f(\mathbf{d}_{supp}^i)$ . We will next describe how this network is trained and then provide details about its architecture.

#### 3.3.1. Training SE-Net

SE-Net is trained using scanpath data from the base training set  $\mathcal{D}_{base}$ , starting with the creation of triplets  $(\mathbf{d}, \mathbf{d}_+, \mathbf{d}_-)$ , where  $\mathbf{d}$  is a scanpath drawn from  $\mathcal{D}_{base}$ ,  $\mathbf{d}_+$  is another scanpath randomly drawn from  $\mathcal{D}_{base}$  but from the same subject as  $\mathbf{d}$ , and  $\mathbf{d}_-$  is a scanpath randomly drawn from a different subject. The training loss for this triplet is the combination of three classification losses and a contrastive loss term:

$$\mathcal{L}_{cls}(\mathbf{d}) + \mathcal{L}_{cls}(\mathbf{d}_{+}) + \mathcal{L}_{cls}(\mathbf{d}_{-}) + \mathcal{L}_{contrast}.$$
 (1)

The classification loss  $\mathcal{L}_{cls}$  is computed separately for each of d, d+, and  $d_-$ . It is defined as classifying the subject given their scanpath on an image by adding a classification head to the subject embedding layer. The contrastive loss  $\mathcal{L}_{contrast}$  is based on triplet loss [39]:

$$\max (\|f(\mathbf{d}) - f(\mathbf{d}_{+})\|^{2} - \|f(\mathbf{d}) - f(\mathbf{d}_{-})\|^{2} + m, 0).$$
(2)

The margin m controls the distance between subject embeddings. A smaller margin allows scanpaths from different subjects to be similar, typically observed in top-down attention scenarios. Conversely, a larger margin is preferable in scenarios like free-viewing, characterized by significant diversity among subjects. Refer to the supplementary for more detailed explanations.

#### 3.3.2. Network architecture

**Feature extraction.** Our semantic feature extractor F follows the design of HAT [57]. We encode images and scanpaths using hierarchical feature maps from an image encoder (ResNet [19]) and decoder (Deformable attention [61]) and obtain image tokens  $F_I \in \mathbb{R}^{\left(\frac{H}{32} \cdot \frac{W}{32}\right) \times C}$  and scanpath tokens  $F_S \in \mathbb{R}^{L \times c}$  (H,W are the height and width of image, L is the max length of scanpaths, c is embedding dimension). The details are described in the supplementary material.

<span id="page-3-3"></span><span id="page-3-0"></span>![](_page_3_Figure_0.jpeg)

Figure 2. Overview of ISP-SENet: Our method for few-shot personalized scanpath prediction has two stages. In the training stage, we train two models on a large amount of image-scanpath pairs  $\mathcal{D}_{base}$ , corresponding to a set of seen subjects. Initially, we train the Subject Embedding Network (SE-Net) to obtain embeddings for seen subjects, followed by training ISP-SENet to predict scanpaths using these embeddings. In the inference phase, both models are frozen, and we extract embeddings for unseen subjects from the support set,  $\mathcal{D}_{supp}$ , which consists of n-shot images sampled from the base set. These unseen subject embeddings then guide ISP-SENet in predicting scanpaths for unseen subjects using the query set,  $\mathcal{D}_{query}$ , which includes a collection of unseen images.

![](_page_3_Figure_2.jpeg)

Figure 3. **Structure of SE-Net.** SE-Net employs a feature extractor F to derive image and scanpath semantic features,  $F_I$  and  $F_S$ , respectively. The CSE module then processes task and duration features, updating the scanpath embedding constrained by all extracted features. An initialized embedding learns human attention information from  $F_S'$  to produce the subject embedding e. This triplet network assesses the distances among  $e^{d_+}$ ,  $e^{d_+}$ ,  $e^{d_-}$ , and the UP module predicts the subject ID. All CSE modules share the same weights, as do the USD and UP modules.

Context-Scanpath Encoder(CSE). Human attention can be divided into bottom-up and top-down processes, either freely viewing the image or viewing it with a specific task, such as searching for a target. Different tasks influence attention toward specific objects mentioned in the task and related objects. To make SE-Net task-aware, we design an multi-modal visual-task encoder for SE-Net, which takes task and image tokens as input, highlighting the image content relevant to the task. The task is first mapped to a task embedding  $t \in \mathbb{R}^c$  by a language model (RoBERTa [30]), then the visual-task encoder is designed as Eq. (3):

$$C = SelfAttn(\{t \cdot W, F_I\}). \tag{3}$$

Here, C represents the context embedding, capturing information from both the image and the task. W is a linear layer.  $\{,\}$  denotes concatenation along the sequence dimension, and SelfAttn is multiple self-attention layers.

To obtain the final scanpath embedding, we first ensure the model is aware of each fixation's duration. Position information reflects where the subject tends to focus, while duration indicates the relative interest in specific objects. We

use 2D sinusoidal position embeddings [29, 57] to encode the position information and 1D positional embeddings for durations. Given that durations range from 0 to 5000 ms, using 5000 different positional encodings would be redundant, as small differences (*e.g.*, 200 ms vs. 203 ms) are often negligible. Therefore, we collected all durations in the base set and uniformly grouped them into 10 bins, replacing the actual duration with its corresponding bin ID. This replacement is only used in SE-Net, while ISP-SENet still predicts the raw duration. Find details in the supplementary.

<span id="page-3-1"></span>Combining the context embedding C, duration embeddings T, and position embeddings Pos, we update the scanpath tokens  $F_S'$  representing viewing behavior refined by image content and task as shown in Eq. (4), w/o C denotes that tokens associated with C are discarded, Isolating image content from subject embeddings to reduce scene-specific bias.

<span id="page-3-2"></span>
$$F'_{S} = SelfAttn(\{C, F_{S} + \mathcal{T} + Pos\})_{w/o C}$$
 (4)

User-Scanpath Decoder (USD). The main focus of USD is to capture subject attention traits from the scanpath embedding. We initialize a subject token  $e \in \mathbb{R}^c$  and use a cross-

<span id="page-4-1"></span>attention mechanism to extract the subject embedding from the scanpath embedding. Here, the subject token serves as the query, enabling it to attend to the scanpath embedding (used as key and value) and extract specific patterns that reflect the subject's unique attention characteristics. The detailed implementation is shown in Eq. [\(5\)](#page-4-0):

<span id="page-4-0"></span>
$$e = \text{ReLU}(\text{Linear}(e + \text{CrossAttn}(e, F_S'))). \tag{5}$$

To enable the updated subject embedding to distinguish between different subjects, we add a subject Predictor (UP), a series of linear layers with ReLU activation, to classify the subject ID:

Subject ID = Linear(ReLU(Linear(
$$e$$
)) (6)

# 4. Experiment

### 4.1. Dataset Setting

Dataset. We train and evaluate ISP-SENet on three datasets: OSIE [\[50\]](#page-9-13), COCO-Freeview [\[12\]](#page-8-2), and COCO-Search18 [\[54\]](#page-9-2) (target-present). OSIE collects scanpaths under a free-viewing condition from 15 subjects on 700 images. COCO-Freeview collects scanpaths under a freeviewing condition from 10 subjects on 6202 images. COCO-Search18 has same images as COCO-FreeView, but the scanpaths are collected under a search task on 18 categories. We follow the same data split of OSIE as ISP [\[10\]](#page-8-1), and the same split of COCO-Search18 and COCO-FreeView as HAT[\[57\]](#page-10-0).

Unseen subject selection. For each dataset, we split the full subject set into around 70% seen and 30% unseen. We repeat experiments twice with different splits, and the result of another split is shown in the supplementary.

n-shot sampling. We selected three values, n = 1, 5, 10, as criteria for choosing the support set. For each value of n, we randomly sample n image-scanpath pairs from unseen subjects within the training set and assessed the model's performance across the entire test set of unseen subjects. This process was repeated 10 times with various support sets to ensure robustness, and the average performance metrics were reported. Find margin of error in the supplementary.

## 4.2. Experiment Setting

Baselines. We use ChenLSTM-ISP and Gazeformer-ISP[\[10\]](#page-8-1) as baselines designed for personalized scanpath prediction. Both models are pretrained on a base set and fully fine-tuned on a support set to jointly learn subject embeddings and scanpaths. Since new subject embeddings are not available initially, fine-tuning is required. To reduce overfitting, we also implement ChenLSTM-ISP-S and Gazeformer-ISP-S, which only fine-tune the subject embedding layer and freeze all other parameters.

Metrics. We evaluated ISP-SENet using value-based metrics as outlined by ISP [\[10\]](#page-8-1), including ScanMatch (SM), MultiMatch (MM), and String-Edit Distance (SED), assess the similarity between predicted and actual scanpaths. ScanMatch [\[14\]](#page-8-10) evaluates the overall similarity, Multi-Match [\[1,](#page-7-2) [16\]](#page-8-21) analyzes five dimensions of scanpath similarity: shape, direction, length, position, and duration, while String-Edit Distance [\[4\]](#page-8-22) identifies structural differences.

Implementation details. For SE-Net, the number of transformer layers in visual-task encoder and self-attention of CSE, and the number of USD layers are set to 3. The embedding dimension is set to 384. We use AdamW [\[31\]](#page-9-25) with learning rate 0.0001. We train SE-Net for 25 epochs with a batch size of 16. The max length of scanpaths in OSIE and COCO-FreeView is set to 20, and the max length in COCO-Search18 is set to 10. For ISP-SENet, adhering to the GazeformerISP [\[10\]](#page-8-1) configuration, we matched the maximum scanpath lengths and embedding dimensions with SE-Net, retaining other hyperparameters. The fine-tuning on support set takes 10 epochs for supervised training and 10 epochs for self-critical sequence training (SCST) [\[9\]](#page-8-0).

# 4.3. Main Results

We compared ISP-SENet with four baselines across three datasets as shown in Tab. [1.](#page-5-0) The consistent performance across n = 1, 5, 10 indicates ISP-SENet's effectiveness in minimizing the impact of individual image instances, allowing embeddings to focus on attention patterns rather than being biased by the support set.

Results on OSIE. ISP-SENet outperforms the second-best approach by 5.9% and 2.5% in SM and SED, respectively. It maintains stable performance in the 5-shot and 10-shot settings and achieves a competitive SM score of 0.368 with just one image-scanpath pair.

Results on COCO-FreeView. Both ISP methods exhibit severe overfitting, particularly as COCO-FreeView is a larger dataset with more complex scenes and scanpaths. Nonetheless, ISP-SENet shows substantial enhancements in all settings, outperforming the baseline by 7.9% and 4.5% in SM and SED.

Results on COCO-Search18. We achieved 6.0% and 2.5% on SM and SED compared with the second-best. The consistent performance of COCO-Search18 highlights ISP-SENet's capability in capturing varied attention patterns across different search tasks. This success is attributed to the visual-task encoder, detailed further in Sec. [5.](#page-6-0)

We observed that in the fine-tuning stage of Gazeformer-ISP-S and ChenLSTM-ISP-S, the unseen subject embeddings change minimally from seen subject embeddings. This suggests that simple learnable embeddings (instead

<span id="page-5-3"></span><span id="page-5-0"></span>

|        |                        |       | OSIE            |       |       | COCO-FreeView   |              | COCO-Search18   |       |
|--------|------------------------|-------|-----------------|-------|-------|-----------------|--------------|-----------------|-------|
| n-shot | Method                 |       | SM ↑ MM ↑ SED ↓ |       |       | SM ↑ MM ↑ SED ↓ |              | SM ↑ MM ↑ SED ↓ |       |
|        | ChenLSTM-ISP           | 0.282 | 0.763           | 7.832 | 0.287 | 0.805           | 13.307 0.371 | 0.760           | 2.756 |
| n = 1  | Gazeformer-ISP         | 0.327 | 0.792           | 7.873 | 0.244 | 0.787           | 15.118 0.342 | 0.770           | 2.818 |
|        | ChenLSTM-ISP-S 0.328   |       | 0.793           | 7.601 | 0.339 | 0.814           | 12.523 0.448 | 0.803           | 2.394 |
|        | Gazeformer-ISP-S 0.354 |       | 0.801           | 7.503 | 0.333 | 0.817           | 12.538 0.446 | 0.802           | 2.463 |
|        | ISP-SENet              | 0.368 | 0.805           | 7.413 | 0.369 | 0.832           | 12.227 0.475 | 0.814           | 2.333 |
| n = 5  | ChenLSTM-ISP           | 0.319 | 0.773           | 7.855 | 0.320 | 0.815           | 12.950 0.386 | 0.773           | 2.489 |
|        | Gazeformer-ISP         | 0.340 | 0.791           | 7.920 | 0.286 | 0.800           | 14.630 0.353 | 0.774           | 2.980 |
|        | ChenLSTM-ISP-S 0.329   |       | 0.791           | 7.649 | 0.338 | 0.814           | 12.540 0.449 | 0.803           | 2.380 |
|        | Gazeformer-ISP-S 0.354 |       | 0.801           | 7.499 | 0.333 | 0.817           | 12.539 0.445 | 0.803           | 2.457 |
|        | ISP-SENet              | 0.376 | 0.803           | 7.337 | 0.368 | 0.829           | 12.017 0.484 | 0.815           | 2.354 |
| n = 10 | ChenLSTM-ISP           | 0.322 | 0.777           | 7.740 | 0.323 | 0.819           | 12.541 0.393 | 0.781           | 2.394 |
|        | Gazeformer-ISP         | 0.345 | 0.794           | 7.916 | 0.317 | 0.805           | 14.224 0.370 | 0.785           | 2.765 |
|        | ChenLSTM-ISP-S 0.328   |       | 0.791           | 7.637 | 0.340 | 0.814           | 12.532 0.449 | 0.803           | 2.379 |
|        | Gazeformer-ISP-S 0.354 |       | 0.802           | 7.505 | 0.333 | 0.816           | 12.545 0.446 | 0.802           | 2.464 |
|        | ISP-SENet              | 0.375 | 0.803           | 7.318 | 0.367 | 0.828           | 11.956 0.482 | 0.815           | 2.359 |

Table 1. Performance Comparison on Different Datasets under different few-shot settings for FS-PSP.

of SE-Net) lack adaptability to unseen subjects. Their relatively better performance compared with fine-tuning all parameters likely stems from the similarity between seen and unseen subjects.

Adaptation time. A significant achievement of ISP-SENet is its rapid adaptation time to new subjects. Adaptation time refers to the duration required to update the scanpath prediction model, enabling it to infer the scanpaths of new subjects. For baseline methods, adaptation time is equivalent to the duration of fine-tuning on a support set, which is 161 and 267 seconds for ChenLSTM-ISP and Gazeformer-ISP respectively. In contrast, ISP-SENet's adaptation time is simply the time taken to obtain subject embeddings from SE-Net, which is 3.62 seconds. ISP-SENet achieves nearly real-time adaptation, enhancing efficiency and making the method more practical for real-world applications.

<span id="page-5-1"></span>

| Dataset                                             | ISP-SENet-Seen |                                 | ISP-SENet-Unseen  |  |       |  |
|-----------------------------------------------------|----------------|---------------------------------|-------------------|--|-------|--|
|                                                     |                | SM ↑ MM ↑ SED ↓ SM ↑ MM ↑ SED ↓ |                   |  |       |  |
| OSIE                                                | 0.390 0.812    |                                 | 7.309 0.375 0.803 |  | 7.318 |  |
| COCO-FreeView 0.401 0.841 11.581 0.367 0.828 11.956 |                |                                 |                   |  |       |  |
| COCO-Search18 0.492 0.815                           |                |                                 | 2.156 0.482 0.815 |  | 2.359 |  |

Table 2. Performance Comparison. Our method fully trained on all subjects (ISP-SENet-Seen), and trained on seen subjects and adapt to unseen subjects under 10-shot setting (ISP-SENet-unseen).

To establish the upper bound of prediction performance on unseen subjects (ISP-SENet-Unseen), we fully trained ISP-SENet using all available training data and evaluated it on the same subjects designated as unseen in the fewshot setting. The results are presented in Tab. [2.](#page-5-1) Remarkably, even without training on the unseen subjects, ISP- SENet's performance approaches the upper bound on OSIE and COCO-Search18. While ISP-SENet-Unseen underperforms compared to ISP-SENet-Seen on COCO-FreeView, it is trained on a considerably larger dataset of 43,143 scanpaths featuring more complex scenes and patterns. Even in more complex datasets, ISP-SENet can extract meaningful personalized embeddings from unseen subjects.

We employed scanpath accuracy, same as R@1 implemented in [\[10\]](#page-8-1), to measure the accuracy of the predicted scanpath for a subject by determining if it is the most similar to the ground truth among all predictions made on the same image across different subjects, reflecting its ability to distinguish subjects. Distinguishing unseen from seen subjects is challenging due to the dataset's limited subject pool, potentially biasing the model towards known subjects. Nevertheless, our method showed considerable improvement in Tab. [3,](#page-5-2) illustrating that our unseen subject embeddings effectively discern among subjects despite the limited training subjects.

<span id="page-5-2"></span>

|                  |       | COCO-FreeView COCO-Search18 OSIE |       |
|------------------|-------|----------------------------------|-------|
| ChenLSTM-ISP     | 33.66 | 34.28                            | 21.14 |
| Gazeformer-ISP   | 31.99 | 31.73                            | 21.36 |
| ChenLSTM-ISP-S   | 31.95 | 33.17                            | 20.17 |
| Gazeformer-ISP-S | 31.87 | 33.53                            | 18.85 |
| ISP-SENet        | 35.57 | 35.25                            | 22.85 |

Table 3. Scanpath accuracy (higher is better) shows the model's ability to distinguish predicted scanpaths from different subjects.

Qualitative results. In Fig. [4,](#page-7-3) we showcase ISP-SENet's distinct scanpath predictions for the same query image across various unseen subjects, highlighting its ability to capture individual attention patterns. GT 1 and GT 2 represent ground truth scanpaths for different subjects. The first row shows ISP-SENet discerning varying fixation orders across objects. The second row demonstrates how ISP-SENet captures the diverse attention distributions: Subject 1 conducts a thorough scene scan, whereas Subject 2 focuses more centrally. The third row reveals Subject 2's distraction by peripheral objects during the search. These observations confirm ISP-SENet's effectiveness in identifying unique attention traits among unseen subjects for personalized scanpath prediction. Find more results in supplementary.

#### <span id="page-6-0"></span>5. Analysis

This section examines the impact of the number of seen subjects on performance, and the interpretability of the model.

<span id="page-6-1"></span>

| num seen     |       | Subject               | 1     | Subject 2 |                       |       |  |
|--------------|-------|-----------------------|-------|-----------|-----------------------|-------|--|
| 114111 50011 | SM ↑  | $\mathbf{MM}\uparrow$ | SED↓  | SM ↑      | $\mathbf{MM}\uparrow$ | SED↓  |  |
| 3 (20%)      | 0.339 | 0.806                 | 7.871 | 0.354     | 0.781                 | 6.740 |  |
| 7 (50%)      | 0.365 | 0.812                 | 7.457 | 0.364     | 0.790                 | 6.522 |  |
| 10 (67%)     | 0.374 | 0.814                 | 7.420 | 0.385     | 0.794                 | 6.460 |  |
| 13 (93%)     | 0.370 | 0.815                 | 7.504 | 0.387     | 0.793                 | 6.482 |  |

Table 4. Performance Comparison of 10-shot setting on OSIE with different numbers of seen subjects in training stage. The table shows the performance for unseen subjects 1 and 2 with varying numbers of seen subjects.

**Size of the base training set.** Intuitively, training a model with a greater number of subjects enables it to learn a broader range of attention patterns. However, if trained on only a few subjects, the model might struggle with novel subjects whose scanpaths significantly deviate from those of seen subjects, thereby limiting its prediction accuracy. We assessed scanpath prediction performance for unseen subjects with varying numbers of training subjects, as detailed in Tab. 4. We consistently treated two subjects (Subject 1 and 2 in the table) as unseen, sampling the support set (n = 10) five times for each and averaging the results. Findings indicate that few-shot performance improves with an increasing number of seen subjects, although the difference between 10 and 13 subjects is minimal, likely due to performance saturation because of the potential similarity between some of OSIE dataset's subjects. See supplementary for COCO-Search18 results.

# Interpretability – which fixations reflect attention trait? SE-Net also functions as a tool for quantitatively analyzing individual visual patterns. By examining the attention weights from the last cross-attention layer of the USD module, we can identify which fixation tokens the subject embeddings consider most critical. We illustrate this analysis in Fig. 5, showcasing the two fixations with the highest attention weights for each subject. For example, in the first row, the highlighted fixations focus on a kid, keyboard, and

cables, which are key to distinguishing the three subjects' attention distributions. The second row captures the initial focus of three individuals on a stop sign, warning tag, and one-way tag in a traffic setting, hinting at SE-Net's potential to further study varied driving behaviors across different subjects.

Role of visual-task encoder. COCO-Search18 features 18 search tasks, with participants typically focusing their final fixations on the search target in each image and task. The visual-task encoder informs the model about task-relevant image areas, providing prior knowledge of the search target for the CSE module to adjust its scanpath embedding. Although not tailored for object detection, the task embeddings consistently direct attention to target areas, as shown in the visualizations in Fig. 6. See supplementary for quantitative results.

<span id="page-6-2"></span>

| loss         |       | OSIE  |       | COCO-Search18 |       |       |  |  |
|--------------|-------|-------|-------|---------------|-------|-------|--|--|
| 1033         | SM↑   | MM ↑  | SED↓  | SM↑           | MM ↑  | SED↓  |  |  |
| cls          | 0.361 | 0.801 | 7.546 | 0.448         | 0.812 | 2.621 |  |  |
| contrast     | 0.372 | 0.795 | 7.389 | 0.442         | 0.812 | 2.400 |  |  |
| cls+contrast | 0.375 | 0.803 | 7.318 | 0.482         | 0.815 | 2.359 |  |  |

Table 5. Ablation on the effect of different losses. cls is solely using classification loss, contrast is solely using contrastive loss, cls+contrast combines both loss, consistent with the settings used in all main experiments.

Ablation study on different losses. To evaluate the impact of different losses, we train SE-Net with classification loss only, contrastive loss only, and their combination (as in the main experiments) on OSIE and COCO-Search18. Results in Tab. 5 show that contrastive loss is crucial for capturing subject-specific attention patterns, while classification loss significantly accelerates convergence. On COCO-Search18, combining classification and triplet loss is more effective due to the similarity of attention patterns across subjects. Classification loss alone overlooks these similarities, while triplet loss struggles to find subtle differences.

Ablation study on different modules. To ablate the effect of different modules in SE-Net, we train SE-Net under two settings on COCO-Search18. 1) w/o task encoder: we remove task encoder, and direct pass task embedding extracted from text encoder to self-attention in CSE module. 2) w/o duration: do not use duration to update scanpath embedding. Results shown in Tab. 6 indicate the importance of each module in inferring unseen subject embeddings.

#### **6. Conclusions and Discussion**

We highlighted the significance of generating personalized scanpath predictions for novel subjects with minimal support samples and introduced the Few-Shot Personalized Scanpath Prediction (FS-PSP) task. To tackle this challenge, we created a pipeline that independently learns

<span id="page-7-3"></span>![](_page_7_Figure_0.jpeg)

Figure 4. Qualitative examples of scanpath prediction for different unseen subjects. GT is the ground truth scanpaths of different unseen subjects. Red circle is the end fixation. In the third row, the subject is searching for "bowl". The results indicates our method is able to capture the temporal order of fixations, fixation distributions, and distractions, while baseline keeps predicting similar scanpaths of different subjects.

<span id="page-7-4"></span>![](_page_7_Figure_2.jpeg)

Figure 5. **Model interpretability.** By analyzing a large dataset of seen subject-scanpath pairs, SE-Net determines the most influential fixations in shaping unseen subject embeddings. Fixations highlighted in blue represent the two with the highest weights. This analysis demonstrates that ISP-SENet can effectively identify which fixations are crucial for distinguishing between subjects.

<span id="page-7-5"></span>![](_page_7_Figure_4.jpeg)

Figure 6. Correspondence between task and image. The visualization highlights the areas of the image that the visual-task encoder focuses on. The first and third columns display the attention weights extracted from the visual-task encoder, and the second and fourth columns show the ground truth scanpath visualizations from the support set. Fixations marked in red represent the last fixation, indicating the search target's location. This indicates the visual-task encoder's effectiveness in identifying task-relevant information within the image.

subject embeddings to capture individual attention patterns and uses these embeddings to predict scanpaths. SE-Net is engineered to extract subject embeddings, separate individual attention traits from the image content, and facilitate generalization to unseen subjects with very few examples,

<span id="page-7-6"></span>

| Module           | SM↑   | ММ↑   | SED↓  |
|------------------|-------|-------|-------|
| w/o Task Encoder | 0.459 | 0.814 | 2.459 |
| w/o Duration     | 0.446 | 0.814 | 2.539 |
| ISP-SENet        | 0.482 | 0.815 | 2.359 |

Table 6. Ablation on the effect of different modules of SE-Net on COCO-Search18.

thereby minimizing biases linked to restricted scenes. By utilizing these robust unseen subject embeddings, our scanpath prediction model markedly outperforms methods that require relearning personalization for new subjects with limited data. Furthermore, we have set a benchmark for FS-PSP, encouraging additional research to develop more comprehensive subject embeddings that are effective across diverse eye-tracking tasks and scenarios.

A limitation of SE-Net is, While it distinguishes subjects, it may overemphasize unique traits while overlooking shared patterns beneficial for scanpath prediction. Addressing this could enhance the robustness and generalizability of the method.

**Acknowledgements.** This project is supported by US National Science Foundation grant IIS-2123920.

#### References

- <span id="page-7-2"></span>[1] Nicola C Anderson, Fraser Anderson, Alan Kingstone, and Walter F Bischof. A comparison of scanpath comparison methods. *Behavior research methods*, 47:1377–1392, 2015.
- <span id="page-7-0"></span>[2] Marc Assens, Xavier Giro-i Nieto, Kevin McGuinness, and Noel E O'Connor. Pathgan: Visual scanpath prediction with generative adversarial networks. In *Proceedings of the Eu*ropean Conference on Computer Vision (ECCV) Workshops, pages 0–0, 2018. 2
- <span id="page-7-1"></span>[3] Marc Assens Reina, Xavier Giro-i Nieto, Kevin McGuin-

- ness, and Noel E O'Connor. Saltinet: Scan-path prediction on 360 degree images using saliency volumes. In *Proceedings of the IEEE International Conference on Computer Vision Workshops*, pages 2331–2338, 2017. [2](#page-1-0)
- <span id="page-8-22"></span>[4] Stephan A Brandt and Lawrence W Stark. Spontaneous eye movements during visual imagery reflect the content of the visual scene. *Journal of cognitive neuroscience*, 9(1):27–38, 1997. [5](#page-4-1)
- <span id="page-8-5"></span>[5] Romuald Carette, Mahmoud Elbattah, Federica Cilia, Gilles Dequen, Jean-Luc Guerin, and Jer´ ome Bosche. Learning to ˆ predict autism spectrum disorder based on the visual patterns of eye-tracking scanpaths. In *HEALTHINF*, pages 103–112, 2019. [1](#page-0-0)
- <span id="page-8-13"></span>[6] Zhuoqing Chang, J Matias Di Martino, Qiang Qiu, Steven Espinosa, and Guillermo Sapiro. Salgaze: Personalizing gaze estimation using visual saliency. In *Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops*, pages 0–0, 2019. [2](#page-1-0)
- <span id="page-8-8"></span>[7] Li Chen, Wanling Cai, Dongning Yan, and Shlomo Berkovsky. Eye-tracking-based personality prediction with recommendation interfaces. *User Modeling and User-Adapted Interaction*, 33(1):121–157, 2023. [1](#page-0-0)
- <span id="page-8-14"></span>[8] Shi Chen, Nachiappan Valliappan, Shaolei Shen, Xinyu Ye, Kai Kohlhoff, and Junfeng He. Learning from unique perspectives: User-aware saliency modeling. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 2701–2710, 2023. [2](#page-1-0)
- <span id="page-8-0"></span>[9] Xianyu Chen, Ming Jiang, and Qi Zhao. Predicting human scanpaths in visual question answering. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 10876–10885, 2021. [1,](#page-0-0) [2,](#page-1-0) [5](#page-4-1)
- <span id="page-8-1"></span>[10] Xianyu Chen, Ming Jiang, and Qi Zhao. Beyond average: Individualized visual scanpath prediction. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 25420–25431, 2024. [1,](#page-0-0) [2,](#page-1-0) [3,](#page-2-0) [5,](#page-4-1) [6,](#page-5-3) [4](#page-3-3)
- <span id="page-8-12"></span>[11] Yupei Chen, Zhibo Yang, Seoyoung Ahn, Dimitris Samaras, Minh Hoai, and Gregory Zelinsky. Coco-search18 fixation dataset for predicting goal-directed attention control. *Scientific reports*, 11(1):8776, 2021. [2](#page-1-0)
- <span id="page-8-2"></span>[12] Yupei Chen, Zhibo Yang, Souradeep Chakraborty, Sounak Mondal, Seoyoung Ahn, Dimitris Samaras, Minh Hoai, and Gregory Zelinsky. Characterizing target-absent human attention. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 5031–5040, 2022. [1,](#page-0-0) [2,](#page-1-0) [5](#page-4-1)
- <span id="page-8-3"></span>[13] Jiyong Chung, Hyeokmin Lee, Hosang Moon, and Eunghyuk Lee. The static and dynamic analyses of drivers' gaze movement using vr driving simulator. *Applied Sciences*, 12(5):2362, 2022. [1](#page-0-0)
- <span id="page-8-10"></span>[14] Filipe Cristino, Sebastiaan Mathot, Jan Theeuwes, and ˆ Iain D Gilchrist. Scanmatch: A novel method for comparing fixation sequences. *Behavior research methods*, 42:692–700, 2010. [2,](#page-1-0) [5](#page-4-1)
- <span id="page-8-23"></span>[15] Christine A Curcio, Kenneth R Sloan, Robert E Kalina, and Anita E Hendrickson. Human photoreceptor topography. *Journal of comparative neurology*, 292(4):497–523, 1990. [1](#page-0-0)

- <span id="page-8-21"></span>[16] Richard Dewhurst, Marcus Nystrom, Halszka Jarodzka, Tom ¨ Foulsham, Roger Johansson, and Kenneth Holmqvist. It depends on how you look at it: Scanpath comparison in multiple dimensions with multimatch, a vector-based approach. *Behavior research methods*, 44:1079–1100, 2012. [5](#page-4-1)
- <span id="page-8-16"></span>[17] Guangxing Han, Shiyuan Huang, Jiawei Ma, Yicheng He, and Shih-Fu Chang. Meta faster r-cnn: Towards accurate few-shot object detection with attentive feature alignment. In *Proceedings of the AAAI Conference on Artificial Intelligence*, pages 780–789, 2022. [2](#page-1-0)
- <span id="page-8-19"></span>[18] Junfeng He, Khoi Pham, Nachiappan Valliappan, Pingmei Xu, Chase Roberts, Dmitry Lagun, and Vidhya Navalpakkam. On-device few-shot personalization for real-time gaze estimation. In *Proceedings of the IEEE/CVF international conference on computer vision workshops*, pages 0–0, 2019. [2](#page-1-0)
- <span id="page-8-20"></span>[19] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 770–778, 2016. [3,](#page-2-0) [1](#page-0-0)
- <span id="page-8-9"></span>[20] Yiren Jian, Chongyang Gao, and Soroush Vosoughi. Contrastive learning for prompt-based few-shot language learners. In *Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, pages 5577–5587, Seattle, United States, 2022. Association for Computational Linguistics. [2](#page-1-0)
- <span id="page-8-7"></span>[21] Yue Jiang, Zixin Guo, Hamed Rezazadegan Tavakoli, Luis A Leiva, and Antti Oulasvirta. Eyeformer: predicting personalized scanpaths with transformer-guided reinforcement learning. In *Proceedings of the 37th Annual ACM Symposium on User Interface Software and Technology*, pages 1–15, 2024. [1,](#page-0-0) [2](#page-1-0)
- <span id="page-8-6"></span>[22] Tilke Judd, Fredo Durand, and Antonio Torralba. A bench- ´ mark of computational models of saliency to predict human fixations. 2012. [1](#page-0-0)
- <span id="page-8-4"></span>[23] Sebastian Kapp, Michael Barz, Sergey Mukhametov, Daniel Sonntag, and Jochen Kuhn. Arett: Augmented reality eye tracking toolkit for head mounted displays. *Sensors*, 21(6): 2234, 2021. [1](#page-0-0)
- <span id="page-8-11"></span>[24] Matthias Kummerer, Matthias Bethge, and Thomas SA Wal- ¨ lis. Deepgaze iii: Modeling free-viewing human scanpaths with deep learning. *Journal of Vision*, 22(5):7–7, 2022. [2](#page-1-0)
- <span id="page-8-15"></span>[25] Aoqi Li and Zhenzhong Chen. Personalized visual saliency: Individuality affects image perception. *IEEE Access*, 6: 16099–16109, 2018. [2](#page-1-0)
- <span id="page-8-17"></span>[26] Gen Li, Varun Jampani, Laura Sevilla-Lara, Deqing Sun, Jonghyun Kim, and Joongkyu Kim. Adaptive prototype learning and allocation for few-shot segmentation. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 8334–8343, 2021. [2](#page-1-0)
- <span id="page-8-24"></span>[27] Mengtang Li, Jie Zhu, Zhixin Huang, and Chao Gou. Imitating the human visual system for scanpath predicting. In *ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, pages 3745–3749. IEEE, 2024. [1](#page-0-0)
- <span id="page-8-18"></span>[28] Ruirui Li, Xian Wu, Xian Wu, and Wei Wang. Few-shot learning for new user recommendation in location-based so-

- cial networks. In *Proceedings of The Web Conference 2020*, pages 2472–2478, 2020. [2](#page-1-0)
- <span id="page-9-24"></span>[29] Yang Li, Si Si, Gang Li, Cho-Jui Hsieh, and Samy Bengio. Learnable fourier features for multi-dimensional spatial positional encoding. *Advances in Neural Information Processing Systems*, 34:15816–15829, 2021. [4](#page-3-3)
- <span id="page-9-23"></span>[30] Yinhan Liu. Roberta: A robustly optimized bert pretraining approach. *arXiv preprint arXiv:1907.11692*, 364, 2019. [4](#page-3-3)
- <span id="page-9-25"></span>[31] I Loshchilov. Decoupled weight decay regularization. *arXiv preprint arXiv:1711.05101*, 2017. [5](#page-4-1)
- <span id="page-9-0"></span>[32] Sounak Mondal, Zhibo Yang, Seoyoung Ahn, Dimitris Samaras, Gregory Zelinsky, and Minh Hoai. Gazeformer: Scalable, effective and fast prediction of goal-directed human attention. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 1441– 1450, 2023. [1,](#page-0-0) [2](#page-1-0)
- <span id="page-9-1"></span>[33] Sounak Mondal, Seoyoung Ahn, Zhibo Yang, Niranjan Balasubramanian, Dimitris Samaras, Gregory Zelinsky, and Minh Hoai. Look hear: Gaze prediction for speech-directed human attention. In *European Conference on Computer Vision*, pages 236–255. Springer, 2025. [1,](#page-0-0) [2](#page-1-0)
- <span id="page-9-4"></span>[34] Jordan Navarro, Otto Lappi, Franccois Osiurak, Emma Hernout, Catherine Gabaude, and Emanuelle Reynaud. Dynamic scan paths investigations under manual and highly automated driving. *Scientific reports*, 11(1):3776, 2021. [1](#page-0-0)
- <span id="page-9-21"></span>[35] Takumi Nishiyasu and Yoichi Sato. Gaze scanpath transformer: Predicting visual search target by spatiotemporal semantic modeling of gaze scanpath. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 625–635, 2024. [2](#page-1-0)
- <span id="page-9-20"></span>[36] Seonwook Park, Shalini De Mello, Pavlo Molchanov, Umar Iqbal, Otmar Hilliges, and Jan Kautz. Few-shot adaptive gaze estimation. In *Proceedings of the IEEE/CVF international conference on computer vision*, pages 9368–9377, 2019. [2](#page-1-0)
- <span id="page-9-10"></span>[37] Jean Pfiffelmann, Nathalie Dens, and Sebastien Soulez. Per- ´ sonalized advertisements with integration of names and photographs: An eye-tracking experiment. *Journal of Business Research*, 111:196–207, 2020. [1](#page-0-0)
- <span id="page-9-26"></span>[38] Brian J Scholl. Objects and attention: The state of the art. *Cognition*, 80(1-2):1–46, 2001. [1](#page-0-0)
- <span id="page-9-22"></span>[39] Florian Schroff, Dmitry Kalenichenko, and James Philbin. Facenet: A unified embedding for face recognition and clustering. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 815–823, 2015. [3](#page-2-0)
- <span id="page-9-8"></span>[40] ABM Fahim Shahriar, Mahedee Zaman Moon, Hasan Mahmud, and Kamrul Hasan. Online product recommendation system by using eye gaze data. In *Proceedings of the International Conference on Computing Advancements*, pages 1–7, 2020. [1](#page-0-0)
- <span id="page-9-17"></span>[41] Jake Snell, Kevin Swersky, and Richard Zemel. Prototypical networks for few-shot learning. *Advances in neural information processing systems*, 30, 2017. [2,](#page-1-0) [3](#page-2-0)
- <span id="page-9-9"></span>[42] Hyejin Song and Nammee Moon. Eye-tracking and social behavior preference-based recommendation system. *The Journal of Supercomputing*, 75:1990–2006, 2019. [1](#page-0-0)

- <span id="page-9-6"></span>[43] Yingjie Song, Zhi Liu, Gongyang Li, Jiawei Xie, Qiang Wu, Dan Zeng, Lihua Xu, Tianhong Zhang, and Jijun Wang. Ems: A large-scale eye movement dataset, benchmark, and new model for schizophrenia recognition. *IEEE Transactions on Neural Networks and Learning Systems*, 2024. [1](#page-0-0)
- <span id="page-9-15"></span>[44] Florian Strohm, Mihai Bace, and Andreas Bulling. Learning ˆ user embeddings from human gaze for personalised saliency prediction. *Proceedings of the ACM on Human-Computer Interaction*, 8(ETRA):1–16, 2024. [2](#page-1-0)
- <span id="page-9-5"></span>[45] Xiangjie Sui, Yuming Fang, Hanwei Zhu, Shiqi Wang, and Zhou Wang. Scandmm: A deep markov model of scanpath prediction for 360deg images. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 6989–6999, 2023. [1,](#page-0-0) [2](#page-1-0)
- <span id="page-9-14"></span>[46] Wanjie Sun, Zhenzhong Chen, and Feng Wu. Visual scanpath prediction using ior-roi recurrent mixture density network. *IEEE transactions on pattern analysis and machine intelligence*, 43(6):2101–2118, 2019. [2](#page-1-0)
- <span id="page-9-18"></span>[47] Kaixin Wang, Jun Hao Liew, Yingtian Zou, Daquan Zhou, and Jiashi Feng. Panet: Few-shot image semantic segmentation with prototype alignment. In *proceedings of the IEEE/CVF international conference on computer vision*, pages 9197–9206, 2019. [2](#page-1-0)
- <span id="page-9-7"></span>[48] Yao Wang, Andreas Bulling, et al. Scanpath prediction on information visualisations. *IEEE Transactions on Visualization and Computer Graphics*, 2023. [1,](#page-0-0) [2](#page-1-0)
- <span id="page-9-11"></span>[49] Jingyi Xu and Hieu Le. Generating representative samples for few-shot classification. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 9003–9013, 2022. [2](#page-1-0)
- <span id="page-9-13"></span>[50] Juan Xu, Ming Jiang, Shuo Wang, Mohan S. Kankanhalli, and Qi Zhao. Predicting human gaze beyond pixels. *Journal of Vision*, 14(1):1–20, 2014. [2,](#page-1-0) [5](#page-4-1)
- <span id="page-9-12"></span>[51] Jingyi Xu, Hieu Le, Mingzhen Huang, ShahRukh Athar, and Dimitris Samaras. Variational feature disentangling for fine-grained few-shot classification. In *Proceedings of the IEEE/CVF international conference on computer vision*, pages 8812–8821, 2021. [2](#page-1-0)
- <span id="page-9-19"></span>[52] Jingyi Xu, Hieu Le, and Dimitris Samaras. Generating features with increased crop-related diversity for few-shot object detection. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 19713– 19722, 2023. [2](#page-1-0)
- <span id="page-9-16"></span>[53] Yanyu Xu, Shenghua Gao, Junru Wu, Nianyi Li, and Jingyi Yu. Personalized saliency and its prediction. *IEEE transactions on pattern analysis and machine intelligence*, 41(12): 2975–2989, 2018. [2](#page-1-0)
- <span id="page-9-2"></span>[54] Zhibo Yang, Lihan Huang, Yupei Chen, Zijun Wei, Seoyoung Ahn, Gregory Zelinsky, Dimitris Samaras, and Minh Hoai. Predicting goal-directed human attention using inverse reinforcement learning. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 193–202, 2020. [1,](#page-0-0) [2,](#page-1-0) [5](#page-4-1)
- <span id="page-9-3"></span>[55] Zhibo Yang, Sounak Mondal, Seoyoung Ahn, Gregory Zelinsky, Minh Hoai, and Dimitris Samaras. Target-absent human attention. In *European Conference on Computer Vision*, pages 52–68. Springer, 2022. [1,](#page-0-0) [2](#page-1-0)

- <span id="page-10-2"></span>[56] Zhanyuan Yang, Jinghua Wang, and Ying J. Zhu. Few-shot classification with contrastive learning. In *European Conference on Computer Vision*, 2022. [2](#page-1-0)
- <span id="page-10-0"></span>[57] Zhibo Yang, Sounak Mondal, Seoyoung Ahn, Ruoyu Xue, Gregory Zelinsky, Minh Hoai, and Dimitris Samaras. Unifying top-down and bottom-up scanpath prediction using transformers. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 1683– 1693, 2024. [1,](#page-0-0) [2,](#page-1-0) [3,](#page-2-0) [4,](#page-3-3) [5](#page-4-1)
- <span id="page-10-3"></span>[58] Han-Jia Ye, Hexiang Hu, De-Chuan Zhan, and Fei Sha. Few-shot learning via embedding adaptation with set-to-set functions. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 8808–8817, 2020. [2](#page-1-0)
- <span id="page-10-1"></span>[59] Feng Yuan, Matthieu Perreira Da Silva, and Alexandre Bruckert. Personalized visual scanpath prediction using iorroi weighted attention network. In *Proceedings of the 2023 ACM International Conference on Interactive Media Experiences Workshops*, pages 66–68, 2023. [1,](#page-0-0) [2](#page-1-0)
- <span id="page-10-4"></span>[60] Wenqi Zhong, Linzhi Yu, Chen Xia, Junwei Han, and Dingwen Zhang. Spformer: Spatio-temporal modeling for scanpaths with transformer. In *Proceedings of the AAAI Conference on Artificial Intelligence*, pages 7605–7613, 2024. [2](#page-1-0)
- <span id="page-10-5"></span>[61] Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, and Jifeng Dai. Deformable detr: Deformable transformers for end-to-end object detection. *arXiv preprint arXiv:2010.04159*, 2020. [3,](#page-2-0) [1](#page-0-0)

# **Few-shot Personalized Scanpath Prediction**

# Supplementary Material

#### 7. Overview

This supplementary meterial is arranged as:

- Sec. 8 shows the implementation details of ISP-SENet.
- Sec. 9 shows statistics of Tab. 1 in the main paper.
- Sec. 10 shows the supplementary evaluation of ISP-SENet.
- Sec. 11 shows ablation study on more parameters and modules.
- Sec. 12 shows more qualitative results. In the experiments, unless specified otherwise, we sample the 10-shot support set for 10 times.

## <span id="page-11-0"></span>8. Implementation Details

#### **8.1. Feature Extractor** *F*

Humans perceive images through a high-resolution foveal region and a low-resolution peripheral region [15], creating distinct focal and contextual areas. This principle also guides scanpath prediction models [27, 55, 57]. Following HAT [57], we encode images and scanpaths using hierarchical feature maps from the image encoder and decoder. The image encoder produces multi-scale feature maps based on ResNet [19]. To better align with human object-centric attention [38], deformable attention [61] pre-trained on segmentation tasks is utilized to generate hierarchical feature maps that capture semantic object information. The output of image decoder is four-scale hierarchical feature maps, where we utilize two feature maps with lowest and highest resolution  $(P_l \in \mathbb{R}^{\left(\frac{H}{32} \cdot \frac{W}{32}\right) \times C})$  and  $P_h \in \mathbb{R}^{\left(\frac{H}{4} \cdot \frac{W}{4}\right) \times C}$ , respectively).  $P_l$  is flattened and directly used as image tokens  $F_I$ , simulating the peripheral region of human attention. We select corresponding location of all fixations from  $P_h$  and obtain  $F_S \in \mathbb{R}^{L \times C}$ , where L is the length of scanpath, resembling the foveated regions of human attention.

#### 8.2. Duration

It should be noted that the duration strategy is exclusively implemented in SE-Net, whereas ISP-SENet utilizes raw durations.

This strategy involves categorizing each fixation duration into one of ten bins, ensuring that each bin contains approximately the same number of fixation durations, a method known as quantile-based intervals. This approach is motivated by two main reasons:

Significance of Duration: Fixation duration is indicative of the importance attributed to a point in an image, as longer durations generally reflect greater interest by

<span id="page-11-1"></span>![](_page_11_Figure_16.jpeg)

![](_page_11_Figure_17.jpeg)

Figure 7. **Duration distribution.** Figure (a) shows the long-tail distribution of all fixation durations in the base set of seen subjects. Figure (b) visualizes the number of points in each bin. The mean value of each bin is shown on top of each bar. In SE-Net, the bin index replaces the raw duration and is encoded using 1D sinusoidal positional encoding.

the viewer. By grouping durations into bins, we aim to quantitatively represent the significance, or the underlying importance, of each fixation.

2. **Distribution Characteristics:** The fixation durations exhibit a long-tail distribution, as evidenced in Fig. 7 (we collect all fixation durations values across all fixations). Employing a quantile strategy prevents the highly frequent shorter durations from clustering excessively in the initial bins. Instead, it ensures a more balanced distribution across the bins, with larger values being more evenly dispersed among them.

The number of bins is set to 10, and the visualization of each

bin's statistics is shown in Fig. [7.](#page-11-1) The ablation of duration strategy is discussed in Sec. [11](#page-15-0) and Tab. [13.](#page-15-1)

## <span id="page-12-0"></span>9. Statistics of Main Results

#### 9.1. Margin of Error

To show the stabilization of our method, we show the margin of error at 95% confidence level in Tab. [7.](#page-13-0) It is obtained by sampling the support set 10 times, and ensuring each sampling set is exclusive. From the results, ISP-SENet experienced more stable performance across different support set sampling, while the performance of baselines experienced more variance, suffering from the different image content in the support set.

#### 9.2. Second Seen-Unseen Split

To ensure the result is not biased on subjects due to the model's ability may various on different subjects, we conduct one more split of seen-unseen subjects, which still follows the rule of 70% seen and 30% unseen. To specify, 10 seen subjects and 5 unseen subjects for OSIE, 7 seen and 3 unseen subjects for both COCO-Search18 and COCO-FreeView. We ensure this second split contains different unseen subjects compared with the split shown in the main paper. The result is shown in Tab. [8.](#page-14-0) From the results in Tab. [7](#page-13-0) and Tab. [8,](#page-14-0) we observe that ISP-SENet demonstrates stability across different seen-unseen splits, as indicated by relatively consistent performance metrics. In contrast, the performance variations in the two baselines are significant. This difference is largely attributed to the fine-tuning process, where performance is heavily dependent on the small support set. With only 10 images from each subject, substantial variation arises due to biases in image content and the specific human attention related to individual scenes. These observations further underscore the limitations of existing methods in few-shot settings.

# <span id="page-12-1"></span>10. Supplementary Evaluation

In this section, we evaluate the performance of ISP-SENet in three aspect:

In Sec. [10.1,](#page-12-2) we compare the performance of ISP-SENet and baselines on seen subject.

In Sec. [10.2,](#page-12-3) we use subject embeddings learned from SE-Net to replace the subject embedding of ChenLSTM-ISP, and compare with baseline fine-tuned on full training set of unseen subjects.

In Sec. [10.3,](#page-12-4) we develop a new evaluation method to validate that ISP-SENet can distinct different subject embeddings.

## <span id="page-12-2"></span>10.1. Results on Seen Set

In Tab. [9,](#page-14-1) we evaluate the performance of ISP-SENet and baselines on seen subjects, same as the split defined in the main paper. Notably, although the subject embeddings generated by SE-Net are frozen during the training process of ISP-SENet, indicating that they are not tailored for scanpath prediction, the performance on COCO-Search18 is significantly better. Moreover, it achieved comparable results with OSIE and COCO-FreeView. This suggests that the seen subject embeddings learned by SE-Net, despite being optimized for distinguishing different subjects rather than specifically for scanpath prediction, effectively retain individual attention traits and excel in personalized scanpath prediction.

## <span id="page-12-3"></span>10.2. ISP-SENet with Different Scanpath Prediction Models

To demonstrate the adaptability of the subject embeddings learned from SE-Net across different scanpath prediction models, we substituted the original subject embeddings in ChenLSTM-ISP with those learned from SE-Net. The performance of this configuration, referred to as ISP-SENet(ChenLSTM-ISP), is shown in the fourth row of Tab. [10.](#page-14-2)

Further, to compare the performance of ISP-SENet with baselines, we fine-tuned both Gazeformer-ISP and ChenLSTM-ISP on the complete training set for unseen subjects. The results, labeled as Gazeformer-ISP-FT and ChenLSTM-ISP-FT, are presented in the first and third rows of Tab. [10.](#page-14-2)

These results highlight that, without any fine-tuning on unseen subjects, ISP-SENet achieves comparable results compared with baseline models, which are fully fine-tuned on full training set of unseen subjects.

## <span id="page-12-4"></span>10.3. Cross-subject embedding Evaluation

To confirm that our unseen subject embeddings capture unique attention patterns rather than a global optimum applicable to all subjects, we implement a cross-subject embedding evaluation. For each unseen subject uk, we first calculate the SM, MM, and SED metrics using the subject's own embedding e<sup>k</sup> for predicting scanpaths on the query set. Then we replace e<sup>k</sup> with embeddings e<sup>i</sup> from m different subjects u<sup>i</sup> , where u<sup>i</sup> ∈ Uunseen,i̸=k, and compute the average SM, MM, and SED. The differences in these metrics underscore the uniqueness of each embedding. For simplicity, we define m = 3 and randomly sampled 5 different support sets.

In Tab. [11,](#page-15-2) the symbol × represents that the subject embedding and prediction correspond to the same subject, while ✓ indicates they belong to different subjects. To better understand the model performance of crosssubject embedding evaluation, we include comparisons with ISP(Seen) and ISP-SENet(Seen). ISP(Seen) evaluates Gazeformer-ISP's ability to differentiate among embeddings of seen subjects. As ISP-SENet is built upon (a) OSIE

<span id="page-13-0"></span>

|                     |                                    | (a) OSIE                                                          |                                                                   |                                                                   |
|---------------------|------------------------------------|-------------------------------------------------------------------|-------------------------------------------------------------------|-------------------------------------------------------------------|
| n-shot              | Method                             | SM ↑                                                              | MM ↑                                                              | SED↓                                                              |
| n = 1               | ChenLSTM-ISP                       | $0.282 \pm 0.009$                                                 | $0.763 \pm 0.006$                                                 | $7.832 \pm 0.181$                                                 |
|                     | Gazeformer-ISP                     | $0.327 \pm 0.007$                                                 | $0.792 \pm 0.003$                                                 | $7.873 \pm 0.134$                                                 |
|                     | ChenLSTM-ISP-S                     | $0.328 \pm 0.001$                                                 | $0.793 \pm 0.001$                                                 | $7.601 \pm 0.039$                                                 |
|                     | Gazeformer-ISP-S                   | $0.354 \pm 0.000$                                                 | $0.801 \pm 0.000$                                                 | $7.503 \pm 0.003$                                                 |
|                     | ISP-SENet                          | $\textbf{0.368} \pm \textbf{0.003}$                               | $0.805 \pm 0.002$                                                 | $7.413 \pm 0.033$                                                 |
| n = 5               | ChenLSTM-ISP                       | $0.319 \pm 0.005$                                                 | $0.773 \pm 0.004$                                                 | $7.855 \pm 0.116$                                                 |
|                     | Gazeformer-ISP                     | $0.340 \pm 0.003$                                                 | $0.791 \pm 0.002$                                                 | $7.920 \pm 0.082$                                                 |
|                     | ChenLSTM-ISP-S<br>Gazeformer-ISP-S | $\begin{array}{c} 0.329 \pm 0.001 \\ 0.354 \pm 0.000 \end{array}$ | $\begin{array}{c} 0.801 \pm 0.000 \\ 0.791 \pm 0.001 \end{array}$ | $7.499 \pm 0.028 \\ 7.699 \pm 0.003$                              |
|                     | ISP-SENet                          | $0.376 \pm 0.000$                                                 | $0.791 \pm 0.001$<br>$0.803 \pm 0.001$                            | $7.649 \pm 0.003$<br>$7.649 \pm 0.028$                            |
| n = 10              | ChenLSTM-ISP                       | $0.322 \pm 0.005$                                                 | $0.777 \pm 0.002$                                                 | $7.740 \pm 0.079$                                                 |
|                     | Gazeformer-ISP                     | $0.345 \pm 0.003$                                                 | $0.794 \pm 0.002$                                                 | $7.916 \pm 0.054$                                                 |
|                     | ChenLSTM-ISP-S                     | $0.328 \pm 0.005$                                                 | $0.791 \pm 0.001$                                                 | $7.637 \pm 0.060$                                                 |
|                     | Gazeformer-ISP-S                   | $0.354 \pm 0.000$                                                 | $0.802 \pm 0.000$                                                 | $7.505 \pm 0.003$                                                 |
|                     | ISP-SENet                          | $0.375 \pm 0.001$                                                 | $0.803 \pm 0.001$                                                 | $7.318 \pm 0.017$                                                 |
|                     |                                    | (b) COCO-FreeVie                                                  |                                                                   |                                                                   |
| n-shot              | Method                             | SM ↑                                                              | MM ↑                                                              | SED↓                                                              |
| n = 1               | ChenLSTM-ISP                       | $0.287 \pm 0.014$                                                 | $0.805 \pm 0.003$                                                 | $13.307 \pm 0.195$                                                |
|                     | Gazeformer-ISP                     | $0.244 \pm 0.021$                                                 | $0.787 \pm 0.011$                                                 | $15.118 \pm 0.510$                                                |
|                     | ChenLSTM-ISP-S<br>Gazeformer-ISP-S | $\begin{array}{c} 0.339 \pm 0.000 \\ 0.333 \pm 0.000 \end{array}$ | $\begin{array}{c} 0.814 \pm 0.000 \\ 0.817 \pm 0.000 \end{array}$ | $12.523 \pm 0.029$                                                |
|                     | ISP-SENet                          | $0.369 \pm 0.002$                                                 | $0.817 \pm 0.000$<br>$0.832 \pm 0.001$                            | $12.538 \pm 0.012$<br>$12.227 \pm 0.134$                          |
| n = 5               |                                    |                                                                   |                                                                   |                                                                   |
| $\Pi = \mathcal{I}$ | ChenLSTM-ISP<br>Gazeformer-ISP     | $0.320 \pm 0.009$<br>$0.286 \pm 0.012$                            | $\begin{array}{c} 0.815 \pm 0.005 \\ 0.800 \pm 0.005 \end{array}$ | $12.950 \pm 0.190$<br>$14.630 \pm 0.310$                          |
|                     | ChenLSTM-ISP-S                     | $0.338 \pm 0.000$                                                 | $0.814 \pm 0.000$                                                 | $12.540 \pm 0.023$                                                |
|                     | Gazeformer-ISP-S                   | $0.333 \pm 0.000$                                                 | $0.817 \pm 0.000$                                                 | $12.539 \pm 0.008$                                                |
|                     | ISP-SENet                          | $\textbf{0.368} \pm \textbf{0.001}$                               | $\textbf{0.829} \pm \textbf{0.001}$                               | $\textbf{12.017} \pm \textbf{0.058}$                              |
| n = 10              | ChenLSTM-ISP                       | $0.323 \pm 0.010$                                                 | $0.819 \pm 0.005$                                                 | $12.541 \pm 0.114$                                                |
|                     | Gazeformer-ISP                     | $0.317 \pm 0.002$                                                 | $0.805 \pm 0.002$                                                 | $14.224 \pm 0.207$                                                |
|                     | ChenLSTM-ISP-S                     | $0.340 \pm 0.000$                                                 | $0.814 \pm 0.000$                                                 | $12.532 \pm 0.025$                                                |
|                     | Gazeformer-ISP-S ISP-SENet         | $0.333 \pm 0.000$<br>$0.367 \pm 0.001$                            | $0.816 \pm 0.000$<br>$0.828 \pm 0.001$                            | $12.545 \pm 0.006$<br>$11.956 \pm 0.010$                          |
|                     | ISI SEITE                          |                                                                   |                                                                   | 11.520 ± 0.010                                                    |
| n-shot              | Method                             | (c) COCO-Search1                                                  | MM ↑                                                              | SED↓                                                              |
|                     |                                    |                                                                   | · · · · · · · · · · · · · · · · · · ·                             | <u>·</u>                                                          |
| n = 1               | ChenLSTM-ISP                       | $0.371 \pm 0.024$                                                 | $0.760 \pm 0.029$                                                 | $2.756 \pm 0.464$                                                 |
|                     | Gazeformer-ISP<br>ChenLSTM-ISP-S   | $0.342 \pm 0.018$<br>$0.448 \pm 0.000$                            | $0.770 \pm 0.008$                                                 | $2.818 \pm 0.216$                                                 |
|                     | Gazeformer-ISP-S                   | $0.446 \pm 0.000$<br>$0.446 \pm 0.001$                            | $\begin{array}{c} 0.803 \pm 0.001 \\ 0.802 \pm 0.001 \end{array}$ | $2.394 \pm 0.013$<br>$2.463 \pm 0.002$                            |
|                     | ISP-SENet                          | $0.475 \pm 0.007$                                                 | $0.814 \pm 0.001$                                                 | $2.333 \pm 0.063$                                                 |
| n = 5               | ChenLSTM-ISP                       | $0.386 \pm 0.015$                                                 | $0.773 \pm 0.008$                                                 | $2.489 \pm 0.058$                                                 |
|                     | Gazeformer-ISP                     | $0.353 \pm 0.028$                                                 | $0.774 \pm 0.011$                                                 | $2.980 \pm 0.292$                                                 |
|                     | ChenLSTM-ISP-S                     | $0.449 \pm 0.001$                                                 | $0.803 \pm 0.001$                                                 | $2.380 \pm 0.014$                                                 |
|                     | Gazeformer-ISP-S                   | $0.445 \pm 0.001$                                                 | $0.803 \pm 0.001$                                                 | $2.457 \pm 0.002$                                                 |
|                     | ISP-SENet                          | $\textbf{0.484} \pm \textbf{0.005}$                               | $0.815 \pm 0.001$                                                 | $2.354 \pm 0.044$                                                 |
| n = 10              | ChenLSTM-ISP                       | $0.393 \pm 0.006$                                                 | $0.781 \pm 0.004$                                                 | $2.394 \pm 0.038$                                                 |
|                     | Gazeformer-ISP                     | $0.370 \pm 0.007$                                                 | $0.785 \pm 0.006$                                                 | $\begin{array}{c} 2.765 \pm 0.128 \\ 2.379 \pm 0.019 \end{array}$ |
|                     | ChenLSTM-ISP-S<br>Gazeformer-ISP-S | $0.449 \pm 0.000$<br>$0.446 \pm 0.001$                            | $\begin{array}{c} 0.803 \pm 0.001 \\ 0.802 \pm 0.001 \end{array}$ | $2.379 \pm 0.019$<br>$2.464 \pm 0.002$                            |
|                     | ISP-SENet                          | $0.440 \pm 0.001$<br>$0.482 \pm 0.002$                            | $0.802 \pm 0.001$<br>$0.815 \pm 0.001$                            | $2.359 \pm 0.002$                                                 |
|                     |                                    |                                                                   |                                                                   |                                                                   |

Table 7. Margin of error for Tab. 1 in the main paper.

Gazeformer-ISP, the distinction ability of these two models will not have significant differences. **ISP-SENet(Seen)** assesses ISP-SENet's cross-subject embedding performance

on seen subjects, indicative of the potential upper limit of our model's discriminative capability. **ISP-SENet(Unseen)** represents our cross-subject embedding evaluation for un-

| (a) OSIE |
|----------|
|----------|

<span id="page-14-0"></span>

| Method         | SM ↑          | MM ↑          | SED ↓         |
|----------------|---------------|---------------|---------------|
| ChenLSTM-ISP   | 0.288 ± 0.009 | 0.780 ± 0.004 | 7.350 ± 0.127 |
| Gazeformer-ISP | 0.318 ± 0.006 | 0.789 ± 0.002 | 8.363 ± 0.155 |
| ISP-SENet      | 0.384 ± 0.001 | 0.813 ± 0.001 | 7.460 ± 0.022 |

#### (b) COCO-FreeView

| Method         | SM ↑          | MM ↑          | SED ↓          |
|----------------|---------------|---------------|----------------|
| ChenLSTM-ISP   | 0.296 ± 0.007 | 0.823 ± 0.001 | 12.534 ± 0.030 |
| Gazeformer-ISP | 0.275 ± 0.008 | 0.801 ± 0.006 | 14.266 ± 0.286 |
| ISP-SENet      | 0.364 ± 0.001 | 0.835 ± 0.001 | 12.342 ± 0.019 |

#### (c) COCO-Search18

| Method         | SM ↑          | MM ↑          | SED ↓         |
|----------------|---------------|---------------|---------------|
| ChenLSTM-ISP   | 0.333 ± 0.006 | 0.766 ± 0.006 | 2.712 ± 0.042 |
| Gazeformer-ISP | 0.403 ± 0.010 | 0.803 ± 0.005 | 2.734 ± 0.116 |
| ISP-SENet      | 0.465 ± 0.001 | 0.812 ± 0.001 | 2.286 ± 0.020 |

<span id="page-14-1"></span>Table 8. Results from the second seen-unseen split under the 10-shot setting. The unseen subject set in this split is distinct from the unseen set used in the main paper's split.

| Methods        |       | OSIE  |       |       | COCO-FreeView |        |       | COCO-Search18 |       |  |
|----------------|-------|-------|-------|-------|---------------|--------|-------|---------------|-------|--|
|                | SM ↑  | MM ↑  | SED ↓ | SM ↑  | MM ↑          | SED ↓  | SM ↑  | MM ↑          | SED ↓ |  |
| ChenLSTM-ISP   | 0.373 | 0.814 | 7.171 | 0.373 | 0.828         | 12.126 | 0.475 | 0.820         | 2.128 |  |
| Gazeformer-ISP | 0.382 | 0.813 | 7.077 | 0.380 | 0.835         | 11.707 | 0.480 | 0.815         | 2.204 |  |
| ISP-SENet      | 0.382 | 0.816 | 7.127 | 0.375 | 0.833         | 11.872 | 0.517 | 0.825         | 2.086 |  |

<span id="page-14-2"></span>Table 9. Performance Comparison of methods on seen subjects. All methods are trained on all training data of seen subjects, and test on the test set of seen subjects.

| Methods                    | OSIE  |       |       |       | COCO-FreeView |        |       | COCO-Search18 |       |  |
|----------------------------|-------|-------|-------|-------|---------------|--------|-------|---------------|-------|--|
|                            | SM ↑  | MM ↑  | SED ↓ | SM ↑  | MM ↑          | SED ↓  | SM ↑  | MM ↑          | SED ↓ |  |
| Gazeformer-ISP-FT          | 0.372 | 0.803 | 7.614 | 0.383 | 0.834         | 11.443 | 0.479 | 0.815         | 2.330 |  |
| ISP-SENet (Gazeformer-ISP) | 0.375 | 0.803 | 7.318 | 0.367 | 0.828         | 11.956 | 0.482 | 0.815         | 2.359 |  |
| ChenLSTM-ISP-FT            | 0.371 | 0.801 | 7.449 | 0.387 | 0.832         | 11.422 | 0.475 | 0.813         | 2.159 |  |
| ISP-SENet (ChenLSTM-ISP)   | 0.369 | 0.800 | 7.574 | 0.366 | 0.824         | 12.241 | 0.467 | 0.810         | 2.272 |  |

Table 10. ISP-SENet with Different Scanpath Prediction Models, and comparsion between ISP-SENet without fine-tuning on unseen subjects, with ISP[\[10\]](#page-8-1) fine-tuned on full training set of unseen subjects.

seen subjects. The results demonstrate that ISP-SENet's capacity to distinguish unseen subjects exceeds the baseline's performance with seen subjects and is comparable with ISP-SENet's performance on seen subjects.

#### 10.4. Quantatitive results on visual-task encoder

To demonstrate that the visual-task encoder effectively captures the alignment between the task and image content, we evaluate the similarity between its cross-attention maps and the ground truth bounding boxes using Correlation Coefficient (CC) and AUC. For SE-Net, we achieve a CC of 0.31 and an AUC of 0.76. While the CC is sensitive to false positives—such as attention allocated to relevant peripheral objects—our model still outperforms ChenLSTM-ISP's taskguidance map m0(CC = 0.07, AUC = 0.63), averaged across channels. This indicates stronger target understanding, despite our model not being explicitly designed for object detection.

<span id="page-15-2"></span>

| Method            | cross-subject embedding | OSIE |                 |       |  |
|-------------------|-------------------------|------|-----------------|-------|--|
|                   |                         |      | SM ↑ MM ↑ SED ↓ |       |  |
|                   | ×                       |      | 0.386 0.814     | 7.003 |  |
| ISP(Seen)         | ✓                       |      | 0.379 0.812     | 7.163 |  |
|                   | ×                       |      | 0.387 0.815     | 7.009 |  |
| ISP-SENet(Seen)   | ✓                       |      | 0.373 0.810     | 7.360 |  |
|                   | ×                       |      | 0.376 0.802     | 7.286 |  |
| ISP-SENet(Unseen) | ✓                       |      | 0.361 0.800     | 7.340 |  |

Table 11. Cross-embedding evaluation on the distinction ability between subject embeddings. The symbol × denotes that the subject embedding and prediction correspond to the same subject, while ✓ indicates that the subject embeddings and prediction belong to different subjects.

## 10.5. More analysis on size of base set

In Tab. [12,](#page-15-3) we analyze the impact of varying the number of seen subjects in the base set during training on COCO-Search18, supplementing the results in Table 4 of the main paper. We consistently select one subject as unseen and vary the number of seen subjects selected from the remaining ones. With fewer seen subjects, SE-Net struggles to infer the attention traits of new subjects based on its learned experience. Performance improves when increasing the number of seen subjects from 7 (as in the main paper) to 9, suggesting that ISP-SE-Net benefits from additional subjects.

<span id="page-15-3"></span>

| num seen  | SM ↑  |       | MM ↑ SED ↑ |
|-----------|-------|-------|------------|
| 4(40%)    | 0.472 | 0.819 | 2.542      |
| 9(90%)    | 0.489 | 0.826 | 2.145      |
| Ours(70%) | 0.487 | 0.823 | 2.333      |

Table 12. Performance comparison of 10-shot setting on COCO-Search18 with different numbers of seen subjects in training stage.

## <span id="page-15-0"></span>11. More Ablation Results

# 11.1. Duration

In Tab. [13,](#page-15-1) we ablate the effect of duration encoding strategy on OSIE in three settings: (1) No duration encoding in scanpath embeddings. (2) Encoding raw duration without assigning bin index. (3) Assigning durations to 10 bins of equal width, without employing the quantile strategy. (4) Assigning durations to 100 bins using the quantile strategy. (5) Assigning durations to 300 bins using the quantile strategy. (6) Assigning durations to 10 bins using the quantile strategy as in the main paper.

The result indicates that: (1) Duration is crucial for understanding subject attention traits. (2) Raw durations offer limited information as they introduce redundancy. For instance, 200 ms and 201 ms are treated as distinct durations, despite their negligible difference, which does not accurately reflect varying importance levels between two fixations. (3) Without the quantile strategy, the bins fail to manage the long-tail effect effectively, resulting in sparse distribution of higher durations across most bins while smaller durations crowd into a few bins due to their higher frequency. (4) and (5) demonstrate how varying the number of bins impacts performance.

<span id="page-15-1"></span>

| Duration Strategy   | SM ↑  | MM ↑  | SED ↓ |
|---------------------|-------|-------|-------|
| w/o Duration        | 0.365 | 0.797 | 7.634 |
| Raw duration        | 0.367 | 0.800 | 7.534 |
| Uniform bin width   | 0.369 | 0.799 | 7.431 |
| 100 bins            | 0.374 | 0.801 | 7.377 |
| 300 bins            | 0.370 | 0.801 | 7.474 |
| ISP-SENet (10 bins) | 0.375 | 0.803 | 7.318 |

Table 13. Ablation on performance with different duration encoding strategy on OSIE.

## 11.2. Margin in Contrastive Loss

We ablate the effect of different margins m in the contrastive loss. The margin is a predefined threshold that specifies how much farther the negative example should be from the anchor compared to the positive example. As shown in Tab. [14,](#page-16-1) lower or higher margins decrease the prediction performance. A possible reason is lower margin prevents SE-Net from distinguishing different subjects.

The performance decreases associated with a higher margin can be attributed to the characteristics of human attention. Despite differences among subjects, their scanpaths often share similarities, such as a focus on foreground objects like humans. Such similarity is critical for SE-Net to learn the subject embedding, and plays a key role in infering embeddings for unseen subjects. We anticipate that embeddings for unseen subjects will benefit from seen subjects with similar attention patterns. Thus, setting a higher margin may overlook these essential similarities.

Tab. [14](#page-16-1) shows that, for COCO-Search18 dataset where viewing patterns between people are more similar (higher Human Consistency(HC)[\[9\]](#page-8-0)), a smaller margin of 1 performs better. For OSIE dataset with more diverse patterns (lower HC), a larger margin of 5 performs better. Also the effect of margin is more significant for higher HC.

# 11.3. Embedding Dimension

In Tab. [15](#page-16-2) we explore different embedding dimensions of SE-Net and ISP-SENet. All layers in SE-Net and ISP-SENet shares the same embedding dimensions. The results indicate that varying the embedding dimensions does not significantly impact performance.

<span id="page-16-1"></span>

| margin | OSIE(HC=0.39) |       | COCO-Search18(HC=0.52) |       |       |       |
|--------|---------------|-------|------------------------|-------|-------|-------|
|        | SM ↑          | ММ↑   | SED↓                   | SM ↑  | MM ↑  | SED↓  |
| 1      | 0.369         | 0.804 | 7.506                  | 0.482 | 0.815 | 2.359 |
| 5      | 0.375         | 0.803 | 7.318                  | 0.467 | 0.815 | 2.455 |
| 10     | 0.367         | 0.809 | 7.546                  | 0.445 | 0.813 | 2.563 |

Table 14. Ablation on performance with different m in contrastive loss on OSIE and COCO-Search18. In the main paper, we use m=5 for OSIE and m=1 for COCO-Search18.

<span id="page-16-2"></span>

| Embedding Dimension | SM ↑  | ММ↑   | SED↓  |
|---------------------|-------|-------|-------|
| 128                 | 0.374 | 0.804 | 7.324 |
| 384                 | 0.375 | 0.803 | 7.318 |

Table 15. Ablation on performance with different embedding dimensions.

# <span id="page-16-0"></span>12. Qualitative Results

We show more qualitative results of ISP-SENet on OSIE, COCO-FreeView and COCO-Search18 in Fig. 8, Fig. 9, Fig. 10. In most cases, ISP-SE-Net successfully captures the variation in viewed objects across different subjects. Notably, on COCO-FreeView, it learns global attention patterns such as centralized or scattered focus. In the search task, our model also identifies subjects influenced by distractions, outperforming the baselines in capturing such behaviors.

<span id="page-17-0"></span>![](_page_17_Figure_0.jpeg)

Figure 8. More Qualitative examples of scanpath prediction for different unseen subjects on OSIE. GT is the ground truth scanpaths of different unseen subjects. Red circle is the end fixation. Each two rows of the same image are scanpaths belonging to two different unseen subjects.

<span id="page-18-0"></span>![](_page_18_Figure_0.jpeg)

Figure 9. More Qualitative examples of scanpath prediction for different unseen subjects on COCO-FreeView. GT is the ground truth scanpaths of different unseen subjects. Red circle is the end fixation. Each two rows of the same image are scanpaths belonging to two different unseen subjects.

<span id="page-19-0"></span>![](_page_19_Figure_0.jpeg)

Figure 10. More Qualitative examples of scanpath prediction for different unseen subjects on COCO-Search18. GT is the ground truth scanpaths of different unseen subjects. Red circle is the end fixation. Each two rows of the same image are scanpaths belonging to two different unseen subjects. The search targets are mouse, car, chair, respectively.
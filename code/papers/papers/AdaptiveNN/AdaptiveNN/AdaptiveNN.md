![](_page_0_Picture_2.jpeg)

# **Emulating Human-like Adaptive Vision for Efficient and Flexible Machine Visual Perception**

**Yulin Wang**† **Yang Yue**† **Yang Yue**† **Huanqian Wang Haojun Jiang Yizeng Han Zanlin Ni Yifan Pu Minglei Shi Rui Lu Qisen Yang Andrew Zhao Zhuofan Xia Shiji Song Gao Huang** Learning And Perception (LEAP) Lab, Department of Automation, Tsinghua University † Equal Contribution. Corresponding Authors: Shiji Song, Gao Huang. {yulin-wang, shijis, gaohuang}@tsinghua.edu.cn.

Vision is fundamental to our interpretation of the intricate physical world [\[1,](#page-34-0) [2,](#page-34-1) [3,](#page-34-2) [4,](#page-34-3) [5,](#page-34-4) [6,](#page-34-5) [7,](#page-34-6) [8,](#page-34-7) [9\]](#page-34-8). Computationally replicating human visual perception capabilities is crucial for modern artificial intelligence (AI), such as multimodal large language models (MLLM) [\[10,](#page-34-9) [11,](#page-34-10) [12,](#page-34-11) [13\]](#page-34-12), embodied AI agents [\[14,](#page-34-13) [15,](#page-34-14) [16,](#page-34-15) [17\]](#page-34-16), and medical AI [\[18,](#page-34-17) [19,](#page-34-18) [20,](#page-35-0) [21\]](#page-35-1), and also carries significant implications for cognitive science [\[22,](#page-35-2) [23,](#page-35-3) [24\]](#page-35-4). Current methods have demonstrated notable success in addressing challenging vision tasks by continuously scaling up input complexity (*e.g.,* spatial-temporal resolutions) and model size [\[25,](#page-35-5) [26,](#page-35-6) [27,](#page-35-7) [28,](#page-35-8) [29\]](#page-35-9), at the price of dramatically growing resource demands. Informationrich, high-dimensional visual inputs, large-scale neural networks, and efficiency have converged to an 'impossible triangle' that impedes both future advancements of computer vision and its adoption in diverse real-world scenarios such as robotics, wearable devices, and industrial inspections, even posing a risk to human life by making high-latency decisions in safety-critical domains like autonomous vehicles and medical robots. Here we find this dilemma may be rooted in the current prevailing representation learning paradigm established decades ago [\[30,](#page-35-10) [31,](#page-35-11) [32,](#page-35-12) [33\]](#page-35-13): the model passively receives an input, and processes the whole input in its entirety at once, yielding computational and memory costs that scale linearly or quadratically with pixel numbers. To address this, we take inspiration from the human visual system and introduce an AdaptiveNN framework, aiming to drive a paradigm shift from 'passive' to 'active' vision models. AdaptiveNN formulates visual perception as a coarse-to-fine sequential decision-making process, progressively identifying and fixating on regions pertinent to a given task, incrementally combining information across fixations, and actively concluding its observation when sufficient to accomplish the task. Hence, akin to human vision, large models can be employed for superior capabilities, yet their inference remains low-cost since they only process a minimally necessary subset of regions within the complex scenes. We introduce a novel theoretical analysis integrating representation learning with self-rewarding reinforcement learning, which enables training the non-differentiable AdaptiveNN in end-to-end without relying on specialized task structures or additional annotations beyond standard objectives. AdaptiveNN is assessed across 17 benchmarks organized into 9 different tasks, including large-scale visual understanding, fine-grained recognition, visual search, processing images from real driving and medical scenarios, MLLM for language-driven embodied AI, and multiple side-by-side comparisons with humans. AdaptiveNN reduces the inference cost of well-performing models by up to 28× without sacrificing accuracy, especially effective for processing complicated real-world scenes, and for employing large models. It also exhibits marked behavioral flexibility to adapt to varying task instructions and fluctuating resource availability without re-training, and achieves strong interpretability through analyzing its fixation patterns. Furthermore, the perceptual behaviors of AdaptiveNN are indistinguishable from people in many cases, uncovering its potential as a useful instrument for investigating human visual cognition. Our findings reveal that incorporating human-like adaptiveness offers a promising avenue toward the next generation of efficient, flexible, and interpretable machine vision paradigms, while also providing valuable insights for the cognitive science community. Code is available at <https://github.com/LeapLabTHU/AdaptiveNN>.

## **1. Main**

Through visual perception, humans interpret complex surrounding environments, learn knowledge about how the physical world works, connect language or concepts with tangible objects and scenes, and guide their behaviors [\[1,](#page-34-0) [2,](#page-34-1) [3,](#page-34-2) [4,](#page-34-3) [5,](#page-34-4) [6,](#page-34-5) [7,](#page-34-6) [8,](#page-34-7) [9,](#page-34-8) [34\]](#page-35-14). Computationally acquiring these visual perception capabilities of humans has been crucial for advancing modern artificial intelligence, such as developing multimodal large language models (MLLM) that understand visual inputs [\[10,](#page-34-9) [11,](#page-34-10) [12,](#page-34-11) [13\]](#page-34-12), embodied AI agents that perceive and interact with the real world [\[14,](#page-34-13) [15,](#page-34-14) [16,](#page-34-15) [17\]](#page-34-16), and AI applications in pathology, radiology, and medical robots [\[18,](#page-34-17) [19,](#page-34-18) [20,](#page-35-0) [21\]](#page-35-1). Computer vision also presents significant opportunities for exploring fundamental questions in cognitive science, such as the role of innateness in human vision [\[22,](#page-35-2) [23,](#page-35-3) [24\]](#page-35-4).

Over the past decades, computational visual perception models have exhibited substantial progress, approaching or even exceeding expert-level performance across a broad range of fields, including large-scale image recognition [\[35,](#page-36-0) [36,](#page-36-1) [37,](#page-36-2) [38,](#page-36-3) [27\]](#page-35-7), object detection [\[39\]](#page-36-4), open-world visual recognition [\[40\]](#page-36-5), medical image analysis [\[41,](#page-36-6) [42,](#page-36-7) [43,](#page-36-8) [19,](#page-34-18) [20,](#page-35-0) [21\]](#page-35-1), and multimodal content understanding [\[10,](#page-34-9) [11,](#page-34-10) [12,](#page-34-11) [13\]](#page-34-12). These achievements are founded on the paradigm of representation learning, where parameterized functions are learned to transform raw pixelated images into semantically meaningful representations. This idea originated at least four decades ago [\[30\]](#page-35-10), but has become dramatically more powerful today because of the breakthroughs in algorithms and hardware, enabling training vastly deeper and larger neural networks to effectively harness large-scale, fine-grained digital visual signals with much higher spatial-temporal resolution [\[25,](#page-35-5) [26,](#page-35-6) [27,](#page-35-7) [28,](#page-35-8) [29\]](#page-35-9). These advancements have sparked interest in deploying deep networks in diverse real-world applications, such as generalist multimodal AI copilots [\[11,](#page-34-10) [12,](#page-34-11) [13\]](#page-34-12), autonomous vehicles and robotics [\[18,](#page-34-17) [14,](#page-34-13) [15,](#page-34-14) [16,](#page-34-15) [17\]](#page-34-16), wearable devices [\[44,](#page-36-9) [45\]](#page-36-10), mobile applications [\[46,](#page-36-11) [47,](#page-36-12) [48\]](#page-36-13), and edge computing [\[49,](#page-36-14) [50,](#page-36-15) [51\]](#page-37-0).

However, models achieving state-of-the-art accuracy often fall short in meeting the demands of real-world applications that extend beyond simple performance metrics. For example, in scenarios like robotics, mobile AI copilots, and industrial inspections, the hardware deployment platforms typically face constraints on computational capability, memory space, and battery capacity, yet the AI systems usually necessitate acting in real-time and performing low-latency interactions with human users and physical environments. In contrast, the inference of large computer vision models demands substantial resources, as it involves activating millions or billions of parameters to process high-resolution images with high frame rates, leading to tremendous power consumption, considerable GPU memory requirements, and nontrivial time delays. These limitations make it challenging to deploy highly capable, scaled-up models in real systems, and may even pose a risk to human life by making high-latency decisions in safety-critical domains like autonomous driving and medical robots. While cloud computing could offer some solutions, it introduces notable network latency and dependence on high-bandwidth, real-time wireless communications. Furthermore, large-scale inference requests of computationally intensive models ultimately translate to a significant rise in carbon emissions, which should be minimized for environmental reasons [\[52\]](#page-37-1).

A foundational source of these aforementioned inefficiencies is rooted in a prevalent routine within the current computer vision community, which stems from a straightforward extension of the basic representation learning paradigm established decades ago [\[30,](#page-35-10) [31,](#page-35-11) [32,](#page-35-12) [33\]](#page-35-13): models usually process a whole image or video in its entirety at once, where all pixels of the input are fed into a model simultaneously and processed parallelly, equivalent in computation (Fig. [1a](#page-2-0)). Consequently, the computational complexity and memory requirements of the same model, for both training and inference, scale linearly with the number of pixels, and thus quadratically with respect to the image height or width. Historically, this posed little concern two or three decades ago, when small neural networks with merely thousands of parameters were employed to classify tiny images, such as 28×28 black and white handwritten digits [\[53,](#page-37-2) [31,](#page-35-11) [32,](#page-35-12) [33\]](#page-35-13). However, this has evolved into a critical limitation in modern contexts, as current models have been enlarged by 5-6 orders of magnitude in terms of parameters, necessitating proficiency in processing complex scenes or videos sourced from real-world environments or the internet [\[27,](#page-35-7) [29\]](#page-35-9). For example, compared with 28×28 images, 224×224, a small yet reasonable size for common web images depicting individual objects or animals [\[35\]](#page-36-0), results in a 64 times increase in computational and memory demands, while 900×1600, a relatively small size for depicting common urban driving scenes, incurs a more than 1,800 times larger resource cost. The challenge of inefficiency becomes even more exacerbated with the recent revealing of the scaling laws of neural networks [\[54,](#page-37-3) [11,](#page-34-10) [55,](#page-37-4) [29\]](#page-35-9), that is, continuously scaling up model size may be essential for acquiring strong, generalizable capabilities across diverse tasks. This finding, coupled with the introduction of

![](_page_2_Figure_1.jpeg)

<span id="page-2-0"></span>Figure 1. The 'impossible triangle' faced by the current paradigm of computational visual perception. (a) (top half) The current prevailing paradigm established decades ago [30, 31, 32, 33]: a model processes the whole image in its entirety at once, with all pixels fed into neural networks simultaneously and processed parallelly, extracting features from all regions for downstream applications. All regions are equivalent in computation. (a) (bottom half) However, an 'impossible triangle' has emerged under this paradigm, which impedes both future advancements and adoption in diverse real-world scenarios. Specifically, continuously scaling up model size and input complexity (e.g., spatial-temporal resolutions) yields superior capabilities for addressing challenging real-world vision tasks, but usually compromises efficiency, leading to dramatically growing resource demands. (b) The human visual system circumvents this 'impossible triangle' by utilizing an active and adaptive perception strategy, which does not process everything everywhere all at once. Instead, human vision only acquires information when and where it is needed, which is implemented by sequentially sampling the optic array, progressively directing a high-resolution fovea toward a few regions of interest through eye movement, until the observation is sufficient.

high-resolution inputs, is driving computational and memory requirements to unaffordable levels. In summary, the increasing demands of higher spatial-temporal resolution for inputs, the rise of larger-scale models, and the necessity of efficiency in real-world applications have formed an 'impossible triangle' (Fig. 1a), which emerges as a major bottleneck faced by the current paradigm of machine visual perception, and its impact is expected to further markedly intensify in the future. A paradigm shift may be necessary for either addressing the immediate pressing application needs or facilitating future advancements.

In this article, we seek to draw inspiration from the human visual system to break through the effectiveness-efficiency trade-off dilemma inherent in the current computer vision paradigm. When interpreting the complex surrounding environments, unlike prevailing neural networks, human vision does not process everything everywhere all

at once. Instead, human vision adopts a much smarter active and selective strategy (Fig. 1b), sequentially sampling visual inputs by shifting a small, high-resolution fovea toward a few regions or objects of interest, and constructing a perception of the visual environment by combining information from different fixations over time [1, 3, 5, 7, 8, 9]. This evolved visual system enables the effective filtration of pertinent signals from extraneous information [56, 6, 57, 58], markedly diminishing the complexity encountered in processing the vast spectrum of visual data presented by the environment [59, 60]. Ultimately, regardless of the complexity of the original visual environment, the resource demands of human visual perception are generally determined by the 'bandwidth' and total number of fixations: the former has been pre-defined as a proper size for efficient processing, while the latter can be minimized by only acquiring information when and where it is essential for specific tasks. Thus, the human visual system not only incorporates tremendous numbers of neurons and demonstrates remarkable capabilities, but can also efficiently process the highly complex visual scenes presented by the real world, without being affected by the 'impossible triangle' limitation (Fig. 1a) faced by modern computer vision models.

As early as 2015, LeCun, Bengio, and Hinton (in the 'The future of deep learning' section in ref. [61]) have famously argued that: in the future, computer vision systems of AI are expected to attain much progress by emulating human vision to sequentially and actively decide where to look in an intelligent, task-specific way. However, nearly a decade later, the significant potential of developing human-like adaptive visual systems has not yet received adequate attention. Early studies [62, 63] have preliminarily indicated the promise of this direction using small models and tiny experiments, such as classifying handwritten digits, but there remains a huge gap between these initial efforts and modern large-scale neural networks and real-world-level application scenarios. More recently, several works have also sought to introduce adaptiveness into computer vision models [64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75], but most of them consider only an incomplete modeling of the adaptive capabilities of the human visual system, usually resulting in only modest improvements in computational efficiency. Besides, these approaches tend to focus on technical solutions tailored for specific network architectures or tasks, without offering in-depth, widely applicable theoretical and empirical insights that could guide the design and training of adaptive models in broader fields. More details of existing works can be found in Supplementary Section A. In conclusion, there is an urgent need to establish a unified computational framework that is underpinned by clear motivations, offers flexible generalizability across diverse network architectures and tasks, and is grounded in sound theoretical learning principles, demonstrating how the AI and computer vision communities can leverage adaptive visual perception models to address the effectiveness-efficiency trade-off dilemma behind the 'impossible triangle' challenge (Fig. 1a).

In response to these pressing needs, we develop a novel AdaptiveNN framework. AdaptiveNN presents a new computer vision paradigm that inherently incorporates human-like adaptive perception behaviors, and its formulation is general enough to be compatible with various network architectures (e.g., Transformers and convolutional neural networks) and tasks (e.g., as stand-alone perceptual models or as the basis of multimodal large language models, being applied to static images and videos, or interacting with dynamic environments, such as for robotics). In specific, given a visual environment (an image or video frame), AdaptiveNN formulates visual perception as a multi-step dynamic decision-making process, sequentially fixating on the regions of interest, incrementally combining information across fixations to build up a continuously updated internal representation, and actively determining when is sufficient to conclude observation. At each step, the model summarizes past information to decide if further observation is warranted; if so, the position of the next visual fixation will be selected, and the region around there will be processed with a high-capacity feature-extraction module to update the internal representation of AdaptiveNN accordingly. Similar to the human visual system, AdaptiveNN focuses resources selectively on some important parts of the visual environment captured by several fixations, whose number is dynamically adjusted depending on the difficulty of accomplishing the task at hand on top of each specific sample. The resource demands of its inference process are independent of the size, or complexity, of the visual environment to perceive. Hence, compared with the current prevailing paradigm that processes the full visual environment all at once, AdaptiveNN enables preserving the superior accuracy of large-scale neural networks with high-resolution inputs, but remains low-cost during inference by strategically selecting 'where to look', thus minimally suffering from the effectiveness-efficiency trade-off dilemma in prior methods. We build a theoretical analysis that enables training AdaptiveNN directly in end-to-end to maximize the performance measure of a given task, typically defined as a loss function. This optimization procedure is general as it does not rely on specialized task formats or additional annotations beyond the task objective itself. On the contrary, we prove that when considering optimizing perceptual behavior distributions for an arbitrary vision task, an integration of

representation learning and self-rewarding reinforcement learning naturally emerges as a major learning principle. The former trains the feature-extraction and representation-updating components of AdaptiveNN, while the latter addresses the non-differentiability of learning to select fixation positions and adaptively conclude observation.

Our comprehensive experimental evaluation reveals that AdaptiveNN, across a diverse spectrum of scenarios, demonstrates markedly improved energy efficiency, flexibility, and interpretability. These features align with the recognized advantages of human visual systems [56, 59, 7, 57, 62, 58, 76, 60]. Our key results include:

- First, on seven popular benchmarks, ranging from general large-scale real-world visual understanding to fine-grained visual recognition, AdaptiveNN with active perception capabilities achieves a computational cost reduction of 4 8× without compromising accuracy, compared to existing 'passive' [61] vision models. When considering a more general visual perception scenario in the wild processing non-object-centric road-scene images collected on real moving vehicles, AdaptiveNN exhibits a remarkable speedup of 28× in the sign recognition task.
- Second, our model is distinctive in its human-like flexibility. Confronting circumstances where the quantities of available resources vary dynamically, AdaptiveNN can adjust its inference cost online (by simply varying the statistical distributions of fixation numbers) without necessitating additional training, yielding a favorable efficiency-effectiveness trade-off among a wide range. This enables AdaptiveNN to dynamically make full use of all available resources or obtain the required performance with minimal power consumption. Besides, to mimic the cases where the demands of vision tasks are highly diversified, we consider a visual search scenario where the categories and numbers of targets can be flexibly changed. AdaptiveNN maintains an average success of  $\sim 90\%$  consistently across all different tasks, while existing methods [62, 63] usually struggle for  $\sim 20\%$ .
- Third, visualization results uncover that the model's visual fixation patterns offer a critical window into interpreting the decision-making processes of our model, which is also a major strategy for understanding human vision [56, 76, 77]. Moreover, we demonstrate the applicability of AdaptiveNN in applications where interpretability holds vital importance, such as medical diagnosis. In pneumonia detection, for example, the visual fixations of AdaptiveNN learned with only classification labels succeed in localizing lung lesions, and these results are consistent with the judgment of human clinicians.
- Last but not least, to thoroughly uncover the potential of our framework, we employ AdaptiveNN to establish an embodied multimodal large language model (MLLM). The MLLM receives text instructions as control signals, adaptively perceives the visual environments, and accordingly interacts with the environment to execute complex robot manipulation tasks. In alignment with previous findings, AdaptiveNN reduces inference cost by  $4-6\times$  without sacrificing performance, exhibits marked behavioral flexibility to adapt to diverse task instructions and fluctuating resource availability, and offers enhanced interpretability.

In summary, we believe that these superiorities of AdaptiveNN demonstrate a practical new avenue toward the next generation of energy-efficient, flexible, and interpretable computational visual perception paradigms.

Beyond the aforementioned merits, we believe that AdaptiveNN also emerges as a potent computational instrument for probing into human behavioral and learning processes. For example, we evaluate humans and AdaptiveNN side by side on the same tests of visual perception behaviors, where our model is learned exclusively on large-scale, object-centric visual recognition tasks. Our quantitative analysis reveals that, in many cases, AdaptiveNN performs mostly consistent with human vision, in terms of both the locations of visual regions it fixates on and the difficulty level it assesses to accomplish the given task based on each individual visual environment. AdaptiveNN produces nuanced human-like patterns such as being attracted by faces, hands, human bodies, or human actions. Hence, highly human-like behaviors of actively observing objects and scenes are learnable through being trained to efficiently fulfill routine vision tasks like recognition, without the guidance of other innate inductive biases (*e.g.*, biases concerning objects, agents, space, and biological motion [78, 79, 80, 81, 82, 83, 84, 85]). These insights suggest AdaptiveNN's potential as a valuable tool for advancing the understanding of some fundamental questions in human visual cognition.

![](_page_5_Figure_0.jpeg)

<span id="page-6-1"></span>Figure 2. Schematic overview of AdaptiveNN. (a) The architecture and inference procedure of AdaptiveNN. The model iteratively identifies new valuable regions to fixate on, and actively determines the appropriate time to conclude its observation. The information from all processed fixations is incrementally combined, forming a dynamic, evolving internal vision representation. The full sequential perception procedure initiates with a quick glance at the visual environment, emulating the coarse-to-fine paradigm of the conscious perception of human vision [86, 87, 88, 89, 90, 91]. (b) AdaptiveNN is compatible with a broad range of vision tasks, including both pre-defined static tasks and tasks with variable demands specified by prompt inputs (e.g., text). The behaviors of AdaptiveNN are learned under task-driven supervision signals. (c) The training of AdaptiveNN is challenging as it incorporates both continuous and discrete optimization. We address this by developing a novel theoretical analysis that decomposes the expected loss into an integration of representation learning and self-rewarding reinforcement learning objectives. Our method enables training AdaptiveNN in end-to-end without relying on specialized task formats or additional annotations beyond standard objectives. (d) In implementation, we formulate the vision agent as the combination of a policy network  $\pi$  and a value network  $V^{\pi}$ , which addresses 'where to look' and 'when to conclude observation' simultaneously, and also facilitates a stabilized reinforcement learning process.

## 2. AdaptiveNN

In this section, we first briefly describe the inference procedure and major components of our proposed AdaptiveNN framework (Section 2.1). Then we introduce its theoretical learning principles (Section 2.2). More details can be found in Section 5.

#### <span id="page-6-0"></span>2.1. Framework

**Inference of AdaptiveNN**. We start by describing AdaptiveNN's overall inference procedure. AdaptiveNN aims to drive a paradigm shift from 'passive' to 'active and adaptive' computer vision models. The key insight behind AdaptiveNN is to mimic the human visual system, modeling visual perception as a coarse-to-fine sequential decision-making process, rather than only receiving inputs 'passively' and processing all input regions in parallel.

Specifically, as shown in Fig. 2a, consider a generic visual environment structured as an  $H \times W$  scene. AdaptiveNN constructs a perception of it by recurrently attending to several selected locations within it, and incrementally combining information from these fixations over time to build up a dynamic, evolving internal vision representation of the scene. This procedure is formulated as a sequential decision-making process. At each step, a Vision Agent processes the current composite vision representation, and determines whether the observation on the environment is sufficient enough to be terminated based on the information of previous steps and the task demands. If more information needs to be acquired from the environment, Vision Agent will select the next location to fixate on; conversely, the perception process on the environment will not proceed, with the current vision representation leveraged to address the task of interest. Each selected visual fixation will be processed by a high-capacity representation learning neural network (*Perception Net*) for extracting discriminative local features to update the internal vision representation. Notably, without loss of generality, a visual fixation is defined as a  $P \times P$ patch (P < H, W) to be compatible with most modern deep learning scenarios [61, 36, 37, 38]. Furthermore, the full sequential process initiates with a quick glance, where a network coarsely processes an unknown scene in a down-sampled scale to establish an initial representation. This design is introduced inspired by the prominent theory that human vision operates in a global-to-local, coarse-to-fine manner [86, 87, 88, 89, 90, 91], where humans' initial conscious perception (vision at a glance [88]) matches a high-level, generalized, abstract scene interpretation, while later vision guides serial eye movements to attend to low-level, specific, fine receptive fields, incorporating the detailed information available there into conscious perception.

In short, mimicking human vision, AdaptiveNN observes a complex visual environment through iteratively localizing and processing visual fixations, and actively deciding when its knowledge about the scene is adequate for fulfilling the given task.

**Components of AdaptiveNN.** Here we briefly describe the major components of AdaptiveNN. Their detailed architectures are deferred to Section 5.3.2.

**Visual fixations**  $l_1, \ldots, l_t$  (at  $1^{\text{st}}, \ldots, t^{\text{th}}$  steps). AdaptiveNN never senses the visual environment in its entirety. In contrast, it extracts information from a sequence of smaller, bandwidth-limited inputs corresponding to certain local regions of the environment, named visual fixations, denoted by  $l_1, \ldots, l_t$ . AdaptiveNN actively determines the locations of  $l_1, \ldots, l_t$  step by step, under the goal of maximizing their contributions to the task

of interest, until sufficient information has been acquired. The small bandwidth of visual fixations ensures that the resource demands of AdaptiveNN can be controlled independently of the size, or complexity of the original visual environments, and will not grow dramatically with higher spatial-temporal input resolution. Consequently, visual perception can be efficient even when employing large-scale neural networks to perceive intricate real-world scenes with high frame rates. Furthermore, since the fixations are strategically localized to focus on the important visual content and new fixations will be continuously introduced until the observation is sufficient, the model performance can be maximally preserved. In some scenarios, the performance may even be improved by eliminating task-irrelevant information interference. Additionally, although we consider the most general form of square patches as fixations to ensure the generality of our framework, more advanced fixation formats may be adopted for optimization toward specific models of tasks (*e.g.*, multi-scale mixed visual fixations).

**Perception Net**  $f_{\text{rep}}$  is a representation learning backbone network that converts raw pixelated image inputs into deep representations with semantic meanings. As aforementioned, high-capacity, large-scale models can be employed as  $f_{\text{rep}}$ , to obtain strong visual processing capabilities. Since  $f_{\text{rep}}$  only needs to process the bandwidth-limited visual fixation, its inference still enjoys superior efficiency.

**Internal vision representation**  $s_1, \ldots, s_t$  is maintained during the whole visual perception process, and dynamically updated utilizing the features extracted from each visual fixation by  $f_{\text{rep}}$ , namely

$$\mathbf{s}_t = \Psi(\mathbf{s}_{t-1}, f_{\text{rep}}(\mathbf{l}_t)), \tag{1}$$

where  $\Psi(\cdot,\cdot)$  denotes the updating operator (see Section 5.3.2 for its implementation details). The internal representation  $s_t$  summarizes the information from the history of all past observations, encoding the model's current knowledge of the environment. It serves two critical purposes. First, as shown in Fig. 2b,  $s_t$  is the output of the AdaptiveNN framework, and the information within it will be utilized to fulfill the given vision task (feeding  $s_t$  into a task-specific head, detailed in Section 5.3.2). Second,  $s_t$  provides necessary information for decision-making in the sequential adaptive visual perception process, *i.e.*, deciding whether to conclude observation now, and where to look next. Both these two abilities are acquired through being trained to accomplish the vision task of interest (Fig. 2b).

Vision agent is a decision-making neural network that receives the internal vision representation  $s_1, \ldots, s_t$  as inputs. At each step of the sequential perception process, it makes two decisions: assessing whether to terminate the ongoing observation and, if necessary, determining the subsequent visual fixation location. To achieve both of them simultaneously, we formulate the vision agent as the combination of a policy network  $\pi$  and a value network  $V^{\pi}$  (Fig. 2d). This formulation is naturally derived from the theoretical learning principles of AdaptiveNN, which will be discussed in Section 2.2 coupled with the training algorithm of  $\pi$  and  $V^{\pi}$ . Here we first introduce the inference process of  $\pi$  and  $V^{\pi}$ . At  $t^{\text{th}}$  step of inference, the outputs of  $\pi$  parameterize a distribution from which we can sample the location of  $t_{t+1}$ , namely

<span id="page-7-0"></span>
$$\boldsymbol{l}_{t+1} \sim p_{\boldsymbol{\pi}}(\boldsymbol{l}_{t+1}|\boldsymbol{s}_t). \tag{2}$$

Paired with  $\pi$ , the value network  $V^{\pi}$  employs  $s_t$  to predict the expected gains of performing further observation on top of  $s_t$  (i.e., further updating  $s_t$ ) using  $\pi$ , yielding a state value  $V^{\pi}(s_t)$ . We compare  $V^{\pi}(s_t)$  with a threshold  $\eta_t$ . If  $V^{\pi}(s_t) \leq \eta_t$ , we are indicated that further observing is not valuable enough, and the sequential perception process will be concluded. Otherwise,  $V^{\pi}(s_t) > \eta_t$  reveals that more fixations may yield significant improvements, and thus the new fixation  $l_{t+1}$  will be processed, evoking the  $(t+1)^{\text{th}}$  step. The value of  $\eta_t$  is solved on the validation data, and can be adjusted online to vary the average resource demands of AdaptiveNN without additional training (see Section 5.3.1). Notably, the outputs of  $\pi$  and  $V^{\pi}$  consider both the current specific situations as the observation on each particular visual environment progresses, as well as the demands of the given vision task. The former has been encoded into  $s_t$ , while the latter is attained through the training process (see Section 2.2), where  $\pi$  and  $V^{\pi}$  can either learn to address a pre-defined static task, or learn to adapt to variable task demands on top of a prompt input (e.g., text), as depicted in Fig. 2b. Moreover, it is noteworthy that  $V^{\pi}(s_t)$  reflects the model's subjective assessments, namely whether the perception process of AdaptiveNN itself is worth proceeding, while  $\eta_t$  determining if  $V^{\pi}(s_t)$  is sufficiently small represents the objective constraints imposed by the external environment, e.g., the extent to which the overall available resources for visual perception

are adequate in the current circumstance. This decoupled modeling of subjective and objective factors enables more flexible usage of our framework.

Compatibility with various network architectures and vision tasks. Importantly, the major goal of developing AdaptiveNN is to facilitate a paradigm shift toward active and adaptive visual perception models. Therefore, its formulation has been designed to be general and flexible. For example, various off-the-shelf network architectures, such as Transformers and convolutional neural networks, can be conveniently deployed as the feature-extraction module of AdaptiveNN. Moreover, as shown in Fig. 2b, the internal vision representation of AdaptiveNN does not adopt a strong assumption on its application scenario, and may be implemented under diverse task settings, *e.g.*, employing AdaptiveNN as stand-alone perceptual models or as the basis of multimodal large language models, being applied to static images and videos, or interacting with dynamic environments, such as for robotics. In Section 3, we provide comprehensive evaluation results to support these claims.

#### <span id="page-8-0"></span>2.2. Theoretical learning principles

Training AdaptiveNN incorporates both continuous (*e.g.*, extracting feature from visual fixations) and discrete (*e.g.*, learning to select fixation positions and adaptively conclude observation) optimization. This can not be straightforwardly solved by standard algorithms like gradient back-propagation. To address optimization challenges, we present a theorem that enables training AdaptiveNN in end-to-end (Fig. 2c).

**Formulation.** Given an AdaptiveNN model parameterized by  $\boldsymbol{\theta}$  and a visual environment  $\boldsymbol{X}$  to perceive, we refer to the distribution of the locations of visual fixation  $\boldsymbol{l}_1,\ldots,\boldsymbol{l}_t$  as  $p(\boldsymbol{l}_{1:t}|\boldsymbol{\theta},\boldsymbol{X})$ . On top of this, given a vision task, the model's outputs at  $t^{th}$  step for accomplishing the task (stemming from the internal vision representation  $\boldsymbol{s}_t$ ) are denoted as  $q(\boldsymbol{\theta},\boldsymbol{X},\boldsymbol{l}_{1:t})$ , e.g., output logits for classification. Then, for a label y associated with  $\boldsymbol{X}$ , which is defined upon the task, assume that we have a performance measure (typically a loss function)  $\mathcal{L}(y,q(\boldsymbol{\theta},\boldsymbol{X},\boldsymbol{l}_{1:t}))$ , such as the cross-entropy loss for classification and the mean squared error for regression.

**Optimization objective**. During training, AdaptiveNN focuses on learning a model capable of sequentially attending to proper visual fixations within a complex visual environment, and extracting information from these fixations to accomplish the vision task of interest. Its optimization objective is defined as minimizing the expected performance measure of the task, namely

<span id="page-8-1"></span>
$$\text{minimize} \quad \mathbf{L}(\boldsymbol{\theta}) = \mathbb{E}_{\boldsymbol{X}, y, t_{o} \sim p(t_{o})} \int_{\boldsymbol{l}_{1:t_{o}}} p(\boldsymbol{l}_{1:t_{o}} | \boldsymbol{\theta}, \boldsymbol{X}) \mathcal{L}(y, q(\boldsymbol{\theta}, \boldsymbol{X}, \boldsymbol{l}_{1:t_{o}})). \tag{3}$$

Here  $t_o \sim p(t_o)$ ,  $t_o \in \{1, \dots, T\}$  indicates that during training, the total length  $t_o$  of the sequential perception process is sampled from a fixed prior distribution  $p(t_o)$ , which reflects the training process's statistical-level preference on the perception procedure's length. This consideration is introduced to add analytical flexibility to our model. Besides, note that we do not explicitly formulate the actions of actively concluding observation in Eq. (3). Conversely, we will demonstrate that the ability to evaluate when the observation is sufficient can be conveniently acquired on top of the model learned by minimizing Eq. (3).

<span id="page-8-3"></span>**Theorem 1.** (see Section 5.1 for proof) The gradients of  $L(\theta)$  can be decomposed into a combination of representation learning and self-rewarding reinforcement learning objectives:

$$\nabla_{\boldsymbol{\theta}} L(\boldsymbol{\theta}) = \nabla_{\boldsymbol{\theta}} L_{rep}(\boldsymbol{\theta}) + \nabla_{\boldsymbol{\theta}} L_{rl}(\boldsymbol{\theta}), \tag{4}$$

<span id="page-8-2"></span>where

$$\nabla_{\boldsymbol{\theta}} \mathbf{L}_{\text{rep}} = \underbrace{\mathbb{E}_{\boldsymbol{X}, y, \boldsymbol{l}_{1:T}} \sum_{t=1}^{T} P(t_{\text{o}} = t) \nabla_{\boldsymbol{\theta}} \mathcal{L}(y, q(\boldsymbol{\theta}, \boldsymbol{X}, \boldsymbol{l}_{1:t}))}_{\text{representation learning}},$$

$$\nabla_{\boldsymbol{\theta}} \mathbf{L}_{\text{rl}} = \underbrace{-\mathbb{E}_{\boldsymbol{X}, y, \boldsymbol{l}_{1:T}} \sum_{t=1}^{T} \left[ \left( \sum_{t'=t}^{T} r_{t'} \right) \nabla_{\boldsymbol{\theta}} \log p(\boldsymbol{l}_{t} | \boldsymbol{\theta}, \boldsymbol{X}, \boldsymbol{l}_{1:(t-1)}) \right]}_{\text{self-rewarding reinforcement learning}},$$

$$r_{t'} = -P(t_{\text{o}} = t') \mathcal{L}(y, q(\boldsymbol{\theta}, \boldsymbol{X}, \boldsymbol{l}_{1:t'})).$$
(5)

In Eq. (5),  $\nabla_{\theta} L_{rep}$  is a standard form of representation learning, namely minimizing the task loss over the features extracted from  $l_1, \ldots, l_t$  by the model. Additionally,  $\nabla_{\theta} L_{rl}$  boiling down to a form of policy gradients in reinforcement learning [92], where  $p(l_t|\theta, \boldsymbol{X}, l_{1:(t-1)})$  is the action distribution,  $r_{t'}$  is the reward received at each time step, and  $\sum_{t'=t}^{T} r_{t'}$  is the cumulative reward following the execution of an action  $l_t$ . Since  $r_{t'}$  is defined using the negative values of task loss of the model itself, we name  $L_{rl}$  as the self-rewarding reinforcement learning objective.

In conclusion, Theorem 1 reveals that when considering minimizing the expected loss of AdaptiveNN over a vision task, an integration of representation learning and self-rewarding reinforcement learning objectives naturally emerges. The former trains the model to extract deep representations from input visual fixations, while the latter guides the model to strategically select fixation locations within the complex visual environment to minimize the loss. Notably, both of them only leverage the standard task loss, without relying on specialized task formats or additional annotations.

**Specific learning algorithm.** Given Theorem 1,  $\nabla_{\theta}L_{rep}$  can be directly utilized as the gradient signals for learning feature-extraction modules. For the policy gradients  $\nabla_{\theta}L_{rl}$ , as reinforcement learning problems are usually more challenging to solve, we propose an augmented version of its basic formulation. First, we introduce a pre-defined discount factor  $\gamma \in [0,1]$  [93, 92, 94] and a differential form of rewards, aiming to achieve a flexible modeling of balancing long-term and short-term returns, as well as to stabilize the training process. Thus, on top of Eq. (2) and Eq. (5), the policy gradient rule for updating the model can be expressed as

$$\nabla_{\boldsymbol{\theta}} \mathbf{L}_{\text{rl}} = -\mathbb{E}_{\boldsymbol{X}, y, \boldsymbol{l}_{1:T}} \sum_{t=1}^{T} \left[ \left( \sum_{t'=t}^{T} \gamma^{t'-t} \left( r_{t'} - r_{t'-1} \right) \right) \nabla_{\boldsymbol{\theta}} \log p_{\boldsymbol{\pi}}(\boldsymbol{l}_{t} | \boldsymbol{s}_{t-1}) \right],$$

$$r_{t'} = -P(t_{0} = t') \mathcal{L}(y, q(\boldsymbol{\theta}, \boldsymbol{X}, \boldsymbol{l}_{1:t'})),$$
(6)

<span id="page-9-1"></span><span id="page-9-0"></span>where we have (see Section 5.1 for the proof)

$$\lim_{\gamma \to 0} \nabla_{\boldsymbol{\theta}} \mathcal{L}_{\text{rl}} = -\mathbb{E}_{\boldsymbol{X}, y, \boldsymbol{l}_{1:T}} \sum_{t=1}^{T} r_{t} \nabla_{\boldsymbol{\theta}} \log p_{\boldsymbol{\pi}}(\boldsymbol{l}_{t} | \boldsymbol{s}_{t-1}),$$

$$\lim_{\gamma \to 1} \nabla_{\boldsymbol{\theta}} \mathcal{L}_{\text{rl}} = -\mathbb{E}_{\boldsymbol{X}, y, \boldsymbol{l}_{1:T}} \sum_{t=1}^{T} r_{T} \nabla_{\boldsymbol{\theta}} \log p_{\boldsymbol{\pi}}(\boldsymbol{l}_{t} | \boldsymbol{s}_{t-1}).$$
(7)

When  $\gamma \to 0$ , the strategy for selecting the next visual fixation tends to be fully short-sighted and is only optimized to maximize the immediate reward  $r_t$ . Conversely,  $0 < \gamma < 1$  tends to encourage perception strategies that maximally attain the goal within a limited number of fixations. When  $\gamma = 1$ , AdaptiveNN only focuses on maximizing the final reward  $r_T$ , corresponding to the scenarios where abundant resources or energy are available, while the perception process can leverage as many visual fixations as possible to accomplish the task.

Moreover, we introduce a value network  $V^{\pi}$  to offer a baseline function for reinforcement learning [95, 94], which can effectively stabilize training by reducing gradient estimation variance [93, 96]. The learning objective of  $V^{\pi}$  is to predict the expected gains of further observing at each step:

minimize 
$$\mathbb{E}\left[V^{\pi}(s_{t-1}) - \sum_{t'=t}^{T} \gamma^{t'-t} (r_{t'} - r_{t'-1})\right]^{2}$$
. (8)

Besides, with this goal,  $V^{\pi}(s_{t-1})$  provides a reasonable proxy measure for adaptive termination, as stated in Section 2.1. For example, a relatively small  $V^{\pi}(s_{t-1})$  indicates that even if the model processes more visual fixations, the loss  $\mathcal{L}$  measuring the performance of the given task will not show notable further reduction. Hence, it is reasonable to consider concluding observation at that time.

![](_page_10_Figure_1.jpeg)

<span id="page-10-0"></span>Figure 3. Results of ImageNet large-scale real-world visual understanding. (a) Qualitative assessment showcasing the visual fixations localized by AdaptiveNN(-DeiT-S), with boxes marking the locations of fixations and colors indicating the model's decision to conclude (green) or continue (red) observation at each step. Step indices are presented at the top left of the boxes. Ground truth labels are displayed at the bottom left of the images. (b-c) Quantitative comparisons of AdaptiveNN and traditional non-adaptive models on top of identical backbones: Top-1 validation accuracy versus average computational cost for inferring the model. To obtain non-adaptive models with varying costs, we consider two common approaches: adjusting model sizes and input resolutions. (d) Relationship between validation accuracy and the number of visual fixations, assuming that all samples utilize the same number of visual fixations. (e) Proportions of data that utilize different numbers of visual fixations, set against different budget constraints for computational costs. \*One-way analysis of variance (ANOVA) with Tukey's honestly significant difference (HSD) test. Error bars show the standard deviations of five independent trials with different random seeds.

## <span id="page-11-0"></span>**3. Results**

We comprehensively evaluate AdaptiveNN on 17 benchmarks organized into 9 different tasks. These benchmarks include large-scale visual understanding, fine-grained visual recognition, visual search, processing images from real driving and medical scenarios, multimodal large language models for language-driven embodied robot execution, and side-by-side comparisons of humans' visual perception behaviors with our model's capabilities. See Section [5.2](#page-22-0) for more details of these tasks (including data collection, annotations, metrics, and other setups). Employing a wide spectrum of evaluation tasks enables us to arrive at a more complete picture of the characteristics, efficacy, and potential values of AdaptiveNN.

## **3.1. Large-scale real-world visual understanding**

Our first evaluation of AdaptiveNN considers the foundational visual understanding task, namely mapping pixelated visual signals (*e.g.*, objects and scenes) to abstract concepts. We employ the ImageNet image recognition benchmark [\[35\]](#page-36-0), which is widely acknowledged for its critical role in assessing the efficacy of machine learning methods [\[36,](#page-36-1) [37,](#page-36-2) [38\]](#page-36-3). Comprising over 1.28 million images classified into 1,000 categories according to the WordNet hierarchy [\[97\]](#page-39-9), ImageNet encompasses a diverse array of visual content, including various objects, buildings, humans, animals, scenes, etc., offering a robust platform for evaluating visual understanding capabilities. Our model is trained to correctly classify the input image. Notably, the feature-extraction networks within AdaptiveNN are compatible with most existing backbones (see Section [5.3.2](#page-26-0) for details). To demonstrate the generalizability of AdaptiveNN, we deploy two examples, ResNet (convolutional network) [\[36\]](#page-36-1) and DeiT (vision Transformer) [\[98\]](#page-39-10), each representing a wide range of popular architectures.

Fig. [3a](#page-10-0) illustrates the learned visual perception behaviors of AdaptiveNN when applied to the ImageNet visual recognition task. Observations reveal that both the locations of visual fixations and the length of observation processes for different samples are reasonable and interpretable. Our model acquires the capability of fixating on the class-discriminative regions, such as the heads of animals, the principal structures of musical instruments, and the functional parts like knobs and nozzles on coffee machines. Moreover, in scenarios involving complex or atypical visual inputs, AdaptiveNN adjusts by extending the duration of observation to enhance the accuracy of its predictions. This adaptive behavior is particularly evident when the objects of interest are small, located at a significant distance from the camera, or depicted from uncommon perspectives, showcasing only parts of their entirety.

Quantitatively, introducing human-like adaptive visual perception to computer vision models substantially enhances both their energy efficiency and adaptability. This can be illustrated by Fig. [3b-3c](#page-10-0), and Supplementary Data Tab. 2-5, where the performance of our model, equipped with AdaptiveNN, is compared against the traditional, non-adaptive counterparts on top of the same backbones. The sole distinction lies in the implementation of AdaptiveNN. DeiT-S and ResNet-50 maximally achieve validation accuracies of 81.6% and 79.1% at the computational costs of 15.5 and 12.1 GFLOPs per image. In contrast, AdaptiveNN-DeiT-S and AdaptiveNN-ResNet-50 performs on par with them at 2.86 and 3.37 GFLOPs per image, which are 5.4× and 3.6× more efficient, respectively. Moreover, the computational cost of AdaptiveNN can be flexibly adjusted online, resulting in a favorable balance between efficiency and effectiveness across broad ranges. This adaptability is in contrast to non-adaptive models, which typically require retraining to achieve similar performance adjustments.

Fig. [3d](#page-10-0) and Supplementary Data Tab. 6-7 report the validation accuracies with all samples processed using the same number of visual fixations. Progressively leveraging more fixations improves accuracy significantly (all P*<*0.005), yet the effects gradually diminish, indicating increased difficulty in further boosting accuracy upon a decent performance. Fig. [3e](#page-10-0) and Supplementary Data Tab. 8-9 illustrate how AdaptiveNN adapts to the dynamically varied quantities of available resources. In scenarios where computational resources are abundant, the model can afford to allocate numerous visual fixations to most samples, thereby optimizing the overall accuracy. In contrast, when computational resources are constrained, prioritization is given to the more challenging samples, while other samples are allocated fewer resources to compensate for the limited budget. This strategic allocation underscores AdaptiveNN's adaptability in managing resource distribution to maximize performance efficiency.

#### 3.2. Fine-grained visual recognition

Beyond the general-purpose large-scale visual understanding task, we further probe into AdaptiveNN's nuanced visual discriminative capabilities using six fine-grained recognition tasks. These tasks are characterized by the small differences between classes and significant variations within each class, such as differentiating between visually very close species of birds or pets against highly diversified backgrounds. Accomplishing them necessitates AdaptiveNN to localize and identify minor, task-dependent signals out of an extensive or even overwhelming multitude of irrelevant visual information. This ability to filter and focus on pertinent details mirrors a key strength of human visual systems [59, 7, 57, 62, 58, 76, 60], revealing the model's potential to approach the nuanced perceptual capabilities observed in human cognition.

Extended Data Fig. A1a and Supplementary Data Tab. 10-15 summarize the quantitative evaluation results. Similar to Fig. 3b and 3c, our model's performance in terms of energy efficiency and adaptability is benchmarked against that of conventional, non-adaptive models. Both AdaptiveNN and the baselines are fine-tuned from the checkpoints pre-trained on ImageNet. AdaptiveNN dramatically saves the computational cost of the model without sacrificing accuracy (multiple of reduction:  $6.2 \times, 6.1 \times, 7.6 \times, 8.2 \times, 5.8 \times, 6.3 \times$ ). This efficiency gain surpasses those observed on ImageNet, underscoring our model's human-like proficiency in fixating on and leveraging nuanced discriminative features.

Furthermore, the behaviors of our model demonstrate good interpretability. As depicted in Extended Data Fig. A1b-A1e, AdaptiveNN autonomously learns to localize the details valuable for fine-grained recognition, such as the beaks of birds, the car lights, the airplane engines, the propellers, etc. This proficiency is notably achieved through training that is guided solely by image-level category labels, without explicit instructions on the spatial details to focus on. In some difficult scenarios, where the primary discriminative features may be concealed or indistinct, our model can actively determine to observe with more fixations, seeking additional secondary features as alternative cues for accurate classification.

#### 3.3. Efficient processing of visual data from real driving scenarios

The ImageNet and fine-grained recognition datasets are standard visual understanding benchmarks collected from the Internet. Consequently, in general, many images within them have been centered toward the relevant objects or content by human photographers and users. Nevertheless, AdaptiveNN does not rely on this object-centric precondition. Similar to human visual systems, AdaptiveNN is applicable to more general and complex scenarios, for example, efficiently processing non-object-centric images collected in the wild without specified pre-processing. To demonstrate this, we evaluate our model with the traffic sign recognition task on the Swedish traffic signs dataset (STSD) [99]. The dataset consists of high-resolution road-scene images collected on real moving vehicles. The traffic signs of interest are usually very small, distributed diversely, and not clear in many cases, presenting a realistic challenge.

For this more generalized task of visual perception in natural environments, AdaptiveNN markedly outperforms traditional non-adaptive models, achieving efficiency gains greater than an order of magnitude. Illustrated in Fig. 4a and Supplementary Data Tab. 16-17, the strongest baseline, ResNet-50 with 960 $^2$  inputs, acquires an accuracy of 90.2% with  $\sim$ 76 GFLOPs/image, while our model performs on par with it using only  $\sim$ 2.7 GFLOPs/image, yielding an 27.9× reduction of inference cost. This substantial enhancement can be elucidated through the qualitative analysis in Fig. 4b. The visual fixations localized by our model adaptively center its 'retina' on the small, task-relevant regions within the expansive, intricate, and cluttered visual scenes, mirroring the efficiency characteristic of human visual perception. Conversely, conventional non-adaptive models typically process all pixels equivalently, which is inefficient and vulnerable to overfitting. Moreover, when AdaptiveNN initially misidentifies the location of traffic signs, it tends to recognize its error, infer the signs' true locations based on the current information, and attempt to rectify this mistake in subsequent fixations.

#### 3.4. Addressing vision tasks with flexible requirements

Even when confronting the same visual environment, humans can flexibly adjust their visual perception behaviors, such as the locations and numbers of fixations, in response to the specific requirements of the task at hand [7, 57, 9]. To investigate whether AdaptiveNN can acquire such human-like adaptability, we considered a visual search

![](_page_13_Figure_0.jpeg)

<span id="page-13-0"></span>Figure 4. Assessment of AdaptiveNN in more general visual perception scenarios: processing images from real driving and medical scenarios, and visual search tasks with variable demands. (a) Comparisons of AdaptiveNN and conventional non-adaptive models in processing complicated, non-object-centric real-world scenes: Top-1 validation accuracy versus average computational cost for inferring the model (log-scale). We consider the traffic sign recognition task on the Swedish traffic signs dataset (STSD) [99], composed of 960×1,280 road-scene images collected on real moving vehicles. ResNets are deployed as backbones since convolutional networks tend to be more efficient for processing high-resolution inputs. (c) Average success rates of visual search tasks. Here 'n digits' indicates the number of target digits, while the bars indicate the mean  $\pm$  standard deviations of five randomly generated visual search tasks with various target categories (yet maintaining a constant number of targets). Success is defined as accurately retrieving exactly all the digits specified by a given task. (e) Area under the receiver operating characteristic curve (AUROC) of the RSNA pneumonia detection task [100]. All models are trained to predict the presence or absence of pneumonia based on image-level labels. Here we do not perform adaptive termination in AdaptiveNN and mainly focus on the AUROC after processing all fixations, since efficiency may not be a major focus of medical diagnosis tasks. (b-f) Qualitative evaluation results corresponding to (a-c). Boxes represent visual fixation locations, with colors indicating the model's decision to either continue (red) or terminate (green) observation at that step. Step indices are annotated at the upper left corner of each box. Particularly, in f, lighter boxes show the pneumonia regions annotated by human clinicians (this localization information is not utilized for training). \*Independent samples t-test. Except for (c), all error bars show the standard deviations of five independent trials with different random seeds.

scenario: we generate 2242 images, each randomly populated with 6 to 10 digits against a black background without repetition of digits. A model is trained to identify the locations of certain given digits within each input, where the categories and number of targets are assumed to be flexibly changed. Each specified setup of targets is defined as an individual visual search task.

Fig. [4c](#page-13-0) and Supplementary Data Tab. 18 summarizes the quantitative evaluation results. We estimate the success rate of retrieving exactly all the targets demanded by a visual search task in various visual environments, and report its expected value over different tasks. AdaptiveNN maintains an average success accuracy of approximately 90% consistently with a varying number of searching targets. On the contrary, existing popular models that aim to mimic human sequential visual perception, like RAM [\[62\]](#page-37-11) and DRAM [\[63\]](#page-37-12), generally do not exceed a success rate of around 20%, markedly underperforming AdaptiveNN by *>*4.5× in most instances. Moreover, Fig. [4d](#page-13-0) illustrates AdaptiveNN's capability to adaptively modulate its fixation selection and observation termination strategies conditioned on each input and specific visual task. It does not fixate on more regions after all targets have been localized, and intriguingly, it learns to efficiently identify two adjacent targets using a single fixation. In general, these observations show that AdaptiveNN can acquire a robust human-like adaptability in task-specific visual perception behaviors, considerably outperforming previous works.

## **3.5. Interpretability-critical tasks: image processing in medical scenarios**

A predominant merit of AdaptiveNN is its capacity for enhanced interpretability through examining its visual fixation patterns (as shown in Fig. [3a](#page-10-0), [4b](#page-13-0), [4d](#page-13-0), and Extended Data Fig. [A1b-A1e](#page-30-0)). Built upon this insight, we further evaluate our model's utility in vision tasks where interpretability is of vital importance. Specifically, we take the medical diagnosis scenario of detecting pneumonia from chest X-ray images as a representative example [\[100\]](#page-39-12). AdaptiveNN is trained only using the image-level labels indicating the presence or absence of pneumonia. As demonstrated in Fig. [4e](#page-13-0), it exhibits a significantly superior AUROC (area under the receiver operating characteristic curve) on validation data than the conventional non-adaptive model (P*<*0.0001). Furthermore, despite the absence of explicit localization guidance during training, the visual fixations identified by AdaptiveNN (see Fig. [4f](#page-13-0)) align closely with the pulmonary opacity regions annotated by human clinicians (18 board-certified radiologists from 16 institutions). This concordance reveals the potential value of AdaptiveNN in developing AI applications that demand not only precision but also good interpretability, such as medical applications.

## **3.6. Embodied multimodal large language models based on AdaptiveNN**

The formulation of AdaptiveNN is general enough to be deployed as the perceptual module of an embodied agent that interacts with dynamic physical environments. To demonstrate its versatility, we deploy AdaptiveNN on top of a multimodal large language model (MLLM). As illustrated in Fig. [5a](#page-15-0), the MLLM receives language as prompt inputs to specify the task of interest, observe the current environment (with or without AdaptiveNN), and update the recurrent policy network to execute an appropriate action, subsequently affecting the environments, evoking the next observation-to-action process. The MLLM's network architecture is based on RoboFlamingo [\[101\]](#page-39-13), as detailed in Extended Data Fig. [A4a](#page-33-0) and Section [5.3.3.](#page-27-0) We assess the performance of AdaptiveNN-based MLLM using the CALVIN Long-Horizon Multi-Task Language Control benchmark (LH-MTLC) [\[102\]](#page-39-14), where an agent aims to successfully complete task sequences, each comprising five subtasks described in natural language. The model performance is quantified as the average successful length (0 to 5) across 1000 task sequences, as detailed in Extended Data Fig. [A4b](#page-33-0). We consider two benchmark settings using identical validation tasks and different training data scales (*i.e.*, D→D, ABCD→D).

Fig. [5b](#page-15-0) and Supplementary Data Tab. 19-20 demonstrate that AdaptiveNN saves the average computational cost by 4*.*4 − 5*.*9× without sacrificing effectiveness compared to the non-adaptive baselines. Besides, AdaptiveNN is notably more flexible in adjusting its computational cost online without necessitating retraining. Supplementary Data Tab. 21-22 further compare AdaptiveNN and the baselines in terms of the success rates of different types of tasks. Fig. [5c](#page-15-0) and Supplementary Data Tab. 23-24 report the performance corresponding to each fixed number of visual fixations, depicting a progressively increasing trend of average successful length, which is more pronounced for large-scale, diverse training data such as ABCD→D. Representative qualitative results are presented in Fig. [5d](#page-15-0). AdaptiveNN succeeds in learning to fixate on the task-relevant objects specified by the input language prompts, as well as to capture their interactions with the robotic operational components. In other words, AdaptiveNN dynamically determines 'where to look' based on both the visual environments themselves and the variable

![](_page_15_Figure_1.jpeg)

<span id="page-15-0"></span>Figure 5. Performance of the embodied multimodal large language models (MLLM) based on AdaptiveNN. (a) A schematic overview of the embodied MLLM. The prompt input specifics the task, and the MLLM iteratively perceives the environment to execute appropriate robotic actions, i.e., six-degree-of-freedom (6-DoF) transformation vectors in 3D space. The next observation reflects the outcome of the preceding actions. A recurrent policy head integrates information from all previous observations. (b) Comparisons of AdaptiveNN-based MLLM and non-adaptive MLLM using identical backbones on CALVIN: average successful length (of 1000 5-task sequences) versus average computational cost for inferring the model. For the non-adaptive models, computational costs are modulated by adjusting model sizes. D→D and ABCD→D indicate different scales of training data. (c) Relationship between average successful rates of each subtask within task sequences and the number of visual fixations, assuming that all samples utilize the same number of visual fixations. (d) Qualitative assessment of two representative 5-task sequences. Boxes marks the fixation locations and colors indicates the model's decision to conclude (green) or continue (red) observation at each step. Step indices are presented at the top left of the boxes. The prompt inputs for specifying tasks are displayed within the black boxes. All error bars show the standard deviations of five independent trials with different random seeds.

task prompts. Moreover, when the required action involves precise and fine-grained control, our model tends to leverage more visual fixations for more careful perception. Otherwise, relatively fewer fixations will be adopted to maximally save the cost. In summary, AdaptiveNN exhibits significantly improved computational efficiency, adaptability, and interpretability. These marked merits are consistent with our previous findings.

#### 3.7. Comparisons between AdaptiveNN and human visual perception

AdaptiveNN also emerges as a potent computational tool for probing human visual cognition under controlled experimental conditions. This can be uncovered with the marked consistency between humans and AdaptiveNN in side-by-side evaluations on the same tests of visual perception behaviors. Our experimental protocols are detailed in Section 5.2.3.

First, we examine the consistency of the locations of visual regions that humans and AdaptiveNN fixate on. We assess the spatial-wise adaptiveness of human vision using the SALICON benchmark dataset [103], which comprises images each observed by approximately 60 participants recruited via Amazon Mechanical Turk (AMT). These participants were asked to freely view each image for 5 seconds without specific instructions on where to direct their gaze, allowing for an unbiased recording of their visual gazing center points. In Fig. 6a, we utilize the aggregate density map of all  $\sim$ 60 observers' gazing points as the ground truth, against which we evaluate the probability that the real focal centers of human vision fall into the visual fixation regions localized by AdaptiveNN. Here AdaptiveNN adopts the same experimental paradigm as humans: it produces a fixed number of fixations, and it is learned on ImageNet, having never seen or been trained using the data in SALICON. In addition to our model, we also report the performance corresponding to the expected behavior of selecting visual fixation regions following the gazing locations of an arbitrary single human (one of  $\sim$ 60 observers) or uniformly at random, denoted by 1.0 (human) and 0.0 (random) on the y-axis of Fig. 6a, yielding a metric named 'normalized human-like score' (see Section 5.2.3 for details).

Fig. 6a and Supplementary Data Tab. 25-26 summarize the quantitative results on the two splits of SALICON. In terms of the alignment with the ground truth spatial-adaptive human visual perception behaviors, AdaptiveNN performs on par with or surpasses that of an average individual human observer, consistently registering normalized human-like scores exceeding 1.0. In contrast, pre-defined fixation localization policies and CAM-based methods yield scores ranging from -0.1 to 0.4, generally failing to significantly outperform a random selection strategy. Furthermore, Fig. 6b offers a qualitative comparison between the ground truth density maps of human gazing locations (heat maps) and the fixation regions selected by AdaptiveNN (boxes). Our model produces human-like patterns in many cases, frequently being attracted by faces, hands, human bodies, human actions, or objects intimately associated with human activity, such as food, computers, skateboards, tennis rackets, and buses. This underscores our model's ability to emulate sophisticated perceptual behaviors that reflect complex human visual strategies. Rather interestingly, these human-like patterns emerge from solely being trained on the ImageNet image recognition task, without reliance on the typical inductive biases innate to human cognition (*e.g.*, biases concerning objects, agents, space, and biological motion [78, 79, 80, 81, 82, 83, 84, 85]).

Second, orthogonal to investigating spatial adaptiveness, we examine the extent to which AdaptiveNN aligns with human judgments in assessing which visual environments are more challenging for a given task and necessitate more thorough scrutiny. Specifically, human participants (n=10) were tasked with rating each image within six representative categories of the ImageNet validation set, according to the difficulty of classifying each image. In Fig. 6c and Supplementary Data Tab. 27, these human-assessed difficulty scores are normalized on an individual basis, averaged across participants, and then compared against the normalized state values predicted by the AdaptiveNN learned on ImageNet, which reflect our model's assessments of each image's difficulty level. The evaluations made by AdaptiveNN demonstrate a strong correlation with human judgments (all P<0.0001; Pearson correlation coefficient  $\rho \in [0.54, 0.80]$ ). Fig. 6d further presents qualitative examples of the relatively 'easy' and 'difficult' data identified by our model. The visualizations are generally reasonable; images depicted from typical perspectives with clear, relevant content tend to be deemed 'easy'. These findings suggest that our model closely approximates human-like proficiency in dynamically allocating visual perception resources across varied visual environments – a critical characteristic of human visual systems.

Finally, we establish several 'visual Turing tests' [22] to compare AdaptiveNN with human vision. In these tests, human judges are given paired examples of visual perception behaviors from humans and our model, and

![](_page_17_Figure_0.jpeg)

<span id="page-18-0"></span>Figure 6. Behavioral comparisons between AdaptiveNN and human vision. (a) Normalized human-like scores, which quantify the probability that the ground truth gazing centers of human vision (whose distribution is estimated by averaging across ~60 observers' visual perception behaviors) fall into the visual fixation regions localized by AdaptiveNN or comparative strategies. The raw results are normalized with respect to the expected performance of selecting fixation regions with the gazing locations of an individual human observer (1.0 on the y-axis) and uniformly at random (0.0 on the y-axis). Each point represents the result over a mini-batch of data, while boxplots depict the distribution of results. The evaluation is based on the SALICON dataset [103]. Our model is trained on ImageNet, having never seen the data in SALICON. This 'zero-shot' paradigm evaluates directly transferring AdaptiveNN's perceptual behaviors to novel, complex environments, with a fixed number of fixations, mirroring the collection procedure of human gazing centers. Baselines for comparison incorporate selecting fixation regions using i) pre-defined rules; ii) class activation maps (CAMs); iii) CAMs augmented with a Gaussian mixture model (GMM). See Supplementary Section B for the details of these baselines. †: GMM introduces additional computation. \*Independent samples t-test. (b) Qualitative comparisons between the ground truth density maps of human gazing centers (heat maps) and AdaptiveNN fixation regions (boxes). Boxes indicate fixation locations, with step indices annotated at the upper left corner of each box. (c) Correlation of human-assessed difficulty scores (averaged across n=10 subjects) and difficulty levels (state values) evaluated by the Vision Agent of AdaptiveNN. Without loss of generality, the state values are taken from the first step of sequential perception processes. Results are based on six representative categories of data in the ImageNet validation set.  $\rho$ : Pearson correlation coefficients. \*Correlation t-test. (d) Visualization examples of the 'easy' and 'difficult' data identified by AdaptiveNN. (e-f) Results of 'visual Turing tests'. Human judges (n=39) are randomly given paired examples of visual perception behaviors from 'humans' and 'one within {AdaptiveNN, humans, random behaviors}'. They are instructed to identify the machine (even in cases when the pairs of 'random v.s. human' or 'human v.s. human' are given, which serve as control groups for comparison). Bars show the mean accuracy across human judges and the corresponding 95% confidence interval. Ideal performance is 50%, where the machine is indistinguishable from human behaviors in these binary choice tasks.

instructed to identify which come from the machine. The results are evaluated using the accuracy of human judgments: 50% indicates perfectly human-like behaviors that are indistinguishable from humans, while 100% represents the worst case. Each participant (n=39) has completed 216 trials with blocked feedback to investigate both the spatial-wise visual fixation behaviors and the sample-wise visual difficulty assessment behaviors of AdaptiveNN, with the judgments analyzed individually and in aggregate. Importantly, we randomly replace the 'machine' behaviors of some trials with 'human' or 'random' behaviors without letting participants know, and separately evaluate the accuracy of these trials, establishing two randomized control groups as baselines for comparison. The detailed procedure of 'visual Turing tests' is illustrated in Extended Data Fig. A3a. Some examples of the trials can be found in Supplementary Data Fig. 1-2.

The results are summarized in Fig. 6e-6f, Extended Data Fig. A3b, Supplementary Data Tab. 28-29, and Supplementary Data Fig. 3-4. In both scenarios, human judges achieve only 50-51% accuracies in correctly identifying 'AdaptiveNN v.s. human', which do not acceptably outperform random guessing in statistics (t(38) = 0.90, -0.09, P = 0.37, 0.93). Additionally, these 'machine v.s. human' judgments do not exhibit a significant difference from the 49-50% accuracies of the 'human v.s. human' baselines (t(38) = 0.97, 0.40, P = 0.33, 0.69). In contrast, 'random v.s. human' results in considerably easier Turing test tasks (accuracies  $\geq 80\%$ ). These observations demonstrate that in general, AdaptiveNN approaches an indistinguishable level from the adaptive perceptual behaviors of human vision.

## 3.8. Ablation studies and analysis of the design of AdaptiveNN

In pursuit of a comprehensive understanding of our work, Extended Data Fig. A2 and Supplementary Data Tab. 30-49 establishes a series of evaluations uncovering that the components of AdaptiveNN function as we expect, and that our design markedly outperforms alternative choices.

Extended Data Fig. A2a and Supplementary Data Tab. 30-37 examine the effectiveness of a broad array of possible strategies for localizing visual fixations. The reinforcement learning algorithm of AdaptiveNN achieves significantly higher validation accuracies than the most competitive baseline across all the scenarios (all P<0.0001), especially with limited numbers of fixations. Intriguingly, although GardCAM has been widely used as a feasible algorithm to visualize the regions relevant to the decision-making of deep networks [43, 104], its application in selecting visual fixations does not yield competitive performance against AdaptiveNN, even though it is augmented with a Gaussian mixture model and additional computation. Moreover, other possible methods for

training the fixation selection policy, such as spatial transformer net and Gumbel-Softmax, do not exhibit the potential to approach reinforcement learning. They fail to secure noteworthy gains over pre-defined non-adaptive policies like random or Gaussian sampling.

In Extended Data Fig. [A2b](#page-31-0) and Supplementary Data Tab. 38-45, we show that the state values predicted by the *Vision Agent* of AdaptiveNN are strongly correlated with the test loss of the validation data. This correlation indicates that, for a given test sample whose label is unknown, we can leverage its associated state values as reliable proxies of how far the outputs of our model are from the accurate prediction. This phenomenon is highly consistent with our goal of introducing the value network (Fig. [2d](#page-6-1)). In this sense, Extended Data Fig. [A2c](#page-31-0) and Supplementary Data Tab. 46-49 provide further evidence supporting that the strategy of concluding the observation processes of the samples exhibiting smaller rather than larger state values is beneficial for a higher overall computational efficiency. This strategy underscores the efficacy of the value network in guiding the allocation of computational resources toward optimizing model performance.

Extended Data Fig. [A2d-A2e](#page-31-0) present system-level comparisons against representative state-of-the-art methods for enhancing the energy efficiency of deep networks. Extended Data Fig. [A2d](#page-31-0) focuses on the recently proposed algorithms that leverage the spatial redundancy of visual data, whereas Extended Data Fig. [A2e](#page-31-0) considers existing multi-exit models characterized by an online-adjustable computational cost. AdaptiveNN outperforms all of them by marked margins when consuming less or comparable amounts of computation, even though the major motivation of our work is to emulate the visual perception behaviors of humans to drive a paradigm shift from 'passive' to 'active and adaptive' vision models, instead of attaining optimal engineering performance.

## **4. Discussion**

Human vision is distinguished by its remarkable flexibility to adapt to spatial regions with different content, varying complexities of visual environments, diverse task demands, and fluctuating resource availability for perception. In contrast, current machine vision models mainly adopt 'passive' paradigms, which usually perceive everything everywhere in parallel with an identical computational graph, regardless of the specific characteristics of variable visual environments, tasks and resources. This lack of adaptive adjustment results in an 'impossible triangle' formed by high-dimensional visual inputs, large-scale neural networks, and efficiency: under the scaling laws, the first two tend to be essential for complex real-world vision problems, but they significantly compromise efficiency. This inherent limitation impedes both future advancements and application in diverse real-world scenarios. In this article, we aim to address this issue by enabling neural networks to computationally emulate the adaptive behaviors of human visual systems, driving a paradigm shift from 'passive' to 'active and adaptive' vision models.

Specifically, we establish an AdaptiveNN framework that perceives scenes by sequentially fixating on pertinent regions, incrementally integrating information across fixations, and actively concluding its observation to accomplish the task of interest. Theoretical analyses suggest that such models can be trained using reinforcement learning without specialized supervision, relying solely on simple task-driven objectives. Our resulting method, AdaptiveNN, substantially reduces the computational cost of well-performing computer vision models by up to 28× without sacrificing accuracy (Fig. [3b](#page-10-0), [3c](#page-10-0), [4a](#page-13-0), [5b](#page-15-0), and Extended Data Fig. [A1a](#page-30-0)). Moreover, it exhibits human-like flexibility in adjusting its inference cost online without necessitating additional training (Fig. [3b](#page-10-0), [3c](#page-10-0), [4a](#page-13-0), [5b](#page-15-0), and Extended Data Fig. [A1a](#page-30-0)), as well as in customizing its perceptual strategies conditioned on variable task demands through modifying the training objective (Fig. [4c](#page-13-0), [4d](#page-13-0)) or introducing natural language prompts as inputs (Fig. [5\)](#page-15-0). Additionally, AdaptiveNN is also distinctive in its enhanced interpretability through analyzing its fixation patterns (Fig. [3a](#page-10-0), [4b](#page-13-0), [4d](#page-13-0), [4f](#page-13-0), [5d](#page-15-0), and Extended Data Fig. [A1b-A1e](#page-30-0)), in a manner akin to understanding human visual systems [\[56,](#page-37-5) [105,](#page-40-1) [106,](#page-40-2) [77\]](#page-38-8). These aforementioned favorable attributes align closely with the extensively recognized advantages of human visual systems [\[56,](#page-37-5) [59,](#page-37-8) [7,](#page-34-6) [57,](#page-37-6) [62,](#page-37-11) [58,](#page-37-7) [76,](#page-38-7) [107,](#page-40-3) [60\]](#page-37-9). In this sense, we believe that our work significantly advances the resolution of the open challenge raised by LeCun, Bengio, and Hinton [\[61\]](#page-37-10), opening up a new horizon for developing more energy-efficient, adaptable, and interpretable computer vision models. These properties are critical for realistic scenarios such as wearable devices, mobile phones, robotics, embedded devices, autonomous vehicles, and medical AI applications.

The design and theoretical analyses of AdaptiveNN have carefully avoided involving strong assumptions or specialized implementation configurations. As a consequence, it is compatible with a wide array of state-of-the-art representation learning backbones [\[108,](#page-40-4) [36,](#page-36-1) [37,](#page-36-2) [98,](#page-39-10) [109,](#page-40-5) [110,](#page-40-6) [111\]](#page-40-7), which can be readily incorporated as the feature-extraction module within our model. Furthermore, the outputs of AdaptiveNN interface seamlessly with various vision tasks, such as recognition, medical diagnosis or prognosis [\[43\]](#page-36-8), segmentation [\[41\]](#page-36-6), locating visual objects [\[112,](#page-40-8) [113\]](#page-40-9), and embodied multimodal large language models (MLLM) [\[101\]](#page-39-13). In this article, we first demonstrate our model's generalizability using several representative backbone networks: ResNet (convolutional neural network) [\[36\]](#page-36-1) and DeiT (vision Transformer) [\[98\]](#page-39-10), through the lens of two common, foundational elements of diverse vision tasks: 'what' and 'where' [\[114\]](#page-40-10), namely semantic understanding and element localization (Fig. [3,](#page-10-0) [4,](#page-13-0) and Extended Data Fig. [A1\)](#page-30-0). To demonstrate the versatility of AdaptiveNN, we further deploy it as the perceptual module of a language-driven embodied MLLM (Fig. [5\)](#page-15-0). Our considerations of building up a general framework not only verify the extensive applicability of our findings, but also facilitate a focused and comprehensive examination of the benefits derived from formulating human-like adaptive visual perception. Additionally, we believe that the sufficient flexibility of our approach offers promising avenues for further extensions.

Our results are also appealing in their contributions to the ongoing discourse on human visual cognition, particularly concerning the debate over the role of innateness in learning perceptual behaviors. This debate has persisted for centuries, questioning whether certain visual behaviors are inherent at birth or learned through experience [\[115,](#page-40-11) [116,](#page-40-12) [117,](#page-40-13) [23\]](#page-35-3). Some developmental psychologists have postulated that innate biases, such as those related to objects, agents, space, and biological motion [\[78,](#page-38-9) [79,](#page-38-10) [80,](#page-38-11) [81,](#page-38-12) [82,](#page-38-13) [83,](#page-38-14) [84,](#page-38-15) [85\]](#page-38-16), may shape the process of learning from the environment. Conversely, others argue that visual capabilities can develop in the absence of such biases, heavily influenced by the richness of the developing child's experience [\[118,](#page-40-14) [23\]](#page-35-3). Our efforts revisit this age-old 'nature versus nurture' debate from a modern perspective: we demonstrate the possibility of investigating these aforementioned claims via rigorous computational simulations. AdaptiveNN emerges as a general, scalable, and sufficiently human-like proxy that learns from visual data with maximally eliminating the innate biases or abilities in humans. By being trained solely on real-world visual tasks like ImageNet object recognition, AdaptiveNN not only achieves beyond human-level accuracies [\[35\]](#page-36-0) (Fig. [3\)](#page-10-0), but also exhibits mostly indistinguishable behaviors from humans (Fig, [6\)](#page-18-0), in terms of either the 'eye movement' patterns in novel scene observation or assessing the 'difficulty levels' of various visual environments. These findings suggest that many adaptive behaviors and basic capabilities of human vision could indeed be acquired through routine visual tasks, without necessitating strong innate biases. In this sense, we are among the first to leverage advanced AI methods like deep networks and reinforcement learning to explore fundamental cognitive science questions under controlled experimental conditions [\[119,](#page-40-15) [120,](#page-40-16) [23,](#page-35-3) [24\]](#page-35-4). Additionally, we hope that our work will inspire new interdisciplinary collaborations between machine learning and broader fields, given the critical role of eye movements in probing into human vision and mind [\[121,](#page-40-17) [122,](#page-41-0) [106,](#page-40-2) [123,](#page-41-1) [7,](#page-34-6) [124,](#page-41-2) [77\]](#page-38-8), and their widespread applications to various research communities, such as visual content analysis [\[105\]](#page-40-1), graphic or web designs [\[125,](#page-41-3) [126\]](#page-41-4), driving [\[127\]](#page-41-5), gaming [\[128\]](#page-41-6), and medical research [\[129,](#page-41-7) [130\]](#page-41-8).

In conclusion, the encouraging findings of this article demonstrate a new avenue for developing deep learning methodologies inspired by human vision. The efficacy of AdaptiveNN and its behavioral alignment with human visual systems underscore its potential. We anticipate future explorations in this direction to benefit both the AI and cognitive science communities – promising not only to foster the creation of next-generation computer vision models that are efficient, adaptable, and interpretable, but also to provide powerful computational tools for investigating human behavioral and learning processes.

We also believe our work offers valuable insights into equipping computer vision models with adaptive sequential 'reasoning'-like perception capabilities using reinforcement learning, analogous to the approach employed in DeepSeek-R1 [\[131\]](#page-41-9). Towards this direction, we demonstrate how to model visual perception tasks as sequential decision procedures, and reveal why and how such models should be trained using reinforcement learning. Our resulting models can adaptively employ a larger number of strategically selected visual fixations to solve more challenging vision tasks.

## <span id="page-20-1"></span><span id="page-20-0"></span>**5. Methods**

## **5.1. Theoretical learning principles of AdaptiveNN**

Here we present the proof of Theorem [1](#page-8-3) and Eq. [\(7\)](#page-9-0).

**Theorem.** The gradients of  $L(\theta)$  can be decomposed into a combination of representation learning and self-rewarding reinforcement learning objectives:

$$\nabla_{\theta} L(\theta) = \nabla_{\theta} L_{\text{rep}}(\theta) + \nabla_{\theta} L_{\text{rl}}(\theta), \tag{9}$$

where

$$\nabla_{\boldsymbol{\theta}} \mathbf{L}_{\text{rep}} = \underbrace{\mathbb{E}_{\boldsymbol{X}, y, \boldsymbol{l}_{1:T}} \sum_{t=1}^{T} P(t_{0} = t) \nabla_{\boldsymbol{\theta}} \mathcal{L}(y, q(\boldsymbol{\theta}, \boldsymbol{X}, \boldsymbol{l}_{1:t}))}_{\text{representation learning}}$$

$$\nabla_{\boldsymbol{\theta}} \mathbf{L}_{\text{rl}} = \underbrace{-\mathbb{E}_{\boldsymbol{X}, y, \boldsymbol{l}_{1:T}} \sum_{t=1}^{T} \left[ \left( \sum_{t'=t}^{T} r_{t'} \right) \nabla_{\boldsymbol{\theta}} \log p(\boldsymbol{l}_{t} | \boldsymbol{\theta}, \boldsymbol{X}, \boldsymbol{l}_{1:(t-1)}) \right]}_{\text{self-rewarding reinforcement learning}},$$

$$(10)$$

$$r_{t'} = -P(t_0 = t')\mathcal{L}(y, q(\boldsymbol{\theta}, \boldsymbol{X}, \boldsymbol{l}_{1:t'}))$$

**Proof:** Taking derivatives of  $L(\theta)$  with respect to  $\theta$ , we have

$$\nabla_{\boldsymbol{\theta}} \mathbf{L} = \mathbb{E}_{\boldsymbol{X}, y, t_{o} \sim p(t_{o})} \left[ \int_{\boldsymbol{I}_{1:t_{o}}} p(\boldsymbol{l}_{1:t_{o}} | \boldsymbol{\theta}, \boldsymbol{X}) \frac{\partial \mathcal{L}(y, q(\boldsymbol{\theta}, \boldsymbol{X}, \boldsymbol{l}_{1:t_{o}}))}{\partial \boldsymbol{\theta}} \right.$$

$$+ \int_{\boldsymbol{l}_{1:t_{o}}} \mathcal{L}(y, q(\boldsymbol{\theta}, \boldsymbol{X}, \boldsymbol{l}_{1:t_{o}})) \frac{\partial p(\boldsymbol{l}_{1:t_{o}} | \boldsymbol{\theta}, \boldsymbol{X})}{\partial \boldsymbol{\theta}} \right]$$

$$= \mathbb{E}_{\boldsymbol{X}, y, t_{o} \sim p(t_{o})} \int_{\boldsymbol{l}_{1:t_{o}}} p(\boldsymbol{l}_{1:t_{o}} | \boldsymbol{\theta}, \boldsymbol{X}) \left[ \frac{\partial \mathcal{L}(y, q(\boldsymbol{\theta}, \boldsymbol{X}, \boldsymbol{l}_{1:t_{o}}))}{\partial \boldsymbol{\theta}} \right.$$

$$+ \mathcal{L}(y, q(\boldsymbol{\theta}, \boldsymbol{X}, \boldsymbol{l}_{1:t_{o}})) \frac{\partial \log p(\boldsymbol{l}_{1:t_{o}} | \boldsymbol{\theta}, \boldsymbol{X})}{\partial \boldsymbol{\theta}} \right]$$

$$= \mathbb{E}_{\boldsymbol{X}, y, t_{o} \sim p(t_{o})} \int_{\boldsymbol{l}_{1:t_{o}}} \int_{\boldsymbol{l}_{t_{o}+1:T}} p(\boldsymbol{l}_{t_{o}+1:T} | \boldsymbol{\theta}, \boldsymbol{X}, \boldsymbol{l}_{1:t_{o}})$$

$$p(\boldsymbol{l}_{1:t_{o}} | \boldsymbol{\theta}, \boldsymbol{X}) \left[ \frac{\partial \mathcal{L}(y, q(\boldsymbol{\theta}, \boldsymbol{X}, \boldsymbol{l}_{1:t_{o}}))}{\partial \boldsymbol{\theta}} \right]$$

$$+ \mathcal{L}(y, q(\boldsymbol{\theta}, \boldsymbol{X}, \boldsymbol{l}_{1:t_{o}})) \frac{\partial \log p(\boldsymbol{l}_{1:t_{o}} | \boldsymbol{\theta}, \boldsymbol{X})}{\partial \boldsymbol{\theta}} \right]$$

$$= \mathbb{E}_{\boldsymbol{X}, y, t_{o} \sim p(t_{o})} \int_{\boldsymbol{l}_{1:T}} p(\boldsymbol{l}_{1:T} | \boldsymbol{\theta}, \boldsymbol{X}) \left[ \frac{\partial \mathcal{L}(y, q(\boldsymbol{\theta}, \boldsymbol{X}, \boldsymbol{l}_{1:t_{o}}))}{\partial \boldsymbol{\theta}} \right.$$

$$+ \mathcal{L}(y, q(\boldsymbol{\theta}, \boldsymbol{X}, \boldsymbol{l}_{1:t_{o}})) \frac{\partial \log p(\boldsymbol{l}_{1:t_{o}} | \boldsymbol{\theta}, \boldsymbol{X})}{\partial \boldsymbol{\theta}} \right],$$

where T is the maximum possible value of  $t_0$ . Since  $t_0$  and  $\boldsymbol{l}_{1:t_0}$  are mutually independent random variables, we have

$$\nabla_{\boldsymbol{\theta}} \mathbf{L} = \mathbb{E}_{\boldsymbol{X}, y, \boldsymbol{l}_{1:T}} \left[ \mathbb{E}_{t_{0} \sim p(t_{0})} \frac{\partial \mathcal{L}(y, q(\boldsymbol{\theta}, \boldsymbol{X}, \boldsymbol{l}_{1:t_{0}}))}{\partial \boldsymbol{\theta}} + \mathbb{E}_{t_{0} \sim p(t_{0})} \mathcal{L}(y, q(\boldsymbol{\theta}, \boldsymbol{X}, \boldsymbol{l}_{1:t_{0}})) \frac{\partial \log p(\boldsymbol{l}_{1:t_{0}} | \boldsymbol{\theta}, \boldsymbol{X})}{\partial \boldsymbol{\theta}} \right].$$
(12)

<span id="page-21-0"></span>Moreover, note that  $\log p(\boldsymbol{l}_{1:t_0}|\boldsymbol{\theta},\boldsymbol{X})$  can be factorized as:

$$\log p(\mathbf{l}_{1:t_{o}}|\boldsymbol{\theta}, \mathbf{X}) = \log p(\mathbf{l}_{1}|\boldsymbol{\theta}, \mathbf{X}) + \log p(\mathbf{l}_{2}|\boldsymbol{\theta}, \mathbf{X}, \mathbf{l}_{1}) + \dots + \log p(\mathbf{l}_{t_{o}}|\boldsymbol{\theta}, \mathbf{X}, \mathbf{l}_{1:t_{o}-1}),$$
(13)

<span id="page-21-1"></span>which can be considered as solving the state distribution over a Markov chain. Then, we have:

$$\mathbb{E}_{t_{0} \sim p(t_{0})} \mathcal{L}(y, q(\boldsymbol{\theta}, \boldsymbol{X}, \boldsymbol{l}_{1:t_{0}})) \frac{\partial \log p(\boldsymbol{l}_{1:t_{0}} | \boldsymbol{\theta}, \boldsymbol{X})}{\partial \boldsymbol{\theta}}$$

$$= \sum_{t'=1}^{T} \left[ P(t_{0} = t') \mathcal{L}(y, q(\boldsymbol{\theta}, \boldsymbol{X}, \boldsymbol{l}_{1:t'})) \sum_{t=1}^{t'} \frac{\partial \log p(\boldsymbol{l}_{t} | \boldsymbol{\theta}, \boldsymbol{X}, \boldsymbol{l}_{1:(t-1)})}{\partial \boldsymbol{\theta}} \right]$$

$$= \sum_{t=1}^{T} \left[ \left( \sum_{t'=t}^{T} P(t_{0} = t') \mathcal{L}(y, q(\boldsymbol{\theta}, \boldsymbol{X}, \boldsymbol{l}_{1:t'})) \right) \frac{\partial \log p(\boldsymbol{l}_{t} | \boldsymbol{\theta}, \boldsymbol{X}, \boldsymbol{l}_{1:(t-1)})}{\partial \boldsymbol{\theta}} \right]$$

$$(14)$$

Furthermore, combining Eq. (12) and Eq. (14), we finally obtain

$$\nabla_{\boldsymbol{\theta}} \mathbf{L} = \underbrace{\mathbb{E}_{\boldsymbol{X}, y, \boldsymbol{l}_{1:T}} \sum_{t=1}^{T} P(t_{o} = t) \frac{\partial \mathcal{L}(y, q(\boldsymbol{\theta}, \boldsymbol{X}, \boldsymbol{l}_{1:t}))}{\partial \boldsymbol{\theta}}}_{\text{representation learning objective, } \nabla_{\boldsymbol{\theta}} \mathbf{L}_{\text{rep}}} + \underbrace{\mathbb{E}_{\boldsymbol{X}, y, \boldsymbol{l}_{1:T}} \sum_{t=1}^{T} \left[ \left( \sum_{t'=t}^{T} P(t_{o} = t') \mathcal{L}(y, q(\boldsymbol{\theta}, \boldsymbol{X}, \boldsymbol{l}_{1:t'})) \right) \frac{\partial \log p(\boldsymbol{l}_{t} | \boldsymbol{\theta}, \boldsymbol{X}, \boldsymbol{l}_{1:(t-1)})}{\partial \boldsymbol{\theta}} \right]},$$

$$(15)$$

which proves Theorem 1.

<span id="page-22-1"></span>**Proof of Eq. (7)**. It is obvious that

$$\lim_{\gamma \to 0} \sum_{t'=t}^{T} \gamma^{t'-t} (r_{t'} - r_{t'-1}) = r_t - r_{t-1},$$

$$\lim_{\gamma \to 1} \sum_{t'=t}^{T} \gamma^{t'-t} (r_{t'} - r_{t'-1}) = r_T - r_{t-1}.$$
(16)

<span id="page-22-2"></span>On top of Eq. (6), we actually have

$$\mathbb{E}_{l_t} r_{t-1} \nabla_{\theta} \log p_{\pi}(l_t | s_{t-1})$$

$$= r_{t-1} \int_{l_t} p_{\pi}(l_t | s_{t-1}) \frac{1}{p_{\pi}(l_t | s_{t-1})} \nabla_{\theta} p_{\pi}(l_t | s_{t-1})$$

$$= r_{t-1} \frac{\partial \int_{l_t} p_{\pi}(l_t | s_{t-1})}{\partial \theta} = 0.$$
(17)

Eq. (7) can be obtained by combining Eq. (16) and Eq. (17).

#### <span id="page-22-0"></span>5.2. Evaluation tasks for AdaptiveNN

Here we describe the 9 different tasks used for evaluating AdaptiveNN, each associated with one or more datasets, yielding 17 benchmarks in total. For all tasks, we held out 20% of training data to perform a hyper-parameter search, and then put this data back to the training set, reporting final results. When involved, we consider the number of floating point operations (FLOPs) as the measure of computational cost for the inference of a model.

#### 5.2.1. Computer vision tasks

Large-scale real-world visual understanding: ImageNet. ImageNet is a large-scale and diverse dataset of high-quality Internet images [35]. Each image is annotated with a label of its category. The categories are organized according to the WordNet hierarchy [97], covering a wide range of common visual content, including objects, buildings, humans, animals, scenes, etc. ImageNet is a very popular benchmark for evaluating deep learning methodologies [36, 37, 38, 23], and has been instrumental in advancing computer vision and machine learning research. In this article, we adopted the standard training-validation split, with  $\sim$ 1,280,000 images for training, 50,000 images for validation, and 1,000-class annotations. Following the common practice, we used validation accuracy as the performance metric.

**Fine-grained visual recognition: six benchmarks**. To examine whether AdaptiveNN has similar capabilities to humans in terms of filtering subtle task-relevant information out of large quantities of noise, we considered six fine-grained visual recognition tasks. These tasks are marked by small inter-class variations (such as distinguishing among visually closely related bird species), and large intra-class variations (such as highly diversified backgrounds and viewpoints). Here we describe the six corresponding datasets we used. For all of them, we adopted the standard training-validation split, and use validation accuracy as the performance metric

- Caltech-UCSD birds-200-2011 (CUB-200-2011) [132] is one of the most widely-used fine-grained categorization dataset. It consists of 11,788 images of 200 subcategories belonging to birds, 5,994 for training and 5,794 for testing.
- North America Birds (NABirds) [133] contains 48,562 annotated photographs of 400 species of commonly observed birds in North America. Each species has more than 100 photographs, including annotations for males, females, and juveniles. All the data is divided into 555 visual categories.

- Oxford-IIIT Pet [134] is a 37 category pet dataset with  $\sim$ 200 images for each class. The images are highly diversified in scale, pose, and lighting.
- Stanford Dogs [135] contains 20,580 images of 120 breeds of dogs from around the world. The dataset is divided into 12,000 images for training and 8,580 images for validation.
- Stanford Cars [136] contains 16,185 images of 196 classes of cars. The data is divided into 8,144 training images and 8,041 validation images. The categories are typically built at the level of make, model, and year.
- FGVC-Aircraft [137] is a benchmark that contains 10,200 images of 102 different classes of aircraft, where each class has 100 images. The data is organized in a four-level hierarchy, namely model, variant, family, and manufacturer.

Efficient processing of visual data from real driving scenarios: STSD. Similar to human visual systems, AdaptiveNN is not only able to process relatively object-centric visual data such as ImageNet and fine-grained classification datasets, but also applicable to more general visual perception scenarios. For example, it can process non-object-centric, complex images collected in the wild without specified pre-processing. As a representative example, we considered the task of recognizing traffic signs on the Swedish traffic signs dataset (STSD) [99]. The dataset consists of  $960 \times 1,280$  road-scene images, captured from real moving vehicles, and the task is to recognize the existence and types of the speed limit signs. Note that the targets of interest are generally small, diversely distributed, and sometimes not clear. In this article, we used two subsets comprising 747 and 648 images for training and validation, respectively. Validation accuracy is used as the performance metric.

Visual search with diversified task demands: localizing arbitrary digits in multi-digit images. To investigate whether AdaptiveNN has the human-like adaptability of customizing visual perception behaviors conditioned on different task demands, we considered a visual search scenario where the categories and number of targets are assumed to be flexibly changed. Specifically, we created a digit localization dataset by generating 224×224 images, each randomly populated with 6 to 10 28×28 MNIST digits [53] against a black background without repetition of digits. We established a large-scale dataset with 500,000 images for training and 50,000 images for validation. To define a visual search task, we specified arbitrary numbers and classes of digits, and trained our model to identify the locations of these specified digits within each input image. This requires a model to not only recognize correct targets, but also accurately localize multiple targets in a single image. To measure the performance of a given model, we randomly defined many visual tasks, and obtained the average success rate on the validation set. Notably, one success means retrieving exactly all the digits demanded by a task from an input, while the success rate of a task is defined as the number of successes divided by the number of all samples.

Image processing in medical scenarios: RSNA pneumonia detection. To demonstrate the efficacy of AdaptiveNN in applications where interpretability holds vital importance, a pneumonia detection scenario was considered. We used the RSNA Pneumonia dataset, which consists of  $\sim 30,000$  frontal view chest radiographs [100]. Each image in the dataset is annotated with image-level labels indicating the presence or absence of pneumonia, as well as bounding boxes for pulmonary opacity which are visual signals for the disease. The annotations are provided by 18 board-certified radiologists from 16 institutions. In this article, we leveraged the image-level labels to train AdaptiveNN, and compared the locations it fixates on with the pulmonary opacity identified by clinicians to assess its interpretability. The dataset was randomly divided into training and validation sets, following a ratio of 85% and 15%, respectively. The model's diagnostic accuracy was quantified through the area under the receiver operating characteristic curve (AUROC) on the validation data.

#### 5.2.2. Embodied AI tasks

CALVIN long-horizon multi-task language control benchmarks. We adopt CALVIN [102] to construct the benchmarks for validating the performance of our multi-task, language-guided embodied agent. Within CALVIN, the agent is tasked with executing sequences of actions, each consisting of five subtasks defined through natural language instructions. The model's effectiveness is measured by its average successful length across 1,000 task sequences, with scores ranging from 0 to 5 based on the number of subtasks completed successfully, as detailed in Extended Data Fig. A4b. The CALVIN dataset is organized into four distinct environmental subsets, labeled A through D, each characterized by unique visual backgrounds and object arrangements. Each of these subsets encompasses approximately 24,000 robot manipulation trajectories accompanied by language annotations. We

train our embodied multimodal large language models on these language-annotated trajectories. To thoroughly evaluate the model's ability to imitate and generalize, we conduct experiments under two scenarios: 1)  $D \rightarrow D$ : training and testing within the same environment, and 2) ABCD $\rightarrow D$ : training on data from all four environments while testing on a single target domain.

#### <span id="page-24-0"></span>5.2.3. Comparisons with human visual perception behaviors

To demonstrate the potential of AdaptiveNN as a valuable tool for investigating human visual cognition, we evaluated humans and AdaptiveNN side by side on the same tests of visual perception behaviors. Specifically, these tests were designed under two goals, namely 1) spatial-wise, examining the locations of visual regions that a human/model fixates on; and 2) sample-wise, examining the difficulty level that a human/model assesses to accomplish the given task based on each individual visual environment. To attain these goals, we conducted three groups of experiments, as described below.

First, through the lens of spatial-wise adaptiveness, we investigated the consistency of the locations of visual fixations selected by AdaptiveNN and humans. We employed the saliency in context (SALICON) benchmark [103], which consists of 10,000 training images and 5,000 validation images. Every image is annotated with a map of the centers of human gazing. Each gazing center is treated as a single point in the map. The maps of gazing centers were collected based on the paid Amazon Mechanical Turk (AMT) crowdsourcing marketplace, with each image observed by  $\sim$ 60 subjects. All participants had normal or corrected-to-normal vision and normal color vision. The images were presented to each subject in a random order, where each image was presented for 5 seconds. The subjects were instructed to explore the image freely by looking at anywhere they wanted to look, with no further instructions on where they should look in the images. The gazing center locations were obtained by a 100 Hz resampling and processed by excluding the fast-moving data corresponding to saccade processes.

To compare humans and AdaptiveNN in terms of spatial-wise adaptive visual perception, a metric named 'normalized human-like score' was defined. For each image, the average density map of the gazing centers of all  $\sim\!60$  observers was used as the ground truth distribution of the real focal centers of human vision. We let AdaptiveNN select n visual fixation regions on top of each image, mimicking the process of freely observing the image for a fixed length of time. Then, we obtained the probability that the ground truth human gazing centers fall into the visual fixation regions localized by AdaptiveNN, denoted as  $p_n^{\rm AdaNN}$ . Similarly, consider sampling visual fixation regions following the gazing center distribution of an arbitrary single person (within  $\sim\!60$  observers), or fully randomly, and then taking the expectations. As a result, we had the corresponding expected probabilities  $\mathbb{E}[p_n^{\rm Single-human}]$  and  $\mathbb{E}[p_n^{\rm Random}]$ , respectively. Built upon this, we defined that

<span id="page-24-1"></span>
$$\text{normalized human-like score} = \frac{p_n^{\text{AdaNN}} - \mathbb{E}[p_n^{\text{Random}}]}{\mathbb{E}[p_n^{\text{Single-human}}] - \mathbb{E}[p_n^{\text{Random}}]}. \tag{18}$$

Notably, Eq. (18) = 1 indicates that the consistency between AdaptiveNN and the average characteristics of the spatial-wise visual fixation behaviors of people is approximately the same as the level of an average individual human observer. On the other hand, Eq. (18) = 0 provides a baseline of randomly fixating on the visual environments. In our implementation, normalized human-like scores were calculated upon mini-batches of data sampled from the dataset to reasonably reflect their values across different sets of visual environments. We adopted n = 3 and a batch size of 64. Moreover, we reported the results on top of the two splits of SALICON (split-1/2 corresponds to the train/val split of SALICON). They were not particularly distinguished since our model had never been trained on SALICON.

Second, through the lens of sample-wise adaptiveness, we investigated whether our model is consistent with humans in judging which visual environments are relatively easier or more difficult for a given task, and should be paid less or more attention to observing them. To achieve this, we started by measuring the judgments of difficulty level from humans. Specifically, ten volunteers (aged between 18 and 35) participated in our experiment (we verified that further increasing the number of subjects does not significantly affect our findings). All of them had normal or corrected-to-normal vision and normal color vision. Our studies were approved by the THU S&T Ethics Committee (AI), protocol THU-03-2024-0006, and obtained informed consent. We selected six representative categories of images from the ImageNet validation set. The participants were instructed to assign a 0-to-100 score to each image according to the difficulty level of the visual recognition task built upon this image,

where smaller scores mean easier. The order of different categories and the order of images within each category were both randomized for each participant. Each image was presented to a participant for 5 seconds, after which a corresponding difficulty score was recorded. There was a practice session before formal trials for the participants to get familiar with our experimental paradigm, which was identical to the formal trials in all configurations but the scores were not recorded. After the experiment, the scores of each category were normalized on a per-participant basis and averaged across participants. This human-assessed difficulty level was compared with the normalized state values predicted by AdaptiveNN, which reflect our model's judgments on the difficulty level of each visual environment.

*Third*, we further developed several 'visual Turing tests' [\[22\]](#page-35-2), leveraging the straightforward human judgments to compare the visual perception behaviors of AdaptiveNN with those of humans. In these tests, real human judges tried to identify the machine, given paired examples of human and machine behaviors. Driven by the previous discussions, our 'visual Turing tests' probed into both the spatial-wise visual fixation behaviors and the sample-wise visual difficulty-assessment behaviors of our model. For the former, we took the ground truth density map of human gazing centers for each image from SALICON, and sampled a sequence of three visual fixation regions, as human behaviors against the machine behaviors of the three fixations selected by AdaptiveNN. For the latter, the normalized and averaged human-assessed difficulty scores acquired as aforementioned, and the normalized state values predicted by AdaptiveNN, were rescaled to [0, 100] on a per-class basis, as human and machine behaviors, respectively. In each trial, a human judge was given two paired groups of images (three in each) in a random order, one group comprising human behaviors and the other comprising machine behaviors. The human judge was informed to identify 'which group of images reflect the visual perception behaviors of a machine'. See Supplementary Data Fig. 1-2 for the representative examples of our trials.

The full procedure of 'visual Turing tests' is deatailed in Extended Data Fig. [A3a](#page-32-0). For each 'visual Turing test' concerning the spatial-wise or sample-wise adaptive visual perception behaviors, we considered three types of trials: i) human v.s. machine, as described above; ii) human v.s. human; and iii) human v.s. random. For each trial of ii) and iii), we replaced the group of images corresponding to 'machine' with samples depicting the behaviors of human vision or randomly generated behaviors, yet the participant was still told to distinguish between human v.s. machine. We established 36 trials for each of i)-iii), yielding totally 108 trials for each of the two 'visual Turing tests'. These 108 trials were shuffled for every participant, such that ii) and iii) provided randomized control groups as baselines for comparison, and also offered information to validate whether our experimental setups were reasonable. 39 volunteers (aged between 18 and 40), with normal or corrected-to-normal vision and normal color vision, participated in our experiment. Our studies were approved by the THU S&T Ethics Committee (AI), protocol THU-03-2024-0006, and obtained informed consent. We verified that further increasing the number of subjects does not significantly affect our findings. There was a practice session before real trials. After all trials, each accuracy of i)-iii) was calculated per participant and aggregated across participants. Notably, 50% accuracy indicates that the sort of behaviors is indistinguishable from those of humans (perfectly human-like), while 100% suggests the inverse case.

## **5.3. Implementation details of AdaptiveNN**

In this section, we describe the implementation details of our method, including its inference procedure, network architectures, and training algorithms. These materials are organized for ease of understanding.

## <span id="page-25-0"></span>5.3.1. Inference procedure

**Termination criteria for the sequential perception process**. The values of {*η*1*, η*2*,* · · · } reflect whether the overall quantity of available resources for visual perception, such as computation, time, or energy, is sufficient in the current circumstance. When {*η*1*, η*2*,* · · · } are large, AdaptiveNN generally tends to leverage relatively fewer fixations to observe various visual environments. Conversely, their small values indicate that our model can employ more fixations for visual processing on average. Notably, this only corresponds to the overall situation. In both cases, AdaptiveNN can efficiently perform uneven computation allocation across different visual environments by utilizing the predicated state values of *Vision Agent*.

Incorporating these considerations of the effectiveness-efficiency trade-off, we argue that the values of {*η*1*, η*2*,* · · · } should be determined through maximizing the performance of AdaptiveNN under a fixed amount of total cost.

Specifically, consider a set of visual environments  $\mathcal{D}$  and the metrics of performance and costs,  $\mathcal{P}(\cdot)$  and  $\mathcal{C}(\cdot)$ , which are defined with respect to  $\mathcal{D}$ ,  $\{\eta_1, \eta_2, \cdots\}$  and an AdaptiveNN model parameterized by  $\boldsymbol{\theta}$ . Given a budget B > 0,  $\{\eta_1, \eta_2, \cdots\}$  can be obtained by solving the optimization problem:

<span id="page-26-1"></span>
$$\underset{\eta_1,\eta_2,\cdots}{\text{maximize}} \quad \mathcal{P}(\boldsymbol{\theta}, \mathcal{D}, \{\eta_1, \eta_2, \cdots\}), \quad \text{subject to } \mathcal{C}(\boldsymbol{\theta}, \mathcal{D}, \{\eta_1, \eta_2, \cdots\}) \leq B.$$
 (19)

Notably, by considering a series of varied B, we can collect a group of different thresholds  $\{\eta_1, \eta_2, \cdots\}$  associated with the model  $\theta$ . As a consequence, the cost of AdaptiveNN can be flexibly adjusted online without additional training by simply adjusting these thresholds. In our implementation, we instantiate  $\mathcal{D}, \mathcal{P}(\cdot)$ , and  $\mathcal{C}(\cdot)$ , as the training set of vision tasks, the accuracy, area under curve (AUC) score, or negative expected mean squared error, and the amount of computation for inference. However, the flexibility of our formulation allows these definitions to adapt to more diversified demands of various tasks, such as introducing task-specific performance metrics as  $\mathcal{P}(\cdot)$ , or leveraging the latency or energy consumption on given hardware devices as  $\mathcal{C}(\cdot)$ . Besides, we solve problem (19) using the genetic algorithm [69].

#### <span id="page-26-0"></span>5.3.2. Network architecture for computer vision tasks

**Perception networks**. In AdaptiveNN, the perception net  $f_{\text{rep}}$  is formulated as feature extractors with flexible architectures. In general, most existing deep learning backbones can be deployed as them. In our implementation, we mainly consider two representative examples, ResNet [36] and DeiT [98]. ResNet processes input images with the alternatively stacked convolutional blocks and pooling layers, while DeiT splits images into 2D patches, each of which is embedded into a token and processed through multi-head self-attention layers and multilayer perceptron [138]. Both of them leverage residual connections [36]. These architectures represent a wide range of popular modern deep networks for extracting embeddings from visual data. In visual recognition and medical diagnosis tasks, we adopt the first three network stages of ResNet or the first eight blocks of DeiT for the initial processing of the down-sampled glance inputs. We employ another full ResNet/DeiT network for processing visual fixations, due to its markedly different scales from the glance inputs. The internal vision representation of AdaptiveNN is fed into a task-specific head whose architecture adopts the final stage of ResNet or four DeiT blocks, in each corresponding scenario. In the visual search scenario, for fair comparisons with the baseline methods [62, 63], the two perception nets consist of two and three convolutional layers, respectively, while the task-specific head is a multilayer perceptron.

Vision agent. The Vision Agent is defined on top of the internal vision representation  $s_t$  with the size  $C \times H_f \times W_f$ . It is formulated as the combination of a policy network  $\pi$  and a value network  $V^{\pi}$ . The architecture of  $V^{\pi}$  and  $\pi$  is both a sequential composition of a  $C \to C$  depth-wise convolutional layer with  $3 \times 3$  kernels, a  $C \to 128$  dense convolutional layer with  $1 \times 1$  kernels, a feature-flatten layer, and a multilayer perceptron with corresponding different output neurons. A Gaussian error linear unit (GELU) [38] is added after each intermediate convolutional or linear layer for introducing nonlinearity. The output of  $V^{\pi}$  is a scalar  $V^{\pi}(s_t)$ . The outputs of  $\pi$  parameterize a distribution  $p_{\pi}(\cdot|s_t)$ , from which we sample the location  $l_{t+1}$  of the  $(t+1)^{th}$  visual fixation  $l_{t+1}$ . During training, we consider  $p_{\pi}(\cdot|s_t)$  as a Gaussian distribution, whose mean is output by  $\pi$  and standard deviation is pre-defined as a hyperparameter. At test time,  $p_{\pi}(\cdot|s_t)$  is set to be a Dirac delta distribution centered at the outputs of  $\pi$  for a deterministic inference process.

Feature updating and reusing. As aforementioned, the perception network  $f_{\text{rep}}$  is activated on top of the  $P \times P$  visual fixation  $\boldsymbol{l}_t$ . This yields local features  $\boldsymbol{s}_t^{\text{local}} = f_{\text{rep}}(\boldsymbol{l}_t) \in \mathbb{R}^{C \times P_{\text{f}} \times P_{\text{f}}}$ , which is employed to update the internal vision representation  $\boldsymbol{s}_{t-1}$  to obtain  $\boldsymbol{s}_t$ . For simplicity, assume that  $\tilde{\boldsymbol{s}}_t^{\text{local}} \in \mathbb{R}^{C \times P_{\text{f}}^2}$  and  $\tilde{\boldsymbol{s}}_{t-1} \in \mathbb{R}^{C \times H_{\text{f}}W_{\text{f}}}$  denote the flattened versions of  $\boldsymbol{s}_t^{\text{local}}$  and  $\boldsymbol{s}_{t-1}$ , respectively. Then the updating operation  $\boldsymbol{s}_t = \Psi(\boldsymbol{s}_{t-1}, f_{\text{rep}}(\boldsymbol{l}_t))$  can be expressed as

<span id="page-26-2"></span>
$$\tilde{\mathbf{s}}_t = \tilde{\mathbf{s}}_{t-1} + \tilde{\mathbf{s}}_t^{\text{local}} \cdot \mathbf{W}, \quad \mathbf{W} \in \mathbb{R}^{P_{\text{f}}^2 \times H_{\text{f}} W_{\text{f}}}.$$
 (20)

We find that this simple rule is able to work reasonably well in various scenarios, where it is combined with the normalization layers and nonlinear blocks introduced by other components of AdaptiveNN.

To construct the transformation matrix  $\mathbf{W}$ , we mainly consider two principles: spatial-wise correlations and semantic-level feature importance. For the former, we constrain that the features within  $s_t^{\text{local}}$  can only be utilized to update the features in  $s_{t-1}$  that are spatially close to them. Specifically, given the feature located at  $i^{\text{th}}$  row and  $j^{\text{th}}$  column of  $s_t^{\text{local}}$ , we can always find its corresponding coordinates  $(x_{ij}, y_{ij})$  at  $s_{t-1}$ , since  $s_{t-1}$  is the overall representation of the visual environment X while  $s_t^{\text{local}}$  is the embeddings extracted from a region of X. Suppose that (i', j') denotes the coordinates of a feature in  $s_{t-1}$ . We let

<span id="page-27-1"></span>
$$\mathbf{W}_{(i-1)P_{f}+j,(i'-1)W_{f}+j'} = 0, \quad \neg(|x_{ij} - i'| \le n^{\text{update}} \land |y_{ij} - j'| \le n^{\text{update}}). \tag{21}$$

By examining all possible values of i, i', j, j', Eq. (21) ensures that each feature vector within  $s_t^{\text{local}}$  can only have effects on its  $(2n^{\text{update}} + 1) \times (2n^{\text{update}} + 1)$  surrounding features in  $s_{t-1}$ , and hence introduces the constraints of spatial-wise correlations to Eq. (20). In our implementation, we simply fix  $n^{\text{update}} = 2$ . For the nonzero elements of  $\mathbf{W}$ , we fill them with feature-conditional weights, aiming to model the diverse semantic-level importance of different features. We take the  $k^{\text{th}}$  column of  $\tilde{s}_t^{\text{local}}$ ,  $(\tilde{s}_t^{\text{local}})_{:,k}$ , and feed it into a multilayer perceptron to obtain a weight matrix  $\mathbf{v}^k$ :

$$\tilde{\mathbf{v}}^{k} = \text{MLP}\left( (\tilde{\mathbf{s}}_{t}^{\text{local}})_{:,k} \right) \in \mathbb{R}^{(2n^{\text{update}} + 1)^{2}}, 
\mathbf{v}^{k} = \text{reshape}(\tilde{\mathbf{v}}^{k}) \in \mathbb{R}^{(2n^{\text{update}} + 1) \times (2n^{\text{update}} + 1)}.$$
(22)

<span id="page-27-2"></span>Then, on top of Eq. (21), we further define

$$\mathbf{W}_{(i-1)P_{\mathbf{f}}+j,(i'-1)W_{\mathbf{f}}+j'} = \mathbf{v}_{\lfloor x_{ij}-i' \rceil + n^{\text{update}} + 1, \lfloor y_{ij}-j' \rceil + n^{\text{update}} + 1}^{(i-1)P_{\mathbf{f}}+j},$$

$$|x_{ij} - i'| \leq n^{\text{update}} \wedge |y_{ij} - j'| \leq n^{\text{update}}.$$

$$(23)$$

Combining Eq. (21) and Eq. (23), the final form of **W** updates  $s_{t-1}$  using the features from  $s_t^{\text{local}}$  corresponding to the neighboring image regions of each element of  $s_{t-1}$ , modeling the inherent spatial continuity of visual data. In addition, the intensity of this updating operation is flexible as it is learnable on top of each specific feature vector in  $s_t^{\text{local}}$ .

Furthermore, built upon a similar idea of leveraging spatial-wise correlations, we note that before processing the next visual fixation  $l_{t+1}$ , some information pertinent to  $l_{t+1}$  have already been incorporated into the corresponding locations of  $s_t$ , e.g., by previous steps of sequential perception. Therefore, we propose to accomplish the feature-extracting process of  $s_{t+1}^{\rm local} = f_{\rm rep}(l_{t+1})$  on the basis of reusing the existing relevant information in  $s_t$ , rather than fully from scratch. To implement this idea, we take the features from  $s_t$  following the same relative sizes and locations as these of  $l_{t+1}$  with respect to  $I_t$ . Then, we feed the features into a multilayer perceptron and add the outputs to the tokens of  $l_{t+1}$  after the input layer of  $l_{t+1}$  as learnable context embeddings. Interestingly, this design is not limited to a technique that facilitates the efficient reuse of computation. As a matter of fact, it is inspired by modeling the presaccadic attention in human vision – the phenomenon of automatically deploying attention to the upcoming fixation location before the eyes start to move, improving visual sensitivity at the saccade target at the price of lowered perceptual sensitivity at other (non-target) locations [139, 60].

#### <span id="page-27-0"></span>5.3.3. Network architecture for embodied AI tasks

The architecture of our embodied multimodal large language models mainly follows RoboFlamingo [101]. A pre-trained OpenFLamingo 3B [140] is utilized as the backbone network. As detailed in Extended Data Fig. A4a, each two adjacent network blocks coupled with the shared vision encoder are employed as the perception net of AdaptiveNN. The visual tokens from both fixation and glance inputs, along with language tokens, are fed into the layers of the large language model (LLM) to extract a joint vision-language representation for the robot tasks. The robotic policy head adopts an LSTM network followed by a multilayer perceptron [101]. The architecture of the policy network and value network of AdaptiveNN's vision agent is both a multilayer perceptron.

#### 5.3.4. Training algorithm

Representation learning for computer vision tasks. Here we describe how we carry out the representation learning of AdaptiveNN ( $L_{rep}$  in Eq. (5)) in computer vision tasks. Without loss of generality, we assume that  $t_0 \sim p(t_0)$  follows a uniform distribution. Note that we can also consider more complex distributions to develop algorithms tailored for the specified demands of application scenarios, which may contribute to further performance improvements. However, we find that simple uniform distribution can already work reasonably well across various scenarios, and the results are sufficient to support our findings. Therefore, in this work, we always adopt  $t_0 \sim \text{unif}\{1, T\}$ . This also demonstrates that our model's effectiveness does not rely on strong assumptions relevant to  $p(t_0)$ .

In our implementation, we augment  $L_{\text{rep}}$  with two advanced representation learning techniques. The final loss to minimize can be written as:

<span id="page-28-0"></span>
$$L_{\text{rep}} + \alpha \mathcal{L}_{\text{regularization}}(y, f_{\text{rep}}(\boldsymbol{X}_{\text{d}})) + \beta \sum_{t=1}^{T-1} \mathcal{L}_{\text{self-distillation}}(q_T, q_t), \tag{24}$$

where  $\alpha, \beta$  are coefficients, and we simply fix  $\alpha = 2, \beta = 1$  in this article. The second term  $\mathcal{L}_{\text{regularization}}$  in Eq. (24) is a regularization loss for the perception network  $f_{\text{rep}}$  that processes visual fixations [141]. We feed the down-sampled version  $\mathbf{X}_{\text{d}}$  of  $\mathbf{X}$  into  $f_{\text{rep}}$  and obtain a direct loss using its outputs and the label y. This regularization technique addresses the slow convergence issue of  $f_{\text{rep}}$  caused by only seeing visual fixations  $\{\mathbf{l}_1,\mathbf{l}_2,\ldots\}$  during training, at the price of slightly increasing the train-test discrepancy of the inputs of  $f_{\text{rep}}$ . The third term  $\mathcal{L}_{\text{self-distillation}}$  in Eq. (24) corresponds to a self-distillation technique [142], where  $q_t = q(\theta, \mathbf{X}, \mathbf{l}_{1:t})$  denotes the outputs of the model at  $t^{\text{th}}$  step. This technique leverages the final outputs  $q_T$  of AdaptiveNN to guide the learning of intermediate outputs  $q_1,\ldots,q_{T-1}$ . It improves the performance of AdaptiveNN equipped with a smaller number of fixations, while only introducing negligible additional training costs.

**Reinforcement learning for computer vision tasks**. Similar to representation learning, we assume  $t_0 \sim \text{unif}\{1, T\}$ . The off-the-shelf proximal policy optimization algorithm (PPO) [94] using generalized advantage estimation [92] is deployed to accomplish the reinforcement learning procedure.

**Training algorithms for embodied AI tasks**. The training of the embodied multimodal large language models basically follows RoboFlamingo [101]. For AdaptiveNN, we simply adopt Eq. (5) as the training objective, which works reasonably well. Other implementation details are the same as computer vision tasks.

#### Data availability

Most data used in this study are publicly available, including ImageNet [35] (https://www.image-net.org/), CUB-200-2011 [132] (https://www.vision.caltech.edu/datasets/cub\_200\_2011/), NABirds [133] (https://dl.allaboutbirds.org/nabirds), Oxford-IIIT Pet [134] (https://www.robots.ox.ac.uk/~vgg/data/pets/), Stanford Dogs [135] (https://paperswithcode.com/dataset/stanford-cars), FGVC-Aircraft [137] (https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/), STSD [99] (https://www.cvl.isy.liu.se/research/datasets/traffic-signs-dataset/), MNIST [53] (https://paperswithcode.com/dataset/mnist), RSNA pneumonia detection [100] (https://www.rsna.org/rsnai/ai-image-challenge/rsna-pneumonia-detection-challenge-2018), CALVIN [102] (https://github.com/mees/calvin), SALICON [103] (http://salicon.net), and MIT1003 [143] (https://saliency.tuebingen.ai/). A minimum dataset for our 'visual Turing tests' is provided in Supplementary Fig. 12-13.

#### Code availability

Implementation code is available at https://github.com/LeapLabTHU/AdaptiveNN [144].

## **Acknowledgements**

G.H. is supported by the National Key R&D Program of China under Grant 2024YFB4708200, the National Natural Science Foundation of China under Grants U24B20173 and 62276150, and the Scientific Research Innovation Capability Support Project for Young Faculty under Grant ZYGXQNJSKYCXNLZCXM-I20. S.S. is supported by the National Natural Science Foundation of China under Grant 42327901. We thank S. Zhang, M. Yao and Y. Wu for helpful discussions and comments on an earlier version of this paper.

## **Author contributions**

G.H. and S.S. initiated and supervised the project. Y.W., Y.Y. (le-y22@mails.tsinghua.edu.cn), Y.Y. (yueyang22@mails.tsinghua.edu.cn) and G.H. contributed to the conception and design of the work. Y.W., Y.Y. (le-y22@mails.tsinghua.edu.cn), Y.Y. (yueyang22@mails.tsinghua.edu.cn), H.W., H.J., Y.H. and Z.N. contributed to the technical implementation. Y.W., Y.Y. (le-y22@mails.tsinghua.edu.cn), Y.P., M.S., R.L. and Q.Y. contributed to the data acquisition and organization. Y.W., Y.Y. (le-y22@mails.tsinghua.edu.cn), Y.Y. (yueyang22@mails.tsinghua.edu.cn), H.W., A.Z. and Z.X. analyzed the results. All authors contributed to the drafting and revising of the manuscript.

## **Competing interests**

The authors declare no competing interests.

![](_page_30_Figure_1.jpeg)

<span id="page-30-0"></span>*Extended Data Fig. A1.* **Results on six fine-grained visual recognition benchmarks. (a)** Quantitative comparisons of AdaptiveNN and conventional non-adaptive models: Top-1 validation accuracy versus average computational cost for inferring the model. Datasets: CUB-200-2011 [\[132\]](#page-41-10), NABirds [\[133\]](#page-41-11), Oxford-IIIT Pet [\[134\]](#page-41-12), Stanford Dogs [\[135\]](#page-41-13), Stanford Cars [\[136\]](#page-41-14), FGVC-Aircraft [\[137\]](#page-41-15). Error bars show the standard deviations of five independent trials with different random seeds. Non-adaptive models with varying costs are obtained by modifying model sizes and input resolutions. Here we set the maximum fixation number to be two, which is generally sufficient to accomplish the recognition tasks. **(b-e)** Qualitative evaluation of the visual fixations chosen by AdaptiveNN-DeiT-S across four datasets: CUB-200-2011, Oxford-IIIT Pet, Stanford Cars, and FGVC-Aircraft. The visualizations adhere to the setups established in Fig. [3a](#page-10-0).

![](_page_31_Figure_1.jpeg)

<span id="page-31-0"></span>Extended Data Fig. A2. Investigation and ablation studies of the design principles of AdaptiveNN. All the results are reported on ImageNet. See Supplementary Section B for the details of comparative baselines. (a) Efficacy of different methodologies for establishing the fixation localization strategy within AdaptiveNN. For a clean comparison, we train a classifier using only the features from visual fixations, and assume all samples use the same number of fixations, such that the resulting validation accuracy serves as a well-controlled measure to assess the effectiveness of each variant. Moreover, we consider an extensive variety of baselines for comparison, including selecting fixations using i) pre-defined rules; ii) class activation maps (CAMs); iii) CAMs augmented with a Gaussian mixture model (GMM); and iv) policy networks learned using other algorithms. (b) Average test loss corresponding to the validation data with different state values predicted by the Vision Agent in AdaptiveNN. We examine the state values taken from every step of sequential perception processes. (c) Comparisons of different termination criteria for concluding the sequential perception process of AdaptiveNN. The term 'anti-' refers to the inverse of our proposed method (detailed in Section 5.3.1), namely terminating the observation process for samples with relatively higher state values. (d-e) Comparisons with representative methodologies designed to improve deep learning models' computational efficiency. Specifically, (d) evaluates against baselines that leverage spatial redundancy in visual data. (e) examines models with multi-exit architectures that allow for online computational cost adjustments. \*Independent samples t-test. Error bars show the standard deviations of five independent trials with different random seeds.

![](_page_32_Figure_1.jpeg)

<span id="page-32-0"></span>*Extended Data Fig. A3.* **Details of 'visual Turing tests'. (a)** The full procedure of 'visual Turing tests'. We first collect the visual perception behaviors from real humans, machine (AdaptiveNN), and random generation. Then, we construct multiple trials, each including paired examples of perceptual behaviors. We consider three types of trials: i) human v.s. machine; ii) human v.s. human; and iii) human v.s. random, each corresponding to *N* trials (we use *N*=36), yielding totally 3*N* trials for each 'visual Turing test'. Finally, these 3*N* trials are mixed and shuffled for every human judge (*n*=39). The participants are only instructed to identify the machine behaviors within each trial (for all i)-iii)). Each accuracy of i)-iii) is calculated per participant and aggregated across participants. As a result, i) offers the Turing test results, while ii) and iii) provide randomized control groups as baselines and also validate whether our experimental setups are reasonable. **(b)** Results of the two 'visual Turing tests'. Each data point represents the average identification accuracy of a human judge. Bars show the mean accuracy across human judges and the corresponding 95% confidence interval. Ideal performance is 50%, where the machine is indistinguishable from human behaviors in these binary choice tasks.

![](_page_33_Figure_1.jpeg)

<span id="page-33-0"></span>*Extended Data Fig. A4.* **Details of the experiments based on embodied multimodal large language models (MLLM). (a)** The network architecture and inference procedure of the AdaptiveNN-based embodied MLLM, which mainly follows RoboFlamingo [\[101\]](#page-39-13). The backbone network is based on a pre-trained OpenFLamingo 3B [\[140\]](#page-42-1). Each two adjacent network blocks coupled with the shared vision encoder are employed as the perception net of AdaptiveNN. **(b)** The metric employed in our experiments on CALVIN. The model performance is quantified as the average successful length (0 to 5) across 1000 5-task sequences.

## **References**

- <span id="page-34-0"></span>[1] Irving Biederman. Perceiving real-world scenes. *Science*, 177(4043):77–80, 1972.
- <span id="page-34-1"></span>[2] George Sperling and Melvin J Melchner. The attention operating characteristic: Examples from visual search. *Science*, 202(4365):315–318, 1978.
- <span id="page-34-2"></span>[3] Dov Sagi and Bela Julesz. "where" and "what" in vision. *Science*, 228(4704):1217–1219, 1985.
- <span id="page-34-3"></span>[4] Jeffrey Moran and Robert Desimone. Selective attention gates visual processing in the extrastriate cortex. *Science*, 229(4715):782–784, 1985.
- <span id="page-34-4"></span>[5] Bence P Ölveczky, Stephen A Baccus, and Markus Meister. Segregation of object and background motion in the retina. *Nature*, 423(6938):401–408, 2003.
- <span id="page-34-5"></span>[6] Tirin Moore and Katherine M Armstrong. Selective gating of visual signals by microstimulation of frontal cortex. *Nature*, 421(6921):370–373, 2003.
- <span id="page-34-6"></span>[7] Jiri Najemnik and Wilson S Geisler. Optimal eye movement strategies in visual search. *Nature*, 434(7031):387–391, 2005.
- <span id="page-34-7"></span>[8] Marisa Carrasco. Visual attention: The past 25 years. *Vision research*, 51(13):1484–1525, 2011.
- <span id="page-34-8"></span>[9] Jeremy M Wolfe and Todd S Horowitz. Five factors that guide attention in visual search. *Nature Human Behaviour*, 1(3):0058, 2017.
- <span id="page-34-9"></span>[10] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. *Advances in neural information processing systems*, 35:23716–23736, 2022.
- <span id="page-34-10"></span>[11] OpenAI. Gpt-4 technical report. Technical report, OpenAI, 2023.
- <span id="page-34-11"></span>[12] Gemini Team Google. Gemini: a family of highly capable multimodal models. Technical report, Google, 2023.
- <span id="page-34-12"></span>[13] Ming Y Lu, Bowen Chen, Drew FK Williamson, Richard J Chen, Melissa Zhao, Aaron K Chow, Kenji Ikemura, Ahrong Kim, Dimitra Pouli, Ankush Patel, et al. A multimodal generative ai copilot for human pathology. *Nature*, pages 1–3, 2024.
- <span id="page-34-13"></span>[14] Elia Kaufmann, Leonard Bauersfeld, Antonio Loquercio, Matthias Müller, Vladlen Koltun, and Davide Scaramuzza. Champion-level drone racing using deep reinforcement learning. *Nature*, 620(7976):982–987, 2023.
- <span id="page-34-14"></span>[15] Brianna Zitkovich, Tianhe Yu, Sichun Xu, Peng Xu, Ted Xiao, Fei Xia, Jialin Wu, Paul Wohlhart, Stefan Welker, Ayzaan Wahid, Quan Vuong, Vincent Vanhoucke, et al. RT-2: Vision-language-action models transfer web knowledge to robotic control. In *7th Annual Conference on Robot Learning*, 2023.
- <span id="page-34-15"></span>[16] Open X-Embodiment Collaboration, Abby O'Neill, Abdul Rehman, Abhinav Gupta, Abhiram Maddukuri, Abhishek Gupta, Abhishek Padalkar, Abraham Lee, Acorn Pooley, et al. Open X-Embodiment: Robotic learning datasets and RT-X models. <https://arxiv.org/abs/2310.08864>, 2023.
- <span id="page-34-16"></span>[17] Daniel Gehrig and Davide Scaramuzza. Low-latency automotive vision with event cameras. *Nature*, 629(8014):1034–1040, 2024.
- <span id="page-34-17"></span>[18] Alvin I Chen, Max L Balter, Timothy J Maguire, and Martin L Yarmush. Deep learning robotic guidance for autonomous vascular access. *Nature Machine Intelligence*, 2(2):104–115, 2020.
- <span id="page-34-18"></span>[19] Hanwen Xu, Naoto Usuyama, Jaspreet Bagga, Sheng Zhang, Rajesh Rao, Tristan Naumann, Cliff Wong, Zelalem Gero, Javier González, Yu Gu, et al. A whole-slide foundation model for digital pathology from real-world data. *Nature*, pages 1–8, 2024.

- <span id="page-35-0"></span>[20] Xiyue Wang, Junhan Zhao, Eliana Marostica, Wei Yuan, Jietian Jin, Jiayu Zhang, Ruijiang Li, Hongping Tang, Kanran Wang, Yu Li, et al. A pathology foundation model for cancer diagnosis and prognosis prediction. *Nature*, pages 1–9, 2024.
- <span id="page-35-1"></span>[21] Raphael Schäfer, Till Nicke, Henning Höfener, Annkristin Lange, Dorit Merhof, Friedrich Feuerhake, Volkmar Schulz, Johannes Lotz, and Fabian Kiessling. Overcoming data scarcity in biomedical imaging with a foundational multi-task model. *Nature Computational Science*, 4(7):495–509, 2024.
- <span id="page-35-2"></span>[22] Brenden M Lake, Ruslan Salakhutdinov, and Joshua B Tenenbaum. Human-level concept learning through probabilistic program induction. *Science*, 350(6266):1332–1338, 2015.
- <span id="page-35-3"></span>[23] A Emin Orhan and Brenden M Lake. Learning high-level visual representations from a child's perspective without strong inductive biases. *Nature Machine Intelligence*, pages 1–13, 2024.
- <span id="page-35-4"></span>[24] Wai Keen Vong, Wentao Wang, A Emin Orhan, and Brenden M Lake. Grounded language acquisition through the eyes and ears of a single child. *Science*, 383(6682):504–511, 2024.
- <span id="page-35-5"></span>[25] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. Masked autoencoders are scalable vision learners. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 16000–16009, 2022.
- <span id="page-35-6"></span>[26] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 4015–4026, 2023.
- <span id="page-35-7"></span>[27] Mostafa Dehghani, Josip Djolonga, Basil Mustafa, Piotr Padlewski, Jonathan Heek, Justin Gilmer, Andreas Peter Steiner, Mathilde Caron, Robert Geirhos, et al. Scaling vision transformers to 22 billion parameters. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors, *Proceedings of the 40th International Conference on Machine Learning*, volume 202, pages 7480–7512, 2023.
- <span id="page-35-8"></span>[28] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 26296–26306, 2024.
- <span id="page-35-9"></span>[29] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel HAZIZA, Francisco Massa, Alaaeldin El-Nouby, Mido Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Herve Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. DINOv2: Learning robust visual features without supervision. *Transactions on Machine Learning Research*, 2024.
- <span id="page-35-10"></span>[30] David E Rumelhart, Geoffrey E Hinton, and Ronald J Williams. Learning representations by backpropagating errors. *nature*, 323(6088):533–536, 1986.
- <span id="page-35-11"></span>[31] Yann LeCun, Bernhard Boser, John Denker, Donnie Henderson, Richard Howard, Wayne Hubbard, and Lawrence Jackel. Handwritten digit recognition with a back-propagation network. *Advances in neural information processing systems*, 2, 1989.
- <span id="page-35-12"></span>[32] Yann LeCun, Yoshua Bengio, et al. Convolutional networks for images, speech, and time series. *The handbook of brain theory and neural networks*, 3361(10):1995, 1995.
- <span id="page-35-13"></span>[33] Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11):2278–2324, 1998.
- <span id="page-35-14"></span>[34] Yann LeCun. A path towards autonomous machine intelligence version 0.9. 2, 2022-06-27. *Open Review*, 62(1):1–62, 2022.

- <span id="page-36-0"></span>[35] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, et al. Imagenet large scale visual recognition challenge. *International journal of computer vision*, 115:211–252, 2015.
- <span id="page-36-1"></span>[36] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 770–778, 2016.
- <span id="page-36-2"></span>[37] Gao Huang, Zhuang Liu, Laurens Van Der Maaten, and Kilian Q Weinberger. Densely connected convolutional networks. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 4700–4708, 2017.
- <span id="page-36-3"></span>[38] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In *International Conference on Learning Representations*, 2021.
- <span id="page-36-4"></span>[39] Zhengxia Zou, Keyan Chen, Zhenwei Shi, Yuhong Guo, and Jieping Ye. Object detection in 20 years: A survey. *Proceedings of the IEEE*, 111(3):257–276, 2023.
- <span id="page-36-5"></span>[40] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In *International conference on machine learning*, pages 8748–8763. PMLR, 2021.
- <span id="page-36-6"></span>[41] Fabian Isensee, Paul F Jaeger, Simon AA Kohl, Jens Petersen, and Klaus H Maier-Hein. nnu-net: a self-configuring method for deep learning-based biomedical image segmentation. *Nature methods*, 18(2):203–211, 2021.
- <span id="page-36-7"></span>[42] Ekin Tiu, Ellie Talius, Pujan Patel, Curtis P Langlotz, Andrew Y Ng, and Pranav Rajpurkar. Expertlevel detection of pathologies from unannotated chest x-ray images via self-supervised learning. *Nature Biomedical Engineering*, 6(12):1399–1406, 2022.
- <span id="page-36-8"></span>[43] Yukun Zhou, Mark A Chia, Siegfried K Wagner, Murat S Ayhan, Dominic J Williamson, Robbert R Struyven, Timing Liu, Moucheng Xu, Mateo G Lozano, Peter Woodward-Court, et al. A foundation model for generalizable disease detection from retinal images. *Nature*, 622(7981):156–163, 2023.
- <span id="page-36-9"></span>[44] Zidong Du, Robert Fasthuber, Tianshi Chen, Paolo Ienne, Ling Li, Tao Luo, Xiaobing Feng, Yunji Chen, and Olivier Temam. Shidiannao: Shifting vision processing closer to the sensor. In *Proceedings of the 42nd annual international symposium on computer architecture*, pages 92–104, 2015.
- <span id="page-36-10"></span>[45] Jinqiang Bai, Shiguo Lian, Zhaoxiang Liu, Kai Wang, and Dijun Liu. Smart guiding glasses for visually impaired people in indoor environment. *IEEE Transactions on Consumer Electronics*, 63(3):258–266, 2017.
- <span id="page-36-11"></span>[46] Andrew G Howard. Mobilenets: Efficient convolutional neural networks for mobile vision applications. *arXiv preprint arXiv:1704.04861*, 2017.
- <span id="page-36-12"></span>[47] Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, and Liang-Chieh Chen. Mobilenetv2: Inverted residuals and linear bottlenecks. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 4510–4520, 2018.
- <span id="page-36-13"></span>[48] Gao Huang, Shichen Liu, Laurens Van der Maaten, and Kilian Q Weinberger. Condensenet: An efficient densenet using learned group convolutions. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 2752–2761, 2018.
- <span id="page-36-14"></span>[49] Jiasi Chen and Xukan Ran. Deep learning with edge computing: A review. *Proceedings of the IEEE*, 107(8):1655–1674, 2019.
- <span id="page-36-15"></span>[50] Xiaofei Wang, Yiwen Han, Victor CM Leung, Dusit Niyato, Xueqiang Yan, and Xu Chen. Convergence of edge computing and deep learning: A comprehensive survey. *IEEE Communications Surveys & Tutorials*, 22(2):869–904, 2020.

- <span id="page-37-0"></span>[51] MG Sarwar Murshed, Christopher Murphy, Daqing Hou, Nazar Khan, Ganesh Ananthanarayanan, and Faraz Hussain. Machine learning at the network edge: A survey. *ACM Computing Surveys (CSUR)*, 54(8):1–37, 2021.
- <span id="page-37-1"></span>[52] Katherine Bourzac. Fixing ai's energy crisis. *Nature*, 2024.
- <span id="page-37-2"></span>[53] Yann LeCun. The mnist database of handwritten digits. *http://yann. lecun. com/exdb/mnist/*, 1998.
- <span id="page-37-3"></span>[54] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. *arXiv preprint arXiv:2001.08361*, 2020.
- <span id="page-37-4"></span>[55] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 24185–24198, 2024.
- <span id="page-37-5"></span>[56] David J Ward and David JC MacKay. Fast hands-free writing by gaze direction. *Nature*, 418(6900):838–838, 2002.
- <span id="page-37-6"></span>[57] Wei Ji Ma, Vidhya Navalpakkam, Jeffrey M Beck, Ronald van den Berg, and Alexandre Pouget. Behavior and neural basis of near-optimal visual search. *Nature neuroScience*, 14(6):783–790, 2011.
- <span id="page-37-7"></span>[58] John M Henderson and Taylor R Hayes. Meaning-based guidance of attention in scenes as revealed by meaning maps. *Nature human behaviour*, 1(10):743–747, 2017.
- <span id="page-37-8"></span>[59] Jeremy M Wolfe and Todd S Horowitz. What attributes guide the deployment of visual attention and how do they do it? *Nature reviews neuroScience*, 5(6):495–501, 2004.
- <span id="page-37-9"></span>[60] Nina M Hanning, Antonio Fernández, and Marisa Carrasco. Dissociable roles of human frontal eye fields and early visual cortex in presaccadic attention. *Nature communications*, 14(1):5381, 2023.
- <span id="page-37-10"></span>[61] Yann LeCun, Yoshua Bengio, and Geoffrey Hinton. Deep learning. *Nature*, 521(7553):436–444, 2015.
- <span id="page-37-11"></span>[62] Volodymyr Mnih, Nicolas Heess, Alex Graves, et al. Recurrent models of visual attention. In *Advances in Neural Information Processing Systems (NeurIPS)*, 2014.
- <span id="page-37-12"></span>[63] Jimmy Ba, Volodymyr Mnih, and Koray Kavukcuoglu. Multiple object recognition with visual attention. In *International Conference on Learning Representations (ICLR)*, 2015.
- <span id="page-37-13"></span>[64] Gao Huang, Danlu Chen, Tianhong Li, Felix Wu, Laurens van der Maaten, and Kilian Weinberger. Multi-scale dense networks for resource efficient image classification. In *International Conference on Learning Representations*, 2018.
- <span id="page-37-14"></span>[65] Hao Li, Hong Zhang, Xiaojuan Qi, Ruigang Yang, and Gao Huang. Improved techniques for training adaptive deep networks. In *Proceedings of the IEEE/CVF international conference on computer vision*, pages 1891–1900, 2019.
- <span id="page-37-15"></span>[66] Le Yang, Yizeng Han, Xi Chen, Shiji Song, Jifeng Dai, and Gao Huang. Resolution adaptive networks for efficient inference. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 2369–2378, 2020.
- <span id="page-37-16"></span>[67] Yulin Wang, Kangchen Lv, Rui Huang, Shiji Song, Le Yang, and Gao Huang. Glance and focus: a dynamic approach to reducing spatial redundancy in image classification. In *Advances in Neural Information Processing Systems*, volume 33, pages 2432–2444, 2020.
- <span id="page-37-17"></span>[68] Gao Huang, Yulin Wang, Kangchen Lv, Haojun Jiang, Wenhui Huang, Pengfei Qi, and Shiji Song. Glance and focus networks for dynamic visual recognition. *IEEE transactions on pattern analysis and machine intelligence*, 45(4):4605–4621, 2022.

- <span id="page-38-0"></span>[69] Yulin Wang, Rui Huang, Shiji Song, Zeyi Huang, and Gao Huang. Not all images are worth 16x16 words: Dynamic transformers for efficient image recognition. In *Advances in Neural Information Processing Systems*, 2021.
- <span id="page-38-1"></span>[70] Bowen Pan, Rameswar Panda, Yifan Jiang, Zhangyang Wang, Rogerio Feris, and Aude Oliva. Ia-red2 : Interpretability-aware redundancy reduction for vision transformers. *Advances in Neural Information Processing Systems*, 34:24898–24911, 2021.
- <span id="page-38-2"></span>[71] Yongming Rao, Wenliang Zhao, Benlin Liu, Jiwen Lu, Jie Zhou, and Cho-Jui Hsieh. Dynamicvit: Efficient vision transformers with dynamic token sparsification. In *Advances in neural information processing systems*, volume 34, pages 13937–13949, 2021.
- <span id="page-38-3"></span>[72] Yifan Xu, Zhijie Zhang, Mengdan Zhang, Kekai Sheng, Ke Li, Weiming Dong, Liqing Zhang, Changsheng Xu, and Xing Sun. Evo-vit: Slow-fast token evolution for dynamic vision transformer. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 36, pages 2964–2972, 2022.
- <span id="page-38-4"></span>[73] Mohsen Fayyaz, Soroush Abbasi Koohpayegani, Farnoush Rezaei Jafari, Sunando Sengupta, Hamid Reza Vaezi Joze, Eric Sommerlade, Hamed Pirsiavash, and Jürgen Gall. Adaptive token sampling for efficient vision transformers. In *European Conference on Computer Vision*, pages 396–414. Springer, 2022.
- <span id="page-38-5"></span>[74] Hongxu Yin, Arash Vahdat, Jose M Alvarez, Arun Mallya, Jan Kautz, and Pavlo Molchanov. A-vit: Adaptive tokens for efficient vision transformer. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 10809–10818, 2022.
- <span id="page-38-6"></span>[75] Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, and Judy Hoffman. Token merging: Your vit but faster. In *International Conference on Learning Representations*, 2023.
- <span id="page-38-7"></span>[76] Jacqueline Gottlieb and Pierre-Yves Oudeyer. Towards a neuroscience of active sampling and curiosity. *Nature Reviews Neuroscience*, 19(12):758–770, 2018.
- <span id="page-38-8"></span>[77] Nachiappan Valliappan, Na Dai, Ethan Steinberg, Junfeng He, Kantwon Rogers, Venky Ramachandran, Pingmei Xu, Mina Shojaeizadeh, Li Guo, Kai Kohlhoff, et al. Accelerating eye movement research via accurate and affordable smartphone eye tracking. *Nature communications*, 11(1):4553, 2020.
- <span id="page-38-9"></span>[78] Philip J Kellman and Elizabeth S Spelke. Perception of partly occluded objects in infancy. *Cognitive psychology*, 15(4):483–524, 1983.
- <span id="page-38-10"></span>[79] Elizabeth S Spelke, Karen Breinlinger, Janet Macomber, and Kristen Jacobson. Origins of knowledge. *Psychological review*, 99(4):605, 1992.
- <span id="page-38-11"></span>[80] Elizabeth Spelke. Initial knowledge: Six suggestions. *Cognition*, 50(1-3):431–445, 1994.
- <span id="page-38-12"></span>[81] Cassia Viola Macchi, Chiara Turati, and Francesca Simion. Can a nonspecific bias toward top-heavy patterns explain newborns' face preference? *Psychological Science*, 15(6):379–383, 2004.
- <span id="page-38-13"></span>[82] Francesca Simion, Elisa Di Giorgio, Irene Leo, and Lara Bardi. The processing of social stimuli in early infancy: from faces to biological motion perception. *Progress in brain research*, 189:173–193, 2011.
- <span id="page-38-14"></span>[83] Shimon Ullman, Daniel Harari, and Nimrod Dorfman. From simple innate biases to complex visual concepts. *Proceedings of the National Academy of Sciences*, 109(44):18215–18220, 2012.
- <span id="page-38-15"></span>[84] Aimee E Stahl and Lisa Feigenson. Observing the unexpected enhances infants' learning and exploration. *Science*, 348(6230):91–94, 2015.
- <span id="page-38-16"></span>[85] Greg D Reynolds and Kelly C Roth. The development of attentional biases for faces in infancy: A developmental systems perspective. *Frontiers in psychology*, 9:315789, 2018.
- <span id="page-38-17"></span>[86] David Navon. Forest before trees: The precedence of global features in visual perception. *Cognitive psychology*, 9(3):353–383, 1977.
- <span id="page-38-18"></span>[87] Lin Chen. Topological structure in visual perception. *Science*, 218(4573):699–700, 1982.

- <span id="page-39-0"></span>[88] Shaul Hochstein and Merav Ahissar. View from the top: Hierarchies and reverse hierarchies in the visual system. *Neuron*, 36(5):791–804, 2002.
- <span id="page-39-1"></span>[89] Tzvi Ganel and Melvyn A Goodale. Visual control of action but not perception requires analytical processing of object shape. *Nature*, 426(6967):664–667, 2003.
- <span id="page-39-2"></span>[90] Aude Oliva and Antonio Torralba. Building the gist of a scene: The role of global image features in recognition. *Progress in brain research*, 155:23–36, 2006.
- <span id="page-39-3"></span>[91] Marius V Peelen, Eva Berlot, and Floris P de Lange. Predictive processing of scenes and objects. *Nature Reviews Psychology*, 3(1):13–26, 2024.
- <span id="page-39-4"></span>[92] John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, and Pieter Abbeel. High-dimensional continuous control using generalized advantage estimation. In *Proceedings of the International Conference on Learning Representations (ICLR)*, 2016.
- <span id="page-39-5"></span>[93] Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G Bellemare, Alex Graves, Martin Riedmiller, Andreas K Fidjeland, Georg Ostrovski, et al. Human-level control through deep reinforcement learning. *nature*, 518(7540):529–533, 2015.
- <span id="page-39-6"></span>[94] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. *arXiv preprint arXiv:1707.06347*, 2017.
- <span id="page-39-7"></span>[95] Richard S Sutton, David McAllester, Satinder Singh, and Yishay Mansour. Policy gradient methods for reinforcement learning with function approximation. In *Advances in neural information processing systems*, volume 12, 1999.
- <span id="page-39-8"></span>[96] David Silver, Aja Huang, Chris J Maddison, Arthur Guez, Laurent Sifre, George Van Den Driessche, Julian Schrittwieser, Ioannis Antonoglou, Veda Panneershelvam, Marc Lanctot, et al. Mastering the game of go with deep neural networks and tree search. *nature*, 529(7587):484–489, 2016.
- <span id="page-39-9"></span>[97] George A Miller. Wordnet: a lexical database for english. *Communications of the ACM*, 38(11):39–41, 1995.
- <span id="page-39-10"></span>[98] Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Hervé Jégou. Training data-efficient image transformers & distillation through attention. In *International conference on machine learning*, pages 10347–10357. PMLR, 2021.
- <span id="page-39-11"></span>[99] Fredrik Larsson and Michael Felsberg. Using fourier descriptors and spatial models for traffic sign recognition. In *Image Analysis: 17th Scandinavian Conference, SCIA 2011, Ystad, Sweden, May 2011. Proceedings 17*, pages 238–249. Springer, 2011.
- <span id="page-39-12"></span>[100] George Shih, Carol C Wu, Safwan S Halabi, Marc D Kohli, Luciano M Prevedello, Tessa S Cook, Arjun Sharma, Judith K Amorosa, Veronica Arteaga, Maya Galperin-Aizenberg, et al. Augmenting the national institutes of health chest radiograph dataset with expert annotations of possible pneumonia. *Radiology: Artificial Intelligence*, 1(1):e180041, 2019.
- <span id="page-39-13"></span>[101] Xinghang Li, Minghuan Liu, Hanbo Zhang, Cunjun Yu, Jie Xu, Hongtao Wu, Chilam Cheang, Ya Jing, Weinan Zhang, Huaping Liu, Hang Li, and Tao Kong. Vision-language foundation models as effective robot imitators. In *International Conference on Learning Representations*, 2024.
- <span id="page-39-14"></span>[102] Oier Mees, Lukas Hermann, Erick Rosete-Beas, and Wolfram Burgard. Calvin: A benchmark for languageconditioned policy learning for long-horizon robot manipulation tasks. *IEEE Robotics and Automation Letters (RA-L)*, 7(3):7327–7334, 2022.
- <span id="page-39-15"></span>[103] Ming Jiang, Shengsheng Huang, Juanyong Duan, and Qi Zhao. Salicon: Saliency in context. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 1072–1080, 2015.

- <span id="page-40-0"></span>[104] Liyuan Wang, Xingxing Zhang, Qian Li, Mingtian Zhang, Hang Su, Jun Zhu, and Yi Zhong. Incorporating neuro-inspired adaptability for continual learning in artificial intelligence. *Nature Machine Intelligence*, 5(12):1356–1368, 2023.
- <span id="page-40-1"></span>[105] Laurent Itti and Christof Koch. Computational modelling of visual attention. *Nature reviews neuroscience*, 2(3):194–203, 2001.
- <span id="page-40-2"></span>[106] John M Henderson. Human gaze control during real-world scene perception. *Trends in cognitive sciences*, 7(11):498–504, 2003.
- <span id="page-40-3"></span>[107] Yizeng Han, Gao Huang, Shiji Song, Le Yang, Honghui Wang, and Yulin Wang. Dynamic neural networks: A survey. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 44(11):7436–7456, 2021.
- <span id="page-40-4"></span>[108] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In *Medical image computing and computer-assisted intervention–MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18*, pages 234–241. Springer, 2015.
- <span id="page-40-5"></span>[109] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In *Proceedings of the IEEE/CVF international conference on computer vision*, pages 10012–10022, 2021.
- <span id="page-40-6"></span>[110] Mingxing Tan and Quoc Le. Efficientnetv2: Smaller models and faster training. In *International conference on machine learning*, pages 10096–10106. PMLR, 2021.
- <span id="page-40-7"></span>[111] Xiaoyi Dong, Jianmin Bao, Dongdong Chen, Weiming Zhang, Nenghai Yu, Lu Yuan, Dong Chen, and Baining Guo. Cswin transformer: A general vision transformer backbone with cross-shaped windows. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 12124–12134, 2022.
- <span id="page-40-8"></span>[112] Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali Farhadi. You only look once: Unified, real-time object detection. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 779–788, 2016.
- <span id="page-40-9"></span>[113] Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, and Alexander C Berg. Ssd: Single shot multibox detector. In *Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11–14, 2016, Proceedings, Part I 14*, pages 21–37. Springer, 2016.
- <span id="page-40-10"></span>[114] Hugo Larochelle and Geoffrey E Hinton. Learning to combine foveal glimpses with a third-order boltzmann machine. *Advances in neural information processing systems*, 23, 2010.
- <span id="page-40-11"></span>[115] John Locke. *An essay concerning human understanding*. 1847.
- <span id="page-40-12"></span>[116] Gottfried W Leibniz, William Seager, Jonathan Bennett, and Peter Remnant. New essays on human understanding. *University of Toronto Quarterly*, 67(1):208, 1997.
- <span id="page-40-13"></span>[117] Lorijn Zaadnoordijk, Tarek R Besold, and Rhodri Cusack. Lessons from infant learning for unsupervised machine learning. *Nature Machine Intelligence*, 4(6):510–520, 2022.
- <span id="page-40-14"></span>[118] Jeffrey L Elman. Rethinking innateness: A connectionist perspective on development, volume 10. 1996.
- <span id="page-40-15"></span>[119] Sven Bambach, David Crandall, Linda Smith, and Chen Yu. Toddler-inspired visual object learning. *Advances in neural information processing systems*, 31, 2018.
- <span id="page-40-16"></span>[120] Emin Orhan, Vaibhav Gupta, and Brenden M Lake. Self-supervised learning through the eyes of a child. *Advances in Neural Information Processing Systems*, 33:9960–9971, 2020.
- <span id="page-40-17"></span>[121] Evelyn Fox Keller and Christine R Grontkowski. The mind's eye. In *Discovering reality: Feminist perspectives on epistemology, metaphysics, methodology, and philosophy of science*, pages 207–224. 1983.

- <span id="page-41-0"></span>[122] Keith Rayner. Eye movements in reading and information processing: 20 years of research. *Psychological bulletin*, 124(3):372, 1998.
- <span id="page-41-1"></span>[123] Mary Hayhoe and Dana Ballard. Eye movements in natural behavior. *Trends in cognitive Sciences*, 9(4):188–194, 2005.
- <span id="page-41-2"></span>[124] Michael F Land. Vision, eye movements, and natural behavior. Visual neuroscience, 26(1):51–62, 2009.
- <span id="page-41-3"></span>[125] Jakob Nielsen and Kara Pernice. Eyetracking web usability. 2010.
- <span id="page-41-4"></span>[126] Zoya Bylinskii, Nam Wook Kim, Peter O'Donovan, Sami Alsheikh, Spandan Madan, Hanspeter Pfister, Fredo Durand, Bryan Russell, and Aaron Hertzmann. Learning visual importance for graphic designs and data visualizations. In *Proceedings of the 30th Annual ACM symposium on user interface software and technology*, pages 57–69, 2017.
- <span id="page-41-5"></span>[127] Michael Land and Benjamin Tatler. Looking and acting: vision and eye movements in natural behaviour. 2009.
- <span id="page-41-6"></span>[128] J David Smith and TC Nicholas Graham. Use of eye movements for video game control. In *Proceedings of the 2006 ACM SIGCHI international conference on Advances in computer entertainment technology*, pages 20–es, 2006.
- <span id="page-41-7"></span>[129] Warren Jones, Katelin Carr, and Ami Klin. Absence of preferential looking to the eyes of approaching adults predicts level of social disability in 2-year-old toddlers with autism spectrum disorder. *Archives of general psychiatry*, 65(8):946–954, 2008.
- <span id="page-41-8"></span>[130] Ricardo Bigolin Lanfredi, Mingyuan Zhang, William F Auffermann, Jessica Chan, Phuong-Anh T Duong, Vivek Srikumar, Trafton Drew, Joyce D Schroeder, and Tolga Tasdizen. Reflacx, a dataset of reports and eye-tracking data for localization of abnormalities in chest x-rays. *Scientific data*, 9(1):350, 2022.
- <span id="page-41-9"></span>[131] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. *arXiv preprint arXiv:2501.12948*, 2025.
- <span id="page-41-10"></span>[132] Catherine Wah, Steve Branson, Peter Welinder, Pietro Perona, and Serge Belongie. The caltech-ucsd birds-200-2011 dataset. 2011.
- <span id="page-41-11"></span>[133] Grant Van Horn, Steve Branson, Ryan Farrell, Scott Haber, Jessie Barry, Panos Ipeirotis, Pietro Perona, and Serge Belongie. Building a bird recognition app and large scale dataset with citizen scientists: The fine print in fine-grained dataset collection. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 595–604, 2015.
- <span id="page-41-12"></span>[134] Omkar M Parkhi, Andrea Vedaldi, Andrew Zisserman, and CV Jawahar. Cats and dogs. In 2012 IEEE conference on computer vision and pattern recognition, pages 3498–3505. IEEE, 2012.
- <span id="page-41-13"></span>[135] Aditya Khosla, Nityananda Jayadevaprakash, Bangpeng Yao, and Fei-Fei Li. Novel dataset for fine-grained image categorization: Stanford dogs. In *Proc. CVPR workshop on fine-grained visual categorization* (*FGVC*), volume 2. Citeseer, 2011.
- <span id="page-41-14"></span>[136] Jonathan Krause, Michael Stark, Jia Deng, and Li Fei-Fei. 3d object representations for fine-grained categorization. In *Proceedings of the IEEE international conference on computer vision workshops*, pages 554–561, 2013.
- <span id="page-41-15"></span>[137] Subhransu Maji, Esa Rahtu, Juho Kannala, Matthew Blaschko, and Andrea Vedaldi. Fine-grained visual classification of aircraft. *arXiv preprint arXiv:1306.5151*, 2013.
- <span id="page-41-16"></span>[138] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. *Advances in neural information processing systems*, 30, 2017.

- <span id="page-42-0"></span>[139] Hsin-Hung Li, Jasmine Pan, and Marisa Carrasco. Different computations underlie overt presaccadic and covert spatial attention. *Nature human behaviour*, 5(10):1418–1431, 2021.
- <span id="page-42-1"></span>[140] Anas Awadalla, Irena Gao, Josh Gardner, Jack Hessel, Yusuf Hanafy, Wanrong Zhu, Kalyani Marathe, Yonatan Bitton, Samir Gadre, Shiori Sagawa, et al. Openflamingo: An open-source framework for training large autoregressive vision-language models. *arXiv preprint arXiv:2308.01390*, 2023.
- <span id="page-42-2"></span>[141] Yulin Wang, Yang Yue, Yuanze Lin, Haojun Jiang, Zihang Lai, Victor Kulikov, Nikita Orlov, Humphrey Shi, and Gao Huang. Adafocus v2: End-to-end training of spatial dynamic networks for video recognition. In *2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 20030–20040. IEEE, 2022.
- <span id="page-42-3"></span>[142] Yizeng Han, Dongchen Han, Zeyu Liu, Yulin Wang, Xuran Pan, Yifan Pu, Chao Deng, Junlan Feng, Shiji Song, and Gao Huang. Dynamic perceiver for efficient visual recognition. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 5992–6002, 2023.
- <span id="page-42-4"></span>[143] Tilke Judd, Krista Ehinger, Frédo Durand, and Antonio Torralba. Learning to predict where humans look. In *IEEE International Conference on Computer Vision (ICCV)*, 2009.
- <span id="page-42-5"></span>[144] LeapLab. Leaplabthu/adaptivenn: Official release, August 2025.

 Supplementary materials for 'Emulating Human-like Adaptive Vision for Efficient, Flexible, and Interpretable Machine Visual Perception'

## A Related works

- To position our novel contributions in the state-of-the-art, here we provide a
- systematic comparison between our work and relevant existing literature.

## A.1 Computationally efficient computer vision backbones

 Modern artificial vision models are typically built upon deep neural networks, such as convolutional neural networks [1, 2] and vision Transformers [3, 4]. Innovations in network architectures and learning algorithms have enabled these models to approach or even surpass human-level performance in various vision tasks [5–9]. However, models with state-of-the-art performance usually come at the price of a computationally intensive inference procedure. In many realistic scenarios such as wearable devices, mobile phones, robotics, embed- ded devices, and autonomous vehicles, computation directly translates into power consumption, latency, and carbon emissions, and all of them should be minimized under environmental, safety or economic considerations [10–16].

 To mitigate these issues, numerous recent research efforts have been directed toward developing lightweight backbone networks, such as MobileNets [17–20], CondenseNets [21, 22], ShuffleNets [23, 24], EfficientNets [25, 26], GhostNets [27–29], Mobile-former [30], and EfficientFormers [31, 32]. These models are designed to maintain performance while reducing the computa- tional inference cost. Moreover, motivated by that deep networks may contain a considerable number of redundant parameters, many approaches propose to prune less useful weights [33–37] or quantize the weights [38–40]. Another important technique is knowledge distillation, which trains smaller models to mirror the behaviors of larger models [41, 42].

 In general, our work is orthogonal to these existing methods. The advanced architectures of backbone networks can be conveniently deployed as the the

 feature-extraction modules of AdaptiveNN, while our model can be fur- ther enhanced by leveraging techniques such as pruning, quantization, and knowledge distillation.

## A.2 Dynamic neural networks

 This article is relevant to the dynamic neural networks that allocate computa- tional resources unevenly across different samples or different spatial locations [43]. However, our work differentiates itself from the existing literature in this direction in several critical aspects. First, we attain a natural and elegant com- bination of sample-wise and spatial-wise adaptive computation by mimicking human visual systems. Our model not only enhances computational efficiency but also improves the model's behavioral adaptability and interpretability. In contrast, most existing works only focus on a single sort of adaptive compu- tation and do not secure a biologically plausible motivation, with only one or two out of the three aspects of our gains observed. Second, for the first time, we develop theoretical analyses concerning the learning principles of dynamic neural networks, revealing the natural emergence of reinforcement learning rules under the goal of optimizing perceptual behavior distributions for an arbitrary vision task. Third, to our best knowledge, we present the first work seeking to build up a bridge between dynamic neural networks and human visual cognition. This includes establishing methodologies for designing and learning a human-like deep learning framework, systematically comparing its behaviors with those of human visual systems, and demonstrating its potential for probing into human behavioral and learning processes.

 In the following, we briefly review the current works on dynamic neural networks to further underscore the innovative aspects of our work.

## A.2.1 Sample-wise dynamic neural networks

 Some recent works improve the efficiency of deep networks by customizing the architectures or parameters of the model conditioned on individual inputs. For example, multi-scale DenseNet (MSDNet) [44] and its improved version [45] introduce a multi-scale architecture equipped with multiple classifiers, allowing for utilizing small networks for easier samples and switching to large networks for more challenging ones. Similarly, dynamic vision Transformer (DVT) [46] processes each image with adaptive token numbers by cascading multiple vision Transformers and reusing computation. Dynamic perceiver [47] introduces an addition attention-based pathway that integrates and selectively exits from intermediate network representations based on the task demands. VideoIQ [48] dynamically adjusts the precision of its models based on the importance of each input. Additionally, several approaches have been devel-oped that ensemble multiple models, selectively activating only a subset of

 architectures [49–51] or dynamically bypassing unnecessary layers [52–54] or channels [55]. Beyond architectures, models like RANet [56], DR-Net [57], and AR-Net [58] process different inputs with varying amounts of computation by changing their resolutions.

## A.2.2 Spatial-wise dynamic neural networks

 It has been observed that not all spatial regions within visual data hold equal relevance for specific tasks [59, 60]. Motivated by this phenomenon, some approaches have been developed to improve computational efficiency by allo- cating computation unevenly across different spatial regions according to their relative importance. One such technique, the spatially adaptive computation time (SACT) algorithm [61] dynamically modulates the number of executed network layers conditioned on different image regions. The algorithms pro- posed in [62] and [63] selectively bypass less critical regions of feature maps. In the context of vision Transformers, many works propose to discard or merge less important visual tokens progressively during the inference procedure of the model [64–68], aiming to reduce less necessary computation.

 Furthermore, some methods autonomously identify and focus on the most informative details of the input data. These works tend to mainly consider improving performance, with less attention paid to efficiency. For example, RA-CNN [69], TASN [70], and NTS-Net [71] enhance the classification of fine- grained visual categories by concentrating on discriminative local patterns, such as the afterbrain colors of birds, which are critical for distinguishing sub- tle inter-species differences. S3N [72] and MGE-CNN [73] employ the class activation maps (CAMs) [74, 75] to isolate and emphasize the visual cues essen- tial for fine-grained image recognition. Similar techniques are also explored in diverse applications including vehicle re-identification [76], image translation [77] and zero-shot recognition [78–80].

 More recently, some works focus on augmenting large vision-language mod- els with the capabilities to selectively attend to the task-relative regions within visual data [81–84]. Against this context, the innovative aspects of our work discussed at the beginning of this section still hold. In addition to them, how- ever, our model is also promising in that it does not necessitate any specialized supervision signal to teach the model 'where to look', while these methods or some of their components usually rely on tailored training data with such localization supervision. This attribute potentially enhances the scalability of our approach, making it more adaptable to varied and less constrained cir- cumstances. Moreover, given the flexibility of our framework and its learning algorithm, we believe AdaptiveNN is generally compatible with the vision tasks defined on top of large vision-language models, which may be an interesting topic for future work.

## A.3 Visual attention models

 Attention constitutes one of the fundamental mechanisms of human vision: people sequentially fixate on regions of the visual environment that are relevant for the task of interest [85–89]. In the context of integrating this mechanism into computer vision models and accomplishing vision tasks, some works pro- pose to predict and leverage soft attention masks tailored for each input images [90–92]. These methods mainly focus on modeling the results, or guidance, of human visual attention instead of its procedure. For example, an attention mask is usually generated through a simultaneous and computationally equiv- alent processing of all relevant image regions. Moreover, it may be costly to obtain proper attention masks on top of relatively large-scale inputs. Recent advances in this direction have seen the direct incorporation of these soft- attention mechanisms into the micro-architectures of backbone networks, often as modules that are based on local features with reduced sizes [3, 93–96]. To this end, our work is generally orthogonal to these efforts.

 Another line of works related to us are visual hard attention models. For example, three representative methods, restricted Boltzmann machine (RBM) [97], recurrent attention model (RAM) [98], and deep recurrent attention model (DRAM) [99], share a similar spirit to AdaptiveNN in perceiving visual scenes through recurrent attention. However, they tend to exhibit limited scal- ability in scenarios beyond simple tasks like tiny digit classification [99, 100]. This problem is moderately alleviated by some recent research [100–103]. Nev- ertheless, our work considerably outperforms them in constructing a general, flexible, and scalable framework that emulates human adaptive vision, leading to substantial enhancements in efficiency, adaptability, and interpretability. Furthermore, our work is novel in developing theoretical analyses to reveal the emergence of broadly applicable reinforcement learning principles that do not rely on specialized supervision signals. Meanwhile, for the first time in the modern community of modeling visual hard attention, we comprehen- sively compare the perceptual behaviors of our vision models with those of humans, and discuss the notable potential of our work for probing human visual cognition.

# B Baseline approaches for comparison

 In this article, we compare AdaptiveNN against a broad range of baseline methods, aiming to comprehensively demonstrate the characteristics of our model. Here we briefly introduce these comparative baselines and describe how they are properly configured in our experiments.

 Unless otherwise specified, AdaptiveNN is mainly compared against tra-ditional, non-adaptive deep networks on top of the same backbone models.  According to the common practice of the deep learning community [1–4, 95], these baselines process all regions within visual inputs with an equivalent amount of computation, in a single step, and the baselines with different infer- ence costs are obtained by modifying the model size and input resolution. It is important to note that AdaptiveNN does not leverage any advanced architectural innovations, nor does it employ additional data augmentation or regularization strategies that are not already parts of the baseline models. The only difference between AdaptiveNN and the baselines lies in the introduction of mechanisms designed to emulate the adaptive visual perception processes of humans. These experimental setups contribute to a focused and isolated evaluation of the benefits derived from our core contributions.

 In Fig. 4c, we compare our model with two representative algorithms designed for learning to emulate human adaptive vision, recurrent attention model (RAM) [98] and deep recurrent attention model (DRAM) [99]. To ensure a fair comparison, AdaptiveNN is configured using neural network components analogous to those used in RAM and DRAM: simple convolutional layers com- bined with multilayer perceptrons (see Section 5.3.2 for details). The model sizes of AdaptiveNN, RAM, and DRAM are approximately equivalent (in fact, we find that further enlarging RAM or DRAM only results in minor improve- ments). Moreover, to ensure reproducing their work reasonably, we adopt the default hyper-parameters and training configurations recommended by their original papers as the starting point, and perform the same hyper-parameter search procedure as our model.

 In Fig. 6a and Extended Data Fig. A2a, the fixation selection strategy learned by AdaptiveNN is compared against four categories of alternative design choices, including selecting fixations using i) non-adaptive pre-defined rules, as foundational baselines; ii) class activation maps (CAMs), which are widely acknowledged as a useful tool to localize the regions based on which neural networks make decisions [9, 104]; iii) CAMs augmented with a Gaussian mixture model (GMM), aimed at maximizing the utility of CAMs regardless of the computational cost; and iv) policy networks learned using other algorithms, to assess whether reinforcement learning is a proper algorithm for optimiz- ing fixation selection strategies as we discuss in theory. For i), we examine random/Gaussian: sampling fixation locations following uniform or Gaussian distributions; and center-corner: first fixating on the center of images, and then traversing the corners. For ii), we consider several representative examples of widely used CAM algorithms, including GradCAM [75], GradCAM++ [105], XGradCAM [106], and LayerCAM [107]. For fair comparisons between them and AdaptiveNN, we solve CAMs utilizing the first internal vision representa- tion s<sup>0</sup> of our model, derived from the initial quick glance (nevertheless, we find that solving CAMs using larger models or inputs does not notably influence

 the results). Then we sample subsequent visual fixations based on the intensity values of CAMs. For iii), we further fit CAMs with a Gaussian mixture model [108], selecting the values of means of the top-n principal components as the fixation locations. Note that GMM introduces additional computation, and this approach represents an estimate of the 'upper bound' for the performance of CAM-based methods. For iv), we consider two alternative choices of rein- forcement learning: spatial transformer networks [109] and Gumbel-Softmax [110]. They approximate the non-differentiable fixation selection problem with interpolation and the Gumbel trick, respectively, such that the model can be trained in end-to-end.

 In Extended Data Fig. A2d, A2e, we present a system-level comparison of AdaptiveNN against representative state-of-the-art methods designed to improve the computational efficiency of deep learning models. We mainly consider two categories of baselines relevant to our model, corresponding to Extended Data Fig. A2d and A2e, respectively. Extended Data Fig. A2d focuses on recently proposed algorithms leveraging the inherent spatial redundancy within visual data, including IA-RED<sup>2</sup> [111], DynamicViT [64], Evo-ViT [65], ATS [66], A-ViT [67], and Token Merging [68]. Extended Data Fig. A2e considers existing multi-exit models that can adjust their computational cost online in a similar way to our model, including MSDNet [44], IMTA-MSDNet [45], RANet [56], GFNet [102, 112], and DVT [46].

# C Training hyper-parameters and configurations

 Here we detail the specific configurations and hyper-parameters for training our AdaptiveNN models. For all datasets, we held out 20% of training data to perform a hyper-parameter search, and then put this data back to the training set, reporting final results.

 On ImageNet, we train AdaptiveNN-DeiT-S and AdaptiveNN-ResNet-50 mainly utilizing the training pipelines proposed in the original papers of DeiT [4] and ResNet [1]. Our basic settings are summarized in Extended Data Tab. 1. Notably, all the baseline models adopt the same training configurations as AdaptiveNN to ensure fair comparisons. For the proximal policy optimization (PPO) algorithm [113] we employ to solve the reinforcement learning problem, we set γ=0.5, ϵ=0.2, c1=1, and c2=0. We set λ=0.84 for generalized advantage estimation [114]. For fine-grained classification tasks and the medical diagnosis task, we fine-tune the models pre-trained on ImageNet, where we reduce the batch size and initial learning rate, and increase the stochastic depth rate, to alleviate overfitting.

| Training configs           | AdaptiveNN-DeiT-S               | ${\bf Adaptive NN-ResNet-50}$ |  |  |
|----------------------------|---------------------------------|-------------------------------|--|--|
| Data pre-processing        | Random res                      | sized crop [1, 2]             |  |  |
| Size of images             | $288^{2}$                       | $320^{2}$                     |  |  |
| Glance size                | 1                               | $112^{2}$                     |  |  |
| Size of visual fixations   | 1                               | $112^{2}$                     |  |  |
| Optimizer                  | $\operatorname{AdamW}$          | $\operatorname{SGD}$          |  |  |
| Optimizer hyper-parameters | $\beta_1, \beta_2 = 0.9, 0.999$ | momentum = 0.9                |  |  |
| Initial learning rate      | 4e-3                            | 0.4                           |  |  |
| Learning rate schedule     | Cosine annealing                |                               |  |  |
| Weight decay               | 0.05                            | 1e-4                          |  |  |
| Batch size                 | 4,096                           | 1,024                         |  |  |
| Training epochs            | 300                             | 100                           |  |  |
| Warmup epochs              | 20                              | 5                             |  |  |
| Warmup schedule            | L                               | inear                         |  |  |
| RandAug [115]              | (9, 0.5)                        | _                             |  |  |
| Mixup [116]                | 0.8                             | _                             |  |  |
| Cutmix [117]               | 1.0                             | _                             |  |  |
| Random erasing [118]       | 0.25                            | _                             |  |  |
| Label smoothing [119]      |                                 | 0.1                           |  |  |
| Stochastic depth [120]     | 0.1                             | _                             |  |  |
| Gradient clipping          | 5.0                             | _                             |  |  |

**Supplementary Data Tab. 1** Hyper-parameters and configurations for training AdaptiveNN models on ImageNet-1K.

For the traffic sign recognition task on STSD, we utilize an Adam optimizer with  $\beta_1$ ,  $\beta_2$ =0.9, 0.999, and adopt a batch size of 32. We set the initial learning rate at 0.001 with a cosine annealing schedule, and implement an L2 weight decay coefficient of  $10^{-5}$ . The two backbone networks within our model is initialized from ImageNet pre-trained checkpoints. Training data is augmented with Mixup [116]. Additionally, due to the long-tailed nature of the dataset, class-balanced loss reweighting is utilized for both representation learning objectives and the rewards in reinforcement learning. The original high-resolution image (960×1280) are down-sampled to 192<sup>2</sup> as the glance input, while the vision agent localizes 192<sup>2</sup> visual fixation sequences with a maximum length of 2. We set the discount factor  $\gamma$  as 0.2.

For the visual search tasks, we utilize an SGD optimizer with a momentum of 0.9, and adopt a batch size of 1024. We set the initial learning rate at 0.006 for the vision agent and 0.01 for other components, with a cosine annealing schedule. The model is trained for 200 epochs, with a 10-epoch warmup. We implement an L2 weight decay coefficient of  $10^{-4}$ . For retrieving the locations of digits, we view it as a classification problem over numerous candidates, employ a multilayer perceptron as the classification head for each target digit, and adopt a dropout rate of 0.7 in the classification head to alleviate overfitting. The input images, sized at  $224^2$ , are down-sampled to  $42^2$  as the glance input, while the vision agent localizes  $28^2$  visual fixation sequences with a maximum length of 5. Here only the features of visual fixations are used for visual search since glance inputs tend to provide little useful information for identifying the

 small digits. We set the discount factor γ as 1.0 to assign more attention to the final results incurred by the current action, rather than focusing on instant results (as our objective is to accomplish the task successfully). Additionally, to encourage the policy to focus on digits, we add a penalty term to the reward; specifically, if the visual fixation region selected by the current action contains any non-black pixel, the penalty is zero. Otherwise, a value of -1.5 is added to the current reward. Note that both baselines and our method are equipped with this technique.

# D Supplementary figures and tables

![](_page_51_Figure_3.jpeg)

Supplementary Data Fig. 1 Randomly selected examples of the 'visual Turing test' on the spatial-wise visual fixation behaviors of AdaptiveNN and humans. The three-sample groups in each pair that are obtained with a machine are (by row) 2, 2; 1, 1; 2, 1.

![](_page_52_Figure_2.jpeg)

Supplementary Data Fig. 2 Randomly selected examples of the 'visual Turing test' on the sample-wise visual difficulty assessment behaviors of AdaptiveNN and humans. The threesample groups in each pair that are obtained with a machine are (by row) 2, 1; 2, 1; 2, 2.

![](_page_53_Figure_2.jpeg)

![](_page_53_Figure_3.jpeg)

Supplementary Data Fig. 3 Per-participant results of the 'visual Turing test' on the spatial-wise visual fixation behaviors. '±' indicates the 95% confidence interval. The distribution across all participants is shown on the right.

![](_page_54_Figure_2.jpeg)

![](_page_54_Figure_3.jpeg)

Supplementary Data Fig. 4 Per-participant results of the 'visual Turing test' on the sample-wise visual difficulty assessment behaviors. '±' indicates the 95% confidence interval. The distribution across all participants is shown on the right.

| Model      | Size of<br>Image<br>(fixation)           | #Visual<br>Fixations<br>(average) | Computational<br>Cost<br>(GFLOPs/image) | Accuracy<br>(%) |  |  |  |  |
|------------|------------------------------------------|-----------------------------------|-----------------------------------------|-----------------|--|--|--|--|
|            | Models with<br>fixed computational cost. |                                   |                                         |                 |  |  |  |  |
| DeiT-T     | 2242<br>(–)                              | –                                 | 1.26                                    | 72.2            |  |  |  |  |
| DeiT-T     | 2882<br>(–)                              | –                                 | 2.27                                    | 75.0            |  |  |  |  |
| DeiT-T     | 3842<br>(–)                              | –                                 | 4.70                                    | 76.5            |  |  |  |  |
| DeiT-S     | 2242<br>(–)                              | –                                 | 4.61                                    | 79.9            |  |  |  |  |
| DeiT-S     | 2882<br>(–)                              | –                                 | 7.99                                    | 80.9            |  |  |  |  |
| DeiT-S     | 3842<br>(–)                              | –                                 | 15.52                                   | 81.6            |  |  |  |  |
|            | AdaptiveNN with                          |                                   | online-adjustable computational cost.   |                 |  |  |  |  |
|            |                                          | 0.09                              | 1.20                                    | 75.7<br>± 0.22  |  |  |  |  |
|            |                                          | 0.43                              | 1.61                                    | 78.3<br>± 0.20  |  |  |  |  |
|            |                                          | 0.77                              | 2.01                                    | 79.9<br>± 0.17  |  |  |  |  |
|            |                                          | 1.11                              | 2.42                                    | 80.8<br>± 0.18  |  |  |  |  |
| AdaptiveNN |                                          | 1.46                              | 2.82                                    | 81.4<br>± 0.13  |  |  |  |  |
| DeiT-S     | 2882<br>(1122<br>)                       | 1.80                              | 3.23                                    | 81.7<br>± 0.06  |  |  |  |  |
| (ours)     |                                          | 2.14                              | 3.63                                    | 81.9<br>± 0.05  |  |  |  |  |
|            |                                          | 2.48                              | 4.04                                    | 82.1<br>± 0.09  |  |  |  |  |
|            |                                          | 2.82                              | 4.44                                    | 82.1<br>± 0.11  |  |  |  |  |
|            |                                          | 3.15                              | 4.85                                    | 82.2<br>± 0.11  |  |  |  |  |
|            |                                          | 3.40                              | 5.25                                    | 82.2<br>± 0.12  |  |  |  |  |

Supplementary Data Tab. 2 Quantitative comparisons of AdaptiveNN and traditional non-adaptive models on top of the identical DeiT backbones: Top-1 validation accuracy versus average computational cost for inferring the model. To obtain non-adaptive models with varying costs, we consider two common approaches: adjusting model sizes and input resolutions.

|            | #Visual                | Computational                                         |         | Trial / Accuracy (%) |         |         |         |             |
|------------|------------------------|-------------------------------------------------------|---------|----------------------|---------|---------|---------|-------------|
| Model      | Fixations<br>(average) | Cost<br>(GFLOPs)                                      | st<br>1 | nd<br>2              | rd<br>3 | th<br>4 | th<br>5 | Average     |
|            |                        | AdaptiveNN with online-adjustable computational cost. |         |                      |         |         |         |             |
|            | 0.09                   | 1.20                                                  | 75.3    | 75.8                 | 75.8    | 75.8    | 75.5    | 75.7 ± 0.22 |
|            | 0.43                   | 1.61                                                  | 78.1    | 78.3                 | 78.4    | 78.4    | 78.6    | 78.3 ± 0.20 |
|            | 0.77                   | 2.01                                                  | 79.7    | 79.8                 | 79.9    | 79.9    | 80.2    | 79.9 ± 0.17 |
|            | 1.11                   | 2.42                                                  | 80.7    | 80.6                 | 80.8    | 80.9    | 81.1    | 80.8 ± 0.18 |
|            | 1.46                   | 2.82                                                  | 81.3    | 81.2                 | 81.4    | 81.5    | 81.6    | 81.4 ± 0.13 |
| AdaptiveNN | 1.80                   | 3.23                                                  | 81.7    | 81.6                 | 81.7    | 81.8    | 81.9    | 81.7 ± 0.06 |
| -DeiT-S    | 2.14                   | 3.63                                                  | 82.0    | 81.9                 | 81.9    | 82.0    | 81.9    | 81.9 ± 0.05 |
|            | 2.48                   | 4.04                                                  | 82.2    | 82.1                 | 81.9    | 82.1    | 81.8    | 82.1 ± 0.09 |
|            | 2.82                   | 4.44                                                  | 82.4    | 82.2                 | 82.0    | 82.2    | 82.1    | 82.1 ± 0.11 |
|            | 3.15                   | 4.85                                                  | 82.4    | 82.3                 | 82.1    | 82.3    | 82.2    | 82.2 ± 0.11 |
|            | 3.40                   | 5.25                                                  | 82.4    | 82.4                 | 82.1    | 82.4    | 82.2    | 82.2 ± 0.12 |

Supplementary Data Tab. 3 AdaptiveNN built on DeiT-S: Top-1 validation accuracy versus the average computational cost of model inference, with results averaged over 5 trials.

| Model      | Size of<br>Image<br>(fixation)           | #Visual<br>Fixations<br>(average) | Computational<br>Cost<br>(GFLOPs/image) | Accuracy<br>(%) |  |  |  |  |
|------------|------------------------------------------|-----------------------------------|-----------------------------------------|-----------------|--|--|--|--|
|            | Models with<br>fixed computational cost. |                                   |                                         |                 |  |  |  |  |
| ResNet-18  | 2242<br>(–)                              | –                                 | 1.82                                    | 71.2            |  |  |  |  |
| ResNet-18  | 2882<br>(–)                              | –                                 | 3.01                                    | 72.3            |  |  |  |  |
| ResNet-18  | 3842<br>(–)                              | –                                 | 5.34                                    | 72.7            |  |  |  |  |
| ResNet-50  | 2242<br>(–)                              | –                                 | 4.11                                    | 77.6            |  |  |  |  |
| ResNet-50  | 2882<br>(–)                              | –                                 | 6.80                                    | 78.6            |  |  |  |  |
| ResNet-50  | 3842<br>(–)                              | –                                 | 12.08                                   | 79.1            |  |  |  |  |
|            | AdaptiveNN with                          |                                   | online-adjustable computational cost.   |                 |  |  |  |  |
|            |                                          | 0.08                              | 1.20                                    | 74.0<br>± 0.07  |  |  |  |  |
|            |                                          | 0.44                              | 1.71                                    | 76.3<br>± 0.13  |  |  |  |  |
|            |                                          | 0.81                              | 2.21                                    | 77.6<br>± 0.16  |  |  |  |  |
|            |                                          | 1.17                              | 2.72                                    | 78.4<br>± 0.09  |  |  |  |  |
| AdaptiveNN |                                          | 1.54                              | 3.22                                    | 78.9<br>± 0.07  |  |  |  |  |
| ResNet-50  | 3202<br>(1122<br>)                       | 1.90                              | 3.73                                    | 79.3<br>± 0.08  |  |  |  |  |
| (ours)     |                                          | 2.27                              | 4.23                                    | 79.5<br>± 0.10  |  |  |  |  |
|            |                                          | 2.63                              | 4.74                                    | 79.6<br>± 0.10  |  |  |  |  |
|            |                                          | 2.98                              | 5.24                                    | 79.7<br>± 0.10  |  |  |  |  |
|            |                                          | 3.27                              | 5.75                                    | 79.8<br>± 0.08  |  |  |  |  |
|            |                                          | 3.47                              | 6.25                                    | 79.8<br>± 0.07  |  |  |  |  |

Supplementary Data Tab. 4 Quantitative comparisons of AdaptiveNN and traditional non-adaptive models on top of the identical ResNet backbones.

|            | #Visual                | Computational                                         |         |         |         |         | Trial / Accuracy (%) |             |
|------------|------------------------|-------------------------------------------------------|---------|---------|---------|---------|----------------------|-------------|
| Model      | Fixations<br>(average) | Cost<br>(GFLOPs)                                      | st<br>1 | nd<br>2 | rd<br>3 | th<br>4 | th<br>5              | Average     |
|            |                        | AdaptiveNN with online-adjustable computational cost. |         |         |         |         |                      |             |
|            | 0.08                   | 1.20                                                  | 73.9    | 74.0    | 74.0    | 74.0    | 74.1                 | 74.0 ± 0.07 |
|            | 0.44                   | 1.71                                                  | 76.4    | 76.5    | 76.2    | 76.2    | 76.5                 | 76.3 ± 0.13 |
|            | 0.81                   | 2.21                                                  | 77.6    | 77.7    | 77.4    | 77.5    | 77.7                 | 77.6 ± 0.16 |
|            | 1.17                   | 2.72                                                  | 78.4    | 78.4    | 78.2    | 78.4    | 78.4                 | 78.4 ± 0.09 |
|            | 1.54                   | 3.22                                                  | 78.9    | 78.9    | 78.8    | 78.9    | 79.0                 | 78.9 ± 0.07 |
| AdaptiveNN | 1.90                   | 3.73                                                  | 79.3    | 79.2    | 79.2    | 79.2    | 79.4                 | 79.3 ± 0.08 |
| -ResNet-50 | 2.27                   | 4.23                                                  | 79.5    | 79.5    | 79.4    | 79.5    | 79.7                 | 79.5 ± 0.10 |
|            | 2.63                   | 4.74                                                  | 79.7    | 79.6    | 79.5    | 79.7    | 79.8                 | 79.6 ± 0.10 |
|            | 2.98                   | 5.24                                                  | 79.8    | 79.7    | 79.6    | 79.8    | 79.8                 | 79.7 ± 0.10 |
|            | 3.27                   | 5.75                                                  | 79.9    | 79.7    | 79.7    | 79.8    | 79.8                 | 79.8 ± 0.08 |
|            | 3.47                   | 6.25                                                  | 79.9    | 79.7    | 79.7    | 79.8    | 79.8                 | 79.8 ± 0.07 |

Supplementary Data Tab. 5 AdaptiveNN built on ResNet-50: Top-1 validation accuracy versus the average computational cost of model inference, with results averaged over 5 trials.

|                       | #Visual   |               |                                                    |           | Accuracy (%) |           |             |
|-----------------------|-----------|---------------|----------------------------------------------------|-----------|--------------|-----------|-------------|
| Model                 | Fixations | st trial<br>1 | 2nd trial                                          | 3rd trial | 4th trial    | 5th trial | Average     |
|                       |           |               | AdaptiveNN with fixed numbers of visual fixations. |           |              |           |             |
|                       | No        | 74.0          | 74.4                                               | 74.3      | 74.2         | 73.9      | 74.2 ± 0.22 |
| AdaptiveNN<br>-DeiT-S | 1         | 79.5          | 79.6                                               | 79.6      | 79.5         | 79.6      | 79.6 ± 0.06 |
|                       | 2         | 81.1          | 81.0                                               | 81.0      | 81.0         | 81.1      | 81.0 ± 0.06 |
|                       | 3         | 81.9          | 81.8                                               | 81.8      | 81.8         | 81.8      | 81.8 ± 0.05 |
|                       | 4         | 82.3          | 82.3                                               | 82.0      | 82.2         | 82.0      | 82.2 ± 0.12 |

Supplementary Data Tab. 6 Relationship between validation accuracy and the number of visual fixations for AdaptiveNN-DeiT-S, assuming all samples use the same fixed number of visual fixations.

|                          | #Visual   |               |                                                    |           | Accuracy (%) |           |             |
|--------------------------|-----------|---------------|----------------------------------------------------|-----------|--------------|-----------|-------------|
| Model                    | Fixations | st trial<br>1 | 2nd trial                                          | 3rd trial | 4th trial    | 5th trial | Average     |
|                          |           |               | AdaptiveNN with fixed numbers of visual fixations. |           |              |           |             |
|                          | No        | 72.9          | 72.9                                               | 72.9      | 72.9         | 73.0      | 72.9 ± 0.05 |
|                          | 1         | 77.4          | 77.4                                               | 77.3      | 77.4         | 77.4      | 77.4 ± 0.05 |
| AdaptiveNN<br>-ResNet-50 | 2         | 78.7          | 78.8                                               | 78.7      | 78.8         | 79.0      | 78.8 ± 0.10 |
|                          | 3         | 79.6          | 79.5                                               | 79.5      | 79.4         | 79.6      | 79.5 ± 0.09 |
|                          | 4         | 79.9          | 79.7                                               | 79.7      | 79.7         | 79.8      | 79.8 ± 0.08 |

Supplementary Data Tab. 7 Relationship between validation accuracy and the number of visual fixations for AdaptiveNN-ResNet-50, assuming all samples use the same fixed number of visual fixations.

|            | Computational          | #Visual Fixations / Data Proportion (%) |                                     |            |              |              |  |
|------------|------------------------|-----------------------------------------|-------------------------------------|------------|--------------|--------------|--|
| Model      | Cost<br>(GFLOPs/image) | No                                      | 1                                   | 2          | 3            | 4            |  |
|            | 1.20                   | 91.2 ± 0.37                             | 8.4 ± 0.71                          | 0.4 ± 0.33 | 0.0 ± 0.04   | 0.0 ± 0.01   |  |
|            | 1.41                   |                                         | 75.0 ± 0.68 23.0 ± 1.35             | 1.7 ± 0.71 | 0.2 ± 0.09   | 0.1 ± 0.05   |  |
|            | 1.63                   |                                         | 60.9 ± 0.89 33.9 ± 1.78             | 4.5 ± 0.93 | 0.6 ± 0.32   | 0.1 ± 0.08   |  |
|            | 1.84                   |                                         | 48.7 ± 1.56 41.7 ± 2.46             | 7.9 ± 1.77 | 1.5 ± 1.01   | 0.3 ± 0.25   |  |
|            | 2.05                   |                                         | 38.0 ± 2.91 46.7 ± 5.15 12.1 ± 3.24 |            | 2.7 ± 1.61   | 0.5 ± 0.38   |  |
|            | 2.27                   |                                         | 29.6 ± 2.22 49.3 ± 3.11 14.8 ± 3.11 |            | 5.1 ± 2.21   | 1.2 ± 0.65   |  |
|            | 2.48                   |                                         | 25.8 ± 3.82 46.2 ± 6.01 15.3 ± 1.08 |            | 10.7 ± 2.83  | 2.0 ± 1.06   |  |
|            | 2.69                   |                                         | 19.6 ± 4.41 46.6 ± 5.83 16.4 ± 2.88 |            | 14.2 ± 4.15  | 3.2 ± 2.05   |  |
|            | 2.91                   |                                         | 14.3 ± 1.54 48.1 ± 2.89 14.7 ± 3.20 |            | 16.4 ± 5.28  | 6.5 ± 3.25   |  |
| AdaptiveNN | 3.12                   |                                         | 10.5 ± 1.58 48.3 ± 3.75 15.4 ± 5.38 |            | 11.7 ± 4.29  | 14.1 ± 2.96  |  |
| -DeiT-S    | 3.33                   |                                         | 7.8 ± 3.33 42.8 ± 6.69 19.1 ± 8.05  |            | 13.7 ± 5.13  | 16.6 ± 2.67  |  |
|            | 3.54                   |                                         | 7.7 ± 2.73 35.5 ± 6.29 21.2 ± 8.97  |            | 13.9 ± 3.42  | 21.7 ± 2.98  |  |
|            | 3.78                   |                                         | 6.7 ± 3.53 32.5 ± 7.43 17.6 ± 3.30  |            | 16.3 ± 10.96 | 26.9 ± 6.97  |  |
|            | 3.97                   |                                         | 3.8 ± 1.86 29.5 ± 7.14 19.3 ± 4.61  |            | 15.3 ± 13.32 | 32.1 ± 8.95  |  |
|            | 4.18                   |                                         | 3.9 ± 1.58 25.5 ± 7.35 18.3 ± 8.91  |            | 10.8 ± 10.64 | 41.4 ± 8.68  |  |
|            | 4.40                   |                                         | 2.3 ± 1.31 23.1 ± 7.42 15.1 ± 9.40  |            | 13.0 ± 17.21 | 46.4 ± 10.98 |  |
|            | 4.61                   |                                         | 2.8 ± 0.95 17.7 ± 5.83              | 8.8 ± 5.56 | 22.0 ± 18.59 | 48.7 ± 12.41 |  |
|            | 4.82                   |                                         | 3.4 ± 3.03 10.2 ± 5.02              | 7.7 ± 6.98 | 27.0 ± 21.45 | 51.7 ± 13.35 |  |
|            | 5.04                   | 2.8 ± 3.19                              | 8.4 ± 3.75                          | 5.6 ± 4.24 | 23.5 ± 12.00 | 59.7 ± 9.21  |  |
|            | 5.25                   | 2.8 ± 3.06                              | 5.5 ± 4.26                          | 6.0 ± 5.61 | 20.5 ± 14.28 | 65.3 ± 11.92 |  |

Supplementary Data Tab. 8 For AdaptiveNN-DeiT-S, distribution of data using different numbers of visual fixations across varying computational cost constraints.

|            | Computational          |             |              |              | #Visual Fixations / Data Proportion (%) |              |
|------------|------------------------|-------------|--------------|--------------|-----------------------------------------|--------------|
| Model      | Cost<br>(GFLOPs/image) | No          | 1            | 2            | 3                                       | 4            |
|            | 1.20                   | 92.8 ± 0.16 | 6.9 ± 0.30   | 0.2 ± 0.14   | 0.0 ± 0.02                              | 0.0 ± 0.01   |
|            | 1.47                   | 75.3 ± 0.47 | 23.0 ± 0.78  | 1.4 ± 0.29   | 0.2 ± 0.13                              | 0.1 ± 0.06   |
|            | 1.73                   | 60.0 ± 0.41 | 35.3 ± 0.25  | 3.6 ± 0.86   | 0.9 ± 0.50                              | 0.2 ± 0.16   |
|            | 2.00                   | 48.4 ± 2.44 | 40.9 ± 5.56  | 8.3 ± 3.89   | 1.9 ± 0.91                              | 0.5 ± 0.32   |
|            | 2.26                   | 35.9 ± 2.34 | 48.3 ± 4.51  | 12.1 ± 2.73  | 3.0 ± 0.85                              | 0.7 ± 0.43   |
|            | 2.53                   | 26.8 ± 3.41 | 49.2 ± 7.06  | 18.6 ± 5.16  | 4.4 ± 2.04                              | 1.0 ± 0.69   |
|            | 2.79                   | 18.3 ± 3.70 | 52.1 ± 4.49  | 20.3 ± 4.22  | 7.3 ± 4.72                              | 2.0 ± 1.76   |
|            | 3.06                   | 13.6 ± 5.15 | 47.3 ± 11.04 | 24.9 ± 8.73  | 11.7 ± 5.15                             | 2.4 ± 2.20   |
|            | 3.33                   | 13.8 ± 4.55 | 35.1 ± 10.26 | 31.5 ± 14.13 | 15.2 ± 8.41                             | 4.4 ± 3.34   |
| AdaptiveNN | 3.59                   | 8.1 ± 2.88  | 35.5 ± 8.09  | 29.0 ± 13.73 | 22.9 ± 8.99                             | 4.5 ± 3.04   |
| -ResNet-50 | 3.86                   | 6.0 ± 1.38  | 34.0 ± 5.92  | 22.0 ± 11.37 | 30.7 ± 8.79                             | 7.4 ± 4.74   |
|            | 4.12                   | 2.5 ± 1.42  | 32.8 ± 7.18  | 20.4 ± 8.96  | 32.1 ± 13.33                            | 12.3 ± 8.72  |
|            | 4.39                   | 2.4 ± 2.78  | 26.6 ± 4.78  | 17.3 ± 6.18  | 37.8 ± 21.54                            | 15.9 ± 13.13 |
|            | 4.66                   | 1.0 ± 0.75  | 21.9 ± 6.08  | 15.3 ± 7.11  | 42.6 ± 20.32                            | 19.3 ± 12.32 |
|            | 4.92                   | 1.0 ± 0.64  | 15.2 ± 8.34  | 16.6 ± 10.29 | 40.7 ± 19.47                            | 26.5 ± 12.97 |
|            | 5.19                   | 1.2 ± 1.05  | 12.8 ± 5.24  | 14.3 ± 10.88 | 33.1 ± 14.00                            | 38.5 ± 8.07  |
|            | 5.45                   | 1.2 ± 1.16  | 11.6 ± 5.56  | 10.6 ± 9.85  | 29.0 ± 17.49                            | 47.6 ± 13.43 |
|            | 5.72                   | 1.5 ± 1.52  | 9.0 ± 5.67   | 9.2 ± 7.72   | 23.0 ± 20.60                            | 57.4 ± 17.86 |
|            | 5.98                   | 0.6 ± 0.44  | 4.5 ± 5.36   | 7.5 ± 7.38   | 28.2 ± 19.68                            | 59.2 ± 18.82 |
|            | 6.25                   | 0.4 ± 0.28  | 3.4 ± 5.33   | 8.3 ± 7.16   | 24.7 ± 21.72                            | 63.2 ± 21.47 |

Supplementary Data Tab. 9 For AdaptiveNN-ResNet-50, distribution of data using different numbers of visual fixations across varying computational cost constraints.

| Model           | Size of<br>Computational<br>Image<br>Cost<br>(fixation)<br>(GFLOPs/image) |                                       | Accuracy<br>(%) |
|-----------------|---------------------------------------------------------------------------|---------------------------------------|-----------------|
|                 | Models with                                                               | fixed computational cost.             |                 |
| DeiT-T          | 2242<br>(–)                                                               | 1.26                                  | 81.6            |
| DeiT-T          | 2882<br>(–)                                                               | 2.27                                  | 83.5            |
| DeiT-T          | 3842<br>(–)                                                               | 4.70                                  | 85.7            |
| DeiT-S          | 2242<br>(–)                                                               | 4.61                                  | 84.4            |
| DeiT-S          | 2882<br>(–)                                                               | 7.99                                  | 86.2            |
| DeiT-S          | 3842<br>(–)                                                               | 15.52                                 | 87.3            |
| AdaptiveNN with |                                                                           | online-adjustable computational cost. |                 |
|                 |                                                                           | 1.40                                  | 83.1<br>± 0.22  |
|                 |                                                                           | 1.58                                  | 84.6<br>± 0.24  |
|                 |                                                                           | 1.75                                  | 85.7<br>± 0.19  |
|                 |                                                                           | 1.93                                  | 86.4<br>± 0.33  |
| AdaptiveNN      |                                                                           | 2.10                                  | 86.7<br>± 0.25  |
| DeiT-S          | 2882<br>(1122<br>)                                                        | 2.28                                  | 86.9<br>± 0.39  |
| (ours)          |                                                                           | 2.45                                  | 87.2<br>± 0.24  |
|                 |                                                                           | 2.63                                  | 87.4<br>± 0.15  |
|                 |                                                                           | 2.80                                  | 87.5<br>± 0.13  |
|                 |                                                                           | 2.98                                  | 87.5<br>± 0.09  |
|                 |                                                                           |                                       |                 |

Supplementary Data Tab. 10 Quantitative comparisons of AdaptiveNN and traditional non-adaptive models: Top-1 validation accuracy versus average computational cost for model inference on the CUB-200-2011 dataset.

3.15 87.5 ± 0.09

| Model           | Size of<br>Image<br>(fixation)           | Computational<br>Cost<br>(GFLOPs/image) |                |  |  |  |  |
|-----------------|------------------------------------------|-----------------------------------------|----------------|--|--|--|--|
|                 | Models with<br>fixed computational cost. |                                         |                |  |  |  |  |
| DeiT-T          | 2242<br>(–)                              | 1.26                                    | 78.5           |  |  |  |  |
| DeiT-T          | 2882<br>(–)                              | 2.27                                    | 81.9           |  |  |  |  |
| DeiT-T          | 3842<br>(–)                              | 4.70                                    | 84.4           |  |  |  |  |
| DeiT-S          | 2242<br>(–)                              | 4.61                                    | 80.9           |  |  |  |  |
| DeiT-S          | 2882<br>(–)                              | 7.99                                    | 83.9           |  |  |  |  |
| DeiT-S          | 3842<br>(–)                              | 15.52                                   | 86.2           |  |  |  |  |
| AdaptiveNN with |                                          | online-adjustable computational cost.   |                |  |  |  |  |
|                 |                                          | 1.50                                    | 80.2<br>± 0.13 |  |  |  |  |
|                 |                                          | 1.66                                    | 82.0<br>± 0.25 |  |  |  |  |
|                 |                                          | 1.82                                    | 83.3<br>± 0.28 |  |  |  |  |
|                 |                                          | 1.98                                    | 84.4<br>± 0.09 |  |  |  |  |
| AdaptiveNN      |                                          | 2.14                                    | 85.2<br>± 0.07 |  |  |  |  |
| DeiT-S          | 2882<br>(1122<br>)                       | 2.30                                    | 85.8<br>± 0.10 |  |  |  |  |
| (ours)          |                                          | 2.46                                    | 86.0<br>± 0.20 |  |  |  |  |

Supplementary Data Tab. 11 Quantitative comparisons of AdaptiveNN and traditional non-adaptive models: Top-1 validation accuracy versus average computational cost for model inference on the NABirds dataset.

2.62 86.3 ± 0.14 2.78 86.4 ± 0.11 2.94 86.4 ± 0.11 3.10 86.4 ± 0.10

| Model           | Size of<br>Image<br>(fixation)           | Computational<br>Cost<br>(GFLOPs/image) | Accuracy<br>(%) |  |  |  |
|-----------------|------------------------------------------|-----------------------------------------|-----------------|--|--|--|
|                 | Models with<br>fixed computational cost. |                                         |                 |  |  |  |
| DeiT-T          | 2242<br>(–)                              | 1.26                                    | 91.1            |  |  |  |
| DeiT-T          | 2882<br>(–)                              | 2.27                                    | 92.2            |  |  |  |
| DeiT-T          | 3842<br>(–)                              | 4.70                                    | 92.4            |  |  |  |
| DeiT-S          | 2242<br>(–)                              | 4.61                                    | 93.3            |  |  |  |
| DeiT-S          | 2882<br>(–)                              | 7.99                                    | 93.9            |  |  |  |
| DeiT-S          | 3842<br>(–)                              | 15.52                                   | 94.0            |  |  |  |
| AdaptiveNN with |                                          | online-adjustable computational cost.   |                 |  |  |  |
|                 |                                          | 1.25                                    | 92.4<br>± 0.06  |  |  |  |
|                 |                                          | 1.43                                    | 93.0<br>± 0.02  |  |  |  |
|                 |                                          | 1.60                                    | 93.4<br>± 0.08  |  |  |  |
|                 |                                          | 1.78                                    | 93.7<br>± 0.05  |  |  |  |
| AdaptiveNN      |                                          | 1.95                                    | 93.9<br>± 0.07  |  |  |  |
| DeiT-S          | 2882<br>(1122<br>)                       | 2.13                                    | 94.1<br>± 0.05  |  |  |  |
| (ours)          |                                          | 2.30                                    | 94.2<br>± 0.10  |  |  |  |
|                 |                                          | 2.48                                    | 94.3<br>± 0.03  |  |  |  |
|                 |                                          |                                         |                 |  |  |  |

Supplementary Data Tab. 12 Quantitative comparisons of AdaptiveNN and traditional non-adaptive models: Top-1 validation accuracy versus average computational cost for model inference on the Oxford-IIIT Pet dataset.

2.65 94.4 ± 0.06 2.83 94.4 ± 0.08 3.00 94.4 ± 0.10

| Size of<br>Image<br>(fixation) | Computational<br>Cost<br>(GFLOPs/image) | Accuracy<br>(%)           |
|--------------------------------|-----------------------------------------|---------------------------|
| Models with                    |                                         |                           |
| 2242<br>(–)                    | 1.26                                    | 81.5                      |
| 2882<br>(–)                    | 2.27                                    | 83.0                      |
| 3842<br>(–)                    | 4.70                                    | 84.7                      |
| 2242<br>(–)                    | 4.61                                    | 87.2                      |
| 2882<br>(–)                    | 7.99                                    | 87.8                      |
| 3842<br>(–)                    | 15.52                                   | 88.8                      |
|                                |                                         | fixed computational cost. |

|            |                    | 1.25 | 84.4<br>± 0.19 |
|------------|--------------------|------|----------------|
|            |                    | 1.48 | 86.3<br>± 0.22 |
|            |                    | 1.70 | 87.8<br>± 0.19 |
|            |                    | 1.93 | 88.9<br>± 0.20 |
| AdaptiveNN | 2882<br>(1122<br>) | 2.15 | 89.7<br>± 0.17 |
| DeiT-S     |                    | 2.38 | 90.3<br>± 0.10 |
| (ours)     |                    | 2.60 | 90.7<br>± 0.11 |
|            |                    | 2.83 | 90.8<br>± 0.10 |
|            |                    | 3.05 | 90.9<br>± 0.10 |
|            |                    | 3.28 | 91.0<br>± 0.10 |
|            |                    | 3.50 | 91.1<br>± 0.12 |

Supplementary Data Tab. 13 Quantitative comparisons of AdaptiveNN and traditional non-adaptive models: Top-1 validation accuracy versus average computational cost for model inference on the Stanford Dogs dataset.

| Model           | Size of<br>Image<br>(fixation) | Computational<br>Cost<br>(GFLOPs/image) |                |
|-----------------|--------------------------------|-----------------------------------------|----------------|
|                 | Models with                    | fixed computational cost.               |                |
| DeiT-T          | 2242<br>(–)                    | 1.26                                    | 85.2           |
| DeiT-T          | 2882<br>(–)                    | 2.27                                    | 87.1           |
| DeiT-T          | 3842<br>(–)                    | 4.70                                    | 87.9           |
| DeiT-S          | 2242<br>(–)                    | 4.61                                    | 90.5           |
| DeiT-S          | 2882<br>(–)                    | 7.99                                    | 91.3           |
| DeiT-S          | 3842<br>(–)                    | 15.52                                   | 92.0           |
| AdaptiveNN with |                                | online-adjustable computational cost.   |                |
|                 |                                | 1.30                                    | 87.3<br>± 0.13 |
|                 |                                | 1.52                                    | 88.8<br>± 0.29 |
|                 |                                | 1.74                                    | 89.8<br>± 0.31 |
|                 |                                | 1.96                                    | 90.5<br>± 0.30 |
| AdaptiveNN      |                                | 2.18                                    | 91.1<br>± 0.25 |
| DeiT-S          | 2882<br>(1122<br>)             | 2.40                                    | 91.6<br>± 0.17 |
| (ours)          |                                | 2.62                                    | 91.9<br>± 0.13 |
|                 |                                | 2.84                                    | 92.1<br>± 0.10 |
|                 |                                | 3.06                                    | 92.3<br>± 0.08 |
|                 |                                | 3.28                                    | 92.4<br>± 0.10 |
|                 |                                |                                         |                |

Supplementary Data Tab. 14 Quantitative comparisons of AdaptiveNN and traditional non-adaptive models: Top-1 validation accuracy versus average computational cost for model inference on the Stanford Cars dataset.

3.50 92.4 ± 0.10

| Model  | Size of<br>Image<br>(fixation)                           | Computational<br>Cost<br>(GFLOPs/image) | Accuracy<br>(%) |  |  |  |
|--------|----------------------------------------------------------|-----------------------------------------|-----------------|--|--|--|
|        | Models with                                              | fixed computational cost.               |                 |  |  |  |
| DeiT-T | 2242<br>(–)                                              | 1.26                                    | 83.2            |  |  |  |
| DeiT-T | 2882<br>(–)                                              | 2.27                                    | 84.4            |  |  |  |
| DeiT-T | 3842<br>(–)                                              | 4.70                                    | 86.3            |  |  |  |
| DeiT-S | 2242<br>(–)                                              | 4.61                                    | 84.9            |  |  |  |
| DeiT-S | 2882<br>(–)                                              | 7.99                                    | 85.7            |  |  |  |
| DeiT-S | 3842<br>(–)                                              | 15.52                                   | 88.4            |  |  |  |
|        | AdaptiveNN with<br>online-adjustable computational cost. |                                         |                 |  |  |  |
|        |                                                          | 1.40                                    | 84.7<br>± 0.22  |  |  |  |

|            |                    | 1.40 | 84.7<br>± 0.22 |
|------------|--------------------|------|----------------|
|            |                    | 1.60 | 85.8<br>± 0.55 |
|            |                    | 1.80 | 86.7<br>± 0.57 |
|            |                    | 2.00 | 87.5<br>± 0.46 |
| AdaptiveNN |                    | 2.20 | 88.0<br>± 0.15 |
| DeiT-S     | 2882<br>(1122<br>) | 2.40 | 88.3<br>± 0.10 |
| (ours)     |                    | 2.60 | 88.5<br>± 0.08 |
|            |                    | 2.80 | 88.6<br>± 0.03 |
|            |                    | 3.00 | 88.8<br>± 0.06 |
|            |                    | 3.20 | 88.8<br>± 0.12 |
|            |                    | 3.40 | 88.8<br>± 0.12 |

Supplementary Data Tab. 15 Quantitative comparisons of AdaptiveNN and traditional non-adaptive models: Top-1 validation accuracy versus average computational cost for model inference on the FGVC-Aircraft dataset.

| Model      | Size of<br>Computational<br>Image<br>Cost<br>(fixation)<br>(GFLOPs/image) |                                       | Accuracy<br>(%) |
|------------|---------------------------------------------------------------------------|---------------------------------------|-----------------|
|            | Models with                                                               | fixed computational cost.             |                 |
| ResNet-18  | 3842<br>(–)                                                               | 5.44                                  | 80.1            |
| ResNet-18  | 7202<br>(–)                                                               | 18.98                                 | 86.8            |
| ResNet-18  | 9602<br>(–)                                                               | 33.50                                 | 89.4            |
| ResNet-50  | 3842<br>(–)                                                               | 12.38                                 | 82.6            |
| ResNet-50  | 7202<br>(–)                                                               | 42.79                                 | 89.0            |
| ResNet-50  | 9602<br>(–)                                                               | 75.88                                 | 90.2            |
|            | AdaptiveNN with                                                           | online-adjustable computational cost. |                 |
|            |                                                                           | 2.20                                  | 78.5<br>± 1.39  |
|            |                                                                           | 2.29                                  | 80.0<br>± 1.41  |
|            |                                                                           | 2.38                                  | 82.1<br>± 1.11  |
|            |                                                                           | 2.47                                  | 83.9<br>± 1.10  |
| AdaptiveNN |                                                                           | 2.56                                  | 86.0<br>± 1.22  |
| ResNet-18  | 960 × 1280<br>(1922<br>)                                                  | 2.65                                  | 88.4<br>± 1.27  |
| (ours)     |                                                                           | 2.74                                  | 90.4<br>± 1.09  |
|            |                                                                           | 2.83                                  | 91.1<br>± 0.49  |
|            |                                                                           | 2.92                                  | 91.3<br>± 0.36  |
|            |                                                                           | 3.01                                  | 91.4<br>± 0.30  |
|            |                                                                           |                                       |                 |

Supplementary Data Tab. 16 Comparisons of AdaptiveNN and traditional non-adaptive models in processing complicated, non-object-centric real-world scenes: Top-1 validation accuracy versus average computational cost for inferring the model. We consider the traffic sign recognition task on the Swedish traffic signs dataset (STSD), composed of 960x1,280 road-scene images collected on real moving vehicles.

3.10 91.5 <sup>±</sup> 0.28

| Model      | Computational                                         | Trial / Accuracy (%) |         |         |         |         |             |
|------------|-------------------------------------------------------|----------------------|---------|---------|---------|---------|-------------|
|            | Cost<br>(GFLOPs/image)                                | st<br>1              | nd<br>2 | rd<br>3 | th<br>4 | th<br>5 | Average     |
|            | AdaptiveNN with online-adjustable computational cost. |                      |         |         |         |         |             |
|            | 2.20                                                  | 77.5                 | 80.1    | 79.8    | 77.1    | 77.9    | 78.5 ± 1.39 |
|            | 2.29                                                  | 78.7                 | 82.1    | 80.8    | 79.0    | 79.4    | 80.0 ± 1.41 |
|            | 2.38                                                  | 82.4                 | 83.4    | 82.7    | 81.0    | 80.9    | 82.1 ± 1.11 |
|            | 2.47                                                  | 84.1                 | 84.7    | 85.1    | 83.1    | 82.5    | 83.9 ± 1.10 |
|            | 2.56                                                  | 86.8                 | 86.2    | 87.4    | 85.5    | 84.3    | 86.0 ± 1.22 |
| AdaptiveNN | 2.65                                                  | 89.6                 | 88.8    | 88.9    | 88.1    | 86.3    | 88.4 ± 1.27 |
| -ResNet-18 | 2.74                                                  | 91.8                 | 90.6    | 89.7    | 91.0    | 89.0    | 90.4 ± 1.09 |
|            | 2.83                                                  | 91.9                 | 90.8    | 91.1    | 91.2    | 90.6    | 91.1 ± 0.49 |
|            | 2.92                                                  | 91.9                 | 91.1    | 91.3    | 91.3    | 90.9    | 91.3 ± 0.36 |
|            | 3.01                                                  | 91.9                 | 91.4    | 91.3    | 91.4    | 91.1    | 91.4 ± 0.30 |
|            | 3.10                                                  | 91.9                 | 91.4    | 91.4    | 91.4    | 91.2    | 91.5 ± 0.28 |

Supplementary Data Tab. 17 Comparisons of AdaptiveNN and non-adaptive models on complicated, non-object-centric visual data from real driving scenarios: Top-1 accuracy versus average computational cost for traffic sign recognition on the STSD dataset, using ResNet backbones for high-resolution inputs, with results averaged over 5 trials.

| Visual      |            | Success Rates (%) |                                                  |      |      |      |             |
|-------------|------------|-------------------|--------------------------------------------------|------|------|------|-------------|
| Search Task | Method     | 1                 | st trial 2nd trial 3rd trial 4th trial 5th trial |      |      |      | Average     |
|             | RAM        | 17.5              | 15.0                                             | 16.8 | 16.8 | 15.8 | 16.4 ± 0.88 |
| 1 digit     | DRAM       | 20.4              | 17.8                                             | 23.2 | 20.9 | 19.1 | 20.3 ± 1.81 |
|             | AdaptiveNN | 96.4              | 98.0                                             | 94.0 | 93.2 | 94.2 | 95.2 ± 1.77 |
|             | DRAM       | 22.3              | 17.7                                             | 20.5 | 18.6 | 19.0 | 19.6 ± 1.62 |
| 2 digits    | AdaptiveNN | 93.0              | 97.8                                             | 91.2 | 90.1 | 94.3 | 93.3 ± 2.68 |
|             | DRAM       | 18.9              | 17.3                                             | 19.8 | 18.6 | 20.7 | 19.1 ± 1.15 |
| 3 digits    | AdaptiveNN | 89.2              | 89.7                                             | 91.7 | 92.1 | 90.8 | 90.7 ± 1.12 |
|             | DRAM       | 17.2              | 17.9                                             | 19.5 | 21.1 | 18.8 | 18.9 ± 1.35 |
| 5 digits    | AdaptiveNN | 87.4              | 87.1                                             | 88.4 | 87.9 | 84.4 | 87.0 ± 1.39 |

Supplementary Data Tab. 18 Success rates of visual search tasks.

| Model                  | Size of<br>Image<br>(fixation) | #Visual<br>Fixations<br>(average) | Computational<br>Cost<br>(GFLOPs/action) | Avg. Successful<br>Length |
|------------------------|--------------------------------|-----------------------------------|------------------------------------------|---------------------------|
|                        | Models with                    |                                   | fixed computational cost.                |                           |
| RoboFlamingo 6 layers  | 2242<br>(–)                    | –                                 | 123.2                                    | 2.49                      |
| RoboFlamingo 12 layers | 2242<br>(–)                    | –                                 | 131.0                                    | 2.65                      |
| RoboFlamingo 24 layers | 2242<br>(–)                    | –                                 | 146.6                                    | 2.71                      |
|                        | AdaptiveNN with                |                                   | online-adjustable computational cost.    |                           |
|                        |                                | 0.13                              | 22.8                                     | 2.56<br>± 0.12            |
|                        |                                | 0.30                              | 24.9                                     | 2.71<br>± 0.12            |
|                        |                                | 0.36                              | 25.7                                     | 2.79<br>± 0.12            |
|                        |                                | 0.54                              | 27.9                                     | 2.82<br>± 0.09            |
| AdaptiveNN<br>(ours)   | 2242<br>(702<br>)              | 0.86                              | 31.8                                     | 2.88<br>± 0.12            |
|                        |                                | 1.01                              | 33.6                                     | 2.90<br>± 0.09            |
|                        |                                | 1.16                              | 35.4                                     | 2.93<br>± 0.12            |
|                        |                                | 1.60                              | 38.1                                     | 2.93<br>± 0.09            |
|                        |                                | 2.00                              | 40.6                                     | 2.94<br>± 0.10            |

Supplementary Data Tab. 19 Quantitative comparisons of AdaptiveNN-based MLLM and non-adaptive MLLM on CALVIN: average successful length for D→D versus average computational cost for inferring the model. For the non-adaptive models, computational costs are modulated by adjusting model sizes.

| Model                  | Size of<br>Image<br>(fixation) | #Visual<br>Fixations<br>(average) | Computational<br>Cost<br>(GFLOPs/action) | Avg. Successful<br>Length |
|------------------------|--------------------------------|-----------------------------------|------------------------------------------|---------------------------|
|                        | Models with                    |                                   | fixed computational cost.                |                           |
| RoboFlamingo 6 layers  | 2242<br>(–)                    | –                                 | 123.2                                    | 2.43                      |
| RoboFlamingo 12 layers | 2242<br>(–)                    | –                                 | 131.0                                    | 3.51                      |
| RoboFlamingo 24 layers | 2242<br>(–)                    | –                                 | 146.6                                    | 4.07                      |
|                        | AdaptiveNN with                |                                   | online-adjustable computational cost.    |                           |
|                        |                                | 0.13                              | 22.8                                     | 3.66<br>± 0.10            |
|                        |                                | 0.20                              | 23.7                                     | 3.71<br>± 0.11            |
|                        |                                | 0.28                              | 24.7                                     | 3.78<br>± 0.10            |
|                        |                                | 0.38                              | 25.9                                     | 3.84<br>± 0.12            |
| AdaptiveNN<br>(ours)   | 2242<br>(702<br>)              | 0.43                              | 26.5                                     | 3.90<br>± 0.11            |
|                        |                                | 0.61                              | 28.7                                     | 4.02<br>± 0.09            |
|                        |                                | 0.78                              | 30.8                                     | 4.03<br>± 0.06            |
|                        |                                | 0.95                              | 32.9                                     | 4.05<br>± 0.06            |
|                        |                                | 1.20                              | 35.9                                     | 4.07<br>± 0.10            |

Supplementary Data Tab. 20 Quantitative comparisons of AdaptiveNN-based MLLM and non-adaptive MLLM on CALVIN: average successful length for ABCD→D versus average computational cost for inferring the model. For the non-adaptive models, computational costs are modulated by adjusting model sizes.

|                         |      |      |            |      | Success Rate (%) | Computational Cost (GFLOPs/action) / |
|-------------------------|------|------|------------|------|------------------|--------------------------------------|
| Task                    |      |      | AdaptiveNN |      |                  | Non-adaptive                         |
|                         | 22.9 | 24.9 | 25.7       | 31.8 | 35.4             | 146.6                                |
| close drawer            | 87.7 | 89.3 | 89.5       | 90.1 | 87.3             | 91.1                                 |
| lift blue block drawer  | 84.6 | 70.6 | 63.6       | 84.6 | 92.9             | 100.0                                |
| lift blue block slider  | 69.0 | 75.0 | 70.3       | 67.3 | 67.0             | 62.9                                 |
| lift blue block table   | 87.4 | 88.1 | 89.7       | 87.0 | 89.9             | 81.6                                 |
| lift pink block drawer  | 77.8 | 72.7 | 66.7       | 54.5 | 83.3             | 80.0                                 |
| lift pink block slider  | 72.0 | 80.4 | 77.3       | 74.5 | 80.0             | 75.5                                 |
| lift pink block table   | 78.4 | 76.6 | 80.5       | 75.0 | 82.3             | 76.6                                 |
| lift red block drawer   | 75.0 | 87.5 | 69.2       | 86.7 | 66.7             | 77.8                                 |
| lift red block slider   | 76.9 | 78.8 | 78.6       | 80.0 | 83.2             | 72.9                                 |
| lift red block table    | 76.9 | 79.8 | 83.2       | 83.4 | 85.1             | 84.1                                 |
| move slider left        | 93.2 | 93.0 | 93.1       | 92.5 | 93.2             | 94.5                                 |
| move slider right       | 91.0 | 93.2 | 92.8       | 88.9 | 94.3             | 89.4                                 |
| open drawer             | 87.8 | 84.1 | 84.3       | 83.9 | 87.1             | 85.6                                 |
| place in drawer         | 94.9 | 97.5 | 96.3       | 94.0 | 96.4             | 95.0                                 |
| place in slider         | 48.0 | 51.2 | 58.6       | 60.4 | 59.8             | 49.4                                 |
| push blue block left    | 71.7 | 70.0 | 73.0       | 77.0 | 75.4             | 62.9                                 |
| push blue block right   | 38.7 | 47.6 | 46.2       | 63.1 | 52.3             | 46.2                                 |
| push into drawer        | 64.8 | 58.4 | 55.1       | 64.5 | 69.5             | 59.8                                 |
| push pink block left    | 85.7 | 85.3 | 82.1       | 86.2 | 86.2             | 83.9                                 |
| push pink block right   | 66.7 | 64.9 | 65.6       | 65.5 | 86.2             | 70.7                                 |
| push red block left     | 59.4 | 64.2 | 72.1       | 73.2 | 64.8             | 74.2                                 |
| push red block right    | 70.1 | 53.7 | 60.3       | 51.4 | 54.0             | 52.9                                 |
| rotate blue block left  | 72.4 | 84.1 | 81.2       | 81.0 | 81.0             | 72.1                                 |
| rotate blue block right | 79.1 | 87.7 | 77.9       | 85.5 | 81.2             | 82.4                                 |
| rotate pink block left  | 70.0 | 83.0 | 81.6       | 91.7 | 81.2             | 77.1                                 |
| rotate pink block right | 65.2 | 70.3 | 78.1       | 76.1 | 71.6             | 80.0                                 |
| rotate red block left   | 68.6 | 83.1 | 82.1       | 85.2 | 79.7             | 66.0                                 |
| rotate red block right  | 80.9 | 83.8 | 84.1       | 80.0 | 84.5             | 78.8                                 |
| stack block             | 31.5 | 35.8 | 31.0       | 34.6 | 36.2             | 27.0                                 |
| turn off led            | 90.7 | 89.3 | 93.8       | 91.5 | 92.7             | 86.8                                 |
| turn off lightbulb      | 90.6 | 92.6 | 87.3       | 90.3 | 89.0             | 90.9                                 |
| turn on led             | 94.6 | 93.8 | 93.9       | 97.0 | 94.2             | 92.7                                 |
| turn on lightbulb       | 88.0 | 92.2 | 87.7       | 87.8 | 88.9             | 90.6                                 |
| unstack block           | 87.0 | 82.1 | 84.2       | 88.9 | 91.7             | 93.8                                 |
| Average                 | 75.8 | 77.6 | 76.8       | 78.6 | 79.7             | 76.6                                 |

Supplementary Data Tab. 21 Quantitative comparisons of AdaptiveNN-based MLLM and non-adaptive MLLM: individual success rate of each D→D task versus average computational cost for inferring the model.

|                         |       |       |            | Success Rate (%) |       | Computational Cost (GFLOPs/action) / |
|-------------------------|-------|-------|------------|------------------|-------|--------------------------------------|
| Task                    |       |       | AdaptiveNN |                  |       | Non-adaptive                         |
|                         | 24.7  | 25.9  | 26.5       | 28.7             | 35.9  | 146.6                                |
| close drawer            | 100.0 | 100.0 | 100.0      | 99.5             | 98.6  | 98.0                                 |
| lift blue block drawer  | 85.0  | 93.8  | 100.0      | 94.1             | 94.7  | 89.5                                 |
| lift blue block slider  | 88.6  | 90.6  | 88.5       | 91.4             | 89.0  | 90.7                                 |
| lift blue block table   | 88.3  | 85.7  | 90.6       | 84.2             | 83.9  | 92.3                                 |
| lift pink block drawer  | 92.3  | 93.3  | 92.9       | 85.7             | 93.3  | 86.7                                 |
| lift pink block slider  | 91.5  | 89.4  | 91.5       | 89.6             | 90.3  | 91.2                                 |
| lift pink block table   | 88.0  | 86.3  | 84.2       | 88.8             | 89.4  | 85.5                                 |
| lift red block drawer   | 94.1  | 95.2  | 85.7       | 94.1             | 95.0  | 100.0                                |
| lift red block slider   | 94.6  | 93.9  | 91.8       | 94.0             | 91.0  | 92.6                                 |
| lift red block table    | 96.5  | 95.4  | 96.0       | 96.7             | 96.7  | 93.3                                 |
| move slider left        | 99.1  | 98.8  | 98.8       | 100.0            | 99.6  | 99.6                                 |
| move slider right       | 99.6  | 99.6  | 100.0      | 100.0            | 99.7  | 100.0                                |
| open drawer             | 99.7  | 99.7  | 98.8       | 98.8             | 100.0 | 98.8                                 |
| place in drawer         | 98.8  | 96.8  | 98.1       | 97.6             | 95.4  | 98.2                                 |
| place in slider         | 69.2  | 79.2  | 84.3       | 88.7             | 88.6  | 89.1                                 |
| push blue block left    | 69.2  | 84.4  | 76.9       | 84.8             | 89.7  | 83.1                                 |
| push blue block right   | 71.2  | 83.1  | 80.3       | 83.3             | 81.4  | 87.0                                 |
| push into drawer        | 69.2  | 69.5  | 67.2       | 73.4             | 77.9  | 70.6                                 |
| push pink block left    | 93.2  | 91.9  | 87.8       | 89.3             | 94.7  | 89.5                                 |
| push pink block right   | 86.7  | 87.7  | 90.2       | 85.9             | 87.9  | 83.6                                 |
| push red block left     | 75.0  | 76.9  | 88.3       | 85.9             | 89.6  | 84.4                                 |
| push red block right    | 84.3  | 81.4  | 77.5       | 82.9             | 68.6  | 84.5                                 |
| rotate blue block left  | 80.0  | 80.6  | 83.8       | 86.6             | 88.1  | 91.2                                 |
| rotate blue block right | 79.5  | 78.7  | 75.7       | 77.3             | 87.8  | 86.1                                 |
| rotate pink block left  | 89.1  | 92.5  | 94.6       | 87.7             | 96.4  | 92.6                                 |
| rotate pink block right | 91.0  | 89.7  | 88.9       | 84.5             | 90.3  | 85.5                                 |
| rotate red block left   | 79.7  | 79.7  | 83.9       | 85.5             | 80.6  | 87.1                                 |
| rotate red block right  | 89.9  | 88.9  | 94.3       | 91.8             | 91.7  | 87.5                                 |
| stack block             | 49.7  | 51.1  | 54.5       | 55.5             | 56.9  | 54.5                                 |
| turn off led            | 99.4  | 98.8  | 99.4       | 100.0            | 100.0 | 100.0                                |
| turn off lightbulb      | 98.4  | 98.5  | 100.0      | 98.6             | 98.6  | 99.3                                 |
| turn on led             | 98.7  | 97.5  | 98.3       | 99.4             | 99.4  | 99.4                                 |
| turn on lightbulb       | 99.4  | 98.9  | 100.0      | 99.4             | 100.0 | 98.2                                 |
| unstack block           | 89.2  | 92.1  | 87.8       | 93.5             | 84.1  | 95.1                                 |
| Average                 | 87.6  | 88.8  | 89.1       | 89.7             | 90.3  | 90.1                                 |

Supplementary Data Tab. 22 Quantitative comparisons of AdaptiveNN-based MLLM and non-adaptive MLLM: individual success rate of each ABCD→D task versus average computational cost for inferring the model.

|            | #Visual   |               | Success Rate (%) |           |           |           |             |  |  |
|------------|-----------|---------------|------------------|-----------|-----------|-----------|-------------|--|--|
| Model      | Fixations | st trial<br>1 | 2nd trial        | 3rd trial | 4th trial | 5th trial | Average     |  |  |
|            |           |               |                  | Subtask 1 |           |           |             |  |  |
|            | No        | 93.3          | 91.9             | 93.3      | 91.1      | 92.0      | 92.3 ± 0.87 |  |  |
|            | 1         | 95.1          | 93.3             | 88.8      | 91.5      | 94.6      | 92.7 ± 2.29 |  |  |
|            | 2         | 93.8          | 93.3             | 93.8      | 93.3      | 96.4      | 94.1 ± 1.18 |  |  |
|            |           |               |                  | Subtask 2 |           |           |             |  |  |
|            | No        | 67.0          | 75.9             | 77.0      | 66.5      | 66.1      | 67.9 ± 1.86 |  |  |
|            | 1         | 75.9          | 75.4             | 69.2      | 72.3      | 73.7      | 73.3 ± 2.42 |  |  |
|            | 2         | 71.9          | 77.2             | 70.5      | 73.7      | 79.5      | 74.6 ± 3.33 |  |  |
|            |           |               |                  | Subtask 3 |           |           |             |  |  |
|            | No        | 40.2          | 49.6             | 46.4      | 46.9      | 47.8      | 46.2 ± 3.18 |  |  |
| AdaptiveNN | 1         | 54.9          | 53.6             | 50.4      | 54.5      | 54.0      | 53.5 ± 1.58 |  |  |
|            | 2         | 55.8          | 60.7             | 51.8      | 52.2      | 59.4      | 56.0 ± 3.62 |  |  |
|            |           |               |                  | Subtask 4 |           |           |             |  |  |
|            | No        | 29.0          | 34.8             | 31.7      | 34.4      | 33.5      | 32.7 ± 2.12 |  |  |
|            | 1         | 43.3          | 41.5             | 39.3      | 39.3      | 41.5      | 41.0 ± 1.53 |  |  |
|            | 2         | 42.4          | 46.4             | 40.6      | 38.8      | 47.3      | 43.1 ± 3.28 |  |  |
|            |           |               |                  | Subtask 5 |           |           |             |  |  |
|            | 0         | 23.2          | 25.4             | 23.7      | 22.3      | 25.9      | 24.1 ± 1.35 |  |  |
|            | 1         | 32.6          | 29.5             | 29.9      | 31.7      | 33.0      | 31.3 ± 1.42 |  |  |
|            | 2         | 33.0          | 37.1             | 28.6      | 33.0      | 33.0      | 32.9 ± 2.68 |  |  |

Supplementary Data Tab. 23 Relationship between the success rate of each subtask within the 5-task sequence and the number of visual fixations for AdaptiveNN-based MLLM (D→D), assuming all samples use the same fixed number of visual fixations.

|            | #Visual   |               |           |           | Success Rate (%) |           |             |
|------------|-----------|---------------|-----------|-----------|------------------|-----------|-------------|
| Model      | Fixations | st trial<br>1 | 2nd trial | 3rd trial | 4th trial        | 5th trial | Average     |
|            |           |               |           | Subtask 1 |                  |           |             |
|            | No        | 96.9          | 96.4      | 95.1      | 96.0             | 95.5      | 96.0 ± 0.63 |
|            | 1         | 97.3          | 95.1      | 98.2      | 95.5             | 97.3      | 96.7 ± 1.18 |
|            | 2         | 98.2          | 96.9      | 97.8      | 96.4             | 97.3      | 97.3 ± 0.63 |
|            |           |               |           | Subtask 2 |                  |           |             |
|            | No        | 88.4          | 83.5      | 85.3      | 84.4             | 82.6      | 84.8 ± 2.00 |
|            | 1         | 88.0          | 84.4      | 87.5      | 86.2             | 86.2      | 86.4 ± 1.25 |
|            | 2         | 92.4          | 87.5      | 84.4      | 90.2             | 89.7      | 88.8 ± 2.72 |
|            |           |               |           | Subtask 3 |                  |           |             |
|            | No        | 74.6          | 69.2      | 73.2      | 73.2             | 69.6      | 72.0 ± 2.14 |
| AdaptiveNN | 1         | 79.9          | 76.3      | 75.4      | 77.2             | 75.4      | 76.9 ± 1.66 |
|            | 2         | 85.3          | 82.1      | 77.2      | 79.5             | 79.9      | 80.8 ± 2.72 |
|            |           |               |           | Subtask 4 |                  |           |             |
|            | No        | 62.9          | 60.7      | 57.1      | 62.5             | 59.4      | 60.5 ± 2.12 |
|            | 1         | 72.8          | 68.3      | 65.6      | 70.1             | 62.5      | 67.9 ± 3.55 |
|            | 2         | 76.3          | 71.9      | 67.0      | 69.2             | 68.8      | 70.6 ± 3.26 |
|            |           |               |           | Subtask 5 |                  |           |             |
|            | 0         | 50.4          | 48.7      | 44.2      | 49.6             | 49.1      | 48.4 ± 2.18 |
|            | 1         | 63.4          | 58.5      | 59.8      | 59.4             | 51.3      | 58.5 ± 3.94 |
|            | 2         | 71.0          | 63.8      | 58.9      | 63.8             | 62.9      | 64.1 ± 3.89 |

Supplementary Data Tab. 24 Relationship between the success rate of each subtask within the 5-task sequence and the number of visual fixations for AdaptiveNN-based MLLM (ABCD→D), assuming all samples use the same fixed number of visual fixations.

| Method                | Zero-shot Normalized Human-like Score<br>(ImageNet-1K → SALICON-split-1) |        |  |  |  |
|-----------------------|--------------------------------------------------------------------------|--------|--|--|--|
|                       | Mean ± Std                                                               | Median |  |  |  |
| Gaussian Distribution | 0.15 ± 0.13                                                              | 0.15   |  |  |  |
| Center-Corner         | 0.08 ± 0.12                                                              | 0.08   |  |  |  |
| GradCAM               | −0.09 ± 0.18                                                             | −0.07  |  |  |  |
| LayerCAM              | 0.02 ± 0.18                                                              | 0.03   |  |  |  |
| GradCAM+GMM           | 0.44 ± 0.17                                                              | 0.46   |  |  |  |
| AdaptiveNN-DeiT-S     | 1.09 ± 0.16                                                              | 1.11   |  |  |  |

Supplementary Data Tab. 25 Normalized human-like scores which quantify the probability that human ground truth gaze centers (estimated from ∼60 observers' visual perception behaviors) fall within the visual fixation regions identified by AdaptiveNN or other methods. Evaluated on the SALICON dataset using a 'zero-shot' setting, AdaptiveNN is trained on ImageNet and tested on the unseen SALICON-split-1 data.

| Method                | Zero-shot Normalized Human-like Score<br>(ImageNet-1K → SALICON-split-2) |        |  |  |
|-----------------------|--------------------------------------------------------------------------|--------|--|--|
|                       | Mean ± Std                                                               | Median |  |  |
| Gaussian Distribution | 0.17 ± 0.15                                                              | 0.16   |  |  |
| Center-Corner         | 0.13 ± 0.13                                                              | 0.11   |  |  |
| GradCAM               | −0.09 ± 0.21                                                             | −0.10  |  |  |
| LayerCAM              | −0.02 ± 0.17                                                             | −0.01  |  |  |
| GradCAM+GMM           | 0.48 ± 0.19                                                              | 0.47   |  |  |
| AdaptiveNN-DeiT-S     | 1.11 ± 0.18                                                              | 1.11   |  |  |

Supplementary Data Tab. 26 Normalized human-like scores which quantify the probability that human ground truth gaze centers (estimated from ∼60 observers' visual perception behaviors) fall within the visual fixation regions identified by AdaptiveNN or other methods. Evaluated on the SALICON dataset using a 'zero-shot' setting, AdaptiveNN is trained on ImageNet and tested on the unseen SALICON-split-2 data.

| Human Judge |                   | Pearson correlation coefficient between<br>human-assessed difficulty level and AdaptiveNN predictions |           |       |          |          |  |  |  |  |
|-------------|-------------------|-------------------------------------------------------------------------------------------------------|-----------|-------|----------|----------|--|--|--|--|
|             | Monarch butterfly | Balloon                                                                                               | Speedboat | Ibex  | Snowplow | Cockatoo |  |  |  |  |
| 1           | 0.744             | 0.644                                                                                                 | 0.646     | 0.799 | 0.632    | 0.673    |  |  |  |  |
| 2           | 0.724             | 0.542                                                                                                 | 0.420     | 0.435 | 0.452    | 0.567    |  |  |  |  |
| 3           | 0.650             | 0.614                                                                                                 | 0.530     | 0.558 | 0.525    | 0.515    |  |  |  |  |
| 4           | 0.803             | 0.763                                                                                                 | 0.518     | 0.677 | 0.380    | 0.577    |  |  |  |  |
| 5           | 0.847             | 0.651                                                                                                 | 0.618     | 0.733 | 0.423    | 0.849    |  |  |  |  |
| 6           | 0.674             | 0.600                                                                                                 | 0.329     | 0.540 | 0.362    | 0.715    |  |  |  |  |
| 7           | 0.735             | 0.524                                                                                                 | 0.588     | 0.705 | 0.233    | 0.678    |  |  |  |  |
| 8           | 0.731             | 0.552                                                                                                 | 0.676     | 0.558 | 0.571    | 0.547    |  |  |  |  |
| 9           | 0.799             | 0.655                                                                                                 | 0.427     | 0.576 | 0.247    | 0.724    |  |  |  |  |
| 10          | 0.720             | 0.593                                                                                                 | 0.408     | 0.601 | 0.121    | 0.872    |  |  |  |  |
| aggregate   | 0.809             | 0.697                                                                                                 | 0.626     | 0.763 | 0.544    | 0.791    |  |  |  |  |

Supplementary Data Tab. 27 Correlation of human-assessed difficulty scores and difficulty levels (state values) evaluated by the Vision Agent of AdaptiveNN.

| Human Judge | Accuracy (%) of distinguishing between each target<br>behavior and human behavior |            |            |  |  |  |  |  |  |
|-------------|-----------------------------------------------------------------------------------|------------|------------|--|--|--|--|--|--|
|             | Random                                                                            | Human      | AdaptiveNN |  |  |  |  |  |  |
| 1           | 94.4                                                                              | 63.9       | 41.7       |  |  |  |  |  |  |
| 2           | 86.1                                                                              | 41.7       | 66.7       |  |  |  |  |  |  |
| 3           | 80.6                                                                              | 44.4       | 63.9       |  |  |  |  |  |  |
| 4           | 72.2                                                                              | 41.7       | 52.8       |  |  |  |  |  |  |
| 5           | 69.4                                                                              | 41.7       | 47.2       |  |  |  |  |  |  |
| 6           | 83.3                                                                              | 69.4       | 41.7       |  |  |  |  |  |  |
| 7           | 72.2                                                                              | 44.4       | 41.7       |  |  |  |  |  |  |
| 8           | 63.9                                                                              | 38.9       | 61.1       |  |  |  |  |  |  |
| 9           | 63.9                                                                              | 44.4       | 41.7       |  |  |  |  |  |  |
| 10          | 97.2                                                                              | 55.6       | 36.1       |  |  |  |  |  |  |
| 11          | 97.2                                                                              | 58.3       | 47.2       |  |  |  |  |  |  |
| 12          | 83.3                                                                              | 55.6       | 58.3       |  |  |  |  |  |  |
| 13          | 100.0                                                                             | 63.9       | 61.1       |  |  |  |  |  |  |
| 14          | 58.3                                                                              | 38.9       | 55.6       |  |  |  |  |  |  |
| 15          | 86.1                                                                              | 36.1       | 58.3       |  |  |  |  |  |  |
| 16          | 86.1                                                                              | 50.0       | 52.8       |  |  |  |  |  |  |
| 17          | 80.6                                                                              | 55.6       | 33.3       |  |  |  |  |  |  |
| 18          | 88.9                                                                              | 61.1       | 52.8       |  |  |  |  |  |  |
| 19          | 80.6                                                                              | 58.3       | 55.6       |  |  |  |  |  |  |
| 20          | 58.3                                                                              | 44.4       | 44.4       |  |  |  |  |  |  |
| 21          | 69.4                                                                              | 50.0       | 63.9       |  |  |  |  |  |  |
| 22          | 66.7                                                                              | 52.8       | 44.4       |  |  |  |  |  |  |
| 23          | 86.1                                                                              | 36.1       | 58.3       |  |  |  |  |  |  |
| 24          | 86.1                                                                              | 47.2       | 50.0       |  |  |  |  |  |  |
| 25          | 63.9                                                                              | 52.8       | 44.4       |  |  |  |  |  |  |
| 26          | 91.7                                                                              | 47.2       | 58.3       |  |  |  |  |  |  |
| 27          | 83.3                                                                              | 63.9       | 38.9       |  |  |  |  |  |  |
| 28          | 83.3                                                                              | 41.7       | 47.2       |  |  |  |  |  |  |
| 29          | 75.0                                                                              | 50.0       | 44.4       |  |  |  |  |  |  |
| 30          | 88.9                                                                              | 47.2       | 47.2       |  |  |  |  |  |  |
| 31          | 58.3                                                                              | 52.8       | 55.6       |  |  |  |  |  |  |
| 32          | 77.8                                                                              | 50.0       | 63.9       |  |  |  |  |  |  |
| 33          | 88.9                                                                              | 55.6       | 44.4       |  |  |  |  |  |  |
| 34          | 83.3                                                                              | 50.0       | 50.0       |  |  |  |  |  |  |
| 35          | 80.6                                                                              | 36.1       | 52.8       |  |  |  |  |  |  |
| 36          | 80.6                                                                              | 58.3       | 52.8       |  |  |  |  |  |  |
| 37          | 80.6                                                                              | 33.3       | 50.0       |  |  |  |  |  |  |
| 38          | 80.6                                                                              | 52.8       | 61.1       |  |  |  |  |  |  |
| 39          | 83.3                                                                              | 36.1       | 55.6       |  |  |  |  |  |  |
| Average CI  | 79.8 ± 3.6                                                                        | 49.3 ± 2.9 | 51.2 ± 2.7 |  |  |  |  |  |  |

Supplementary Data Tab. 28 Results of 'Turing Test of the Visual Fixation Behaviors': human judges (n=39) were randomly presented with paired examples of visual perception behaviors from 'humans' and one of AdaptiveNN, humans, random behaviors. Judges were tasked with identifying the machine, including in control pairs ('random v.s. human' or 'human v.s. human'). Values represent individual and mean accuracy across judges.

| Human Judge | Accuracy (%) of distinguishing between each target<br>behavior and human behavior |            |            |  |  |  |  |  |
|-------------|-----------------------------------------------------------------------------------|------------|------------|--|--|--|--|--|
|             | Random                                                                            | Human      | AdaptiveNN |  |  |  |  |  |
| 1           | 86.1                                                                              | 55.6       | 66.7       |  |  |  |  |  |
| 2           | 80.6                                                                              | 52.8       | 58.3       |  |  |  |  |  |
| 3           | 86.1                                                                              | 50.0       | 52.8       |  |  |  |  |  |
| 4           | 88.9                                                                              | 61.1       | 41.7       |  |  |  |  |  |
| 5           | 66.7                                                                              | 50.0       | 55.6       |  |  |  |  |  |
| 6           | 83.3                                                                              | 52.8       | 52.8       |  |  |  |  |  |
| 7           | 88.9                                                                              | 50.0       | 36.1       |  |  |  |  |  |
| 8           | 94.4                                                                              | 47.2       | 61.1       |  |  |  |  |  |
| 9           | 94.4                                                                              | 58.3       | 52.8       |  |  |  |  |  |
| 10          | 72.2                                                                              | 52.8       | 52.8       |  |  |  |  |  |
| 11          | 83.3                                                                              | 41.7       | 63.9       |  |  |  |  |  |
| 12          | 91.7                                                                              | 50.0       | 41.7       |  |  |  |  |  |
| 13          | 80.6                                                                              | 41.7       | 52.8       |  |  |  |  |  |
| 14          | 77.8                                                                              | 50.0       | 36.1       |  |  |  |  |  |
| 15          | 80.6                                                                              | 47.2       | 63.9       |  |  |  |  |  |
| 16          | 86.1                                                                              | 38.9       | 58.3       |  |  |  |  |  |
| 17          | 83.3                                                                              | 38.9       | 55.6       |  |  |  |  |  |
| 18          | 91.7                                                                              | 38.9       | 61.1       |  |  |  |  |  |
| 19          | 77.8                                                                              | 58.3       | 47.2       |  |  |  |  |  |
| 20          | 86.1                                                                              | 47.2       | 61.1       |  |  |  |  |  |
| 21          | 75.0                                                                              | 52.8       | 44.4       |  |  |  |  |  |
| 22          | 77.8                                                                              | 47.2       | 52.8       |  |  |  |  |  |
| 23          | 75.0                                                                              | 52.8       | 55.6       |  |  |  |  |  |
| 24          | 86.1                                                                              | 55.6       | 41.7       |  |  |  |  |  |
| 25          | 75.0                                                                              | 47.2       | 58.3       |  |  |  |  |  |
| 26          | 83.3                                                                              | 47.2       | 30.6       |  |  |  |  |  |
| 27          | 88.9                                                                              | 41.7       | 52.8       |  |  |  |  |  |
| 28          | 75.0                                                                              | 38.9       | 61.1       |  |  |  |  |  |
| 29          | 91.7                                                                              | 38.9       | 50.0       |  |  |  |  |  |
| 30          | 72.2                                                                              | 52.8       | 38.9       |  |  |  |  |  |
| 31          | 80.6                                                                              | 63.9       | 36.1       |  |  |  |  |  |
| 32          | 77.8                                                                              | 38.9       | 47.2       |  |  |  |  |  |
| 33          | 83.3                                                                              | 52.8       | 30.6       |  |  |  |  |  |
| 34          | 72.2                                                                              | 41.7       | 44.4       |  |  |  |  |  |
| 35          | 83.3                                                                              | 38.9       | 58.3       |  |  |  |  |  |
| 36          | 83.3                                                                              | 50.0       | 38.9       |  |  |  |  |  |
| 37          | 75.0                                                                              | 58.3       | 50.0       |  |  |  |  |  |
| 38          | 83.3                                                                              | 55.6       | 33.3       |  |  |  |  |  |
| 39          | 91.7                                                                              | 55.6       | 47.2       |  |  |  |  |  |
| Average CI  | 82.3 ± 2.2                                                                        | 49.1 ± 2.3 | 49.9 ± 3.2 |  |  |  |  |  |

Supplementary Data Tab. 29 Results of 'Turing Test of the Difficulty-assessment Behaviors': human judges (n=39) were randomly presented with paired examples of visual perception behaviors from 'humans' and one of AdaptiveNN, humans, random behaviors. Judges were tasked with identifying the machine, including in control pairs ('random v.s. human' or 'human v.s. human'). Values represent individual and mean accuracy across judges.

| Policy                  |         |                    | Trial / Accuracy (%)     |         |         |             |  |  |
|-------------------------|---------|--------------------|--------------------------|---------|---------|-------------|--|--|
|                         | st<br>1 | nd<br>2            | rd<br>3                  | th<br>4 | th<br>5 | Average     |  |  |
|                         |         | Pre-defined Policy |                          |         |         |             |  |  |
| Random                  | 73.6    | 73.3               | 73.4                     | 73.7    | 73.7    | 73.5 ± 0.16 |  |  |
| Gaussian Distribution   | 74.1    | 74.1               | 74.0                     | 74.3    | 74.1    | 74.1 ± 0.08 |  |  |
| Center-Corner           | 76.5    | 76.3               | 76.2                     | 76.6    | 76.5    | 76.4 ± 0.12 |  |  |
| CAM-based Policy        |         |                    |                          |         |         |             |  |  |
| GradCAM                 | 76.7    | 76.5               | 76.6                     | 76.7    | 76.7    | 76.7 ± 0.07 |  |  |
| GradCAM++               | 75.6    | 75.4               | 75.3                     | 75.5    | 75.7    | 75.5 ± 0.16 |  |  |
| XGradCAM                | 75.6    | 75.3               | 75.3                     | 75.5    | 75.6    | 75.5 ± 0.14 |  |  |
| LayerCAM                | 75.8    | 75.4               | 75.2                     | 75.7    | 75.7    | 75.5 ± 0.23 |  |  |
| GradCAM + GMM           | 77.1    | 77.1               | 77.0                     | 77.0    | 77.0    | 77.0 ± 0.04 |  |  |
|                         |         |                    | Learnable Policy Network |         |         |             |  |  |
| Spatial Transformer Net | 75.9    | 75.7               | 75.6                     | 75.7    | 75.9    | 75.8 ± 0.12 |  |  |
| Gumbel-Softmax          | 73.2    | 73.1               | 73.1                     | 73.4    | 73.2    | 73.2 ± 0.11 |  |  |
| AdaptiveNN (Ours)       | 78.3    | 78.2               | 78.0                     | 78.2    | 78.3    | 78.2 ± 0.11 |  |  |

Supplementary Data Tab. 30 Comparisons of different policies for establishing the fixation localization strategy on ImageNet-1K. 1 visual fixation is used on the top of AdaptiveNN-DeiT-S.

| Policy                  |         |                    |                          | Trial / Accuracy (%) |         |             |  |
|-------------------------|---------|--------------------|--------------------------|----------------------|---------|-------------|--|
|                         | st<br>1 | nd<br>2            | rd<br>3                  | th<br>4              | th<br>5 | Average     |  |
|                         |         | Pre-defined Policy |                          |                      |         |             |  |
| Random                  | 77.0    | 76.9               | 77.0                     | 77.0                 | 77.0    | 77.0 ± 0.04 |  |
| Gaussian Distribution   | 77.2    | 77.2               | 77.4                     | 77.5                 | 77.2    | 77.3 ± 0.11 |  |
| Center-Corner           | 78.3    | 78.1               | 78.1                     | 78.2                 | 78.3    | 78.2 ± 0.09 |  |
| CAM-based Policy        |         |                    |                          |                      |         |             |  |
| GradCAM                 | 79.1    | 79.0               | 79.0                     | 78.7                 | 79.0    | 79.0 ± 0.12 |  |
| GradCAM++               | 78.4    | 78.3               | 78.2                     | 78.3                 | 78.4    | 78.3 ± 0.05 |  |
| XGradCAM                | 78.4    | 78.2               | 78.4                     | 78.1                 | 78.4    | 78.3 ± 0.13 |  |
| LayerCAM                | 78.6    | 78.3               | 78.2                     | 78.5                 | 78.3    | 78.4 ± 0.16 |  |
| GradCAM + GMM           | 79.6    | 79.6               | 79.5                     | 79.5                 | 79.6    | 79.6 ± 0.05 |  |
|                         |         |                    | Learnable Policy Network |                      |         |             |  |
| Spatial Transformer Net | 77.6    | 77.4               | 77.4                     | 77.5                 | 77.5    | 77.5 ± 0.06 |  |
| Gumbel-Softmax          | 77.0    | 76.8               | 77.0                     | 76.9                 | 76.9    | 76.9 ± 0.07 |  |
| AdaptiveNN (Ours)       | 80.5    | 80.3               | 80.5                     | 80.3                 | 80.5    | 80.4 ± 0.10 |  |

Supplementary Data Tab. 31 Comparisons of different policies for establishing the fixation localization strategy on ImageNet-1K. 2 visual fixations are used on the top of AdaptiveNN-DeiT-S.

| Policy                  |                          |                    |         | Trial / Accuracy (%) |         |             |  |
|-------------------------|--------------------------|--------------------|---------|----------------------|---------|-------------|--|
|                         | st<br>1                  | nd<br>2            | rd<br>3 | th<br>4              | th<br>5 | Average     |  |
|                         |                          | Pre-defined Policy |         |                      |         |             |  |
| Random                  | 78.5                     | 78.5               | 78.5    | 78.6                 | 78.5    | 78.5 ± 0.03 |  |
| Gaussian Distribution   | 78.7                     | 78.5               | 78.5    | 78.8                 | 78.6    | 78.6 ± 0.12 |  |
| Center-Corner           | 79.6                     | 79.4               | 79.4    | 79.4                 | 79.6    | 79.5 ± 0.11 |  |
| CAM-based Policy        |                          |                    |         |                      |         |             |  |
| GradCAM                 | 79.8                     | 79.8               | 79.8    | 79.7                 | 79.8    | 79.8 ± 0.06 |  |
| GradCAM++               | 79.5                     | 79.5               | 79.4    | 79.4                 | 79.6    | 79.5 ± 0.08 |  |
| XGradCAM                | 79.6                     | 79.5               | 79.5    | 79.4                 | 79.6    | 79.5 ± 0.08 |  |
| LayerCAM                | 79.8                     | 79.5               | 79.3    | 79.6                 | 79.7    | 79.6 ± 0.15 |  |
| GradCAM + GMM           | 80.4                     | 80.4               | 80.3    | 80.3                 | 80.5    | 80.4 ± 0.07 |  |
|                         | Learnable Policy Network |                    |         |                      |         |             |  |
| Spatial Transformer Net | 78.3                     | 78.4               | 78.2    | 78.3                 | 78.3    | 78.3 ± 0.07 |  |
| Gumbel-Softmax          | 78.8                     | 78.7               | 78.6    | 78.7                 | 78.8    | 78.7 ± 0.07 |  |
| AdaptiveNN (Ours)       | 81.3                     | 81.3               | 81.3    | 81.3                 | 81.3    | 81.3 ± 0.03 |  |

Supplementary Data Tab. 32 Comparisons of different policies for establishing the fixation localization strategy on ImageNet-1K. 3 visual fixations are used on the top of AdaptiveNN-DeiT-S.

| Policy                  |         |                    | Trial / Accuracy (%)     |         |         |             |  |  |
|-------------------------|---------|--------------------|--------------------------|---------|---------|-------------|--|--|
|                         | st<br>1 | nd<br>2            | rd<br>3                  | th<br>4 | th<br>5 | Average     |  |  |
|                         |         | Pre-defined Policy |                          |         |         |             |  |  |
| Random                  | 79.3    | 79.4               | 79.5                     | 79.4    | 79.3    | 79.4 ± 0.05 |  |  |
| Gaussian Distribution   | 79.3    | 79.3               | 79.2                     | 79.4    | 79.2    | 79.3 ± 0.07 |  |  |
| Center-Corner           | 80.4    | 80.2               | 80.1                     | 80.2    | 80.4    | 80.3 ± 0.09 |  |  |
| CAM-based Policy        |         |                    |                          |         |         |             |  |  |
| GradCAM                 | 80.4    | 80.4               | 80.3                     | 80.2    | 80.4    | 80.4 ± 0.07 |  |  |
| GradCAM++               | 80.2    | 80.3               | 80.1                     | 80.0    | 80.3    | 80.2 ± 0.11 |  |  |
| XGradCAM                | 80.2    | 80.0               | 80.2                     | 80.3    | 80.2    | 80.2 ± 0.11 |  |  |
| LayerCAM                | 80.4    | 80.3               | 80.1                     | 80.3    | 80.3    | 80.3 ± 0.11 |  |  |
| GradCAM + GMM           | 80.9    | 80.9               | 80.9                     | 80.9    | 80.9    | 80.9 ± 0.02 |  |  |
|                         |         |                    | Learnable Policy Network |         |         |             |  |  |
| Spatial Transformer Net | 78.7    | 78.7               | 78.5                     | 78.7    | 78.7    | 78.7 ± 0.06 |  |  |
| Gumbel-Softmax          | 79.8    | 79.6               | 79.6                     | 79.6    | 79.8    | 79.7 ± 0.09 |  |  |
| AdaptiveNN (Ours)       | 81.8    | 81.8               | 81.7                     | 81.7    | 81.8    | 81.8 ± 0.05 |  |  |

Supplementary Data Tab. 33 Comparisons of different policies for establishing the fixation localization strategy on ImageNet-1K. 4 visual fixations are used on the top of AdaptiveNN-DeiT-S.

| Policy                  |                          |                    |         | Trial / Accuracy (%) |         |             |  |
|-------------------------|--------------------------|--------------------|---------|----------------------|---------|-------------|--|
|                         | st<br>1                  | nd<br>2            | rd<br>3 | th<br>4              | th<br>5 | Average     |  |
|                         |                          | Pre-defined Policy |         |                      |         |             |  |
| Random                  | 66.3                     | 65.6               | 66.2    | 66.2                 | 66.1    | 66.1 ± 0.26 |  |
| Gaussian Distribution   | 67.6                     | 67.2               | 67.6    | 67.6                 | 67.5    | 67.5 ± 0.15 |  |
| Center-Corner           | 71.1                     | 71.0               | 71.2    | 70.8                 | 70.7    | 70.9 ± 0.20 |  |
| CAM-based Policy        |                          |                    |         |                      |         |             |  |
| GradCAM                 | 71.2                     | 70.7               | 70.8    | 70.8                 | 70.8    | 70.8 ± 0.19 |  |
| GradCAM++               | 70.5                     | 70.3               | 70.5    | 70.4                 | 70.4    | 70.4 ± 0.07 |  |
| XGradCAM                | 71.2                     | 70.7               | 70.8    | 70.8                 | 70.8    | 70.8 ± 0.19 |  |
| LayerCAM                | 70.5                     | 70.4               | 70.5    | 70.5                 | 70.5    | 70.5 ± 0.05 |  |
| GradCAM + GMM           | 71.8                     | 71.4               | 71.7    | 71.5                 | 71.7    | 71.6 ± 0.14 |  |
|                         | Learnable Policy Network |                    |         |                      |         |             |  |
| Spatial Transformer Net | 70.0                     | 69.6               | 69.9    | 69.7                 | 69.7    | 69.8 ± 0.14 |  |
| Gumbel-Softmax          | 68.1                     | 67.8               | 68.1    | 68.0                 | 67.9    | 68.0 ± 0.11 |  |
| AdaptiveNN (Ours)       | 73.5                     | 73.5               | 73.6    | 73.7                 | 73.6    | 73.6 ± 0.08 |  |

Supplementary Data Tab. 34 Comparisons of different policies for establishing the fixation localization strategy on ImageNet-1K. 1 visual fixation is used on the top of AdaptiveNN-ResNet-50.

| Policy                  |         |                          |         | Trial / Accuracy (%) |         |             |  |
|-------------------------|---------|--------------------------|---------|----------------------|---------|-------------|--|
|                         | st<br>1 | nd<br>2                  | rd<br>3 | th<br>4              | th<br>5 | Average     |  |
|                         |         | Pre-defined Policy       |         |                      |         |             |  |
| Random                  | 70.8    | 70.7                     | 70.8    | 71.0                 | 71.0    | 70.8 ± 0.10 |  |
| Gaussian Distribution   | 71.8    | 71.5                     | 71.8    | 71.9                 | 71.9    | 71.8 ± 0.15 |  |
| Center-Corner           | 73.4    | 73.0                     | 73.3    | 73.2                 | 73.2    | 73.2 ± 0.14 |  |
| CAM-based Policy        |         |                          |         |                      |         |             |  |
| GradCAM                 | 74.7    | 74.4                     | 74.6    | 74.6                 | 74.5    | 74.6 ± 0.08 |  |
| GradCAM++               | 74.4    | 74.1                     | 74.4    | 74.3                 | 74.3    | 74.3 ± 0.12 |  |
| XGradCAM                | 74.7    | 74.4                     | 74.6    | 74.6                 | 74.5    | 74.6 ± 0.09 |  |
| LayerCAM                | 74.4    | 74.2                     | 74.3    | 74.4                 | 74.4    | 74.4 ± 0.07 |  |
| GradCAM + GMM           | 75.7    | 75.6                     | 75.8    | 75.7                 | 75.6    | 75.7 ± 0.08 |  |
|                         |         | Learnable Policy Network |         |                      |         |             |  |
| Spatial Transformer Net | 72.2    | 72.1                     | 72.2    | 72.1                 | 72.1    | 72.1 ± 0.08 |  |
| Gumbel-Softmax          | 72.9    | 73.0                     | 73.0    | 72.9                 | 72.9    | 72.9 ± 0.06 |  |
| AdaptiveNN (Ours)       | 77.0    | 77.1                     | 77.2    | 77.2                 | 77.1    | 77.1 ± 0.07 |  |

Supplementary Data Tab. 35 Comparisons of different policies for establishing the fixation localization strategy on ImageNet-1K. 2 visual fixations are used on the top of AdaptiveNN-ResNet-50.

| Policy                  |                          | Trial / Accuracy (%) |         |         |         |             |  |  |
|-------------------------|--------------------------|----------------------|---------|---------|---------|-------------|--|--|
|                         | st<br>1                  | nd<br>2              | rd<br>3 | th<br>4 | th<br>5 | Average     |  |  |
|                         |                          | Pre-defined Policy   |         |         |         |             |  |  |
| Random                  | 73.3                     | 73.1                 | 73.4    | 73.5    | 73.5    | 73.3 ± 0.16 |  |  |
| Gaussian Distribution   | 73.9                     | 73.6                 | 73.9    | 73.8    | 73.8    | 73.8 ± 0.13 |  |  |
| Center-Corner           | 75.4                     | 75.1                 | 75.4    | 75.1    | 75.2    | 75.2 ± 0.11 |  |  |
| CAM-based Policy        |                          |                      |         |         |         |             |  |  |
| GradCAM                 | 76.2                     | 75.8                 | 76.2    | 76.2    | 76.2    | 76.1 ± 0.15 |  |  |
| GradCAM++               | 76.1                     | 75.7                 | 75.9    | 76.0    | 75.9    | 75.9 ± 0.14 |  |  |
| XGradCAM                | 76.2                     | 75.8                 | 76.2    | 76.2    | 76.2    | 76.1 ± 0.15 |  |  |
| LayerCAM                | 76.2                     | 75.7                 | 76.1    | 76.1    | 76.1    | 76.0 ± 0.16 |  |  |
| GradCAM + GMM           | 77.2                     | 76.9                 | 77.1    | 77.1    | 76.9    | 77.0 ± 0.10 |  |  |
|                         | Learnable Policy Network |                      |         |         |         |             |  |  |
| Spatial Transformer Net | 73.3                     | 73.0                 | 73.3    | 73.1    | 73.2    | 73.2 ± 0.12 |  |  |
| Gumbel-Softmax          | 75.3                     | 75.2                 | 75.3    | 75.2    | 75.2    | 75.3 ± 0.05 |  |  |
| AdaptiveNN (Ours)       | 78.3                     | 78.5                 | 78.5    | 78.4    | 78.3    | 78.4 ± 0.07 |  |  |

Supplementary Data Tab. 36 Comparisons of different policies for establishing the fixation localization strategy on ImageNet-1K. 3 visual fixations are used on the top of AdaptiveNN-ResNet-50.

| Policy                  |         | Trial / Accuracy (%) |                          |         |         |             |  |  |
|-------------------------|---------|----------------------|--------------------------|---------|---------|-------------|--|--|
|                         | st<br>1 | nd<br>2              | rd<br>3                  | th<br>4 | th<br>5 | Average     |  |  |
|                         |         | Pre-defined Policy   |                          |         |         |             |  |  |
| Random                  | 74.8    | 74.6                 | 74.9                     | 74.9    | 74.9    | 74.8 ± 0.14 |  |  |
| Gaussian Distribution   | 75.1    | 74.8                 | 75.0                     | 75.1    | 75.0    | 75.0 ± 0.09 |  |  |
| Center-Corner           | 76.7    | 76.5                 | 76.6                     | 76.6    | 76.6    | 76.6 ± 0.09 |  |  |
| CAM-based Policy        |         |                      |                          |         |         |             |  |  |
| GradCAM                 | 77.0    | 76.8                 | 77.1                     | 77.1    | 77.0    | 77.0 ± 0.11 |  |  |
| GradCAM++               | 77.0    | 76.8                 | 77.0                     | 77.0    | 76.9    | 76.9 ± 0.10 |  |  |
| XGradCAM                | 77.0    | 76.8                 | 77.1                     | 77.1    | 77.0    | 77.0 ± 0.11 |  |  |
| LayerCAM                | 77.1    | 76.8                 | 77.1                     | 77.1    | 77.1    | 77.0 ± 0.10 |  |  |
| GradCAM + GMM           | 77.8    | 77.8                 | 77.8                     | 78.0    | 77.8    | 77.8 ± 0.08 |  |  |
|                         |         |                      | Learnable Policy Network |         |         |             |  |  |
| Spatial Transformer Net | 73.9    | 73.5                 | 73.9                     | 73.8    | 73.7    | 73.8 ± 0.12 |  |  |
| Gumbel-Softmax          | 76.5    | 76.2                 | 76.5                     | 76.5    | 76.4    | 76.4 ± 0.12 |  |  |
| AdaptiveNN (Ours)       | 79.0    | 79.2                 | 79.2                     | 79.3    | 79.0    | 79.1 ± 0.09 |  |  |

Supplementary Data Tab. 37 Comparisons of different policies for establishing the fixation localization strategy on ImageNet-1K. 4 visual fixations are used on the top of AdaptiveNN-ResNet-50.

|            | Predicted   |         | Trial / Test Loss |         |         |         |                |  |
|------------|-------------|---------|-------------------|---------|---------|---------|----------------|--|
| Model      | State Value | st<br>1 | nd<br>2           | rd<br>3 | th<br>4 | th<br>5 | Average        |  |
|            | 0.002       | 0.264   | 0.270             | 0.256   | 0.269   | 0.232   | 0.258 ± 0.0156 |  |
|            | 0.077       | 0.546   | 0.511             | 0.515   | 0.518   | 0.520   | 0.522 ± 0.0141 |  |
|            | 0.152       | 0.820   | 0.767             | 0.794   | 0.722   | 0.760   | 0.773 ± 0.0370 |  |
|            | 0.226       | 0.894   | 0.947             | 0.934   | 0.902   | 0.788   | 0.893 ± 0.0626 |  |
|            | 0.301       | 0.947   | 1.085             | 0.960   | 0.986   | 1.069   | 1.009 ± 0.0635 |  |
| AdaptiveNN | 0.376       | 1.085   | 1.082             | 1.191   | 1.352   | 1.178   | 1.178 ± 0.1101 |  |
| -DeiT-S    | 0.451       | 1.239   | 1.271             | 1.224   | 1.333   | 1.264   | 1.266 ± 0.0418 |  |
|            | 0.526       | 1.285   | 1.306             | 1.345   | 1.363   | 1.420   | 1.344 ± 0.0526 |  |
|            | 0.600       | 1.452   | 1.422             | 1.485   | 1.556   | 1.572   | 1.497 ± 0.0648 |  |
|            | 0.675       | 1.465   | 1.692             | 1.613   | 1.519   | 1.507   | 1.559 ± 0.0920 |  |
|            | 0.750       | 1.629   | 1.403             | 1.604   | 1.523   | 1.739   | 1.580 ± 0.1254 |  |

Supplementary Data Tab. 38 Average test loss corresponding to the validation data with different state values predicted by AdaptiveNN-DeiT-S (1-st fixation).

|            | Predicted   |         | Trial / Test Loss |         |         |         |                |  |
|------------|-------------|---------|-------------------|---------|---------|---------|----------------|--|
| Model      | State Value | st<br>1 | nd<br>2           | rd<br>3 | th<br>4 | th<br>5 | Average        |  |
|            | 0.002       | 0.267   | 0.298             | 0.287   | 0.258   | 0.269   | 0.276 ± 0.0162 |  |
|            | 0.052       | 0.708   | 0.667             | 0.713   | 0.710   | 0.702   | 0.700 ± 0.0189 |  |
|            | 0.102       | 0.914   | 0.967             | 1.014   | 1.009   | 1.089   | 0.999 ± 0.0646 |  |
|            | 0.151       | 1.251   | 1.149             | 1.198   | 1.108   | 1.255   | 1.192 ± 0.0640 |  |
|            | 0.201       | 1.285   | 1.453             | 1.433   | 1.318   | 1.413   | 1.380 ± 0.0743 |  |
| AdaptiveNN | 0.251       | 1.682   | 1.598             | 1.664   | 1.555   | 1.695   | 1.639 ± 0.0600 |  |
| -DeiT-S    | 0.301       | 1.767   | 1.735             | 1.722   | 1.793   | 1.821   | 1.767 ± 0.0409 |  |
|            | 0.351       | 1.772   | 1.767             | 1.894   | 1.937   | 1.935   | 1.861 ± 0.0855 |  |
|            | 0.400       | 1.862   | 1.899             | 2.062   | 1.892   | 2.089   | 1.961 ± 0.1059 |  |
|            | 0.450       | 2.008   | 2.097             | 2.123   | 2.084   | 2.420   | 2.146 ± 0.1587 |  |
|            | 0.500       | 2.217   | 2.044             | 2.141   | 2.106   | 2.313   | 2.164 ± 0.1039 |  |

Supplementary Data Tab. 39 Average test loss corresponding to the validation data with different state values predicted by AdaptiveNN-DeiT-S (2-nd fixation).

|            | Predicted   |         | Trial / Test Loss |         |         |         |                |  |  |
|------------|-------------|---------|-------------------|---------|---------|---------|----------------|--|--|
| Model      | State Value | st<br>1 | nd<br>2           | rd<br>3 | th<br>4 | th<br>5 | Average        |  |  |
|            | 0.002       | 0.341   | 0.343             | 0.339   | 0.280   | 0.264   | 0.313 ± 0.0383 |  |  |
|            | 0.032       | 0.674   | 0.671             | 0.719   | 0.734   | 0.843   | 0.728 ± 0.0696 |  |  |
|            | 0.062       | 0.898   | 1.127             | 1.089   | 1.053   | 1.196   | 1.073 ± 0.1110 |  |  |
|            | 0.091       | 1.225   | 1.312             | 1.264   | 1.219   | 1.520   | 1.308 ± 0.1240 |  |  |
|            | 0.121       | 1.281   | 1.387             | 1.571   | 1.500   | 1.597   | 1.467 ± 0.1319 |  |  |
| AdaptiveNN | 0.151       | 1.562   | 1.586             | 1.640   | 1.697   | 1.923   | 1.682 ± 0.1447 |  |  |
| -DeiT-S    | 0.181       | 1.774   | 1.726             | 1.850   | 1.742   | 1.898   | 1.798 ± 0.0735 |  |  |
|            | 0.211       | 1.884   | 1.699             | 1.942   | 1.899   | 2.256   | 1.936 ± 0.2014 |  |  |
|            | 0.240       | 2.056   | 2.155             | 2.226   | 1.990   | 2.357   | 2.157 ± 0.1437 |  |  |
|            | 0.270       | 2.071   | 2.159             | 2.271   | 2.194   | 2.508   | 2.241 ± 0.1660 |  |  |
|            | 0.300       | 2.068   | 2.117             | 2.309   | 2.386   | 2.555   | 2.287 ± 0.1995 |  |  |

Supplementary Data Tab. 40 Average test loss corresponding to the validation data with different state values predicted by AdaptiveNN-DeiT-S (3-rd fixation).

| Model      | Predicted   | Trial / Test Loss |         |         |         |         |                |
|------------|-------------|-------------------|---------|---------|---------|---------|----------------|
|            | State Value | st<br>1           | nd<br>2 | rd<br>3 | th<br>4 | th<br>5 | Average        |
|            | 0.001       | 0.307             | 0.309   | 0.289   | 0.312   | 0.291   | 0.301 ± 0.0109 |
|            | 0.026       | 0.762             | 0.761   | 0.857   | 0.852   | 1.086   | 0.864 ± 0.1330 |
|            | 0.051       | 1.098             | 1.153   | 1.186   | 1.232   | 1.516   | 1.237 ± 0.1634 |
|            | 0.076       | 1.321             | 1.427   | 1.537   | 1.534   | 1.810   | 1.526 ± 0.1820 |
|            | 0.101       | 1.589             | 1.640   | 1.835   | 1.703   | 2.177   | 1.789 ± 0.2356 |
| AdaptiveNN | 0.126       | 1.873             | 1.767   | 1.997   | 2.007   | 2.192   | 1.967 ± 0.1596 |
| -DeiT-S    | 0.150       | 2.053             | 2.055   | 2.183   | 2.022   | 2.414   | 2.145 ± 0.1624 |
|            | 0.175       | 2.319             | 2.306   | 2.294   | 2.356   | 2.508   | 2.356 ± 0.0877 |
|            | 0.200       | 2.417             | 2.420   | 2.438   | 2.453   | 2.601   | 2.466 ± 0.0771 |
|            | 0.225       | 2.446             | 2.449   | 2.582   | 2.544   | 2.647   | 2.534 ± 0.0871 |
|            | 0.250       | 2.474             | 2.478   | 2.634   | 2.631   | 2.647   | 2.573 ± 0.0885 |

Supplementary Data Tab. 41 Average test loss corresponding to the validation data with different state values predicted by AdaptiveNN-DeiT-S (4-th fixation).

|            | Predicted   | Trial / Test Loss |         |         |         |         |                |
|------------|-------------|-------------------|---------|---------|---------|---------|----------------|
| Model      | State Value | st<br>1           | nd<br>2 | rd<br>3 | th<br>4 | th<br>5 | Average        |
|            | 0.002       | 0.252             | 0.343   | 0.301   | 0.277   | 0.332   | 0.301 ± 0.0379 |
|            | 0.062       | 0.561             | 0.519   | 0.510   | 0.525   | 0.579   | 0.539 ± 0.0297 |
|            | 0.122       | 0.731             | 0.863   | 0.677   | 0.811   | 0.707   | 0.758 ± 0.0773 |
|            | 0.181       | 0.996             | 1.101   | 0.858   | 0.959   | 1.051   | 0.993 ± 0.0928 |
|            | 0.241       | 1.234             | 1.203   | 1.090   | 1.082   | 1.208   | 1.163 ± 0.0719 |
| AdaptiveNN | 0.301       | 1.412             | 1.328   | 1.253   | 1.332   | 1.285   | 1.322 ± 0.0597 |
| -ResNet-50 | 0.361       | 1.493             | 1.490   | 1.495   | 1.430   | 1.430   | 1.468 ± 0.0344 |
|            | 0.421       | 1.528             | 1.627   | 1.464   | 1.528   | 1.488   | 1.527 ± 0.0620 |
|            | 0.480       | 1.706             | 1.649   | 1.670   | 1.701   | 1.663   | 1.678 ± 0.0244 |
|            | 0.540       | 1.778             | 1.783   | 1.713   | 1.602   | 1.707   | 1.716 ± 0.0733 |
|            | 0.600       | 1.846             | 1.633   | 1.766   | 1.724   | 1.693   | 1.732 ± 0.0797 |

Supplementary Data Tab. 42 Average test loss corresponding to the validation data with different state values predicted by AdaptiveNN-ResNet-50 (1st fixation).

| Model      | Predicted   | Trial / Test Loss |         |         |         |         |                |
|------------|-------------|-------------------|---------|---------|---------|---------|----------------|
|            | State Value | st<br>1           | nd<br>2 | rd<br>3 | th<br>4 | th<br>5 | Average        |
|            | 0.002       | 0.256             | 0.258   | 0.250   | 0.265   | 0.290   | 0.264 ± 0.0158 |
|            | 0.041       | 0.727             | 0.836   | 0.769   | 0.809   | 0.784   | 0.785 ± 0.0416 |
|            | 0.080       | 1.135             | 1.159   | 1.156   | 1.215   | 1.094   | 1.152 ± 0.0437 |
|            | 0.118       | 1.348             | 1.499   | 1.428   | 1.432   | 1.374   | 1.416 ± 0.0583 |
|            | 0.157       | 1.640             | 1.540   | 1.550   | 1.621   | 1.565   | 1.583 ± 0.0446 |
| AdaptiveNN | 0.196       | 1.587             | 1.770   | 1.700   | 1.707   | 1.820   | 1.717 ± 0.0878 |
| -ResNet-50 | 0.235       | 1.953             | 1.873   | 2.141   | 1.995   | 1.934   | 1.979 ± 0.1003 |
|            | 0.274       | 2.180             | 2.006   | 2.074   | 2.124   | 2.069   | 2.091 ± 0.0654 |
|            | 0.312       | 2.224             | 2.229   | 2.077   | 2.154   | 2.119   | 2.161 ± 0.0658 |
|            | 0.351       | 2.282             | 2.242   | 2.147   | 2.185   | 2.232   | 2.218 ± 0.0525 |
|            | 0.390       | 2.340             | 2.256   | 2.216   | 2.216   | 2.345   | 2.275 ± 0.0642 |

Supplementary Data Tab. 43 Average test loss corresponding to the validation data with different state values predicted by AdaptiveNN-ResNet-50 (2nd fixation).

|            | Predicted   | Trial / Test Loss |         |         |         |         |                |
|------------|-------------|-------------------|---------|---------|---------|---------|----------------|
| Model      | State Value | st<br>1           | nd<br>2 | rd<br>3 | th<br>4 | th<br>5 | Average        |
|            | 0.002       | 0.305             | 0.296   | 0.288   | 0.273   | 0.380   | 0.308 ± 0.0415 |
|            | 0.029       | 0.706             | 0.900   | 0.819   | 0.686   | 0.725   | 0.767 ± 0.0900 |
|            | 0.057       | 1.086             | 1.232   | 1.180   | 1.019   | 1.020   | 1.108 ± 0.0958 |
|            | 0.084       | 1.392             | 1.350   | 1.347   | 1.505   | 1.312   | 1.381 ± 0.0748 |
|            | 0.111       | 1.612             | 1.581   | 1.517   | 1.718   | 1.554   | 1.596 ± 0.0764 |
| AdaptiveNN | 0.139       | 1.757             | 1.800   | 1.801   | 1.756   | 1.785   | 1.780 ± 0.0221 |
| -ResNet-50 | 0.166       | 2.102             | 2.081   | 2.190   | 2.063   | 2.056   | 2.098 ± 0.0540 |
|            | 0.193       | 2.131             | 2.182   | 2.194   | 2.134   | 2.123   | 2.153 ± 0.0327 |
|            | 0.220       | 2.166             | 2.285   | 2.247   | 2.333   | 2.188   | 2.244 ± 0.0686 |
|            | 0.248       | 2.298             | 2.388   | 2.344   | 2.532   | 2.345   | 2.381 ± 0.0901 |
|            | 0.275       | 2.429             | 2.437   | 2.404   | 2.549   | 2.502   | 2.464 ± 0.0596 |

Supplementary Data Tab. 44 Average test loss corresponding to the validation data with different state values predicted by AdaptiveNN-ResNet-50 (3rd fixation).

| Model      | Predicted   | Trial / Test Loss |         |         |         |         |                |
|------------|-------------|-------------------|---------|---------|---------|---------|----------------|
|            | State Value | st<br>1           | nd<br>2 | rd<br>3 | th<br>4 | th<br>5 | Average        |
|            | 0.001       | 0.321             | 0.250   | 0.298   | 0.266   | 0.308   | 0.289 ± 0.0297 |
|            | 0.018       | 0.658             | 0.687   | 0.617   | 0.753   | 0.731   | 0.689 ± 0.0547 |
|            | 0.036       | 0.866             | 1.023   | 0.931   | 1.056   | 0.872   | 0.949 ± 0.0866 |
|            | 0.053       | 1.339             | 1.303   | 1.259   | 1.275   | 1.428   | 1.321 ± 0.0674 |
|            | 0.071       | 1.579             | 1.434   | 1.476   | 1.514   | 1.593   | 1.519 ± 0.0673 |
| AdaptiveNN | 0.088       | 1.792             | 1.826   | 1.678   | 1.833   | 1.755   | 1.777 ± 0.0633 |
| -ResNet-50 | 0.105       | 1.807             | 1.873   | 1.930   | 1.834   | 1.970   | 1.883 ± 0.0671 |
|            | 0.123       | 2.092             | 2.232   | 2.116   | 2.153   | 2.089   | 2.136 ± 0.0592 |
|            | 0.140       | 2.283             | 2.234   | 2.418   | 2.209   | 2.383   | 2.306 ± 0.0917 |
|            | 0.158       | 2.378             | 2.296   | 2.497   | 2.362   | 2.429   | 2.392 ± 0.0752 |
|            | 0.175       | 2.473             | 2.358   | 2.575   | 2.515   | 2.463   | 2.477 ± 0.0797 |

Supplementary Data Tab. 45 Average test loss corresponding to the validation data with different state values predicted by AdaptiveNN-ResNet-50 (4th fixation).

|            |          | Computational Cost (GFLOPs/image)  |                               |  |  |  |  |
|------------|----------|------------------------------------|-------------------------------|--|--|--|--|
| Model      | Accuracy | w.o. Sample-adaptive<br>Allocation | Sample-adaptive<br>Allocation |  |  |  |  |
|            | 79.57    | 2.28                               | 1.90                          |  |  |  |  |
| AdaptiveNN | 81.05    | 3.47                               | 2.55                          |  |  |  |  |
| -DeiT-S    | 81.81    | 4.66                               | 3.37                          |  |  |  |  |
|            | 82.16    | 5.85                               | 4.67                          |  |  |  |  |

Supplementary Data Tab. 46 Effectiveness of sample-adaptive computation allocation on top of AdaptiveNN-DeiT-S, highlighting computational costs required to achieve equivalent accuracies.

| Model      | Computational<br>Cost<br>(GFLOPs/image) | Accuracy (%)         |                                    |                               |  |  |  |
|------------|-----------------------------------------|----------------------|------------------------------------|-------------------------------|--|--|--|
|            |                                         | Random<br>Allocation | Anti-sample-adaptive<br>Allocation | Sample-adaptive<br>Allocation |  |  |  |
|            | 1.20                                    | 74.6 ± 0.22          | 74.2 ± 0.22                        | 75.7 ± 0.22                   |  |  |  |
|            | 1.61                                    | 76.0 ± 0.19          | 74.6 ± 0.24                        | 78.4 ± 0.20                   |  |  |  |
|            | 2.01                                    | 77.0 ± 0.16          | 75.2 ± 0.23                        | 79.9 ± 0.17                   |  |  |  |
|            | 2.42                                    | 77.9 ± 0.09          | 75.9 ± 0.27                        | 80.8 ± 0.18                   |  |  |  |
|            | 2.82                                    | 78.7 ± 0.09          | 76.5 ± 0.32                        | 81.4 ± 0.13                   |  |  |  |
| AdaptiveNN | 3.23                                    | 79.3 ± 0.07          | 77.2 ± 0.32                        | 81.7 ± 0.06                   |  |  |  |
| -DeiT-S    | 3.63                                    | 80.0 ± 0.06          | 77.9 ± 0.28                        | 81.9 ± 0.05                   |  |  |  |
|            | 4.04                                    | 80.5 ± 0.04          | 78.6 ± 0.24                        | 82.1 ± 0.09                   |  |  |  |
|            | 4.44                                    | 81.0 ± 0.07          | 79.4 ± 0.16                        | 82.1 ± 0.11                   |  |  |  |
|            | 4.85                                    | 81.5 ± 0.07          | 80.2 ± 0.14                        | 82.2 ± 0.11                   |  |  |  |
|            | 5.25                                    | 81.8 ± 0.07          | 81.0 ± 0.11                        | 82.2 ± 0.12                   |  |  |  |

Supplementary Data Tab. 47 Comparisons of different sample-wise computation allocation strategies on top of AdaptiveNN-DeiT-S. We report accuracy (%) and standard deviation (±) for each allocation method at various computational costs (GFLOPs/image).

|            |          | Computational Cost (GFLOPs/image)  |                               |  |  |  |  |
|------------|----------|------------------------------------|-------------------------------|--|--|--|--|
| Backbone   | Accuracy | w.o. Sample-adaptive<br>Allocation | Sample-adaptive<br>Allocation |  |  |  |  |
|            | 77.39    | 2.48                               | 2.12                          |  |  |  |  |
| AdaptiveNN | 78.80    | 3.86                               | 3.14                          |  |  |  |  |
| -ResNet-50 | 79.52    | 5.25                               | 4.26                          |  |  |  |  |
|            | 79.77    | 6.63                               | 5.64                          |  |  |  |  |

Supplementary Data Tab. 48 Effectiveness of sample-adaptive computation allocation on top of AdaptiveNN-ResNet-50, highlighting computational costs required to achieve equivalent accuracies.

|            | Computational          | Accuracy (%)         |                                    |                               |  |  |  |
|------------|------------------------|----------------------|------------------------------------|-------------------------------|--|--|--|
| Model      | Cost<br>(GFLOPs/image) | Random<br>Allocation | Anti-sample-adaptive<br>Allocation | Sample-adaptive<br>Allocation |  |  |  |
|            | 1.20                   | 73.2 ± 0.07          | 73.0 ± 0.08                        | 74.0 ± 0.07                   |  |  |  |
|            | 1.71                   | 74.4 ± 0.11          | 73.4 ± 0.04                        | 76.4 ± 0.13                   |  |  |  |
|            | 2.21                   | 75.3 ± 0.13          | 73.8 ± 0.09                        | 77.6 ± 0.16                   |  |  |  |
|            | 2.72                   | 76.2 ± 0.13          | 74.3 ± 0.10                        | 78.4 ± 0.09                   |  |  |  |
|            | 3.22                   | 76.9 ± 0.11          | 74.9 ± 0.10                        | 78.9 ± 0.08                   |  |  |  |
| AdaptiveNN | 3.73                   | 77.5 ± 0.15          | 75.5 ± 0.09                        | 79.3 ± 0.08                   |  |  |  |
| -ResNet-50 | 4.23                   | 78.1 ± 0.10          | 76.1 ± 0.09                        | 79.5 ± 0.10                   |  |  |  |
|            | 4.74                   | 78.5 ± 0.08          | 76.8 ± 0.09                        | 79.7 ± 0.11                   |  |  |  |
|            | 5.24                   | 79.0 ± 0.09          | 77.7 ± 0.11                        | 79.7 ± 0.10                   |  |  |  |
|            | 5.75                   | 79.4 ± 0.04          | 78.5 ± 0.11                        | 79.8 ± 0.08                   |  |  |  |
|            | 6.25                   | 79.7 ± 0.05          | 79.3 ± 0.07                        | 79.8 ± 0.07                   |  |  |  |

Supplementary Data Tab. 49 Comparisons of different sample-wise computation allocation strategies on top of AdaptiveNN-ResNet-50. We report accuracy (%) and standard deviation (±) for each allocation method at various computational costs (GFLOPs/image).

# References

- [1] He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: Pro- ceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 770–778 (2016)
- [2] Huang, G., Liu, Z., Van Der Maaten, L., Weinberger, K.Q.: Densely connected convolu- tional networks. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 4700–4708 (2017)
- [3] Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., Houlsby, N.: An image is worth 16x16 words: Transformers for image recognition at scale. In: International Conference on Learning Representations (2021)
- [4] Touvron, H., Cord, M., Douze, M., Massa, F., Sablayrolles, A., J´egou, H.: Training data- efficient image transformers & distillation through attention. In: International Conference on Machine Learning, pp. 10347–10357 (2021). PMLR
- [5] Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A.A., Veness, J., Bellemare, M.G., Graves, A., Riedmiller, M., Fidjeland, A.K., Ostrovski, G., et al.: Human-level control through deep reinforcement learning. nature 518(7540), 529–533 (2015)
- [6] LeCun, Y., Bengio, Y., Hinton, G.: Deep learning. Nature 521(7553), 436–444 (2015)
- [7] Russakovsky, O., Deng, J., Su, H., Krause, J., Satheesh, S., Ma, S., Huang, Z., Karpathy, A., Khosla, A., Bernstein, M., et al.: Imagenet large scale visual recognition challenge. International journal of computer vision 115, 211–252 (2015)
- [8] Tiu, E., Talius, E., Patel, P., Langlotz, C.P., Ng, A.Y., Rajpurkar, P.: Expert-level detection of pathologies from unannotated chest x-ray images via self-supervised learning. Nature Biomedical Engineering 6(12), 1399–1406 (2022)
- [9] Zhou, Y., Chia, M.A., Wagner, S.K., Ayhan, M.S., Williamson, D.J., Struyven, R.R., Liu, T., Xu, M., Lozano, M.G., Woodward-Court, P., et al.: A foundation model for generalizable disease detection from retinal images. Nature 622(7981), 156–163 (2023)
- [10] Bojarski, M., Del Testa, D., Dworakowski, D., Firner, B., Flepp, B., Goyal, P., Jackel, L.D., Monfort, M., Muller, U., Zhang, J., et al.: End to end learning for self-driving cars. arXiv preprint arXiv:1604.07316 (2016)
- [11] Mohammadi, M., Al-Fuqaha, A., Sorour, S., Guizani, M.: Deep learning for iot big data and streaming analytics: A survey. IEEE Communications Surveys & Tutorials 20(4), 2923–2960 (2018)
- [12] Li, H., Ota, K., Dong, M.: Learning iot in edge: Deep learning for the internet of things with edge computing. IEEE network 32(1), 96–101 (2018)
- [13] Zhang, C., Patras, P., Haddadi, H.: Deep learning in mobile and wireless networking: A survey. IEEE Communications surveys & tutorials 21(3), 2224–2287 (2019)
- [14] Grigorescu, S., Trasnea, B., Cocias, T., Macesanu, G.: A survey of deep learning techniques for autonomous driving. Journal of field robotics 37(3), 362–386 (2020)
- [15] Muhammad, K., Ullah, A., Lloret, J., Del Ser, J., de Albuquerque, V.H.C.: Deep learning for safe autonomous driving: Current challenges and future directions. IEEE Transactions on Intelligent Transportation Systems 22(7), 4316–4336 (2020)
- [16] Wang, Y., Han, Y., Wang, C., Song, S., Tian, Q., Huang, G.: Computation-efficient deep learning for computer vision: A survey. Cybernetics and Intelligence (2024)
- [17] Howard, A.G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., Andreetto, M., Adam, H.: Mobilenets: Efficient convolutional neural networks for mobile vision applications. arXiv preprint arXiv:1704.04861 (2017)
- [18] Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., Chen, L.-C.: Mobilenetv2: Inverted resid- uals and linear bottlenecks. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 4510–4520 (2018)
- [19] Howard, A., Sandler, M., Chu, G., Chen, L.-C., Chen, B., Tan, M., Wang, W., Zhu, Y., Pang, R., Vasudevan, V., et al.: Searching for mobilenetv3. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 1314–1324 (2019)

- [20] Qin, D., Leichner, C., Delakis, M., Fornoni, M., Luo, S., Yang, F., Wang, W., Banbury, C., Ye, C., Akin, B., et al.: Mobilenetv4-universal models for the mobile ecosystem. arXiv preprint arXiv:2404.10518 (2024)
- [21] Huang, G., Liu, S., Van der Maaten, L., Weinberger, K.Q.: Condensenet: An efficient densenet using learned group convolutions. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 2752–2761 (2018)
- [22] Yang, L., Jiang, H., Cai, R., Wang, Y., Song, S., Huang, G., Tian, Q.: Condensenet v2: Sparse feature reactivation for deep networks. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 3569–3578 (2021)
- [23] Zhang, X., Zhou, X., Lin, M., Sun, J.: Shufflenet: An extremely efficient convolutional neural network for mobile devices. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 6848–6856 (2018)
- [24] Ma, N., Zhang, X., Zheng, H.-T., Sun, J.: Shufflenet v2: Practical guidelines for efficient cnn architecture design. In: Proceedings of the European Conference on Computer Vision (ECCV), pp. 116–131 (2018)
- [25] Tan, M., Le, Q.: Efficientnet: Rethinking model scaling for convolutional neural networks. In: International Conference on Machine Learning, pp. 6105–6114 (2019). PMLR
- [26] Tan, M., Le, Q.: Efficientnetv2: Smaller models and faster training. In: International Conference on Machine Learning, pp. 10096–10106 (2021). PMLR
- [27] Han, K., Wang, Y., Tian, Q., Guo, J., Xu, C., Xu, C.: Ghostnet: More features from cheap operations. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 1580–1589 (2020)
- [28] Tang, Y., Han, K., Guo, J., Xu, C., Xu, C., Wang, Y.: Ghostnetv2: Enhance cheap operation with long-range attention. Advances in Neural Information Processing Systems 35, 9969– 9982 (2022)
- [29] Liu, Z., Hao, Z., Han, K., Tang, Y., Wang, Y.: GhostNetV3: Exploring the Training Strate- gies for Compact Models. arXiv e-prints, 2404–11202 (2024) https://arxiv.org/abs/2404. 11202 [cs.CV]. https://doi.org/10.48550/arXiv.2404.11202
- [30] Chen, Y., Dai, X., Chen, D., Liu, M., Dong, X., Yuan, L., Liu, Z.: Mobile-former: Bridging mobilenet and transformer. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 5270–5279 (2022)
- [31] Li, Y., Yuan, G., Wen, Y., Hu, J., Evangelidis, G., Tulyakov, S., Wang, Y., Ren, J.: Efficient- former: Vision transformers at mobilenet speed. Advances in Neural Information Processing Systems 35, 12934–12949 (2022)
- [32] Li, Y., Hu, J., Wen, Y., Evangelidis, G., Salahi, K., Wang, Y., Tulyakov, S., Ren, J.: Rethinking vision transformers for mobilenet size and speed. In: Proceedings of the IEEE International Conference on Computer Vision (2023)
- [33] Li, H., Kadav, A., Durdanovic, I., Samet, H., Graf, H.P.: Pruning filters for effi- cient convnets. In: International Conference on Learning Representations (2017). https: //openreview.net/forum?id=rJqFGTslg
- [34] Liu, Z., Li, J., Shen, Z., Huang, G., Yan, S., Zhang, C.: Learning efficient convolutional networks through network slimming. In: Proceedings of the IEEE International Conference on Computer Vision, pp. 2736–2744 (2017)
- [35] Liu, Z., Sun, M., Zhou, T., Huang, G., Darrell, T.: Rethinking the value of network pruning. In: International Conference on Learning Representations (2019). https://openreview.net/ forum?id=rJlnB3C5Ym
- [36] Frankle, J., Carbin, M.: The lottery ticket hypothesis: Finding sparse, trainable neu- ral networks. In: International Conference on Learning Representations (2019). https: //openreview.net/forum?id=rJl-b3RcF7
- [37] He, Y., Xiao, L.: Structured pruning for deep convolutional neural networks: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence 46(5), 2900–2919 (2024). https://doi.org/10.1109/TPAMI.2023.3334614
- [38] Rastegari, M., Ordonez, V., Redmon, J., Farhadi, A.: Xnor-net: Imagenet classification using binary convolutional neural networks. In: European Conference on Computer Vision, pp.

- 525–542 (2016). Springer
- [39] Hubara, I., Courbariaux, M., Soudry, D., El-Yaniv, R., Bengio, Y.: Binarized neural networks. In: Advances in Neural Information Processing Systems, vol. 29 (2016)
- [40] Jacob, B., Kligys, S., Chen, B., Zhu, M., Tang, M., Howard, A., Adam, H., Kalenichenko, D.: Quantization and training of neural networks for efficient integer-arithmetic-only inference. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 2704–2713 (2018)
- [41] Hinton, G., Vinyals, O., Dean, J.: Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531 (2015)
- [42] Gou, J., Yu, B., Maybank, S.J., Tao, D.: Knowledge distillation: A survey. International Journal of Computer Vision 129(6), 1789–1819 (2021)
- [43] Han, Y., Huang, G., Song, S., Yang, L., Wang, H., Wang, Y.: Dynamic neural networks: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence 44(11), 7436–7456 (2021)
- [44] Huang, G., Chen, D., Li, T., Wu, F., van der Maaten, L., Weinberger, K.: Multi-scale dense networks for resource efficient image classification. In: International Conference on Learning Representations (2018). https://openreview.net/forum?id=Hk2aImxAb
- [45] Li, H., Zhang, H., Qi, X., Yang, R., Huang, G.: Improved techniques for training adaptive deep networks. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 1891–1900 (2019)
- [46] Wang, Y., Huang, R., Song, S., Huang, Z., Huang, G.: Not all images are worth 16x16 words: Dynamic transformers for efficient image recognition. In: Advances in Neural Information Processing Systems (2021)
- [47] Han, Y., Han, D., Liu, Z., Wang, Y., Pan, X., Pu, Y., Deng, C., Feng, J., Song, S., Huang, G.: Dynamic perceiver for efficient visual recognition. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 5992–6002 (2023)
- [48] Sun, X., Panda, R., Chen, C.-F.R., Oliva, A., Feris, R., Saenko, K.: Dynamic network quantization for efficient video inference. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 7375–7385 (2021)
- [49] Bolukbasi, T., Wang, J., Dekel, O., Saligrama, V.: Adaptive neural networks for efficient inference. In: International Conference on Machine Learning, pp. 527–536 (2017). PMLR
- [50] Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., Dean, J.: Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. In: Inter- national Conference on Learning Representations (2017). https://openreview.net/forum? id=B1ckMDqlg
- [51] Mullapudi, R.T., Mark, W.R., Shazeer, N., Fatahalian, K.: Hydranets: Specialized dynamic architectures for efficient inference. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 8080–8089 (2018)
- [52] Veit, A., Belongie, S.: Convolutional networks with adaptive inference graphs. In: Proceed-ings of the European Conference on Computer Vision (ECCV), pp. 3–18 (2018)
- [53] Wang, X., Yu, F., Dou, Z.-Y., Darrell, T., Gonzalez, J.E.: Skipnet: Learning dynamic routing in convolutional networks. In: Proceedings of the European Conference on Computer Vision (ECCV), pp. 409–424 (2018)
- [54] Wu, Z., Nagarajan, T., Kumar, A., Rennie, S., Davis, L.S., Grauman, K., Feris, R.: Block- drop: Dynamic inference paths in residual networks. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 8817–8826 (2018)
- [55] Lin, J., Rao, Y., Lu, J., Zhou, J.: Runtime neural pruning. In: Advances in Neural Information Processing Systems, vol. 30 (2017)
- [56] Yang, L., Han, Y., Chen, X., Song, S., Dai, J., Huang, G.: Resolution adaptive networks for efficient inference. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 2369–2378 (2020)
- [57] Zhu, M., Han, K., Wu, E., Zhang, Q., Nie, Y., Lan, Z., Wang, Y.: Dynamic resolution network. In: Advances in Neural Information Processing Systems, vol. 34, pp. 27319–27330

(2021)

- [58] Meng, Y., Lin, C.-C., Panda, R., Sattigeri, P., Karlinsky, L., Oliva, A., Saenko, K., Feris, R.: Ar-net: Adaptive frame resolution for efficient action recognition. In: Computer Vision– ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part VII 16, pp. 86–104 (2020). Springer
- [59] Wang, Y., Chen, Z., Jiang, H., Song, S., Han, Y., Huang, G.: Adaptive focus for efficient video recognition. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 16249–16258 (2021)
- [60] Han, Y., Huang, G., Song, S., Yang, L., Zhang, Y., Jiang, H.: Spatially adaptive feature refinement for efficient inference. IEEE Transactions on Image Processing 30, 9345–9358 (2021)
- [61] Figurnov, M., Collins, M.D., Zhu, Y., Zhang, L., Huang, J., Vetrov, D., Salakhutdinov, R.: Spatially adaptive computation time for residual networks. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 1039–1048 (2017)
- [62] Xie, Z., Zhang, Z., Zhu, X., Huang, G., Lin, S.: Spatially adaptive inference with stochas- tic feature sampling and interpolation. In: Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part I 16, pp. 531–548 (2020). Springer
- [63] Hua, W., Zhou, Y., De Sa, C.M., Zhang, Z., Suh, G.E.: Channel gating neural networks. In: Advances in Neural Information Processing Systems, vol. 32 (2019)
- [64] Rao, Y., Zhao, W., Liu, B., Lu, J., Zhou, J., Hsieh, C.-J.: Dynamicvit: Efficient vision trans- formers with dynamic token sparsification. In: Advances in Neural Information Processing Systems, vol. 34, pp. 13937–13949 (2021)
- [65] Xu, Y., Zhang, Z., Zhang, M., Sheng, K., Li, K., Dong, W., Zhang, L., Xu, C., Sun, X.: Evo-vit: Slow-fast token evolution for dynamic vision transformer. In: Proceedings of the AAAI Conference on Artificial Intelligence, vol. 36, pp. 2964–2972 (2022)
- [66] Fayyaz, M., Koohpayegani, S.A., Jafari, F.R., Sengupta, S., Joze, H.R.V., Sommerlade, E., Pirsiavash, H., Gall, J.: Adaptive token sampling for efficient vision transformers. In: European Conference on Computer Vision, pp. 396–414 (2022). Springer
- [67] Yin, H., Vahdat, A., Alvarez, J.M., Mallya, A., Kautz, J., Molchanov, P.: A-vit: Adaptive tokens for efficient vision transformer. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10809–10818 (2022)
- [68] Bolya, D., Fu, C.-Y., Dai, X., Zhang, P., Feichtenhofer, C., Hoffman, J.: Token merging: Your vit but faster. In: International Conference on Learning Representations (2023). https: //openreview.net/forum?id=JroZRaRw7Eu
- [69] Fu, J., Zheng, H., Mei, T.: Look closer to see better: Recurrent attention convolutional neural network for fine-grained image recognition. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 4438–4446 (2017)
- [70] Zheng, H., Fu, J., Zha, Z.-J., Luo, J.: Looking for the devil in the details: Learning trilin- ear attention sampling network for fine-grained image recognition. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 5012–5021 (2019)
- [71] Yang, Z., Luo, T., Wang, D., Hu, Z., Gao, J., Wang, L.: Learning to navigate for fine-grained classification. In: Proceedings of the European Conference on Computer Vision (ECCV), pp. 420–435 (2018)
- [72] Ding, Y., Zhou, Y., Zhu, Y., Ye, Q., Jiao, J.: Selective sparse sampling for fine-grained image recognition. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 6599–6608 (2019)
- [73] Zhang, L., Huang, S., Liu, W., Tao, D.: Learning a mixture of granularity-specific experts for fine-grained categorization. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 8331–8340 (2019)
- [74] Zhou, B., Khosla, A., Lapedriza, A., Oliva, A., Torralba, A.: Learning deep features for discriminative localization. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 2921–2929 (2016)
- [75] Selvaraju, R.R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., Batra, D.: Grad-cam:

- Visual explanations from deep networks via gradient-based localization. In: Proceedings of the IEEE International Conference on Computer Vision, pp. 618–626 (2017)
- [76] He, B., Li, J., Zhao, Y., Tian, Y.: Part-regularized near-duplicate vehicle re-identification. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 3997–4005 (2019)
- [77] Ma, S., Fu, J., Chen, C.W., Mei, T.: Da-gan: Instance-level image translation by deep atten- tion generative adversarial networks. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 5657–5666 (2018)
- [78] Li, Y., Zhang, J., Zhang, J., Huang, K.: Discriminative learning of latent features for zero- shot recognition. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 7463–7471 (2018)
- [79] Zhu, Y., Xie, J., Tang, Z., Peng, X., Elgammal, A.: Semantic-guided multi-attention local- ization for zero-shot learning. In: Advances in Neural Information Processing Systems, vol. 32 (2019)
- [80] Chen, B., Deng, W.: Hybrid-attention based decoupled metric learning for zero-shot image retrieval. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 2750–2759 (2019)
- [81] Wu, P., Xie, S.: V\*: Guided visual search as a core mechanism in multimodal llms. arXiv preprint arXiv:2312.14135 (2023)
- [82] Liu, Z., Dong, Y., Rao, Y., Zhou, J., Lu, J.: Chain-of-spot: Interactive reasoning improves large vision-language models. arXiv preprint arXiv:2403.12966 (2024)
- [83] Shao, H., Qian, S., Xiao, H., Song, G., Zong, Z., Wang, L., Liu, Y., Li, H.: Visual cot: Unleashing chain-of-thought reasoning in multi-modal language models. arXiv preprint arXiv:2403.16999 (2024)
- [84] Luan, B., Feng, H., Chen, H., Wang, Y., Zhou, W., Li, H.: Textcot: Zoom in for enhanced multimodal text-rich image understanding. arXiv preprint arXiv:2404.09797 (2024)
- [85] Itti, L., Koch, C.: Computational modelling of visual attention. Nature reviews neuroscience 2(3), 194–203 (2001)
- [86] Wolfe, J.M., Horowitz, T.S.: What attributes guide the deployment of visual attention and how do they do it? Nature reviews neuroScience 5(6), 495–501 (2004)
- [87] Carrasco, M.: Visual attention: The past 25 years. Vision research 51(13), 1484–1525 (2011)
- [88] Henderson, J.M., Hayes, T.R.: Meaning-based guidance of attention in scenes as revealed by meaning maps. Nature human behaviour 1(10), 743–747 (2017)
- [89] Wolfe, J.M., Horowitz, T.S.: Five factors that guide attention in visual search. Nature Human Behaviour 1(3), 0058 (2017)
- [90] Itti, L., Koch, C., Niebur, E.: A model of saliency-based visual attention for rapid scene analysis. IEEE Transactions on pattern analysis and machine intelligence 20(11), 1254–1259 (1998)
- [91] Xu, K., Ba, J., Kiros, R., Cho, K., Courville, A., Salakhudinov, R., Zemel, R., Bengio, Y.: Show, attend and tell: Neural image caption generation with visual attention. In: International Conference on Machine Learning, pp. 2048–2057 (2015). PMLR
- [92] Yang, Z., He, X., Gao, J., Deng, L., Smola, A.: Stacked attention networks for image ques- tion answering. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 21–29 (2016)
- [93] Wang, X., Girshick, R., Gupta, A., He, K.: Non-local neural networks. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 7794–7803 (2018)
- [94] Woo, S., Park, J., Lee, J.-Y., Kweon, I.S.: Cbam: Convolutional block attention module. In: Proceedings of the European Conference on Computer Vision (ECCV), pp. 3–19 (2018)
- [95] Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., Guo, B.: Swin transformer: Hierarchical vision transformer using shifted windows. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 10012–10022 (2021)

- [96] Dong, X., Bao, J., Chen, D., Zhang, W., Yu, N., Yuan, L., Chen, D., Guo, B.: Cswin transformer: A general vision transformer backbone with cross-shaped windows. In: Pro- ceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 12124–12134 (2022)
- [97] Larochelle, H., Hinton, G.E.: Learning to combine foveal glimpses with a third-order boltzmann machine. Advances in neural information processing systems 23 (2010)
- [98] Mnih, V., Heess, N., Graves, A., et al.: Recurrent models of visual attention. In: Advances in Neural Information Processing Systems (NeurIPS) (2014)
- [99] Ba, J., Mnih, V., Kavukcuoglu, K.: Multiple object recognition with visual attention. In: International Conference on Learning Representations (ICLR) (2015)
- [100] Papadopoulos, A., Korus, P., Memon, N.: Hard-attention for scalable image classification. Advances in Neural Information Processing Systems 34, 14694–14707 (2021)
- [101] Elsayed, G., Kornblith, S., Le, Q.V.: Saccader: Improving accuracy of hard attention models for vision. Advances in Neural Information Processing Systems 32 (2019)
- [102] Wang, Y., Lv, K., Huang, R., Song, S., Yang, L., Huang, G.: Glance and focus: a dynamic approach to reducing spatial redundancy in image classification. In: Advances in Neural Information Processing Systems, vol. 33, pp. 2432–2444 (2020)
- [103] Rangrej, S.B., Srinidhi, C.L., Clark, J.J.: Consistency driven sequential transformers atten- tion model for partially observable scenes. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 2518–2527 (2022)
- [104] Wang, L., Zhang, X., Li, Q., Zhang, M., Su, H., Zhu, J., Zhong, Y.: Incorporating neuro-inspired adaptability for continual learning in artificial intelligence. Nature Machine Intelligence 5(12), 1356–1368 (2023)
- [105] Chattopadhay, A., Sarkar, A., Howlader, P., Balasubramanian, V.N.: Grad-cam++: Gen- eralized gradient-based visual explanations for deep convolutional networks. In: 2018 IEEE Winter Conference on Applications of Computer Vision (WACV), pp. 839–847 (2018). IEEE
- [106] Fu, R., Hu, Q., Dong, X., Guo, Y., Gao, Y., Li, B.: Axiom-based grad-cam: Towards accurate visualization and explanation of cnns. arXiv preprint arXiv:2008.02312 (2020)
- [107] Jiang, P.-T., Zhang, C.-B., Hou, Q., Cheng, M.-M., Wei, Y.: Layercam: Exploring hierar- chical class activation maps for localization. IEEE Transactions on Image Processing 30, 5875–5888 (2021)
- [108] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., Duchesnay, E.: Scikit-learn: Machine learning in Python. Journal of Machine Learning Research 12, 2825–2830 (2011)
- [109] Jaderberg, M., Simonyan, K., Zisserman, A., et al.: Spatial transformer networks. Advances in neural information processing systems 28 (2015)
- [110] Jang, E., Gu, S., Poole, B.: Categorical reparameterization with gumbel-softmax. In: International Conference on Learning Representations (2017)
- [111] Pan, B., Panda, R., Jiang, Y., Wang, Z., Feris, R., Oliva, A.: Ia-red<sup>2</sup> : Interpretability-aware redundancy reduction for vision transformers. Advances in Neural Information Processing Systems 34, 24898–24911 (2021)
- [112] Huang, G., Wang, Y., Lv, K., Jiang, H., Huang, W., Qi, P., Song, S.: Glance and focus networks for dynamic visual recognition. IEEE transactions on pattern analysis and machine intelligence 45(4), 4605–4621 (2022)
- [113] Schulman, J., Wolski, F., Dhariwal, P., Radford, A., Klimov, O.: Proximal policy optimiza-tion algorithms. arXiv preprint arXiv:1707.06347 (2017)
- [114] Schulman, J., Moritz, P., Levine, S., Jordan, M., Abbeel, P.: High-dimensional continu- ous control using generalized advantage estimation. In: Proceedings of the International Conference on Learning Representations (ICLR) (2016)
- [115] Cubuk, E.D., Zoph, B., Shlens, J., Le, Q.: Randaugment: Practical automated data augmen- tation with a reduced search space. In: Advances in Neural Information Processing Systems, pp. 18613–18624 (2020)

- [116] Zhang, H., Cisse, M., Dauphin, Y.N., Lopez-Paz, D.: mixup: Beyond empirical risk minimization. In: International Conference on Learning Representations (2018)
- [117] Yun, S., Han, D., Oh, S.J., Chun, S., Choe, J., Yoo, Y.: Cutmix: Regularization strat- egy to train strong classifiers with localizable features. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 6023–6032 (2019)
- [118] Zhong, Z., Zheng, L., Kang, G., Li, S., Yang, Y.: Random erasing data augmentation. In: Proceedings of the AAAI Conference on Artificial Intelligence, pp. 13001–13008 (2020)
- [119] Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., Wojna, Z.: Rethinking the inception archi- tecture for computer vision. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 2818–2826 (2016)
- [120] Huang, G., Sun, Y., Liu, Z., Sedra, D., Weinberger, K.Q.: Deep networks with stochastic depth. In: Proceedings of the European Conference on Computer Vision, pp. 646–661 (2016)
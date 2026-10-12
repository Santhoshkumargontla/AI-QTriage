# AI-QTriage: A Hybrid AI–Quantum Multimodal Framework for Explainable Injury Assessment and Emergency Support

**Authors:** N. Subbulakshmi, Dr. K. Venkatesh, Gontla Santhosh Kumar, Khambam Venkata Sai Girish, Dummu Rajesh  
**Affiliation:** Department of Computer Science and Engineering, Kalasalingam Academy of Research and Education, Krishnankoil, Tamil Nadu, India  
**Correspondence:** santoshkumargontla@gmail.com  
**Conference Paper ID:** ICICNIS-945  

---

## Abstract
AI-QTriage is a hybrid AI–quantum multimodal decision-support framework designed to provide objective preliminary injury risk assessment and coordinated emergency workflow assistance. The architecture integrates three complementary diagnostic modalities: computer vision (YOLO11 for visible lesion localization, EfficientNetV2-S for dermatological pattern classification, and ResNet34-U-Net for pixel-level wound segmentation), a structured patient questionnaire capturing pain intensity and kinetic mechanisms, and continuous 50 Hz smartphone accelerometer telemetry. Extracted features are mapped into a standardized 23-dimensional multimodal representation. Severity classification is executed using a primary classical Extreme Gradient Boosting (XGBoost) model alongside an experimental 4-qubit Variational Quantum Classifier (VQC) simulated via PennyLane's `default.qubit` engine. Interpretability is enforced via Grad-CAM heatmaps and SHAP Shapley attribution values, accompanied by deterministic kinetic safety thresholds and an emergency SOS protocol.

Experimental evaluation on held-out test splits demonstrates strong modular software and diagnostic integration: EfficientNetV2-S achieved a macro-F1 score of 94.34% (accuracy 96.62%, N = 473 photos); ResNet34-U-Net achieved a mean Dice similarity coefficient of 64.18% (IoU 51.78%, N = 69 pairs, fixed binarization threshold $\tau = 0.50$); and the sensor classifier achieved 96.34% accuracy (1,183/1,228 windows, macro-F1 89.85%, SisFall/UCI HAR). On real unconstrained trauma photographs, the primary production YOLO11 detector achieved mAP@50 = 34.54% and mAP@50–95 = 16.74% (Precision = 44.57%, Recall = 35.88%) on 436 held-out test images from an expanded 2,813-image corpus (1,935 train / 442 val / 436 test), and mAP@50 = 36.47% on a 123-image localized photographic baseline (83 train / 17 val / 23 test). On the 23-feature multimodal benchmark, classical XGBoost attained 83.33% accuracy (25/30, 95% Wilson Score CI: [66.44%, 92.66%], Macro-F1 = 0.7850, MCC = 0.7027), outperforming the simulated 4-qubit VQC (56.67%, 17/30, 95% CI: [39.20%, 72.62%], Macro-F1 = 0.4214, MCC = 0.1275 on the canonical grouped test split, and 53.33%, 16/30 on sequential evaluation). In independent modality ablation evaluations, combining questionnaire symptoms with kinetic sensor telemetry reached 90.00% accuracy (+33.33 percentage points over unimodal baselines at 56.67%), while the full multimodal framework achieved 83.33% accuracy with critical spatial lesion localization and wound area estimation. Because the multimodal dataset relies on 200 synthetic feature vectors with rule-derived research labels, AI-QTriage is presented strictly as an academic research prototype rather than a clinically validated medical device.

**Keywords—** Multimodal Deep Learning, Injury Assessment, Variational Quantum Classifier, YOLO11, EfficientNetV2, ResNet34-U-Net, Sensor Telemetry, Explainable AI, Emergency Triage.

---

## I. INTRODUCTION
Acute physical trauma resulting from vehicular collisions, falls, occupational accidents, and athletic injuries represents a substantial global health burden. During pre-hospital or remote emergencies, non-specialist individuals often lack immediate access to certified triage clinicians and struggle to assess injury severity or formulate appropriate initial responses. Conventional automated triage systems commonly rely on a single data modality—such as dermatological photographs—which fails to capture internal biomechanical stress, kinetic impact velocity, or subjective functional impairment.

To address these diagnostic limitations, AI-QTriage introduces a multi-tier framework combining computer vision, symptom questionnaires, and high-frequency inertial sensor telemetry. The architecture processes inputs through specialized model pipelines, fuses them into a canonical 23-dimensional feature space, and evaluates triage severity using classical gradient boosting while benchmarking a four-qubit variational quantum circuit. Explainability is integrated directly via spatial bounding boxes, pixel segmentation masks, Grad-CAM visual heatmaps, and SHAP feature attribution.

Crucially, the system is designed strictly as an academic research prototype and does not claim clinical clearance or autonomous diagnostic authority. The multimodal fusion evaluation operates on synthetic research vectors generated under rule-derived research categories. The VQC runs via classical CPU simulation on PennyLane without claiming quantum computational advantage. This paper establishes the feasibility of the end-to-end multimodal architecture, reconciles conflicting baseline metrics, and provides reproducible benchmarks across all subsystems.

---

## II. LITERATURE REVIEW
Multimodal biomedical AI has emerged as a promising paradigm for integrating heterogeneous health records, clinical imaging, and physiological signals [1], [2], [7]. In dermatological and wound analysis, deep convolutional architectures such as EfficientNetV2 have demonstrated superior parameter efficiency and high classification accuracy across diverse cutaneous lesions [3], [4]. For localized tissue boundary delineation, encoder-decoder architectures with skip connections such as U-Net [5] remain the gold standard for clinical image segmentation. Explainability in computer vision has been largely driven by gradient-weighted class activation mapping (Grad-CAM) [6], which provides transparent visual localization of decision-critical features.

In physical trauma analysis, wearable and smartphone-based inertial measurement units (IMUs) have been widely deployed for automated fall detection using established public benchmarks including SisFall [9], the UCI Human Activity Recognition (HAR) dataset [10], and the UP-Fall multimodal framework [11]. For structured tabular classification, XGBoost [8] offers robust gradient-boosted decision tree optimization amenable to Shapley additive explanations (SHAP) [12]. Recent explorations in quantum machine learning have evaluated parameterized quantum circuits and Variational Quantum Classifiers (VQCs) for binary and multiclass medical data [13]. However, systematic evaluations consistently demonstrate that current noisy intermediate-scale quantum (NISQ) algorithms simulated on classical hardware do not yet outperform classical ensembles. AI-QTriage builds on these foundations by integrating localized object detection, segmentation, whole-image classification, IMU kinetic modeling, and simulated quantum circuits into an integrated triage workflow.

---

## III. PROPOSED METHODOLOGY

### 3.1. Dataset Acquisition and Partitioning
AI-QTriage utilizes four distinct datasets to train and evaluate its component models:
1. **Photographic Trauma Detection Dataset:** The primary production object detector is trained on an expanded corpus of 2,813 photographic wound images (1,935 train / 442 val / 436 held-out test images) spanning cuts, bruises, abrasions, burns, lacerations, and surgical wounds. A localized benchmark dataset of 123 authentic photographic wound images comprising 412 annotated bounding-box instances is partitioned into 83 training images (288 wound, 12 bruise, 15 cut boxes), 17 validation images (70 wound, 1 bruise, 2 cut boxes), and 23 held-out test images (65 wound, 4 bruise, 5 cut boxes). An auxiliary baseline dataset of 126 synthetic drawings (87 train / 20 val / 19 test) was examined during preliminary training.
2. **Photographic Wound Classification Dataset:** Sourced from public Kaggle multiclass wound repositories comprising 2,000+ images across 8 categories (abrasion, bruise, burn, cut, laceration, wound, normal skin, and out-of-distribution reject controls). The independent held-out test split consists of 473 photographic images.
3. **Photographic Wound Segmentation Dataset:** Sourced from AZH, WSEG, and Medetec clinical repositories. To prevent data leakage, exact and near-duplicate images were removed via perceptual hashing (pHash), yielding 414 image-mask pairs divided under strict subject-aware isolation into 276 training, 69 validation, and 69 held-out test pairs.
4. **Real-World Sensor Telemetry Dataset:** Extracted from SisFall and UCI HAR tri-axial accelerometer recordings. Continuous 50 Hz signals were segmented into 2.5-second non-overlapping windows (125 time steps), yielding 6,505 windows partitioned under subject-level segregation into 4,247 training windows, 1,030 validation windows, and 1,228 held-out test windows (720 normal activity, 448 fall, 60 impact).
5. **Synthetic Multimodal Dataset:** A benchmark set of 200 vectors generated via controlled parametric distributions (140 training, 30 validation, 30 held-out test vectors). Labels (LOW, MODERATE, HIGH) were assigned using deterministic research rules based on a linear severity index.

### 3.2. YOLO11 Bounding-Box Detection and ROI Localization
Visible surface trauma is localized using an Ultralytics YOLO11n object detector [14], following recent clinical object-detection paradigms for localized injury identification [15]. Bounding boxes $B = [x_1, y_1, x_2, y_2]$ are predicted with confidence score $p$. Detections are retained when $p \ge \tau_{\text{det}}$, where the production inference threshold is set to $\tau_{\text{det}} = 0.25$. Rather than directly assigning the patient's global triage category, YOLO11 provides spatial bounding coordinates to crop the Region of Interest (ROI) and calculate the lesion bounding area ratio as an auxiliary feature.

### 3.3. EfficientNetV2-S Classification and Grad-CAM
Whole-image and cropped ROI classification is executed by an EfficientNetV2-S convolutional backbone. Softmax probabilities are computed across supported trauma classes. To eliminate severe out-of-distribution (OOD) collapse—wherein blank skin previously misclassified as swelling—the swelling head was deactivated and an explicit OOD reject class was trained. Interpretability is provided by Grad-CAM visual heatmaps, which are automatically withheld if image quality gates (contrast, blur, darkness) fail or if the classifier prediction lacks confidence.

### 3.4. ResNet34-U-Net Lesion Segmentation
Pixel-level lesion area estimation is performed by a ResNet34-U-Net architecture. Given input ROI $x$, the network generates continuous probability map $M_{\text{prob}}(x, y) \in [0, 1]$. Binary segmentation masks are generated using a fixed probability threshold $\tau = 0.50$:
$$M(x, y) = 1 \quad \text{if} \quad M_{\text{prob}}(x, y) \ge 0.50, \quad \text{else} \quad 0.$$
Segmentation quality is evaluated via the Dice similarity coefficient and Intersection over Union (IoU).

### 3.5. Symptom Questionnaire Processing and Pain Normalization
The clinical questionnaire collects contextual variables unavailable from images. Crucially, the raw patient-reported integer pain score $P \in \{0, 1, \dots, 10\}$ is normalized into continuous feature $F_7$ via:
$$F_7 = \frac{P}{10.0} \in [0.0, 1.0].$$
Injury mechanisms are one-hot encoded, and mobility and weight-bearing limitations are mapped to ordinal values: None = 0.0, Mild/Limited = 0.5, and Severe/Immobile = 1.0.

### 3.6. Sensor Telemetry Feature Extraction
Tri-axial accelerometer data $a(t) = (a_x, a_y, a_z)$ are validated for 40–50 Hz sampling continuity. Kinetic parameters are extracted across each time series:
1. **Peak Impact Acceleration:** $\text{peak\_g} = \max_t (\|a(t)\| / 9.80665)$ in $g$ units.
2. **Pre-Impact Velocity Change:** $\Delta v = \int \|a(t)\| dt$ over the 1.0 s window preceding peak impact.
3. **Physical Stabilization Duration:** Time required following impact for acceleration magnitude to remain settled below $1.5g$ for at least 0.5 s.
4. **Optical Lux Drop:** Binary flag indicating whether ambient illuminance dropped below 10.0 lux at impact.
A gradient-boosted decision tree classifies the window into `normal_activity`, `fall`, or `impact`.

### 3.7. Canonical 23-Dimensional Multimodal Feature Vector Schema and Triage Rules
Extracted features are fused into a canonical vector $F = [F_{\text{vision}}, F_{\text{questionnaire}}, F_{\text{sensor}}] \in \mathbb{R}^{23}$ following the Option-A missingness protocol (when a modality is absent, its presence indicator is 0.0 and its corresponding feature dimensions are represented as explicit zeros). The complete 23-feature breakdown is documented in Table I.

Ground truth triage severity labels on the synthetic research benchmark are assigned through a continuous Severity Index $S$ defined as:
$$S = 0.40 \times F_7 + 0.30 \times \min\left(\frac{F_{19}}{10.0}, 1.0\right) + 0.30 \times F_5,$$
where $F_7$ is normalized pain, $F_{19}$ is peak kinetic acceleration (normalized by 10g), and $F_5$ is the U-Net affected wound area ratio. Triage categories are established by deterministic cutoffs:
- **LOW:** $S < 0.32$
- **MODERATE:** $0.32 \le S < 0.58$
- **HIGH:** $S \ge 0.58$

Additionally, deterministic clinical red-flag overrides enforce patient safety: cases exhibiting active cutaneous bleeding ($F_{14} = 1.0$) with severe kinetic impact ($F_{19} \ge 6.0g$), or complete weight-bearing inability ($F_{16} = 1.0$) paired with acute pain ($F_7 \ge 0.80$), are unconditionally escalated to HIGH regardless of index $S$.

#### TABLE I: THE CANONICAL 23 MULTIMODAL FEATURE SCHEMA (F0–F22)
| Feature ID | Feature Name | Modality | Description & Range |
| :--- | :--- | :--- | :--- |
| F0 | vision_present | Vision | Binary presence indicator {0.0, 1.0} |
| F1 | prob_cut | Vision | EfficientNetV2 cut probability [0.0, 1.0] |
| F2 | prob_bruise | Vision | EfficientNetV2 bruise probability [0.0, 1.0] |
| F3 | prob_swelling | Vision | Deactivated legacy feature (hardcoded 0.0) |
| F4 | prob_other | Vision | Max probability across other injury/reject classes [0.0, 1.0] |
| F5 | affected_ratio | Vision | U-Net segmented lesion area / image area [0.0, 1.0] |
| F6 | questionnaire_present | Survey | Binary presence indicator {0.0, 1.0} |
| F7 | pain_level | Survey | Normalized pain level: P_raw / 10.0 in [0.0, 1.0] |
| F8 | mech_fall | Survey | One-hot mechanism: fall / accident {0.0, 1.0} |
| F9 | mech_impact | Survey | One-hot mechanism: direct blow / collision {0.0, 1.0} |
| F10 | mech_sports | Survey | One-hot mechanism: athletic / twisting trauma {0.0, 1.0} |
| F11 | mech_sharp | Survey | One-hot mechanism: sharp object laceration {0.0, 1.0} |
| F12 | mech_other | Survey | One-hot mechanism: other unlisted cause {0.0, 1.0} |
| F13 | direct_impact | Survey | User reported direct blunt impact {0.0, 1.0} |
| F14 | visible_bleeding | Survey | User reported active cutaneous bleeding {0.0, 1.0} |
| F15 | movement_limitation | Survey | Ordinal impairment: None=0.0, Limited=0.5, Immobile=1.0 |
| F16 | weight_bearing | Survey | Ordinal impairment: Full=0.0, Partial=0.5, Unable=1.0 |
| F17 | crack_pop | Survey | Audible/palpable cracking sensation {0.0, 1.0} |
| F18 | sensor_present | Sensor | Binary presence indicator {0.0, 1.0} |
| F19 | peak_g_force | Sensor | Peak acceleration magnitude (g units, >= 0.0) |
| F20 | delta_v | Sensor | Pre-impact velocity change (m/s, >= 0.0) |
| F21 | stabilization_time | Sensor | Post-impact kinetic stabilization duration (s, >= 0.0) |
| F22 | lux_drop | Sensor | Ambient illuminance occlusion drop < 10 lux {0.0, 1.0} |

### 3.8. Classical XGBoost vs. 4-Qubit Variational Quantum Classifier
The 23-dimensional feature vector is supplied to two parallel triage models:
1. **Classical XGBoost Severity Classifier:** An ensemble of gradient-boosted decision trees trained with balanced sample weighting. SHAP TreeExplainer computes feature contributions for interpretability.
2. **4-Qubit Variational Quantum Classifier (VQC):** A parameterized quantum circuit operating on PennyLane's `default.qubit` simulator. The 23-D vector is normalized via StandardScaler and projected to 4 dimensions via Principal Component Analysis (PCA, fitted exclusively on training data). The 4 components are embedded as quantum rotation angles via `AngleEmbedding(rotation='X')`, followed by two `StronglyEntanglingLayers`. Expectation values $\langle Z_i \rangle$ of the Pauli-Z operator are measured across all four qubits and transformed via linear softmax into 3 class probabilities. The VQC is evaluated strictly as an experimental comparative baseline and is completely isolated from main emergency triage decisions.

---

## IV. RESULTS AND DISCUSSION

### 4.1. Comparative Multimodal Benchmark: XGBoost vs. 4-Qubit VQC
Table II summarizes the performance of the classical XGBoost model and the experimental 4-qubit VQC evaluated on the identical held-out test split ($N = 30$ vectors). Classical XGBoost achieved an accuracy of 83.33% (25/30, 95% Wilson Score CI: [66.44%, 92.66%]), Macro Precision of 0.8222, Macro Recall of 0.7693, Macro-F1 of 0.7850, and Matthews Correlation Coefficient (MCC) of 0.7027. Calibration evaluation yielded a multiclass Brier score of 0.3045 and Expected Calibration Error (ECE) of 0.1183.

In contrast, the 4-qubit VQC achieved 56.67% accuracy (17/30, 95% Wilson Score CI: [39.20%, 72.62%], Macro-F1 = 0.4214, MCC = 0.1275) on the canonical grouped test split, and 53.33% accuracy (16/30, Macro-F1 = 0.3606, MCC = 0.0173) under sequential evaluation. While VQC demonstrates successful end-to-end quantum circuit integration within a multimodal data pipeline, its low MCC reflects a strong majority-class bias toward MODERATE (predicting MODERATE in 25 of 30 test cases). No quantum computational advantage is observed on classical simulators, confirming that classical boosting remains vastly superior for tabular triage.

#### TABLE II: MULTIMODAL CLASSIFICATION PERFORMANCE: CLASSICAL XGBOOST VS. 4-QUBIT VQC (HELD-OUT TEST SET N=30)
| Evaluation Metric | Classical XGBoost (Production) | 4-Qubit VQC (Experimental Simulation) |
| :--- | :--- | :--- |
| Correct Predictions / Total | 25 / 30 | 17 / 30 (Grouped) \| 16 / 30 (Sequential) |
| Overall Classification Accuracy | 83.33% (95% CI: [66.44%, 92.66%]) | 56.67% (Grouped) \| 53.33% (Sequential) |
| Macro-Averaged Precision | 0.8222 | 0.3833 (Grouped) \| 0.3667 (Sequential) |
| Macro-Averaged Recall | 0.7693 | 0.4706 (Grouped) \| 0.3556 (Sequential) |
| Macro F1-Score | 0.7850 | 0.4214 (Grouped) \| 0.3606 (Sequential) |
| Matthews Correlation Coefficient (MCC) | 0.7027 | 0.1275 (Grouped) \| 0.0173 (Sequential) |
| Brier Score (Multi-Class) | 0.3045 | 0.5123 |
| Expected Calibration Error (ECE, M=5) | 0.1183 | 0.1365 |
| Execution Hardware | Classical CPU / Server | PennyLane default.qubit (Classical CPU Sim) |
| Triage Authority | Primary Triage Engine | Isolated Benchmark Only (Weight = 0.0) |

### 4.2. Modality Contribution Ablation Study
To rigorously evaluate the individual and synergistic contributions of each input modality, a 7-configuration ablation experiment was executed on the untouched held-out test split ($N = 30$) using independently trained models for each active feature subset. As documented in Table III, unimodal classifiers exhibit constrained performance: Vision Only achieves 40.00% accuracy (MCC = -0.0610), Questionnaire Only achieves 56.67% accuracy (MCC = 0.3340), and Sensor Only achieves 56.67% accuracy (MCC = 0.3325). In contrast, combining Questionnaire symptoms with physical Sensor telemetry yields dominant classification performance: 90.00% accuracy (27/30), 0.8693 Macro-F1, and 0.8251 MCC. The Full Multimodal architecture (F0–F22) achieves 83.33% accuracy (25/30), 0.7850 Macro-F1, and 0.7027 MCC. While Questionnaire and Sensor data drive discrete tabular severity classification, Vision is critical to the clinical system by providing spatial lesion localization (YOLO11 34.54% mAP50 on 436 test photos) and wound surface area quantification (U-Net 64.18% Dice) for clinician review.

#### TABLE III: MODALITY CONTRIBUTION ABLATION: INDEPENDENT MODELS EVALUATED ON HELD-OUT TEST SPLIT (N=30)
| Modality Configuration | Active Features | Test Accuracy (%) | Macro F1 | MCC | Wilson 95% CI |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Vision Only** | F0–F5 (D=6) | 40.00% (12/30) | 0.2886 | -0.0610 | [24.59%, 57.68%] |
| **Questionnaire Only** | F6–F17 (D=12) | 56.67% (17/30) | 0.5036 | +0.3340 | [39.20%, 72.62%] |
| **Sensor Only** | F18–F22 (D=5) | 56.67% (17/30) | 0.5010 | +0.3325 | [39.20%, 72.62%] |
| **Vision + Questionnaire** | F0–F17 (D=18) | 53.33% (16/30) | 0.4732 | +0.2647 | [36.14%, 69.77%] |
| **Vision + Sensor** | F0–F5, F18–F22 (D=11) | 63.33% (19/30) | 0.5321 | +0.4079 | [45.51%, 78.13%] |
| **Questionnaire + Sensor** | F6–F22 (D=17) | **90.00%** (27/30) | **0.8693** | **+0.8251** | [74.38%, 96.54%] |
| **Full Multimodal (AI-QTriage)** | F0–F22 (D=23) | **83.33%** (25/30) | **0.7850** | **+0.7027** | [66.44%, 92.66%] |

### 4.3. Sensor Motion Classifier Performance and Class-Wise Breakdown
The sensor motion classifier was evaluated on 1,228 held-out 2.5-second windows derived from SisFall and UCI HAR under subject-level isolation. Overall accuracy reached 96.34% (1,183/1,228) with a Macro-F1 score of 89.85% and MCC of 0.9296. As detailed in Table IV, the model achieved high detection accuracy on Normal Activity (F1 = 98.82%, n = 720) and Fall events (F1 = 95.10%, n = 448). For the minority Impact class (n = 60), the model achieved 76.27% precision and 75.00% recall (F1 = 75.63%). Lower impact recall reflects the physical subtlety of stumble-and-recovery events compared to high-kinetic vertical drops.

#### TABLE IV: SENSOR MOTION EVENT CLASSIFIER PER-CLASS PERFORMANCE (N=1,228 WINDOWS)
| Motion Class | Test Support (N) | Precision (%) | Recall (%) | F1-Score (%) | Confusion Matrix Row |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Normal Activity** | 720 | 98.89% | 98.75% | 98.82% | [711 TP, 8 Fall, 1 Impact] |
| **Fall** | 448 | 94.89% | 95.31% | 95.10% | [8 Normal, 427 TP, 13 Impact] |
| **Impact** | 60 | 76.27% | 75.00% | 75.63% | [0 Normal, 15 Fall, 45 TP] |
| **Overall / Macro Avg** | 1,228 | 90.02% | 89.69% | 89.85% | Overall Accuracy: 96.34% |

### 4.4. Vision Pipeline Performance: YOLO11, EfficientNetV2-S, and ResNet34-U-Net
Vision model evaluation yielded distinct, task-specific findings:
1. **YOLO11 Detection:** On the primary production evaluation comprising 436 held-out test images from the expanded 2,813-image corpus, YOLO11 achieved mAP@50 = 34.54% and mAP@50–95 = 16.74% (Precision = 44.57%, Recall = 35.88%). On the localized 123-image photographic benchmark (23 test images, 74 bounding boxes), YOLO11 achieved mAP@50 = 36.47% and mAP@50–95 = 18.44% (Precision = 51.59%, Recall = 39.42%). Class-wise performance demonstrated highest accuracy for Abrasion (AP50 = 61.03%), Bruise (60.08%), and Cut (50.67%), while deeper chronic Wounds (19.36%) and Burns (16.07%) suffered from skin texture ambiguity. Laceration AP50 was bounded by extreme class sparsity (122 laceration samples vs. 1,999 wounds). These metrics illustrate that YOLO11 serves as an initial spatial Region-of-Interest (ROI) filter rather than an autonomous diagnostic classifier.
2. **EfficientNetV2-S Classification:** Evaluated across 473 held-out test photos, the model achieved an overall accuracy of 96.62% and Macro-F1 of 94.34% (Macro Precision: 94.26%, Macro Recall: 95.01%), successfully classifying cuts (F1 = 94.12%), bruises (93.33%), and wounds (98.48%) while maintaining zero out-of-distribution false positives.
3. **ResNet34-U-Net Segmentation:** Across 69 held-out photographic wound-mask pairs, ResNet34-U-Net achieved a Mean Dice score of 64.18% and Mean IoU of 51.78% (Pixel Precision = 69.30%, Pixel Recall = 72.33%, fixed binarization threshold $\tau = 0.50$). This performance aligns with published biomedical literature on non-standardized chronic wound photography, where baseline Dice scores typically range between 62% and 72% due to indistinct epithelial margins and variable illumination.

---

## V. CONCLUSION AND RESEARCH LIMITATIONS
AI-QTriage demonstrates a functional, multi-tier multimodal framework that successfully integrates computer vision, high-frequency kinetic telemetry, and symptom questionnaires into an explainable preliminary injury assessment pipeline. Rigorous empirical verification resolved prior manuscript contradictions, standardized real photographic detection metrics (YOLO11 mAP@50 = 34.54% on 436 test images; 36.47% on localized 23-test split), established a 7-configuration modality ablation showing that multimodal fusion achieves 83.33% accuracy (+26.66% to +43.33% over single vision and sensor modalities), and verified sensor classification at 96.34% accuracy on 1,228 real-world IMU windows.

The study also confirms that classical gradient-boosted trees (XGBoost accuracy 83.33%, MCC = 0.7027) remain vastly superior to small-scale simulated quantum classifiers (4-qubit VQC accuracy 56.67% grouped / 53.33% sequential, MCC = 0.1275 / 0.0173), establishing that current parameterized quantum circuits on classical simulators offer no computational advantage for tabular triage.

**Crucial Research Limitations:** AI-QTriage is strictly an academic proof-of-concept. The multimodal tabular dataset comprises 200 synthetic vectors evaluated on a small held-out test split ($N = 30$), and ground truth labels represent rule-derived research heuristics rather than certified clinical diagnoses. Future work will focus on acquiring large-scale, IRB-approved paired clinical trauma cohorts, expanding mobile-edge inference, and evaluating fault-tolerant quantum algorithms on physical QPU hardware.

---

## VI. REFERENCES
[1] F. Krones et al., 'Review of multimodal machine learning approaches in healthcare,' Information Fusion, vol. 104, p. 102148, 2024.  
[2] A. Kline et al., 'Multimodal machine learning in precision health: A scoping review,' npj Digital Medicine, vol. 5, no. 1, p. 171, 2022.  
[3] M. Tan and Q. V. Le, 'EfficientNetV2: Smaller models and faster training,' in Proc. 38th Int. Conf. Machine Learning (ICML), 2021, pp. 10096–10106.  
[4] N. Gessert et al., 'Skin lesion classification using ensembles of multi-resolution EfficientNets with meta data,' MethodsX, vol. 7, p. 100864, 2020.  
[5] O. Ronneberger, P. Fischer, and T. Brox, 'U-Net: Convolutional networks for biomedical image segmentation,' in Proc. MICCAI, 2015, pp. 234–241.  
[6] R. R. Selvaraju et al., 'Grad-CAM: Visual explanations from deep networks via gradient-based localization,' in Proc. IEEE ICCV, 2017, pp. 618–626.  
[7] J. N. Acosta et al., 'Multimodal biomedical AI,' Nature Medicine, vol. 28, no. 9, pp. 1773–1784, 2022.  
[8] T. Chen and C. Guestrin, 'XGBoost: A scalable tree boosting system,' in Proc. 22nd ACM SIGKDD, 2016, pp. 785–794.  
[9] A. Sucerquia, J. D. López, and J. F. Vargas-Bonilla, 'SisFall: A fall and movement dataset,' Sensors, vol. 17, no. 1, p. 198, 2017.  
[10] D. Anguita, A. Ghio, L. Oneto, X. Parra, and J. L. Reyes-Ortiz, 'A public domain dataset for human activity recognition using smartphones,' in Proc. 21th Int. European Symp. Artificial Neural Networks, Computational Intelligence and Machine Learning (ESANN), 2013, pp. 437–442.  
[11] M. Martínez-Villaseñor et al., 'UP-Fall detection dataset: A multimodal approach,' Sensors, vol. 19, no. 9, p. 1988, 2019.  
[12] S. M. Lundberg and S.-I. Lee, 'A unified approach to interpreting model predictions,' in Advances in Neural Information Processing Systems (NeurIPS 30), 2017, pp. 4765–4774.  
[13] V. Bergholm et al., 'PennyLane: Automatic differentiation of hybrid quantum-classical computations,' arXiv preprint arXiv:1811.04968, 2018.  
[14] G. Jocher and J. Qiu, 'Ultralytics YOLO11 (Version 11.0.0),' Ultralytics Inc., 2024. [Online]. Available: https://docs.ultralytics.com  
[15] M. H. Tusar et al., 'AI-powered image-based assessment of pressure injuries using YOLOv8 models,' Wound Repair and Regeneration, vol. 15, no. 5, pp. 271–288, 2026.  

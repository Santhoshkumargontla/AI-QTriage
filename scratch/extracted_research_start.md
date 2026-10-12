
                     AI-QTriage: A Hybrid AI–Quantum Multimodal Framework  for
           Explainable Injury Assessment and Emergency Support

N.Subbulakshmi
    Department of Computer Science and
Engineering,
   Kalasalingam Academy of Research & Education
  Krishnan Koil, Virudhunagar, Tamil Nadu, India
n.subbulakshmi@klu.ac.in
Dr. K.Venkatesh
 Department of Computer Science and
Engineering,
 Kalasalingam Academy of Research & Education
  Krishnan Koil, Virudhunagar, Tamil Nadu, India
 k.venkatesh@klu.ac.in 
Gontla Santhosh Kumar
      Department of Computer Science and Engineering,
           Kalasalingam Academy of Research &
 Education
                Krishnan Koil, Virudhunagar, Tamil Nadu, India
santoshkumargontla@gmail.com


Khambam Venkata Sai Girish
    Department of Computer Science and
Engineering,
   Kalasalingam Academy of Research & Education
  Krishnan Koil, Virudhunagar, Tamil Nadu, India
girishsaikhambam@gmail.com
     Dummu Rajesh
Department of Computer Science and
Engineering,
Kalasalingam Academy of Research & Education
 Krishnan Koil, Virudhunagar, Tamil Nadu, India
dummurajesh05@gmail.com

Abstract— AI-QTriage is a hybrid AI–quantum multimodal framework designed to support preliminary injury assessment and emergency assistance. The framework combines injury images, questionnaire responses, and smartphone sensor data to provide complementary information for injury-related assessment. Its vision pipeline uses YOLO11 for injury detection, EfficientNetV2 for image classification, and ResNet34-U-Net for injury-region segmentation. Grad-CAM and SHAP are incorporated to provide visual and feature-level explanations of model outputs. Sensor data are processed to identify motion events such as falls and impacts. Information from these sources is represented using a standardized 23-feature vector and classified using XGBoost, while a four-qubit Variational Quantum Circuit (VQC) is evaluated as an experimental alternative. A rule-based safety layer and an SOS workflow with optional Twilio test messaging are also included.
Evaluation was performed using the held-out test sets associated with the individual components. XGBoost achieved 83.33% accuracy (25/30) on the synthetic multimodal test set, compared with 53.33% (16/30) for the VQC. YOLO11 achieved 0.3647 mAP@50 and 0.1844 mAP@50–95, while EfficientNetV2 obtained a 0.9375 macro-F1. ResNet34-U-Net achieved a 0.6418 Dice score, and the sensor classifier correctly classified 1183 of 1228 samples (96.34%). These results demonstrate the feasibility of integrating heterogeneous AI components within a single research workflow. However, the multimodal dataset is synthetic, the individual models were evaluated using separate datasets, and no clinical validation was conducted. Therefore, AI-QTriage is presented as a research prototype rather than a clinical diagnostic system.
Keywords: AI-QTriage, Multimodal AI, Explainable AI, XGBoost, Quantum Machine Learning, YOLO11, Smartphone Sensing, Injury Assessment
I. INTRODUCTION
Injuries such as cuts, bruises, burns, and wounds can happen suddenly during sports, accidents, or everyday activities. When medical help is not immediately available, people may find it difficult to judge the seriousness of an injury or decide what to do first. Traditional assessment mainly depends on visual observation and information provided by the injured person, which can be subjective. Recent developments in AI, computer vision, and mobile sensing offer an opportunity to combine these sources for preliminary injury-assessment support.
To explore this possibility, AI-QTriage: A Hybrid AI–Quantum Multimodal Framework for Explainable Injury Assessment and Emergency Support integrates injury images, questionnaire responses, and smartphone sensor data. The vision module uses YOLO11 for injury detection, EfficientNetV2 for image classification, and ResNet34-U-Net for affected-region segmentation. Grad-CAM provides visual explanations for the classification results.
The questionnaire captures information such as pain severity, injury mechanism, bleeding, movement difficulty, weight-bearing ability, and crack or popping sensations. When available, smartphone sensor data provide additional motion and impact information, including acceleration and stabilization characteristics. These inputs are combined into a standardized 23-feature multimodal representation. XGBoost is used as the primary multimodal classifier, while a 4-qubit Variational Quantum Classifier (VQC) using the PennyLane default.qubit simulator is included as an experimental comparison. No quantum advantage is claimed.
To improve transparency, the system uses SHAP, Grad-CAM, YOLO bounding boxes, and U-Net segmentation masks to provide different forms of model evidence. A separate rule-based safety layer generates preliminary guidance from available case information. The prototype also includes local SOS simulation and controlled Twilio test messaging.
The system is implemented using Next.js for the frontend, FastAPI for the backend, and MongoDB for data storage. AI-QTriage is intended as a research prototype rather than a clinical diagnostic system. The individual models use separate datasets, while the multimodal experiments use synthetic data rather than genuinely paired patient records. Therefore, the reported results represent research-level performance, and further validation with larger, real-world, clinically annotated multimodal datasets is required.
II. LITERATURE REVIEW
Multimodal machine learning has become an important research direction in healthcare because medical information is often available in different forms. Krones et al. [1] reviewed multimodal healthcare approaches and highlighted the benefits of combining heterogeneous data, while also identifying challenges such as missing information, data heterogeneity, bias, and clinical validation. Kline et al. [2] similarly examined multimodal learning in precision health and emphasized issues related to data integration, generalization, privacy, and dataset availability. Acosta et al. [7] discussed multimodal biomedical AI and showed how combining imaging with structured biomedical information can provide a broader representation of patient conditions. Warner et al. [13] further reviewed multimodal learning in image-based and clinical biomedicine, highlighting opportunities as well as challenges in data integration and clinical translation.
Deep-learning methods have shown strong potential for medical and wound-image analysis. Gessert et al. [3] investigated multi-resolution EfficientNet models for skin-lesion classification, demonstrating the usefulness of convolutional networks for visual medical classification. Tan and Le [4] introduced EfficientNetV2, which focuses on improving training efficiency and model performance. These studies provide a foundation for using EfficientNetV2 as the image-classification component of AI-QTriage.
For image segmentation, Ronneberger et al. [5] introduced U-Net, an encoder-decoder architecture widely used for biomedical image segmentation. More recent wound-analysis studies have also demonstrated deep-learning approaches for automated wound segmentation. Scebba et al. [17] proposed a detect-and-segment approach for wound images, combining detection and segmentation, while Wang et al. [18] investigated fully automatic wound segmentation using deep convolutional networks. [16] further explored YOLO-based image analysis for pressure-injury assessment. These studies support the use of detection and segmentation as complementary components in AI-QTriage.
Explainability is particularly relevant when machine-learning models are applied to healthcare. Selvaraju et al. [6] introduced Grad-CAM, which highlights image regions that contribute to a deep-learning prediction. Van der Velden et al. [14] reviewed explainable AI techniques for medical image analysis and discussed challenges in interpreting deep-learning models. Champendal et al. [15] further reviewed interpretability and explainability methods in medical imaging and discussed the need for meaningful and reliable explanations. Alhazmi [20] discussed barriers to clinical translation of explainable AI, while Fanale et al. [21] proposed a quantitative framework for evaluating explanations in medical imaging. Čižmárová et al. [22] examined Grad-CAM for medical-image processing, and a recent systematic review [23] summarized major XAI techniques, applications, and challenges in medical imaging. Another review [24] examined XAI applications across imaging, diagnosis, and rehabilitation. These studies motivate the use of multiple explanation methods in AI-QTriage rather than relying only on a final prediction.
For structured-data prediction, Chen and Guestrin [8] introduced XGBoost, a scalable gradient-boosting method based on decision-tree ensembles. Its ability to process structured numerical features makes it suitable for the multimodal feature representation used in AI-QTriage. Lundberg and Lee [12] introduced SHAP, a feature-attribution method for interpreting machine-learning predictions. In the proposed framework, SHAP is used to explain the contribution of individual features to the XGBoost output.
Smartphone and wearable sensing provide another useful source of information for injury-related event analysis. Sucerquia et al. [9] developed the SisFall dataset for fall and movement recognition using sensor measurements. Micucci et al. [10] introduced UniMiB SHAR for human-activity and fall recognition using smartphone acceleration data. Martínez-Villaseñor et al. [11] developed the UP-Fall Detection Dataset using multiple sensing modalities for fall detection. Wertner et al. [19] also investigated mobile-phone sensing for fall detection. These datasets demonstrate the potential of motion and acceleration information for identifying abnormal physical events and support the inclusion of smartphone sensor data in AI-QTriage.
Research Gap
The existing literature covers multimodal healthcare learning [1], [2], [7], [13], medical-image classification [3], [4], segmentation and detection [5], [16]–[18], smartphone-based activity recognition [9]–[11], [19], and explainable AI [6], [12], [14], [15], [20]–[24]. However, these areas are generally studied as separate problems. 
Relatively few studies combine visible injury detection, image classification, segmentation, questionnaire information, smartphone motion data, multimodal fusion, explainability, and experimental quantum learning within a single framework.
AI-QTriage addresses this research gap by integrating these components into a unified prototype. Visual, questionnaire, and sensor information are combined into a 23-feature multimodal representation, which is evaluated using XGBoost and an experimental 4-qubit VQC. SHAP and Grad-CAM provide complementary explanations for the structured and image-based predictions. Since the current multimodal experiments use synthetic rather than genuinely paired patient data, the proposed system is positioned as a research proof of concept, with real-world and clinical validation left for future work.
III. METHODOLOGY
The proposed AI-QTriage system is designed as a multimodal research prototype that combines computer vision, questionnaire-based injury context, smartphone sensor information, classical machine learning, and an experimental quantum model. The purpose of the system is not to provide a medical diagnosis, but to investigate how multiple sources of injury-related evidence can be integrated into a unified decision-support pipeline.
The methodology consists of image acquisition, injury detection, image classification, injury segmentation, questionnaire feature extraction, sensor-data processing, multimodal feature fusion, classical and quantum prediction, explainable AI, and emergency-support logic.
3.1. Dataset Acquisition
Data is collected from multiple publicly available research sources, including wound-image datasets, injury-detection datasets, wound-segmentation datasets, and smartphone motion/fall datasets.
YOLO11: Uses annotated injury images with bounding-box labels for visible injury detection, including cut, bruise, and wound-related categories where supported by the training data.
EfficientNetV2: Uses a public wound-image classification dataset containing labeled injury categories for image-level classification.
ResNet34-U-Net: Uses paired wound images and pixel-level segmentation masks for identifying the affected injury region.
Sensor Classifier: Uses motion and fall-detection datasets containing accelerometer/gyroscope measurements for recognizing normal activity, falls, and impacts.
XGBoost and VQC: Use a 23-feature multimodal dataset combining vision, questionnaire, and sensor features. The current multimodal dataset is synthetic because no publicly available dataset provides all these modalities as genuinely paired records.
The multimodal fusion dataset contains 200 synthetically generated research records, divided into:
Training: 140 records
Validation: 30 records
Held-out test: 30 records
The LOW, MODERATE, and HIGH labels were generated using predefined rules applied to the available multimodal feature values and were used only for experimental model evaluation.
No genuinely paired patient records are available in this dataset. The multimodal labels represent rule-derived research categories and are not clinical ground-truth labels.
3.2. Data Preprocessing
Uploaded images are first validated for supported formats and basic image quality before being passed to the vision models.
Images are resized and normalized according to the input requirements of each model.
YOLO11 processes the image while preserving the spatial information required for bounding-box detection. 
EfficientNetV2 receives a resized image or image crop and applies the required normalization before classification. 
ResNet34-U-Net processes the image to generate a pixel-level probability mask. 
Sensor data is resampled into fixed-length windows and converted into numerical motion features. 
Questionnaire responses are converted into numerical and binary features. 
Missing modalities are handled using the predefined Option-A missingness strategy, where unavailable modality indicators and corresponding features are represented as zero.
Model Pipeline
Dataset
Samples
Target Classes
Train / Val / Test
Split
YOLO11
Roboflow Skin Injury & Wound Bounding Box Dataset
123 images
Cut, Bruise, Swelling, Abrasion, Laceration, Burn
86 / 17 / 20
Group-level
EfficientNetV2-S
MedWound Public Skin Lesion & Cutaneous Wound Dataset
200 images / 40 subjects
Cut, Bruise, Swelling, Other/Normal
140 / 30 / 30
Subject-level
ResNet34-U-Net
Kaggle Wound Boundary Mask Dataset (200 annotated image-mask pairs)
200 mask pairs
Background, Wound Region
140 / 30 / 30
Subject-level
Kinematic Sensor Model
SisFall + UCI HAR Motion Telemetry
200 series / 38 subjects
Low-G Motion, Moderate Impact, High-G Fall
140 / 24 / 36
Subject-level
XGBoost / 4-Qubit VQC
Synthesized Multimodal Baseline Matrix
200 records
LOW, MODERATE, HIGH
140 / 30 / 30
Stratified held-out
TABLE I : DATASET AND EXPERIMENTAL SPLIT SUMMARY 
The multimodal fusion dataset contains 200 synthetically generated research records and does not represent genuinely paired clinical patient records. The labels are rule-derived research categories and are not clinical ground truth.
The sensor classes used in this study were mapped from the available motion and fall activities in the source datasets into the three research categories: Low-G Motion, Moderate Impact, and High-G Fall.
3.3. YOLO-Based Injury Detection
YOLO11 is used to detect and localize visible injuries in the uploaded image.
For every detected region, the model produces:
A bounding box 
A predicted class label 
A confidence score 
The bounding box is represented as:
B=x1y1x2y2
where:
B= bounding box representing the detected injury region 
x1= x-coordinate of the left boundary of the bounding box 
y1= y-coordinate of the top boundary 
x2= x-coordinate of the right boundary 
y2= y-coordinate of the bottom boundary 
Thus, the four coordinates define the location and spatial extent of the detected injury within the image.
Confidence Threshold
YOLO11 produces a confidence score pfor each detection. A detection is retained when:
p≥T
where:
p= predicted confidence score for the detected object 
T= selected confidence threshold 
If:
p≥T
the detection is retained.
If:
p<T
the detection is discarded.
The current research configuration uses approximately:
T=0.25
The model is evaluated using mean Average Precision at IoU 0.50 (mAP@50) and mean Average Precision over IoU thresholds from 0.50 to 0.95 (mAP@50–95).
The current reported YOLO11 performance is:
mAP@50 = 0.3647 (36.47%) 
mAP@50–95 = 0.1844 (18.44%) 
These values represent performance on the available research evaluation dataset and should not be interpreted as clinical accuracy.
3.4. EfficientNetV2 Classification
EfficientNetV2 is used as the image-level classification component of AI-QTriage.
The model produces a score, or logit, for each possible injury class. These scores are converted into probabilities using the softmax function:
Py=k∣x=ezkj=1Kezj
where:
Py=k∣x= probability that input image xbelongs to class k
y= predicted injury class 
k= specific class being evaluated 
x= input image or image crop 
zk= model output score (logit) for class k
e= exponential function 
K= total number of injury classes 
j= index used to iterate through all classes 
∑= summation of the exponential scores for all classes 
The probabilities for all classes are then compared to determine the final predicted class.
Final Class Selection
y=argmaxkPy=k∣x
where:
y= final predicted injury class 
argmax= operation that selects the class having the highest value 
Py=k∣x= predicted probability for class k
k= class index 
Therefore, the class with the highest predicted probability is selected as the final classification output.
The current EfficientNetV2 evaluation achieved:
F1macro=0.9375
or 93.75% macro-F1 on its held-out research evaluation dataset.
This result represents performance on the available research dataset and does not represent clinical diagnostic accuracy.
3.5. ResNet34-U-Net-Based Injury Segmentation
A ResNet34-U-Net architecture is used to estimate the pixel-level region corresponding to the visible injury.
The model produces a probability mask:
Mxy∈01
where:
Mxy= predicted probability that pixel xybelongs to the injury region 
x= horizontal pixel coordinate 
y= vertical pixel coordinate 
01= probability range 
A value close to 1 indicates a higher probability that the pixel belongs to the predicted injury region, while a value close to 0 indicates a lower probability.
Binary Segmentation Mask
The probability mask is converted into a binary mask using a threshold τ:
Mxy=1,Mxy≥τ0,Mxy<τ
where:
Mxy= final binary segmentation value 
Mxy= predicted probability for pixel xy
τ= segmentation threshold 
1= pixel selected as part of the injury region 
0= pixel not selected as part of the injury region 
Dice Coefficient
Segmentation performance is evaluated using the Dice coefficient:
Dice=2∣P∩G∣∣P∣+∣G∣
where:
Dice= segmentation overlap score 
P= predicted segmentation mask 
G= ground-truth segmentation mask 
P∩G= intersection between predicted and ground-truth regions 
∣P∣= number of pixels in the predicted region 
∣G∣= number of pixels in the ground-truth region 
2= Dice overlap factor 
The Dice coefficient ranges from 0 to 1, where a higher value indicates greater overlap between the predicted and ground-truth regions.
The current test result is:
Dice=0.641 or 64.18%.
Intersection over Union
Intersection over Union (IoU) is also calculated as:
IoU=∣P∩G∣∣P∪G∣
where:
IoU= Intersection over Union 
P= predicted segmentation mask 
G= ground-truth segmentation mask 
P∩G= intersection of the predicted and ground-truth regions 
P∪G= union of the predicted and ground-truth regions 
∣P∩G∣= number of pixels in the intersection 
∣P∪G∣= number of pixels in the union 
3.6. Questionnaire Feature Extraction
The questionnaire collects contextual information that cannot reliably be obtained from an image alone.
The collected information includes:
Pain level 
Injury mechanism 
Direct impact 
Visible bleeding 
Movement limitation 
Ability to bear weight 
Crack/pop sensation 
Pain is represented using a numerical scale:
Pain∈010
where:
Pain= reported pain level 
0= minimum reported pain 
10= maximum reported pain 
Categorical injury mechanisms are converted into binary features:
mi∈01
where:
mi= binary feature representing the presence or absence of a particular injury mechanism 
1= condition is present 
0= condition is absent 
These questionnaire features are subsequently incorporated into the multimodal feature vector.
3.7. Sensor Data Processing
Smartphone or sensor telemetry is processed to obtain motion-related features associated with the injury event.
Acceleration magnitude is calculated as:
amag=ax2+ay2+az2
where:
amag= overall acceleration magnitude 
ax= acceleration measured along the x-axis 
ay= acceleration measured along the y-axis 
az= acceleration measured along the z-axis 
ax2,ay2,az2= squared acceleration values along the three axes 
√= square-root operation 
Extracted sensor features include peak g-force, acceleration-related characteristics, velocity change, stabilization time, and other available motion features.
The sensor classifier categorizes motion events into classes such as:
Normal activity 
Fall 
Impact 
The current sensor classifier reports:
Accuracy=11831228=96.34%
where:
1183= number of correctly classified test samples 
1228= total number of test samples 
96.34%= resulting classification accuracy 
This result is based on the available research sensor dataset and should not be considered clinical fall-detection accuracy.
3.8. Multimodal Feature Fusion
The outputs from the vision, questionnaire, and sensor pipelines are combined into a unified feature representation.
For multimodal evaluation, the vision features used in the 23-dimensional vector are generated from the corresponding vision-model outputs without using ground-truth labels from the held-out multimodal test set.
The current system uses 23 canonical features.
The fused representation is expressed as:
F=FvFqFs
where:
F= complete multimodal feature vector 
Fv= features obtained from the vision pipeline 
Fq= features obtained from the questionnaire 
Fs= features obtained from the sensor pipeline 
The resulting feature vector has 23 dimensions:
F∈R23
where:
F= multimodal feature vector 
R= set of real-valued numbers 
23= total number of canonical multimodal features 
Missing modalities are represented using the predefined missing-data strategy. This allows the available information to continue through the fusion pipeline without treating missing information as a genuine observation.
The current multimodal dataset consists of 200 synthetic research records, with 140 training samples, 30 validation samples, and 30 test samples.
3.9. XGBoost-Based Multimodal Classification
The 23-dimensional multimodal feature vector is provided to the XGBoost classifier.
XGBoost combines multiple decision trees to generate the final prediction.
Multiple Decision Trees
yi=k=1Kfkxi
where:
yi= predicted output for sample i
i= index of the input sample 
k= index of an individual decision tree 
K= total number of decision trees 
fk= output produced by the k-th decision tree 
xi= input feature vector for sample i
Σ = summation operator
The overall prediction is obtained by combining the contributions of the individual trees.

Optimization Objective
The XGBoost model minimizes an objective function represented as:
Obj=iLyiyi+kΩfk
where:
Obj= overall optimization objective 
i= sample index 
L= loss function measuring prediction error 
yi= actual research label for sample i
yi= predicted value for sample i
k= decision-tree index 
Ωfk= regularization term for the k-th tree 
∑k= sum of regularization terms across trees 
The current XGBoost model is trained using the synthetic multimodal research dataset.
Held-Out Performance
Accuracy=2530=83.33%
where:
25= correctly classified test samples 
30= total test samples 
The macro-F1 score is:
F1macro=0.7850
or 78.50%.
Because the multimodal dataset is synthetic and does not contain genuinely paired patient records, these results represent an experimental engineering baseline rather than clinical performance.
3.10. Experimental Quantum Classification
A four-qubit Variational Quantum Circuit (VQC) is included as an experimental comparison model.
The 23-dimensional fused features are first scaled and reduced to four dimensions using Principal Component Analysis (PCA) before being encoded into the quantum circuit.
The circuit uses:
AngleEmbedding for feature encoding 
StronglyEntanglingLayers for trainable quantum transformations 
PennyLane default.qubit for classical simulation 
The VQC is therefore executed using a classical quantum simulator, rather than physical quantum hardware.
Held-Out Performance
Accuracy=1630=53.33%
where:
16= correctly classified test samples 
30= total test samples 
The macro-F1 score is:
F1macro=0.4230 or 42.30%.
The VQC result reflects only the current four-qubit architecture, feature-reduction procedure, optimization settings, and synthetic dataset; it does not establish a general comparison between classical and quantum machine learning.
ID
Feature
Modality
Type
F0
vision_present
Vision
Binary
F1
prob_cut
Vision
Continuous [0,1]
F2
prob_bruise
Vision
Continuous [0,1]
F3
prob_swelling
Vision
Continuous [0,1]
F4
prob_other
Vision
Continuous [0,1]
F5
affected_ratio
Vision
Continuous [0,1]
F6
questionnaire_present
Questionnaire
Binary
F7
pain_level
Questionnaire
Continuous [0,1]
F8
mech_fall
Questionnaire
Binary
F9
mech_impact
Questionnaire
Binary
F10
mech_sports
Questionnaire
Binary
F11
mech_sharp
Questionnaire
Binary
F12
mech_other
Questionnaire
Binary
F13
direct_impact
Questionnaire
Binary
F14
visible_bleeding
Questionnaire
Binary
F15
movement_limitation
Questionnaire
Binary
F16
weight_bearing
Questionnaire
Binary
F17
crack_pop
Questionnaire
Binary
F18
sensor_present
Sensor
Binary
F19
peak_g_force
Sensor
Continuous
F20
delta_v
Sensor
Continuous
F21
stabilization_time
Sensor
Continuous
F22
lux_drop
Sensor
Binary
TABLE II : 23-DIMENSIONAL MULTIMODAL FEATURE SPACE 
The 23-dimensional feature vector combines six vision features, twelve questionnaire features, and five sensor features; the resulting vector is used directly by XGBoost and reduced to four dimensions using PCA before VQC classification.
3.11. Explainable AI
AI-QTriage uses multiple explainability techniques to provide evidence associated with model predictions.
Grad-CAM is applied to the EfficientNetV2 model to generate a visual representation of image regions influencing the classification output.
Grad-CAM Weights
The Grad-CAM weights are calculated as:
αkc=1Zij∂yc∂Aijk
where:
αkc= Grad-CAM importance weight for feature map kand class c
c= target class 
k= feature-map index 
Z= normalization factor based on the number of spatial positions 
i= spatial row index 
j= spatial column index 
yc= model output for target class c
Aijk= activation value at spatial position ijin feature map k
∂yc∂Aijk= gradient of the class output with respect to the feature-map activation 
∑ij= summation of gradients across spatial locations 
Final Grad-CAM Activation Map
The final activation map is represented as:
LGrad-CAMc=ReLUkαkcAk
where:
LGrad-CAMc= final Grad-CAM heatmap for class c
ReLU= Rectified Linear Unit operation 
αkc= importance weight for feature map k
Ak= activation feature map k
∑k=summation across feature maps 
c= target class 
The resulting heatmap highlights image regions associated with the classification output.
SHAP is used to explain the contribution of individual multimodal features to the XGBoost prediction. This provides feature-level explanations for the structured 23-dimensional representation and complements the visual explanations generated by Grad-CAM.
3.12. Safety and Emergency Support
The system contains a rule-based safety layer that operates separately from the machine-learning predictions.
Emergency-support conditions are evaluated using predefined combinations of available sensor and questionnaire evidence.
For example, one rule evaluates a high-impact condition using:
PeakG≥4.0g
together with:
StabilizationTime≥1.5s
where:
PeakG= maximum measured acceleration expressed in units of gravitational acceleration g
4.0g= predefined research threshold for a high-impact event 
StabilizationTime= time required for the motion signal to stabilize after the event 
1.5s= predefined research threshold for stabilization duration 
Severe pain combined with a reported crack/pop sensation can also contribute to the emergency-support decision according to the predefined safety rules.
The emergency workflow supports local SOS simulation and controlled test SMS functionality through Twilio. Twilio credentials are configured for test alerts only. No real emergency service is contacted by the system.
The safety layer is therefore intended to demonstrate emergency-support logic within the research prototype and should not be interpreted as a clinically validated emergency decision system.
These thresholds are engineering parameters selected for prototype evaluation and are not clinically validated injury thresholds.

Fig. 1. Block diagram of the proposed AI-QTriage framework.
IV. WORKING PRINCIPLE
AI-QTriage uses a multimodal workflow that combines an injury image, questionnaire responses, and smartphone sensor data when available. The image is analyzed using YOLO11 for injury detection, EfficientNetV2 for classification, and ResNet34-U-Net for segmentation. Questionnaire responses are converted into structured features, while sensor data provide motion- and impact-related information. Missing modalities are marked as unavailable rather than treated as actual observations.
The outputs from these modalities are combined into a standardized 23-dimensional feature vector. This vector is processed by XGBoost for the primary multimodal prediction, while a 4-qubit VQC using the PennyLane default.qubit simulator is included for experimental comparison. The VQC is not used to claim quantum advantage or clinical superiority.
For interpretability, SHAP explains the contribution of multimodal features to the XGBoost prediction, while Grad-CAM, YOLO bounding boxes, and U-Net masks provide visual evidence from the image models. A separate rule-based safety layer uses available questionnaire, visual, and sensor information to generate predefined safety guidance.
The final predictions, visual outputs, explanations, and safety information are combined into a structured case result and stored in MongoDB. The Next.js frontend retrieves these results through the FastAPI backend and presents them to the user in a unified case-results interface.
V. IMPLEMENTATION
AI-QTriage was developed as a full-stack multimodal research prototype using Next.js, FastAPI, Python-based ML models, and MongoDB. The system follows the workflow: input collection → preprocessing → model inference → feature fusion → prediction → explanation → safety assessment → result storage.
A. Frontend and Backend
The Next.js frontend allows users to create cases, upload injury images, enter questionnaire responses, provide sensor data, and view analysis results. FastAPI manages the REST API, coordinates the different models, processes inputs, generates predictions, and stores the results in MongoDB. This modular design allows individual components to be tested and updated independently.
B. Dataset and Model Integration
Different datasets are used for the vision, segmentation, sensor, and multimodal components because no suitable public dataset provides all required modalities in a paired format. The multimodal experiments use 200 synthetic records, divided into 140 training, 30 validation, and 30 test samples. These records are not paired patient data, and their labels are research-level rule-derived categories. Therefore, the multimodal results are treated as prototype benchmarks rather than clinical results.
C. Vision Models
YOLO11 detects visible injury regions and provides bounding boxes, classes, and confidence scores. A detection is retained when its confidence exceeds the configured threshold. The current model achieved 36.47% mAP@50 and 18.44% mAP@50–95.
EfficientNetV2 performs image classification after resizing and normalization. Its logits are converted into class probabilities using
Py=k∣x=ezkj=1Kezj
where zkis the logit for class kand Kis the number of classes. The model achieved a 93.75% macro-F1 score.
ResNet34-U-Net generates a pixel-level injury mask. Segmentation performance is measured using Dice and IoU, with the Dice coefficient given by
Dice=2∣P∩G∣∣P∣+∣G∣
where Pis the predicted region and Gis the ground-truth region. The current model achieved a 64.18% Dice score.
D. Questionnaire and Sensor Processing
Questionnaire responses include pain severity, injury mechanism, impact, bleeding, movement limitation, weight-bearing ability, and crack or popping sensations. Numerical and categorical responses are converted into machine-learning features.
The sensor module extracts motion-related characteristics such as acceleration and stabilization time. Acceleration magnitude is calculated as
amag=ax2+ay2+az2
where ax, ay, and azrepresent acceleration along the three axes. The sensor classifier achieved 96.34% accuracy (1183/1228).
E. Multimodal Fusion and Prediction
The processed information is combined into a standardized 23-dimensional feature vector:
F=FvFqFs
where Fv, Fq, and Fsrepresent visual, questionnaire, and sensor features, respectively.
XGBoost is used as the primary multimodal classifier. It achieved 83.33% accuracy and 78.50% macro-F1 on the synthetic test set. A 4-qubit VQC using PennyLane's default.qubit simulator is included for experimental comparison. It achieved 53.33% accuracy and 42.30% macro-F1. No quantum advantage is claimed.
F. Explainability and Safety
SHAP is used to show the contribution of individual multimodal features to the XGBoost prediction, while Grad-CAM highlights image regions influencing EfficientNetV2 classification. YOLO bounding boxes and U-Net masks provide additional spatial evidence.
After inference, a rule-based safety layer evaluates available injury information to generate predefined guidance. The system also provides local SOS simulation and controlled Twilio SMS testing when valid credentials are configured. It does not contact real emergency services.
G. Storage and Evaluation
Analysis results, including predictions, confidence values, visual outputs, explanations, and safety information, are stored in MongoDB and retrieved by the frontend for display. The models are evaluated using task-specific measures such as mAP, accuracy, macro-F1, Dice, IoU, precision, recall, MCC, and calibration metrics.
Because the components use different datasets and the multimodal experiments rely on synthetic rather than paired patient data, the reported results represent research and prototype performance, not clinical accuracy.
Metric
Classical XGBoost
Experimental 4-Qubit VQC
Correct Predictions
25 / 30
16 / 30
Accuracy
83.33%
53.33%
Macro Precision
0.822222
0.466184
Macro Recall
0.769281
0.432680
Macro F1-Score
0.785020
0.423016
Matthews Correlation Coefficient (MCC)
0.702692
0.095978
Expected Calibration Error (ECE)
0.118284
0.136455
Brier Score (Multi-class)
0.304532
0.512306
TABLE III : PERFORMANCE COMPARISON OF XGBOOST AND FOUR-QUBIT VQCVI. RESULTS AND DISCUSSION
The AI-QTriage prototype was evaluated across its major components, including image classification, image segmentation, object detection, sensor-based classification, and multimodal prediction. Since these components were developed for different tasks and evaluated using different datasets and protocols, their results are reported separately rather than being combined into a single overall accuracy. The reported values therefore represent the performance of individual research components and should not be interpreted as clinical performance.
6.1 Results
A. Overall Model Performance
The evaluated models showed different levels of performance according to their respective tasks. The Sensor Motion Classifier achieved the highest reported accuracy, with 96.34% (1183/1228) correct predictions on its held-out test set. The EfficientNetV2 image-classification model achieved a macro-F1 score of 93.75%, indicating strong classification performance on its evaluated image dataset.
For the multimodal classification experiment, XGBoost correctly classified 25 out of 30 samples, resulting in an accuracy of 83.33%. The experimental 4-qubit VQC correctly classified 16 out of 30 samples, corresponding to 53.33% accuracy. Since both models were evaluated on the same synthetic multimodal test set, their performance can be compared directly within this experiment.
The ResNet34-U-Net segmentation model achieved a Dice score of 64.18%, indicating moderate overlap between the predicted and reference injury regions. YOLO11 obtained 36.47% mAP@50 and 18.44% mAP@50–95, indicating that injury detection and localization remain challenging areas of the current system.
Component
Evaluation Metric
Result
EfficientNetV2
Macro-F1
93.75%
ResNet34-U-Net
Dice
64.18%
YOLO11
mAP@50
36.47%
YOLO11
mAP@50–95
18.44%
SensorMotion Classifier
Accuracy
96.34% (1183/1228)
XGBoost
Accuracy
83.33% (25/30)
XGBoost
Macro-F1
78.50%
VQC
Accuracy
53.33% (16/30)
VQC
Macro-F1
42.30%

TABLE IV :  COMPONENT-WISE EVALUATION RESULTS
These results demonstrate that the models have different strengths depending on the task. In particular, the classification and sensor components show stronger numerical performance than the current object-detection and segmentation components.
 
Fig. 2. Confusion Matrix of VQC
Fig. 3. SHAP Score

                        Fig. 4. Models Trained Information

Fig. 5. Comparison of XGBoost and VQC performance
B. Classification, Precision, and Recall
EfficientNetV2 achieved a 93.75% macro-F1 on the evaluated image dataset. For multimodal classification, XGBoost performed better than the VQC, with 82.22% macro precision, 76.93% macro recall, and 78.50% macro-F1. The VQC obtained 46.62% precision, 43.27% recall, and 42.30% macro-F1. XGBoost also recorded a higher MCC (0.7027) than VQC (0.0960). These results indicate that XGBoost is more effective for the current 23-feature multimodal setup. Since the fusion data are synthetic, these results should not be considered clinical performance.
C. Vision and Sensor Results
YOLO11 achieved 36.47% mAP@50 and 18.44% mAP@50–95, showing that injury detection and localization need further improvement. In individual tests, it detected selected injuries with confidence values of approximately 60% for a cut and 53.8% for a bruise; these examples are not benchmark results.
ResNet34-U-Net achieved a 64.18% Dice score, indicating moderate segmentation performance. The Sensor Motion Classifier achieved 96.34% accuracy (1183/1228), which was the highest reported accuracy among the evaluated components.
D. XGBoost and VQC Comparison
Both multimodal models were tested on the same 30 synthetic held-out samples. XGBoost achieved 83.33% accuracy, 78.50% macro-F1, and 0.7027 MCC, while VQC achieved 53.33% accuracy, 42.30% macro-F1, and 0.0960 MCC. XGBoost therefore performed approximately 30 percentage points better than VQC in the current experiment.
Fig. 4. Comparison of XGBoost and VQC performance
The VQC result reflects only the current 4-qubit architecture, training setup, and synthetic dataset and does not represent the overall capability of quantum machine learning.
E. Overall Interpretation
Overall, the results confirm that AI-QTriage can integrate vision, sensor, questionnaire, multimodal prediction, and explainability components into one workflow. EfficientNetV2 and the sensor classifier produced strong results, U-Net showed moderate segmentation performance, and YOLO11 remains an area for improvement. XGBoost currently provides better multimodal performance than VQC.
However, the results must not be combined into a single overall accuracy, as the models perform different tasks and use different datasets and metrics. The multimodal experiments also rely on synthetic rather than paired patient data. Therefore, the findings should be viewed as prototype-level research results, with further real-world and clinical validation required.
6.2 .   Discussion of Research Outcomes:
The results show that AI-QTriage can combine image analysis, questionnaire data, smartphone sensors, multimodal machine learning, and explainable AI in a single injury-assessment workflow. EfficientNetV2 achieved 93.75% macro-F1, while the Sensor Motion Classifier reached 96.34% accuracy on their respective datasets. ResNet34-U-Net obtained a 64.18% Dice score, whereas YOLO11 achieved 36.47% mAP@50 and 18.44% mAP@50–95, indicating that injury detection and localization still require improvement.
For multimodal prediction, XGBoost outperformed the 4-qubit VQC, achieving 83.33% accuracy and 78.50% macro-F1, compared with 53.33% accuracy and 42.30% macro-F1 for VQC. This comparison reflects the current experimental setup and does not imply that classical methods are generally superior to quantum approaches.
The use of SHAP, Grad-CAM, YOLO bounding boxes, and U-Net masks provides different forms of evidence to make the model outputs easier to interpret. However, the multimodal experiment uses 200 synthetic records with a 140/30/30 split, without genuinely paired patient data. Therefore, the results should be considered prototype-level research findings rather than clinical performance.
Overall, AI-QTriage demonstrates the feasibility of a multimodal injury-assessment framework, but further work is needed. Future studies should use larger, real-world, paired, clinically annotated datasets and focus on improving detection, segmentation, robustness, calibration, and external validation before considering real-world clinical deployment.
VII. CONCLUSION
AI-QTriage is a hybrid AI–quantum research prototype that combines injury images, questionnaire responses, smartphone sensor data, multimodal fusion, explainable AI, and rule-based safety support in a single workflow. The prototype achieved 93.75% macro-F1 with EfficientNetV2, 64.18% Dice with ResNet34-U-Net, 96.34% sensor accuracy, 83.33% accuracy with XGBoost, and 53.33% with the 4-qubit VQC. YOLO11 achieved 36.47% mAP@50 and 18.44% mAP@50–95, indicating that injury detection and localization still need improvement.
XGBoost performed better than the VQC on the current multimodal experiment. However, the reported results come from different datasets and evaluation settings and therefore cannot be combined into a single overall accuracy. In addition, the multimodal experiments use synthetic fused data rather than paired patient records, and the system has not been clinically validated.
Thus, AI-QTriage should be considered a proof-of-concept research system, not a clinical diagnostic or autonomous emergency-response tool. Future work will focus on real paired multimodal datasets, improved detection and segmentation, stronger evaluation methods, better robustness, and external validation to assess its reliability and practical potential.
VIII. REFERENCES 
[1] F. Krones, U. Marikkar, G. Parsons, A. Szmul, and A. Mahdi, “Review of multimodal machine learning approaches in healthcare,” Information Fusion, vol. 114, Art. no. 102690, 2025, doi: 10.1016/j.inffus.2024.102690 
[2] A. Kline, H. Wang, Y. Li, S. Dennis, M. Hutch, Z. Xu, F. Wang, F. Cheng, and Y. Luo, “Multimodal machine learning in precision health: A scoping review,” npj Digital Medicine, vol. 5, 2022, doi: 10.1038/s41746-022-00611-0.
[3] N. Gessert, M. Nielsen, M. Shaikh, R. Werner, and C. B. Schonlieb, “Skin lesion classification using ensembles of multi-resolution EfficientNets with metadata,” MethodsX, vol. 7, Art. no. 100864, 2020, doi: 10.1016/j.mex.2020.100864.
[4] M. Tan and Q. V. Le, “EfficientNetV2: Smaller models and faster training,” in Proc. 38th Int. Conf. Machine Learning (ICML), 2021, pp. 10096–10106.
[5] O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional networks for biomedical image segmentation,” in Medical Image Computing and Computer-Assisted Intervention (MICCAI), 2015, pp. 234–241, doi: 10.1007/978-3-319-24574-4_28.
[6] R. R. Selvaraju, A. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, “Grad-CAM: Visual explanations from deep networks via gradient-based localization,” in Proc. IEEE Int. Conf. Computer Vision (ICCV), 2017, pp. 618–626, doi: 10.1109/ICCV.2017.74.
[7] J. N. Acosta, G. J. Falcone, P. Rajpurkar, and E. J. Topol, “Multimodal biomedical AI,” Nature Medicine, vol. 28, pp. 1773–1784, 2022, doi: 10.1038/s41591-022-01981-2.
[8] T. Chen and C. Guestrin, “XGBoost: A scalable tree boosting system,” in Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining, 2016, pp. 785–794, doi: 10.1145/2939672.2939785.
[9] A. Sucerquia, J. D. López, and J. F. Vargas-Bonilla, “SisFall: A fall and movement dataset,” Sensors, vol. 17, no. 1, Art. no. 198, 2017, doi: 10.3390/s17010198.
[10] D. Micucci, M. Mobilio, and P. Napoletano, “UniMiB SHAR: A dataset for human activity recognition using acceleration data from smartphones,” Applied Sciences, vol. 7, no. 10, Art. no. 1101, 2017, doi: 10.3390/app7101101.
[11] M. Martínez-Villaseñor, H. Ponce, A. Velázquez-García, et al., “UP-Fall Detection Dataset: A multimodal approach,” Sensors, vol. 19, no. 9, Art. no. 1988, 2019, doi: 10.3390/s19091988.
[12] S. M. Lundberg and S.-I. Lee, “A unified approach to interpreting model predictions,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 30, 2017.
[13] E. Warner, J. Lee, W. Hsu, T. Syeda-Mahmood, C. E. Kahn, O. Gevaert, and A. Rao, “Multimodal machine learning in image-based and clinical biomedicine: Survey and prospects,” International Journal of Computer Vision, vol. 132, no. 9, pp. 3753–3769, 2024, doi: 10.1007/s11263-024-02032-8.
[14] B. H. M. van der Velden, H. J. Kuijf, K. G. A. Gilhuijs, and M. A. Viergever, “Explainable artificial intelligence (XAI) in deep learning-based medical image analysis,” Medical Image Analysis, vol. 79, Art. no. 102470, 2022, doi: 10.1016/j.media.2022.102470.
[15] M. Champendal, H. Müller, J. O. Prior, and C. Sá dos Reis, “A scoping review of interpretability and explainability concerning artificial intelligence methods in medical imaging,” European Journal of Radiology, vol. 169, Art. no. 111159, 2023, doi: 10.1016/j.ejrad.2023.111159.
 [16] M. H. Tusar, F. Fayyazbakhsh, N. Zendehdel, E. Mochalin, I. Melnychuk, L. Gould, and M. C. Leu, “AI-powered image-based assessment of pressure injuries using You Only Look Once version 8 (YOLOv8) models,” Advances in Wound Care, vol. 15, no. 5, pp. 271–288, 2026, doi: 10.1089/wound.2024.0245.
[17] G. Scebba, J. Zhang, S. Catanzaro, C. Mihai, O. Distler, M. Berli, and W. Karlen, “Detect-and-segment: A deep learning approach to automate wound image segmentation,” Informatics in Medicine Unlocked, vol. 29, Art. no. 100884, 2022, doi: 10.1016/j.imu.2022.100884. 
[18] C. Wang, D. M. Anisuzzaman, V. Williamson, M. K. Dhar, B. Rostami, J. Niezgoda, S. Gopalakrishnan, and Z. Yu, “Fully automatic wound segmentation with deep convolutional neural networks,” Scientific Reports, vol. 10, Art. no. 21897, 2020, doi: 10.1038/s41598-020-78799-w. 
[19] A. Wertner, P. Czech, and V. Pammer-Schindler, “An open labelled dataset for mobile phone sensing based fall detection,” in Proc. 12th EAI Int. Conf. Mobile and Ubiquitous Systems: Computing, Networking and Services (MOBIQUITOUS), 2015, doi: 10.4108/eai.22-7-2015.2260160.
[20] A. Alhazmi, “Explainable AI in medical imaging and diagnosis: A critical review of methods, clinical translation barriers, and an integrated evaluation framework,” AI in Clinical Medicine, vol. 2, Art. no. e36, 2026, doi: 10.14740/aicm36.
[21] R. Fanale, R. Caldelli, B. Martini, and F. Sciarrone, “A quantitative framework for explainability assessment in medical imaging,” Frontiers in Imaging, vol. 5, Art. no. 1846414, 2026, doi: 10.3389/fimag.2026.1846414.
[22] A. Čižmárová, K. Dostálová, P. Hrkut, and A. Poroshenko, “Explainable artificial intelligent method Grad-CAM in medical images processing,” Advanced Information Systems, 2026, doi: 10.20998/2522-9052.2026.2.09.
[23] F. Ahmed, N. S. Naz, S. Khan, A. U. Rehman, W. M. Ismael, and M. A. Khan, “Explainable artificial intelligence (XAI) in medical imaging: A systematic review of techniques, applications, and challenges,” BMC Medical Imaging, vol. 26, Art. no. 37, 2026, doi: 10.1186/s12880-025-02118-w. 
[24] A. Aravindkumar, M. Ramadoss, S. A. F. Ahmed, V. Sampath, and K. Lakshminarayanan, “Explainable AI in healthcare: A systematic review of XAI use cases in imaging, diagnostics, and rehabilitation,” Frontiers in Artificial Intelligence, vol. 9, Art. no. 1749527, 2026, doi: 10.3389/frai.2026.1749527.
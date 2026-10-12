
AI-QTriage: A Hybrid AI–Quantum Multimodal Framework  for 
Explainable Injury Assessment and Emergency Support


 DR.SUBBALAKSHMI
Department of Computer Science and
Engineering,
Kalasalingam Academy of Research & Education
Krishnan koil,Virudhnagar,India 
GMAIL@gmail.com
     GONTLA SANTHOSH KUMAR
 Department of Computer Science and
Engineering,
Kalasalingam Academy of Research & Education
Krishnan koil,Virudhnagar,India santoshkumargontla@gmail.com
KHAMBAM VENKATA SAI GIRISH
Department of Computer Science and Engineering,
Kalasalingam Academy of Research & Education
Krishnan koil,Virudhnagar,India, 
girishsaikhambam@gmail.com

DUMMU RAJESH
Department of Computer Science and
Engineering,
Kalasalingam Academy of Research & Education
Krishnan koil,Virudhnagar,India dummurajesh05@gmail.com





Abstract-AI-QTriage is a hybrid AI–quantum multimodal framework designed to support preliminary injury assessment and emergency assistance using multiple sources of information. The system combines computer vision, questionnaire responses, and smartphone sensor data to provide a more comprehensive assessment than an image-only approach. The vision pipeline uses YOLO11 for visible injury detection, EfficientNetV2 for injury classification, ResNet34-U-Net for injury-region segmentation, and Grad-CAM and SHAP for model explainability. Sensor data are processed to identify motion events such as normal activity, falls, and impacts. The extracted information is then converted into a 23-feature multimodal representation and analyzed using an XGBoost classifier, while a four-qubit variational quantum circuit is included as an experimental comparison model. The framework also provides rule-based safety guidance and an emergency SOS workflow with optional Twilio integration.
Experimental evaluation demonstrates that the individual components can be integrated into a complete working research prototype. The current XGBoost model achieved 83.33% accuracy and the experimental VQC achieved 53.33% accuracy on a 30-sample synthetic multimodal test set. YOLO11 achieved an mAP50 of 0.8358 and mAP50-95 of 0.7080 on its current evaluation dataset. Real-image verification also demonstrated successful detection of visible injuries such as bruises and cuts. However, the multimodal fusion data are synthetic and the system has not undergone clinical validation. Therefore, AI-QTriage is presented as an experimental research framework rather than a clinical diagnostic system.
SDG GoalsAI-QTriage is mainly aligned with SDG 3 – Good Health and Well-Being, as it aims to support preliminary injury assessment, safety guidance, and emergency assistance. It also supports SDG 9 – Industry, Innovation and Infrastructure by combining artificial intelligence, computer vision, smartphone sensor technology, and experimental quantum computing into an innovative healthcare-support framework. The project additionally relates to SDG 10 – Reduced Inequalities by exploring accessible smartphone-based technology for preliminary injury support.


Keywords:- AI-QTriage, Multimodal Artificial Intelligence, Quantum Machine Learning, YOLO11, EfficientNetV2, U-Net, Computer Vision, Smartphone Sensor Data, Injury Detection, Injury Assessment, Explainable AI, XGBoost, Variational Quantum Circuit, Emergency Support, Risk Assessment, Healthcare Technology
I. INTRODUCTION
Injuries such as cuts, bruises, burns, wounds, and other visible tissue damage can occur unexpectedly during sports, daily activities, road incidents, or household accidents. In many situations, people may not have immediate access to a healthcare professional and may be uncertain about the seriousness of an injury or the appropriate first response. Conventional injury assessment generally depends on visual inspection and information provided by the affected person, which can be subjective and difficult to perform consistently. Recent developments in artificial intelligence (AI), computer vision, and mobile sensing provide an opportunity to develop systems that can assist users by combining different sources of information during a preliminary assessment. However, a useful system should not rely on a single image or model because visible appearance alone may not provide sufficient information about the circumstances and potential risk associated with an injury. 
To address this problem, AI-QTriage: A Hybrid AI–Quantum Multimodal Framework for Explainable Injury Assessment and Emergency Support is proposed as an experimental research framework that combines visual information, questionnaire responses, and smartphone sensor data. The main idea is to treat injury assessment as a multimodal problem in which different sources provide complementary information. The image component analyzes visible injuries using multiple deep-learning models. YOLO11 is used for object detection to identify and localize visible injury regions using bounding boxes. EfficientNetV2 is used as an image-classification component, while ResNet34-U-Net is used to estimate the affected region through image segmentation. Grad-CAM provides visual explanations for the classification model, allowing the system to highlight image regions that contribute to its prediction. These models serve different purposes rather than performing the same task: YOLO11 focuses on where and what type of visible injury is detected, EfficientNetV2 focuses on image-level classification, and U-Net focuses on the approximate affected region.
The second source of information is a structured injury questionnaire. The questionnaire captures information that cannot be reliably obtained from an image, such as reported pain level, mechanism of injury, bleeding, movement limitation, ability to bear weight, and whether a crack or popping sound was experienced. These responses are converted into numerical features so that they can be combined with the visual and sensor information. This component is important because an injury may appear minor in an image while the reported symptoms or mechanism indicate a potentially higher level of concern.
The third component processes smartphone or sensor-based motion data. The system extracts kinetic characteristics such as acceleration and motion-related features and uses a sensor classification model to distinguish events such as normal activity, falls, and impacts when the required sensor information is available. This allows the framework to consider how an incident occurred rather than relying exclusively on the appearance of the resulting injury. In the current research implementation, sensor experiments and the multimodal fusion models include synthetic research data, and therefore their results are not considered evidence of clinical performance.
After the individual modalities are processed, their outputs are transformed into a standardized 23-feature multimodal representation. Missing modalities are handled explicitly rather than being treated as real observations. The resulting feature vector is supplied to an XGBoost multimodal classifier, which provides the primary machine-learning fusion experiment. An experimental four-qubit Variational Quantum Circuit (VQC) is also included to investigate the use of quantum machine-learning techniques within the same multimodal framework. The VQC operates using the PennyLane default.qubit simulator and is therefore a classical simulation rather than execution on quantum hardware. Its role in the current system is experimental and comparative; it is not used as a clinically validated decision-making component.
The framework also incorporates an explainability and safety layer. Model outputs such as YOLO bounding boxes, segmentation masks, Grad-CAM visualizations, and SHAP-based feature contributions can provide evidence about how different components reached their outputs. Based on the available information, the system applies deterministic safety rules to generate preliminary guidance and can initiate an SOS workflow. The emergency component supports local simulation and can optionally integrate with Twilio for SMS communication when correctly configured. Importantly, the system distinguishes between a local SOS simulation and an actual SMS transmission so that an unconfigured messaging service is not represented as a successful emergency notification.
From an implementation perspective, AI-QTriage is organized as a web-based client-server application. The Next.js frontend provides the user interface for creating cases, uploading images, entering questionnaire information, providing sensor data, and viewing analysis results. The FastAPI backend manages case processing, model inference, feature fusion, safety logic, reports, and communication with the database. MongoDB stores the case
information and structured model outputs. This architecture separates the user interface from the machine-learning and data-processing components and allows the individual models to be evaluated independently as well as through the complete inference pipeline.
The primary objective of AI-QTriage is not to replace medical professionals or provide a definitive medical diagnosis. Instead, it investigates whether multiple AI-based modalities can be integrated into a single explainable framework for preliminary injury assessment and emergency support. The current evaluation demonstrates the feasibility of the software and multimodal architecture, while also exposing important limitations related to dataset size, synthetic multimodal data, model generalization, sensor-data provenance, and the absence of clinical validation. Consequently, AI-QTriage is presented as a research prototype, and its experimental model results should not be interpreted as clinical accuracy or evidence that the system is suitable for medical deployment.
II. LITERATURE REVIEW
Krones et al. studied the use of multimodal machine learning in healthcare, focusing on how different forms of medical information can be combined within a single artificial intelligence system. Their work explains that healthcare data can include medical images, clinical records, text, and time-series information, and that combining these sources can provide a more complete representation of a patient's condition. The study also points out challenges such as missing data, data heterogeneity, bias, and difficulties in validating multimodal systems. This research is relevant to AI-QTriage because the proposed system combines injury images, questionnaire information, and smartphone sensor data to support injury assessment.[1]
Kline et al. investigated multimodal machine learning approaches used in precision health through a review of existing research. The study showed that combining multiple healthcare data sources can improve the representation of complex patient information compared with relying on a single modality. However, the authors identified important limitations involving dataset size, data alignment, generalization, privacy, and clinical validation. These findings are relevant to AI-QTriage because its architecture is designed to combine visual, questionnaire, and sensor features while explicitly handling situations where one of the modalities is unavailable.[2]
A Study on multiclass wound-image classification investigated the use of deep convolutional neural networks for automatically identifying different types of wounds from photographs. The research demonstrated the potential of deep-learning-based image classification for assisting wound assessment and showed that visual characteristics can be used to distinguish between different wound categories. This work provides a foundation for the image-classification component of AI-QTriage, where EfficientNetV2 is used to analyse injury images and estimate the corresponding injury category.[3]
Tan and Le proposed EfficientNetV2 as an improved convolutional neural-network architecture designed to achieve efficient training while maintaining strong image-classification performance. The architecture improves the relationship between computational cost, model size, and accuracy through optimized scaling and training strategies. EfficientNetV2 is therefore suitable for applications where image classification needs to be performed using a relatively efficient deep-learning architecture. AI-QTriage applies EfficientNetV2 as an image-classification model alongside YOLO11, allowing the system to obtain classification information separately from object localization.[4]
Ronneberger et al. introduced U-Net for biomedical image segmentation. The architecture uses an encoder-decoder structure with skip connections to preserve spatial information while extracting meaningful image features. U-Net became widely used for biomedical segmentation because it can generate detailed pixel-level predictions from medical images. This approach is relevant to AI-QTriage because the system uses a ResNet34-U-Net model to estimate the affected region of an injury and provide segmentation information in addition to the bounding boxes produced by the object-detection model.[5]
Selvaraju et al. introduced Grad-CAM, a method for generating visual explanations from convolutional neural networks. Grad-CAM uses gradients from the target class to identify image regions that contribute to a model's prediction. This provides researchers with a way to inspect the visual evidence associated with an image classification result instead of relying only on the final predicted class. AI-QTriage incorporates Grad-CAM to produce a visual heatmap associated with the EfficientNetV2 classification output, supporting interpretability of the image-analysis pipeline.[6]
Research on explainable artificial intelligence in medical imaging has examined techniques that make deep-learning predictions easier to interpret. Such methods are particularly important in healthcare because users need to understand the evidence associated with an automated prediction rather than receiving only a classification label. The literature discusses approaches such as Grad-CAM and other attribution techniques for identifying influential image regions. AI-QTriage follows this research direction by providing Grad-CAM visualizations for image-based predictions and SHAP-based feature importance for its structured multimodal model.[7]
 Acosta et al. investigated multimodal biomedical artificial intelligence and described how different biomedical data types can be integrated to address complex healthcare problems. Their work emphasized that medical information can come from multiple sources, including images, clinical information, and other structured or physiological measurements. The study also highlighted challenges related to data quality, interoperability, bias, and clinical translation. This research closely relates to AI-QTriage because the proposed framework combines multiple sources of injury-related evidence instead of depending solely on image classification.[8]
Chen and Guestrin introduced XGBoost, a scalable gradient-boosting framework designed for efficient learning from structured data. XGBoost uses an ensemble of decision trees and has become widely used for classification and regression tasks involving tabular features. Its ability to handle structured numerical information makes it suitable for multimodal feature-fusion applications. In AI-QTriage, XGBoost receives a 23-feature representation generated from image, questionnaire, and sensor information and produces the main experimental multimodal classification result.[9]XGBoost) technique is exploited for classification.[9]

Sucerquia et al. introduced the SisFall dataset for research into fall and movement detection using wearable sensor measurements. The dataset contains activities of daily living together with different types of falls and provides acceleration-related information that can be used to distinguish abnormal movement patterns. The study demonstrated the usefulness of sensor-based machine learning for automatic fall detection. This research is relevant to AI-QTriage because the system analyses smartphone or sensor telemetry to extract kinetic characteristics associated with falls and impacts.[10]
Micucci et al. developed the UniMiB SHAR dataset for human-activity and fall recognition using acceleration data collected from smartphones. The dataset includes both activities of daily living and different fall events, providing a useful benchmark for smartphone-based activity recognition. The study also demonstrates the importance of evaluating models using appropriate subject-independent data splits to reduce overly optimistic performance estimates. AI-QTriage uses a similar concept by processing smartphone motion information and classifying events such as normal activity, fall, and impact.[11]
Martínez-Villaseñor et al. introduced the UP-Fall Detection Dataset, which provides multimodal recordings for studying human activity and fall detection. The dataset combines information from different sensing sources and demonstrates how multiple signals can contribute to identifying potentially dangerous events. This work supports the use of sensor fusion for emergency-event recognition. AI-QTriage extends this general research direction by combining sensor-derived motion information with visual injury evidence and questionnaire responses within a multimodal injury-support framework.[12]
III. PROPOSED METHODOLOGY
The proposed AI-QTriage system is designed as a multimodal research prototype that combines computer vision, questionnaire-based clinical context, smartphone sensor information, classical machine learning, and an experimental quantum model. The purpose of the system is not to provide a medical diagnosis, but to demonstrate how multiple sources of injury-related evidence can be integrated into a single decision-support pipeline. The methodology consists of image acquisition, injury detection, image classification, injury segmentation, questionnaire processing, sensor analysis, multimodal feature fusion, classical and quantum prediction, explainability, and emergency-support logic. 
3.1. Dataset Acquisition
Data is collected from multiple publicly available sources, including wound-image datasets, injury-detection datasets, wound-segmentation datasets, and smartphone motion/fall datasets. 
YOLO11: Uses annotated injury images with bounding-box labels for visible injury detection, including cut, bruise, and wound-related categories where supported by the training data. 
EfficientNetV2: Uses a public wound-image classification dataset containing labeled injury categories for image-level classification. 
ResNet34-U-Net: Uses paired wound images and pixel-level segmentation masks for identifying the affected injury region. 
Sensor Classifier: Uses motion and fall-detection datasets containing accelerometer/gyroscope measurements for recognizing normal activity, falls, and impacts. 
XGBoost and VQC: Use a 23-feature multimodal dataset combining vision, questionnaire, and sensor features. The current multimodal training dataset is synthetic, because no publicly available dataset provides all these modalities as genuinely paired records.
3.2. Data Preprocessing
Uploaded images are validated for supported formats and basic image quality before being passed to the vision models. 
Images are resized and normalized according to the input requirements of each model. 
YOLO11 processes the image while preserving the spatial information required for bounding-box detection. 
EfficientNetV2 receives a resized image/crop and applies the required normalization before classification. 
U-Net processes the image to generate a pixel-level probability mask. 
Sensor data is resampled into fixed-length windows and converted into numerical motion features. 
Questionnaire responses are converted into numerical and binary features. 
Missing modalities are handled using the predefined Option-A missingness strategy, where unavailable modality indicators and corresponding features are represented as zero.
3.3.       YOLO Based Injury Detection
YOLO11 is used to detect and localize visible injuries in the uploaded image. 
For every detected region, the model produces a bounding box, class label, and confidence score. 
The bounding-box representation is: 
B=x1y1x2y2
A detection is retained when its confidence satisfies: 
p≥T
where pis the predicted confidence and Tis the selected detection threshold.
The current research configuration uses approximately 0.25 as the detection threshold. 
The model is evaluated using mAP50 and mAP50-95. 
Current reported YOLO11 performance: 
mAP50 = 83.58% 
mAP50-95 = 70.80% 
These values represent performance on the available research dataset and should not be interpreted as clinical accuracy.
3.4. EfficientNetV2 Classification
EfficientNetV2 is used to classify the overall injury category from the input image. 
The model produces class probabilities using the Softmax function: 
Py=k∣x=ezkj=1Kezj
The final class is selected using: 
y=argmaxkPy=k∣x
Model performance is evaluated using accuracy, precision, recall, and macro-F1 score. 
Current reported performance: 
Macro-F1=93.75%
The result is based on the available held-out research dataset and does not represent clinical diagnostic accuracy.
3.5. U-Net-Based Injury Segmentation
A ResNet34-U-Net architecture is used to identify the pixel-level region corresponding to the visible injury. 
The model generates a probability mask: 
Mxy∈01
The probability mask is converted into a binary mask using a threshold: 
Mxy=1,Mxy≥τ0,Mxy<τ
Segmentation performance is evaluated using the Dice coefficient: 
Dice=2∣P∩G∣∣P∣+∣G∣
Intersection over Union is also calculated: 
IoU=∣P∩G∣∣P∪G∣
Current reported test Dice score: 
 3.6. Questionnaire Feature Extraction
The questionnaire collects contextual information related to the injury. 
The collected information includes pain level, injury mechanism, direct impact, visible bleeding, movement limitation, weight-bearing ability, and crack/pop sensation. 
Pain is represented on a numerical scale: 
Pain∈010
Categorical injury mechanisms are converted into binary features: 
mi∈01
These features are subsequently included in the multimodal feature vector. 
3.7. Sensor Data Processing
Smartphone or sensor telemetry is processed to obtain motion-related features. 
Acceleration magnitude is calculated as: 
amag=ax2+ay2+az2
Extracted features include peak g-force, acceleration-related characteristics, velocity change, stabilization time, and other available motion features. 
The sensor classifier categorizes the motion event into classes such as normal activity, fall, and impact. 
The current sensor classifier reports: 
Accuracy=2836=77.78%
This result is based on the available research sensor dataset and should not be considered clinical fall-detection accuracy.
3.8. Multimodal Feature Fusion
Features obtained from the vision, questionnaire, and sensor pipelines are combined into a unified feature vector. 
The current system uses 23 canonical features. 
F=FvFqFs
where:
Fv= vision features 
Fq= questionnaire features 
Fs= sensor features 
The resulting feature vector is: 
F∈R23
Missing modalities are represented using the predefined missing-data strategy so that the remaining available information can still be processed.
   3.9. XGBoost-Based Multimodal Classification
The 23-dimensional fused feature vector is provided to the XGBoost classifier. 
The model combines multiple decision trees: 
yi=k=1Kfkxi
The optimization objective is: 
Obj=iLyiyi+kΩfk
The current XGBoost model is trained using synthetic multimodal research data. 
Current held-out performance: 
Accuracy=2530=83.33%
Macro-F1: 
F1macro=78.50%
Since the multimodal dataset is synthetic and not composed of paired patient records, these values represent an experimental engineering baseline.
   3.10. Experimental Quantum Classification
A 4-qubit Variational Quantum Circuit (VQC) is included as an experimental comparison model. 
The fused features are scaled and reduced before quantum encoding. 
The circuit uses AngleEmbedding followed by StronglyEntanglingLayers. 
The quantum model operates using the PennyLane default.qubit classical simulator. 
Current test performance is: 
Accuracy=1630=53.33%
Macro-F1: 
F1macro=42.30%
The VQC is therefore treated as an experimental research component, and no quantum advantage is claimed. 
   3.11. Explainable AI
Grad-CAM is applied to the EfficientNetV2 model to generate a visual representation of the regions influencing the classification. 
The Grad-CAM weights are calculated as: 
αkc=1Zij∂yc∂Aijk
The final activation map is: 
LGrad-CAMc=ReLUkαkcAk
SHAP is used to explain the contribution of individual multimodal features to the XGBoost prediction. 
3.12. Safety and Emergency Support
The system contains a rule-based safety layer that operates separately from the machine-learning predictions. 
Emergency conditions are evaluated using predefined combinations of sensor and questionnaire evidence. 
For example, the system can evaluate: 
PeakG≥4.0g
together with:
StabilizationTime≥1.5s
Severe pain combined with a reported crack/pop can also contribute to the emergency-support decision. 
The current Twilio integration is not configured, so the emergency workflow provides local simulation rather than claiming real SMS delivery.
3.13. Final Output Generation
The outputs from all processing stages are stored in MongoDB. 
The stored results include: 
YOLO detections and bounding boxes 
EfficientNetV2 classification 
U-Net segmentation 
Grad-CAM visualization 
Questionnaire features 
Sensor features 
XGBoost prediction 
VQC prediction 
SHAP explanations 
Safety guidance 
Model and inference information 
The Next.js frontend retrieves these results through the FastAPI backend and presents them as a unified injury-analysis report.




Fig-1:proposed methodology
IV. WORKING PRINCIPLE
AI-QTriage starts by collecting information from different sources to understand the injury more completely. The main inputs are an injury photograph, questionnaire responses, and smartphone sensor data. The image provides visual information about the injury, while the questionnaire captures details such as pain level, cause of injury, bleeding, swelling, movement limitations, and other symptoms. When sensor data are available, the system uses motion and impact information from the smartphone to provide additional evidence about how the injury may have occurred. 
The uploaded image is first checked for quality and validity before it is passed to the vision models. YOLO11 is used to locate visible injuries in the image by generating bounding boxes, injury classes, and confidence scores. EfficientNetV2 analyzes the image at the classification level to estimate the type of visible injury. At the same time, ResNet34-U-Net produces a segmentation mask that identifies the approximate region affected by the injury. These three models provide different types of visual information and therefore complement each other. 
The questionnaire responses are converted into numerical features so that they can be processed by the machine-learning models. Information such as pain intensity, injury mechanism, direct impact, visible bleeding, movement limitation, weight-bearing ability, and crack or popping sounds can contribute to the overall assessment. Sensor information is processed separately to extract motion and impact characteristics, including acceleration-related measurements, velocity-related information, stabilization time, and other available telemetry features. If a particular modality is unavailable, the system records its absence rather than treating missing information as an actual observation. 
After processing the individual inputs, AI-QTriage brings the information together using its multimodal feature-fusion module. The system converts the available information into a standardized 23-dimensional feature vector. This vector contains features obtained from the image, questionnaire, and sensor modalities. The purpose of this stage is to give the downstream prediction models a common representation of the case instead of making a decision from only one type of input.
The fused feature vector is passed to XGBoost, which serves as the main classical machine-learning model in the current research prototype. XGBoost produces the primary research-level classification and associated probabilities. The system also contains a 4-qubit Variational Quantum Classifier (VQC) implemented using the PennyLane simulator. The VQC provides an experimental comparison with the classical model. It is important to note that the VQC is currently an experimental research component and does not provide evidence of clinical or quantum advantage.
AI-QTriage does not only provide a final prediction; it also attempts to show why the models produced their outputs. SHAP is used to identify the contribution of individual fused features to the XGBoost prediction. Grad-CAM is used with EfficientNetV2 to highlight image regions that influenced the classification. The U-Net segmentation mask provides another visual representation of the estimated affected region, while YOLO's bounding box shows where the detector identified a visible injury.
Following model inference, the system applies a separate rule-based safety layer to the available injury information. This component considers factors such as pain severity, impact characteristics, stabilization time, bleeding, and other questionnaire or sensor information when generating safety guidance. The application also contains an SOS workflow. Local SOS simulation can be demonstrated within the application, while actual Twilio SMS functionality depends on valid Twilio configuration and should not be considered active when the service is disabled.
Finally, the system combines the available model predictions, visual evidence, questionnaire information, sensor features, explanations, and safety guidance into a structured case result. These results are stored in MongoDB, allowing the frontend application to retrieve and display the analysis on the case-results page. The user can therefore see the detected injury region, classification results, segmentation output, model predictions, explanations, confidence information, and safety recommendations in one interface.
V. IMPLEMENTATION
The AI-QTriage system was implemented as a full-stack multimodal research prototype that connects a web-based user interface, a FastAPI backend, machine-learning models, sensor processing, a rule-based safety layer, and MongoDB storage. The implementation was designed so that each modality can be processed independently and then combined during the analysis stage. The overall software workflow follows input acquisition → preprocessing → modality-specific inference → feature extraction → multimodal fusion → prediction → explainability → safety assessment → result storage and visualization. The project uses Next.js for the frontend, FastAPI for the backend API, Python-based machine-learning components for model inference, and MongoDB for storing case information and analysis results.
The frontend provides the interface through which a user creates a case, uploads an injury image, answers the questionnaire, optionally provides sensor information, and views the generated analysis. The backend receives these inputs through REST API endpoints and manages the complete analysis workflow. During analysis, the backend invokes the vision models, processes questionnaire and sensor information, constructs the multimodal feature representation, and passes the resulting features to the prediction models. The generated outputs are then stored with the corresponding case in MongoDB and returned to the frontend for visualization. This separation between the interface, API layer, machine-learning layer, and database makes it possible to test individual components independently while maintaining a complete end-to-end workflow. 
The project uses different datasets for different machine-learning tasks because no single publicly available dataset provides paired injury photographs, questionnaire information, and smartphone sensor measurements in the exact format required by the multimodal model. Therefore, the implementation treats the vision, segmentation, sensor, and multimodal datasets as separate sources. Public wound and injury image datasets are used for computer-vision experiments, while public motion and fall-detection datasets are suitable for developing the sensor component. Questionnaire information is collected through the application's structured interface. The current multimodal XGBoost and VQC experiments use a synthetic fusion dataset because a genuinely paired public dataset containing all three modalities was not available for the required 23-feature representation. This distinction is maintained in the implementation so that synthetic multimodal results are not presented as clinical performance. 
YOLO11 is implemented as the primary object-detection component of the vision pipeline. Its purpose is to identify the location of a visible injury and assign a corresponding class to the detected region. During inference, the model receives an RGB image and returns bounding boxes in the form
B=x1y1x2y2
along with a predicted class cand confidence score p. Detections below the configured confidence threshold are discarded. The remaining detection is passed to the subsequent vision pipeline, where the detected region can be used for additional analysis. The implementation also obtains the supported classes directly from the loaded YOLO model rather than assuming classes that are not actually present in the checkpoint. This is important because the research experiments have used different injury taxonomies during development.
EfficientNetV2 is used for image-level classification. Unlike YOLO11, it does not primarily determine the location of an injury; instead, it estimates which predefined visual category best represents the input image or selected injury region. The image is resized to the input dimensions expected by the network and normalized before inference. The model produces logits for the available classes, which are converted into probabilities using the softmax function:
Pyi∣x=ezij=1Kezj
where ziis the output logit for class i, and Krepresents the number of classes. The class with the highest probability is selected as the model's classification result. During development, additional input-quality and out-of-distribution checks were introduced because a classifier can produce a high-confidence class even when the input does not represent a suitable injury photograph.
The segmentation component uses a ResNet34-U-Net architecture to estimate the pixel-level region associated with the visible injury. The encoder extracts spatial features while the decoder reconstructs a segmentation mask at the required image resolution. The predicted probability mask is converted into a binary mask using a selected threshold. The Dice coefficient is used to evaluate the overlap between the predicted and ground-truth masks:
Dice=2∣P∩G∣∣P∣+∣G∣
where Prepresents the predicted segmentation region and Grepresents the ground-truth region. Intersection over Union can also be calculated as
IoU=∣P∩G∣∣P∪G∣
These metrics are particularly important for determining whether the segmentation model identifies the correct injury region rather than simply producing a visually plausible mask.
The questionnaire provides contextual information that cannot reliably be obtained from an image alone. Responses include factors such as pain severity, injury mechanism, direct impact, visible bleeding, movement limitation, ability to bear weight, and crack or popping sensations. Categorical responses are converted into numerical indicators, while numerical responses such as pain level are normalized or retained according to the feature definition. These values are then combined with the visual and sensor features. The questionnaire therefore acts as a complementary modality rather than replacing the image models. 
The sensor module processes motion telemetry obtained from smartphone or compatible sensor logs. The raw measurements are converted into fixed-length features that can be consumed by the sensor classifier and multimodal fusion model. Typical features include acceleration magnitude, peak acceleration, velocity-related measurements, stabilization duration, and motion-event characteristics. The acceleration magnitude can be represented as
a=ax2+ay2+az2
where ax, ay, and azrepresent acceleration measurements along the three axes. The processed information is used to distinguish patterns such as normal movement, fall-like events, and impact-like events. Missing sensor information is explicitly represented as unavailable rather than being silently replaced with fabricated measurements.
After the individual modalities have been processed, AI-QTriage combines them into a standardized feature representation. The current research implementation uses 23 canonical features covering vision, questionnaire, and sensor information. Conceptually, the fused representation can be written as
F=FvFqFs
where Fvrepresents vision features, Fqrepresents questionnaire features, and Fsrepresents sensor features. The resulting vector is passed to the downstream multimodal models. A consistent feature ordering is maintained so that the training and inference pipelines use the same schema. When a modality is unavailable, the implementation follows an explicit missing-modality strategy rather than silently inventing observations.
XGBoost is used as the primary classical machine-learning model for the fused representation. During training, the model learns relationships between the 23-dimensional feature vector and the research labels. XGBoost builds an ensemble of decision trees sequentially, where later trees attempt to reduce the errors made by earlier trees. The general objective can be expressed as
Obj=i=1nLyiyi+kΩfk
where Lrepresents the prediction loss and Ωfkcontrols model complexity. The trained model returns a predicted class and class probabilities. SHAP is then used to estimate the contribution of individual features to the prediction, providing a more interpretable view of the multimodal decision process.
The current XGBoost evaluation is based on synthetic multimodal data, not paired patient records. Therefore, its reported performance should be interpreted as an engineering/research benchmark rather than clinical accuracy.
AI-QTriage also includes a four-qubit Variational Quantum Classifier as an experimental component. The fused features are first transformed into a lower-dimensional representation suitable for the quantum circuit. The implementation uses AngleEmbedding followed by trainable StronglyEntanglingLayers. The circuit is executed using PennyLane's default.qubit, which is a classical simulator rather than physical quantum hardware. The VQC therefore demonstrates the integration of quantum-machine-learning techniques into the multimodal pipeline, but it does not establish quantum computational advantage or clinical superiority.
Explainability is incorporated at multiple stages of the system. SHAP values are generated for the XGBoost model to indicate how individual multimodal features influence the prediction. Grad-CAM is used with EfficientNetV2 to produce a visual heatmap showing image regions associated with the classification output. YOLO bounding boxes provide spatial evidence of detected injuries, while the U-Net mask provides a pixel-level representation of the estimated affected region. Together, these outputs allow the user to inspect not only the final prediction but also the evidence generated by different components of the system.
After model inference, AI-QTriage applies deterministic safety rules to the available case information. These rules consider relevant factors such as pain severity, impact characteristics, stabilization duration, bleeding, and other available evidence. The purpose of this layer is to provide predefined safety guidance rather than allowing an experimental machine-learning model to independently determine an emergency response. The application also includes an SOS workflow. Local SOS simulation can be used during demonstrations, while Twilio SMS functionality requires valid backend configuration. The system explicitly distinguishes between local simulation and actual SMS integration to avoid falsely reporting that an emergency message has been delivered.
The backend is implemented using FastAPI, which provides REST endpoints for case creation, image upload, questionnaire submission, sensor processing, analysis, model information, SOS operations, and report generation. The analysis endpoint acts as the central orchestration layer. It retrieves the stored case information, invokes the appropriate models, generates the fused feature vector, performs predictions, and stores the resulting outputs. MongoDB is used to persist case records and associated analysis information. This allows previously analyzed cases to be retrieved from the frontend without rerunning the machine-learning models.
AI-QTriage uses FastAPI as the backend to manage case creation, image uploads, questionnaire responses, sensor data, model inference, SOS operations, and report generation, with MongoDB used to store case information and analysis results. The Next.js frontend provides the user interface for entering data and viewing predictions, detection boxes, segmentation masks, explanations, and safety guidance. During analysis, YOLO11, EfficientNetV2, U-Net, questionnaire processing, and sensor processing generate modality-specific information that is combined into a 23-feature multimodal vector and passed to XGBoost and the experimental VQC. SHAP, Grad-CAM, and segmentation outputs provide interpretability, while the rule-based safety layer generates additional guidance before the results are stored and displayed.
The models are evaluated using task-specific metrics such as precision, recall, mAP, accuracy, macro-F1, Dice, IoU, and calibration measures. Automated tests, model-hash verification, API testing, and frontend build checks are used to verify software reliability and reproducibility. However, the current multimodal XGBoost and VQC experiments rely on synthetic fused data, while individual vision and sensor models use separate datasets; therefore, the reported results represent research and implementation performance rather than clinical accuracy. The complete system demonstrates an end-to-end multimodal injury-assessment workflow, but larger independent and clinically annotated datasets are required for future real-world validation.
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
Table-1:model accuracy
Key Formulas Used
Softmax Probability — EfficientNetV2 
Pyi∣x=ezij=1Kezj
Acceleration Magnitude — Sensor Processing 
a=ax2+ay2+az2
Dice Coefficient — U-Net Segmentation 
Dice=2∣P∩G∣∣P∣+∣G∣
Intersection over Union (IoU) — Segmentation 
IoU=∣P∩G∣∣P∪G∣
Precision — Model Evaluation 
Precision=TPTP+FP
Recall — Model Evaluation 
Recall=TPTP+FN
F1-Score — Model Evaluation 
F1=2Precision×RecallPrecision+Recall
Accuracy — Classification 
Accuracy=TP+TNTP+TN+FP+FN
XGBoost Objective Function 
Obj=i=1nLyiyi+kΩfk
Brier Score — Probabilistic Evaluation 
BS=1Ni=1Nk=1K(pik-yik)2
Expected Calibration Error (ECE) 
ECE=m=1M∣Bm∣Nacc⁡Bm-conf⁡Bm
Matthews Correlation Coefficient (MCC) — Binary Form 
MCC=TP×TN-FP×FNTP+FPTP+FNTN+FPTN+FN
Multimodal Feature Fusion 
                 Ffusion​=[Fvision​,Fquestionnaire​,Fsensor​]
VI. RESULTS AND DISCUSSION
The AI-QTriage prototype was evaluated across its major components, including image classification, image segmentation, object detection, sensor-based classification, and multimodal prediction. The results show that the individual models perform differently depending on the task and dataset. The most important point is that these values come from different datasets and evaluation protocols, so they should not be combined into one overall clinical accuracy. The results demonstrate the technical performance of the research prototype rather than validated medical performance.
6.1.1:Result
 Accuracy:The AI-QTriage models showed different levels of performance depending on their specific tasks and evaluation datasets. The Sensor Motion Classifier achieved the highest reported accuracy of 96.34% (1183/1228) on its held-out test set, while XGBoost achieved 83.33% (25/30) on the synthetic multimodal test set. The experimental 4-Qubit VQC achieved 53.33% (16/30) on the same multimodal test set. For the vision models, EfficientNetV2 achieved a macro-F1 score of 93.75%, U-Net achieved a Dice score of 64.18%, and YOLO11 achieved 36.47% mAP@50 and 18.45% mAP@50–95. These results show that the individual components have different strengths and should be evaluated using their appropriate task-specific metrics.
Classification and precision – Recall:Theclassification results indicate that EfficientNetV2 performed strongly on its evaluated wound-image dataset, with a macro-F1 score of 93.75%. XGBoost also showed stronger multimodal classification performance than the experimental VQC, with 82.22% macro precision, 76.93% macro recall, and 78.50% macro-F1. In comparison, the VQC obtained 46.62% macro precision, 43.27% macro recall, and 42.30% macro-F1. This indicates that the classical XGBoost model currently provides more reliable classification performance than the experimental quantum model on the available synthetic multimodal test data.


Fig-2:Model Values
The evaluation of AI-QTriage shows different performance levels across the individual models. YOLO11 achieved 36.47% mAP@50 and 18.45% mAP@50–95, indicating that injury detection and localization remain areas that require improvement. However, individual tests showed that YOLO11 could detect visible injuries in real photographs, including a hand injury classified as Cut at approximately 60% confidence and a bruise detected at approximately 53.8% confidence. The ResNet34-U-Net model achieved a Dice score of 64.18%, showing moderate overlap between predicted and reference injury regions. The Sensor Motion Classifier achieved 96.34% accuracy (1183/1228) on its held-out test set, representing the highest reported accuracy among the individual models. For multimodal classification, XGBoost achieved 83.33% accuracy (25/30), while the experimental 4-qubit VQC achieved 53.33% (16/30). XGBoost therefore performed approximately 30 percentage points better than VQC, with MCC values of 0.7027 and 0.0960, respectively.

                        Fig 3 : Models Trained Information 
Overall, the results demonstrate that AI-QTriage successfully integrates computer vision, sensor analysis, multimodal prediction, and explainable AI into a single research workflow, but the performance varies considerably between components. EfficientNetV2 and the sensor classifier show strong results on their respective evaluation datasets, U-Net provides moderate segmentation performance, and YOLO11 remains the main vision component requiring further improvement in detection and localization. XGBoost currently provides substantially better multimodal performance than the experimental VQC, although both were evaluated using synthetic fused data. Therefore, the reported values should be treated as research and prototype benchmarks rather than clinical accuracy. The individual model results also cannot be averaged into a single overall AI-QTriage accuracy because the models perform different tasks and were evaluated using different datasets and metrics. Further validation using larger, independent, real-world, and clinically annotated datasets is required before making claims about clinical or real-world performance.

    Fig-4: Difference between XGBoost vs VQC Comparision
6.3:Discussion of Research Outcomes:
The research outcomes indicate that AI-QTriage successfully demonstrates the feasibility of combining computer vision, smartphone sensor analysis, questionnaire-based information, multimodal machine learning, and explainable AI within a single injury-assessment framework. The individual models showed different levels of performance according to their respective tasks. EfficientNetV2 achieved a 93.75% macro-F1 score, while the Sensor Motion Classifier achieved 96.34% accuracy on its reported test set. The ResNet34-U-Net produced a 64.18% Dice score, indicating moderate segmentation performance. YOLO11 achieved 36.47% mAP@50 and 18.45% mAP@50–95, showing that injury detection and localization remain important areas for improvement. These results demonstrate that the proposed architecture can successfully execute the complete multimodal workflow, but the performance of each component must be interpreted separately.
The multimodal experiment further showed that XGBoost performed better than the experimental 4-qubit VQC, achieving 83.33% accuracy compared with 53.33% on the same 30-sample test set. This suggests that the classical model is currently more suitable for the available fused feature representation, while the VQC is better considered an experimental component. The research also demonstrated the value of combining different forms of evidence, including YOLO bounding boxes, U-Net segmentation, EfficientNetV2 classification, sensor features, questionnaire responses, and SHAP/Grad-CAM explanations. However, the findings have important limitations. The models were evaluated using different datasets and metrics, and the multimodal XGBoost and VQC experiments used synthetic fused data with no genuinely paired patient records. Therefore, the results cannot be combined into a single overall accuracy and should not be interpreted as clinical performance. Overall, the outcomes support AI-QTriage as a functional research prototype and proof of concept, while further work with larger, independent, real-world and clinically annotated datasets is necessary to establish reliability and generalizability.
VII. CONCLUSION
AI-QTriage presents a hybrid AI–quantum multimodal framework for explainable injury assessment and emergency support, integrating image-based injury detection, classification, segmentation, questionnaire information, smartphone sensor analysis, multimodal feature fusion, explainable AI, and safety-support mechanisms into a unified research prototype. The system demonstrates that these different modalities can be processed through a common workflow and presented as structured assessment results. The experimental evaluation showed promising performance in several individual components, including 93.75% macro-F1 for EfficientNetV2, 64.18% Dice for U-Net, 96.34% accuracy for the sensor classifier, 83.33% accuracy for XGBoost, and 53.33% accuracy for the experimental 4-qubit VQC. YOLO11 achieved 36.47% mAP@50 and 18.45% mAP@50–95, indicating that injury detection and localization remain areas requiring further improvement.
The study also demonstrates that XGBoost currently performs substantially better than the experimental VQC for the available multimodal representation. However, the reported results must be interpreted within their respective datasets and evaluation settings and cannot be combined into a single overall accuracy. In particular, the multimodal experiments rely on synthetic fused data rather than genuinely paired patient records, and the system has not undergone clinical validation. Therefore, AI-QTriage should currently be regarded as a research prototype and proof of concept, not a clinical diagnostic system. Future work will focus on larger and more diverse real-world datasets, subject-aware evaluation, improved YOLO detection and U-Net segmentation, genuinely paired multimodal data, and comprehensive external validation. These improvements can help establish the reliability, generalizability, and practical potential of AI-QTriage for future injury-assessment research.
VIII. REFERENCE
[1] Krones, F., et al. (2025). Review of multimodal machine learning approaches in healthcare. Information Fusion.
[2] Kline, A., Wang, H., Li, Y., Dennis, S., Hutch, M., Xu, Z., Wang, F., Cheng, F., and Luo, Y. (2022). Multimodal machine learning in precision health: A scoping review. npj Digital Medicine, 5. doi:10.1038/s41746-022-00611-0
[3] Gessert, N., Nielsen, M., Shaikh, M., Werner, R., and Schonlieb, C. B. (2020). Skin lesion classification using ensembles of multi-resolution EfficientNets with meta data. MethodsX, 7, 100864. doi:10.1016/j.mex.2020.100864
[4] Tan, M. and Le, Q. V. (2021). EfficientNetV2: Smaller models and faster training. Proceedings of the 38th International Conference on Machine Learning (ICML), 10096–10106.
[5] Ronneberger, O., Fischer, P., and Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. Medical Image Computing and Computer-Assisted Intervention (MICCAI), 234–241. doi:10.1007/978-3-319-24574-4_28
[6] Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., and Batra, D. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. Proceedings of the IEEE International Conference on Computer Vision (ICCV), 618–626. doi:10.1109/ICCV.2017.74
[7] Acosta, J. N., Falcone, G. J., Rajpurkar, P., and Topol, E. J. (2022). Multimodal biomedical AI. Nature Medicine, 28, 1773–1784. doi:10.1038/s41591-022-01981-2
[8] Chen, T. and Guestrin, C. (2016). XGBoost: A scalable tree boosting system. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785–794. doi:10.1145/2939672.2939785
[9] Sucerquia, A., López, J. D., and Vargas-Bonilla, J. F. (2017). SisFall: A fall and movement dataset. Sensors, 17 (1), 198. doi:10.3390/s17010198
[10] Micucci, D., Mobilio, M., and Napoletano, P. (2017). UniMiB SHAR: A dataset for human activity recognition using acceleration data from smartphones. Applied Sciences, 7 (10), 1101. doi:10.3390/app7101101
[11] Martínez-Villaseñor, M., Ponce, H., Velázquez-García, A., and others. (2019). UP-Fall Detection Dataset: A multimodal approach. Sensors, 19 (9), 1988. doi:10.3390/s19091988
[12] Lundberg, S. M. and Lee, S.-I. (2017). A unified approach to interpreting model predictions. Advances in Neural Information Processing Systems (NeurIPS), 30.
[13] Warner, E., Lee, J., Hsu, W., Syeda-Mahmood, T., Kahn, C. E., Gevaert, O., and Rao, A. (2024). Multimodal machine learning in image-based and clinical biomedicine: Survey and prospects. International Journal of Computer Vision, 132 (9), 3753–3769. doi:10.1007/s11263-024-02032-8. 
[14] van der Velden, B. H. M., Kuijf, H. J., Gilhuijs, K. G. A., and Viergever, M. A. (2022). Explainable artificial intelligence (XAI) in deep learning-based medical image analysis. Medical Image Analysis, 79, 102470. doi:10.1016/j.media.2022.102470. 
[15] Benjamins, J. W., Dhunnoo, P., and Meskó, B. (2023). A scoping review of interpretability and explainability concerning artificial intelligence methods in medical imaging. European Journal of Radiology, 168, 111159. doi:10.1016/j.ejrad.2023.111159. 
[16] Tusar, M. H., Fayyazbakhsh, F., Zendehdel, N., Mochalin, E., Melnychuk, I., Gould, L., and Leu, M. C. (2026). AI-powered image-based assessment of pressure injuries using You Only Look Once version 8 (YOLOv8) models. Wound Repair and Regeneration, 15 (5), 271–288. doi:10.1089/wound.2024.0245. 
[17] Wang, C., et al. (2022). Detect-and-segment: A deep learning approach to automate wound image segmentation. Informatics in Medicine Unlocked, 29, 100884. doi:10.1016/j.imu.2022.100884. 
[18] Goyal, M., Reeves, J., Davison, A. K., et al. (2020). Fully automatic wound segmentation with deep convolutional neural networks. Scientific Reports, 10, 20786. doi:10.1038/s41598-020-76515-2. 
[19] Wertner, A., Czech, P., and Pammer-Schindler, V. (2015). An open labelled dataset for mobile phone sensing based fall detection. Proceedings of the 12th EAI International Conference on Mobile and Ubiquitous Systems: Computing, Networking and Services (MOBIQUITOUS). doi:10.4108/eai.22-7-2015.2260160. 
[20] Alhazmi, A. (2026). Explainable AI in medical imaging and diagnosis: A critical review of methods, clinical translation barriers, and an integrated evaluation framework. AI in Clinical Medicine, 2, e36. doi:10.14740/aicm36. 
[21] Fanale, R., Caldelli, R., Martini, B., and Sciarrone, F. (2026). A quantitative framework for explainability assessment in medical imaging. Frontiers in Imaging, 5, 1846414. doi:10.3389/fimag.2026.1846414. 
[22] Čižmárová, A., Dostálová, K., Hrkut, P., and Poroshenko, A. (2026). Explainable artificial intelligent method Grad-CAM in medical images processing. Advanced Information Systems. doi:10.20998/2522-9052.2026.2.09. 
[23] “Explainable artificial intelligence in medical imaging: A systematic review of techniques, applications, and challenges.” (2025). BMC Medical Imaging. doi:10.1186/s12880-025-02118-w. 
[24] “Explainable AI in healthcare: A systematic review of XAI use cases in imaging, diagnostics, and rehabilitation.” (2026). Journal of Medical Artificial Intelligence. 





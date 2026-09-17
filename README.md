# Emotion Detection from Uploaded Images

A Streamlit application that detects faces in an uploaded photo and classifies the emotion shown, comparing a custom-trained CNN against a pretrained model. Built on the FER-2013 dataset.

## Objective

Develop an end-to-end system where a user uploads an image, the app detects any face(s) present, and classifies the emotion (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral) using Convolutional Neural Networks.

## Dataset

**FER-2013** (via Kaggle: `yusufkorayhasdemir/fer2013csv`) — 35,887 grayscale, 48×48 pixel facial images labeled with 7 emotions, pre-split into Training (28,709), PublicTest/Validation (3,589), and PrivateTest/Test (3,589) sets.

**Class distribution** (Training set):

| Emotion | Count | % of data |
|---|---|---|
| Happy | 8,989 | 25.0% |
| Neutral | 6,198 | 17.3% |
| Sad | 6,077 | 16.9% |
| Fear | 5,121 | 14.3% |
| Angry | 4,953 | 13.8% |
| Surprise | 4,002 | 11.2% |
| **Disgust** | **547** | **1.5%** |

This severe imbalance (Disgust is ~16x rarer than Happy) turned out to be the central challenge of the project, discussed throughout.

## Approach

1. Parsed FER-2013's pixel strings into 48×48 grayscale image arrays
2. Detected faces in uploaded photos using OpenCV's Haar Cascade (chosen over Dlib/Mediapipe due to Dlib's known Windows installation issues; see "Facial Landmarks" below)
3. Trained and compared multiple modeling approaches (below)
4. Built a Streamlit app for live image upload, face detection, and emotion prediction
5. Evaluated every model with accuracy, precision, recall, and F1-score (not accuracy alone, given the class imbalance)

## Modeling Approaches Compared

| Model | Framework | Test Accuracy | Notes |
|---|---|---|---|
| CNN (from scratch) | Keras/TensorFlow | 51.94% | Baseline: 3 Conv+BatchNorm+Pool blocks, Dense head, Dropout |
| MobileNetV2 (transfer learning, frozen base) | Keras/TensorFlow | 49.40% | Underperformed the baseline |
| **CNN (from scratch)** | **PyTorch** | **56.00%** | Same architecture as the Keras baseline, confirms reproducibility — used as the app's main model |
| CNN + class-weighted loss | PyTorch | 35.75% | Attempted fix for class imbalance; destabilized training (see below) |

### Why MobileNetV2 underperformed

Transfer learning with a frozen MobileNetV2 (pretrained on ImageNet) scored *lower* than the from-scratch CNN. Likely explanation: ImageNet's natural-image features (learned from millions of real-world color photos at high resolution) transfer poorly to small (48×48, upscaled to 96×96), low-resolution, grayscale-converted-to-fake-RGB facial expressions. With the base frozen, the model could only reinterpret generic ImageNet features rather than learn face-specific ones — for a domain this different and this low-resolution, a smaller model trained directly on the target data outperformed the frozen pretrained one.

A pretrained Hugging Face ViT model (`ChristopherLi/vit-fer2013-emotion`, tuned for "in-the-wild" faces) is used separately in the live app as a second opinion, since it is likely better suited to real, unconstrained photos than either model trained here.

### Why the Keras and PyTorch CNNs were built and compared

Building the identical CNN architecture in both frameworks and getting closely matching results (51.94% and 56.00%, both in the same range) served as a reproducibility check — it confirms the result reflects the architecture and data, not a framework-specific quirk or bug.

### Attempted fix for class imbalance: weighted loss

To address Disgust's near-zero recall, `CrossEntropyLoss` was re-weighted using `sklearn`'s `compute_class_weight('balanced', ...)`, which assigned Disgust a 9.4x weight relative to the majority class. This destabilized training: several classes (Angry, Disgust, Fear) collapsed to 0% precision/recall, and overall accuracy dropped from 56% to 35.75%. Capping the maximum weight (to 3.0x) and lowering the learning rate did not resolve this. This suggests class weighting, as applied here, needs a larger/more robust model than this custom CNN, or that targeted data augmentation of the minority class would be a more suitable fix than loss reweighting for this architecture. The unweighted model was kept as the final result.

### Facial Landmarks (Dlib)

Dlib's 68-point facial landmark detector was tested (in Colab, to avoid Dlib's known Windows installation issues) as an alternative to Haar Cascade for facial feature extraction. It failed to detect a face in the majority of FER-2013's images even after upscaling from 48×48 to 200×200 — a known limitation of Dlib's HOG-based detector on this dataset's naturally low resolution and tight cropping. Given this high failure rate, landmark-based classification was not pursued further; Haar Cascade was used for face detection in the live app instead.

## Test Set Results — Best Model (PyTorch CNN)

| Emotion | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Angry | 0.48 | 0.40 | 0.44 | 491 |
| Disgust | 0.00 | 0.00 | 0.00 | 55 |
| Fear | 0.41 | 0.27 | 0.32 | 528 |
| Happy | 0.76 | 0.86 | 0.80 | 879 |
| Sad | 0.36 | 0.54 | 0.43 | 594 |
| Surprise | 0.74 | 0.70 | 0.72 | 416 |
| Neutral | 0.55 | 0.49 | 0.52 | 626 |
| **Accuracy** | | | **0.56** | 3589 |
| Macro avg | 0.47 | 0.46 | 0.46 | 3589 |
| Weighted avg | 0.55 | 0.56 | 0.55 | 3589 |

The model performs well on visually distinctive emotions (Happy, Surprise) and struggles on Disgust, whose near-zero recall persisted across every model and technique tried (frozen transfer learning, two frameworks, and weighted loss) — strong evidence this is a data limitation (1.5% class representation), not a fixable modeling error.

## The Streamlit App

- Users upload a `.jpg`, `.jpeg`, or `.png` image (Streamlit's `file_uploader` type restriction blocks any other format)
- OpenCV Haar Cascade detects face(s) in the photo, with 20% padding added around each detected box and contrast normalization (`cv2.equalizeHist`) applied before prediction, to better match FER-2013's cropping/contrast style
- Each detected face is classified two ways side by side: the custom-trained PyTorch CNN (grayscale, 48×48) and a pretrained Hugging Face ViT model (color, "in-the-wild" tuned), for direct comparison
- A confidence score is shown alongside each prediction

## Ethical Considerations

- **Bias**: Every model tested — regardless of framework, transfer learning, or class weighting — failed to reliably recognize Disgust, the most underrepresented emotion in the training data. This is a direct illustration of how a model trained on imbalanced data will systematically underperform for underrepresented classes/groups, a core concern in real-world deployment (e.g., in healthcare or customer service, misreading or ignoring an underrepresented emotional state could have real consequences).
- **Domain gap**: Models trained on FER-2013's lab-cropped, consistent-lighting images performed noticeably worse on real, unconstrained uploaded photos (different lighting, angles, backgrounds), showing that benchmark accuracy does not guarantee real-world reliability.
- **Privacy**: Facial images are sensitive biometric data. This app does not store uploaded images or predictions; processing happens only for the duration of the session. Any production deployment would need explicit user consent, clear data retention policies, and safeguards against storing or reusing uploaded faces.
- **Appropriate use**: Emotion recognition from facial expressions is an imperfect proxy for actual emotional state (a fixed facial expression does not always reflect internal feeling, and expression varies across individuals and cultures). This technology should not be used for high-stakes automated decisions (e.g., hiring, surveillance, law enforcement) without human oversight, given the accuracy and bias limitations demonstrated here.

## Tech Stack

Python, TensorFlow/Keras, PyTorch, OpenCV, Dlib, Streamlit, Hugging Face Transformers, scikit-learn, Pandas, NumPy, Matplotlib, Seaborn, Google Colab (GPU training)

## Key Takeaways

- Compared 4 distinct modeling approaches (from-scratch CNN, transfer learning, cross-framework reproduction, class-weighted loss) with full precision/recall/F1 evaluation, not just accuracy
- Frozen transfer learning underperformed a smaller from-scratch model on this low-resolution, domain-specific dataset — a useful, counter-intuitive finding
- A standard class imbalance fix (weighted loss) was tried, failed, and honestly reported rather than discarded — the failure itself is informative
- Documented a real limitation of Dlib's facial landmark detector on low-resolution datasets, and made a deliberate, explained choice to use Haar Cascade instead
- Built a working end-to-end app comparing a custom PyTorch model against a pretrained one on real, live user photos

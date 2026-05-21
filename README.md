# Person-Independent WiFi CSI Activity Recognition for Passive Home Monitoring

## Overview
This repository contains the hardware data collection pipeline and machine learning models for a device-free human activity recognition (HAR) system. The system utilizes a dual-ESP32 receiver setup and a commodity access point to capture Channel State Information (CSI). The core objective is to address the cross-person generalization failure common in CSI-based HAR by implementing a domain-adversarial neural network (DANN).

## Hardware Configuration
The sensing platform eliminates the need for intermediate edge devices or expensive network interface cards.
* **Transmitter (AP):** One TP-Link TL-WR845N access point operating on Channel 6 (2.4 GHz 802.11n) with 20 MHz bandwidth. The AP is managed via Ethernet to avoid medium interference.
* **Receivers (Nodes):** Two ESP32-DevKitC-1 nodes acting as CSI receivers. A powered USB hub provides stable power delivery to both nodes without drawing from the host's USB bus.
* **Data Extraction:** Custom firmware extracts CSI, per-packet RSSI, and SNR, streaming timestamped packets over USB-UART at 115200 baud to a laptop computer.
* **Spatial Setup:** The system evaluates three physical configurations, including an orthogonal setup for maximum angular diversity and opposite-side placement for maximal spatial separation.

## Data Preprocessing Pipeline
Raw CSV streams from the two independent receivers undergo a rigorous synchronization and denoising pipeline.
* **Synchronization:** Streams are time-aligned by finding a maximum common start time, extracting a 10-second overlap, and linearly resampling both nodes to a uniform 10 Hz grid.
* **Denoising:** Outliers are removed using a Hampel filter, while high-frequency noise is suppressed via Wavelet denoising (db4, level 3, 20% soft threshold).
* **Normalization:** Phase vectors are linearly detrended to cancel offsets, and per-subcarrier z-score normalization centers and scales the data.
* **Windowing:** Data is concatenated into a 436-dimensional feature vector and segmented using a 2-second sliding window with a 0.5-second stride.

## Machine Learning Architecture
The project compares a standard CNN against a DANN to force the extraction of subject-invariant features.
* **Baseline 1D CNN:** An encoder composed of three convolutional blocks extracts hierarchical temporal patterns, followed by a global average pooling layer and a fully connected classifier.
* **DANN:** Shares the baseline encoder but adds a domain head and a Gradient Reversal Layer (GRL). The GRL forces the encoder to produce features discriminative for activity but uninformative about subject identity.
* **Open-Set Anomaly Detection:** An energy-based detector wraps the classifier to compute free energy from output logits, distinguishing known activities from out-of-distribution motion and outputting an "unknown" state.

## Dataset and Results
The dataset consists of 3,600 packets capturing five activities (walking, standing still, sitting still, arm gestures, and empty room) across two subjects.
* **Baseline Performance:** Achieved 68.9% within-subject accuracy but dropped to an average of 53.1% cross-subject, confirming heavy reliance on subject-specific signatures.
* **DANN Performance:** Raised the average cross-subject accuracy to 57.3% and the macro-F1 score from 43.2% to 56.9%. The domain classifier stabilized near chance level, confirming the successful suppression of subject identity by the

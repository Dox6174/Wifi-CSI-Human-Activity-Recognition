
# Person-Independent WiFi CSI Activity Recognition for Passive Home Monitoring

## Overview
[cite_start]This repository contains the hardware data collection pipeline and machine learning models for a device-free human activity recognition (HAR) system[cite: 1, 15]. [cite_start]The system utilizes a dual-ESP32 receiver setup and a commodity access point to capture Channel State Information (CSI)[cite: 7]. [cite_start]The core objective is to address the cross-person generalization failure common in CSI-based HAR by implementing a domain-adversarial neural network (DANN)[cite: 17, 19].

## Hardware Configuration
[cite_start]The sensing platform eliminates the need for intermediate edge devices or expensive network interface cards[cite: 36].
* [cite_start]**Transmitter (AP):** One TP-Link TL-WR845N access point operating on Channel 6 (2.4 GHz 802.11n) with 20 MHz bandwidth[cite: 34]. [cite_start]The AP is managed via Ethernet to avoid medium interference[cite: 39].
* [cite_start]**Receivers (Nodes):** Two ESP32-DevKitC-1 nodes acting as CSI receivers[cite: 34]. [cite_start]A powered USB hub provides stable power delivery to both nodes without drawing from the host's USB bus[cite: 37, 38].
* [cite_start]**Data Extraction:** Custom firmware extracts CSI, per-packet RSSI, and SNR, streaming timestamped packets over USB-UART at 115200 baud to a laptop computer[cite: 35, 44].
* [cite_start]**Spatial Setup:** The system evaluates three physical configurations, including an orthogonal setup for maximum angular diversity and opposite-side placement for maximal spatial separation[cite: 40, 42].

## Data Preprocessing Pipeline
[cite_start]Raw CSV streams from the two independent receivers undergo a rigorous synchronization and denoising pipeline[cite: 8, 44].
* [cite_start]**Synchronization:** Streams are time-aligned by finding a maximum common start time, extracting a 10-second overlap, and linearly resampling both nodes to a uniform 10 Hz grid[cite: 45].
* [cite_start]**Denoising:** Outliers are removed using a Hampel filter, while high-frequency noise is suppressed via Wavelet denoising (db4, level 3, 20% soft threshold)[cite: 47, 48].
* [cite_start]**Normalization:** Phase vectors are linearly detrended to cancel offsets, and per-subcarrier z-score normalization centers and scales the data[cite: 49].
* [cite_start]**Windowing:** Data is concatenated into a 436-dimensional feature vector and segmented using a 2-second sliding window with a 0.5-second stride[cite: 51, 52].

## Machine Learning Architecture
[cite_start]The project compares a standard CNN against a DANN to force the extraction of subject-invariant features[cite: 9, 29].
* [cite_start]**Baseline 1D CNN:** An encoder composed of three convolutional blocks extracts hierarchical temporal patterns, followed by a global average pooling layer and a fully connected classifier[cite: 58, 59, 60, 61].
* [cite_start]**DANN:** Shares the baseline encoder but adds a domain head and a Gradient Reversal Layer (GRL)[cite: 79, 83, 84]. [cite_start]The GRL forces the encoder to produce features discriminative for activity but uninformative about subject identity[cite: 85].
* [cite_start]**Open-Set Anomaly Detection:** An energy-based detector wraps the classifier to compute free energy from output logits, distinguishing known activities from out-of-distribution motion and outputting an "unknown" state[cite: 31, 97, 98, 100].

## Dataset and Results
[cite_start]The dataset consists of 3,600 packets capturing five activities (walking, standing still, sitting still, arm gestures, and empty room) across two subjects[cite: 105, 108].
* [cite_start]**Baseline Performance:** Achieved 68.9% within-subject accuracy but dropped to an average of 53.1% cross-subject, confirming heavy reliance on subject-specific signatures[cite: 10, 115, 116].
* [cite_start]**DANN Performance:** Raised the average cross-subject accuracy to 57.3% and the macro-F1 score from 43.2% to 56.9%[cite: 11, 122]. [cite_start]The domain classifier stabilized near chance level, confirming the successful suppression of subject identity by the encoder[cite: 11, 124, 137].

## Future Work
[cite_start]Subsequent iterations will focus on formally characterizing an adaptive dormancy protocol to manage the latency-overhead tradeoff[cite: 142, 144]. [cite_start]This three-state finite state machine leverages RSSI changes to trigger the sensing pipeline only when motion is likely, expediting the return to a low-power idle state[cite: 143].


# Active CSI collection (Station)
Connects to some Access Point (AP) (Router or another ESP32) and sends packet requests (thus receiving CSI packet responses).
To use run `idf.py flash monitor` from a terminal.


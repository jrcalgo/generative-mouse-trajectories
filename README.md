# Generative Mouse Trajectories

**Generative Mouse Trajectories** is a comprehensive project dedicated to capturing, analyzing, and generating realistic computer mouse cursor movements. It leverages state-of-the-art Generative Adversarial Networks (GANs) to simulate authentic human-like interactions, significantly enhancing applications in accessibility, automation, UX/UI testing, and user behavior analysis.

---

## 🚀 Project Overview

The project consists of three key components:

1. **Mouse Collection Environment**: Captures high-fidelity mouse trajectory data from real users in real-time.
2. **Trajectory Data Analysis**: Computes instantaneous and derived behavioral metrics, ensuring data is clean, meaningful, and actionable.
3. **GAN-based Generation & Demonstration**: Trains GAN models on the collected data to generate realistic synthetic mouse trajectories. A demonstration application showcases a computer autonomously performing these generated cursor movements.

---


## 📥 Mouse Collection Environment

The **Mouse Collection Environment** is built using Rust, `tokio`, and `winit`. Its purpose is to record precise, granular mouse events, capturing essential attributes such as:

- **Temporal Data**: Timestamp, duration between movements, hover times.
- **Spatial Metrics**: Coordinates, path lengths, direction angles.
- **Kinematic Metrics**: Velocity, acceleration, jerk, curvature.
- **Behavioral Metrics**: Mouse clicks, scroll events.

### 🛠️ Key Features:
- **Real-time Data Collection**: Efficient, non-blocking, asynchronous event listening.
- **Detailed Attribute Computation**: Instantly calculates sophisticated metrics.
- **CSV Exporting**: Structured data for ease of access and downstream processing.

### ▶️ Getting Started:
```bash
# Navigate into collection environment directory
cd mouse-collection-environment

# Build and run data collection environment
cargo run --release

# Navigate to build directory and run collection environment
./mouse-collection-environmnet
```

### 📂 Output:

Collected data is stored in data/data.csv, containing extensive trajectory details for analysis and GAN training.

---

### 🤖 GAN-based Mouse Trajectory Generation

We utilize advanced Generative Adversarial Networks (GANs) trained on collected mouse trajectory data. The GAN model captures the complexity of real human cursor movements, including nuanced aspects like pauses, accelerations, and subtle directional shifts.

### 🔑 GAN Architecture:
- **Generator**: Synthesizes realistic trajectories from random latent inputs.
- **Discriminator**: Evaluates authenticity against real collected data.
- **Latent Space Control**: Enables custom trajectory generation with controlled behaviors (speed, path complexity).

---

### 📌 Potential Use Cases

This system provides realistic synthetic cursor movements applicable to various domains:
- **Accessibility Tools**: Assistive technologies that simulate human interactions for users with limited mobility.
- **Automated UI Testing**: Generating user-like interaction sequences for rigorous interface testing.
- **Behavioral Biometrics**: Authenticating users based on mouse-movement patterns.
- **Simulation and Robotics**: Emulating realistic interactions in virtual or robotic environments.

---

### 📜 Licensing

This project is licensed under the MIT License, enabling open collaboration and integration into broader projects. Refer to LICENSE for details.

---

### 👐 Contribution

Contributions are warmly welcomed! Please open an issue or pull request describing your proposed improvements, enhancements, or fixes.
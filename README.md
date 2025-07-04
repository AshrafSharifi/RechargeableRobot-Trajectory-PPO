Optimizing Trajectories for Rechargeable Agricultural Robots in Greenhouse Climatic Sensing Using DRL with PPO

This repository contains the code, models, and data for our work on optimizing the trajectory of rechargeable agricultural robots in greenhouses using Deep Reinforcement Learning (DRL) with Proximal Policy Optimization (PPO).

It supports intelligent trajectory planning to:

Efficiently collect greenhouse climate data by visiting Points of Interest (PoIs),

Manage the robot’s energy by incorporating charging station visits,

Improve spatial microclimate mapping with fewer physical sensors.

📜 Paper Reference
Sharifi, A.; Migliorini, S.; Quaglia, D.
Optimizing Trajectories for Rechargeable Agricultural Robots in Greenhouse Climatic Sensing Using Deep Reinforcement Learning with Proximal Policy Optimization Algorithm.
Future Internet 2025, 17, 296. https://doi.org/10.3390/fi17070296

🚀 Highlights
🔄 DRL with PPO: Adaptive learning of the best trajectory to balance measurement and battery constraints.

⚡ Recharge-aware: Integrates PoIs with charging stations and dynamically schedules recharging.

🌱 Real data tested: Applied on greenhouses in Verona (Italy) across multiple crops.


📊 Results Summary
✅ Compared to our previous methods and baseline approaches, our PPO-based approach:

Reduced climate estimation errors (MAPE improved by ~2-2.5%).

Increased visits to key PoIs while respecting battery constraints.

Dynamically scheduled recharges, avoiding critical battery drops.


📈 Citation
If you use this work in your research, please cite:

bibtex
Copy
Edit
@article{Sharifi2025,
  author = {Sharifi, A. and Migliorini, S. and Quaglia, D.},
  title = {Optimizing Trajectories for Rechargeable Agricultural Robots in Greenhouse Climatic Sensing Using Deep Reinforcement Learning with Proximal Policy Optimization Algorithm},
  journal = {Future Internet},
  year = {2025},
  volume = {17},
  pages = {296},
  doi = {10.3390/fi17070296}
}
🔗 Useful Links
📜 Paper on MDPI
🧑‍💻 Project GitHub
🧹 CleanRL (PPO framework)


# Udacity Projects Master Repo

Welcome to the Udacity Projects Master Repo! If you are an MSc in Artificial Intelligence candidate at the Udacity Institute of AI & Technology, this repo can help you to succeed in your coursework. 

This repo contains my solutions for various Nanodegree Programs (Robotics, Artificial Intelligence, Machine Learning, Deep Reinforcement Learning, Computer Vision, Sensor Fusion, Flying Cars, etc.). Additionally, several of the elective courses are covered.  

The projects were originally sorted by the relative Udacity School, with the more recent completions at the top (my software development skills improved over time :rocket:). However, some of the newer programs that succeeded previous content are intertwined (i.e., the Machine Learning with Pytorch replaced the original Machine Learning Nanodegree material). 

All projects can be used for inspiration, but please follow the **Udacity Honor Code** and reference any "borrowed" material. Caveat emptor: use at your own risk (dura lex sed lex).

## Quick navigation

Jump to:
- [School of Artificial Intelligence](#ai)
- [School of Autonomous Systems](#autonomous-systems)
- [School of Data Science](#data-science)
- [School of Business](#business)
- [School of Programming and Software Development](#programming)

## Skills index

Reinforcement learning, deep learning, computer vision, NLP, multi-agent systems, agents and orchestration, RAG, bioinformatics tooling, statistics and experiment design, SQL and data modeling, ETL and warehousing, ROS and robotics, sensor fusion and estimation, VR with Unity, product and project management, digital marketing analytics.

---

<a id="ai"></a>
## 🧠 School of Artificial Intelligence 🤖

### Agentic AI

P1. [Multi-Agent Orchestrator Logistics System](https://github.com/Ohara124c41/agentic_ai/tree/main/multi-agent_orchestrator_logistics_system)  
High-level goal: coordinate up to five agents to automate inventory, quoting, and fulfillment flows for a paper company.  
Technologies used: Python, Pydantic AI multi-agent graph, SQLite or SQLAlchemy-backed tools, structured outputs, OpenAI proxy integration.  
Scope: uses a compact graph of cooperating agents with shared tools and structured I/O, suited for quote-to-cash style flows.

P2. [AI-Powered Genetic Workflow for Product Management](https://github.com/Ohara124c41/agentic_ai/tree/main/AI-powered_genetic_workflow_for_product_mgmt)  
High-level goal: run persona-aware analysis, routing, and staged evaluation to generate and score product requirements.  
Technologies used: Python, OpenAI-compatible APIs, RAG over CSV knowledge stores, modular agent harness.  
Scope: orchestrates multi-step evaluation and reporting for PM artifacts with traceable intermediate products.

P3. [AI Research Agent - Video Game Industry](https://github.com/Ohara124c41/agentic_ai/tree/main/AI_research_agent_video_game_industry)  
High-level goal: answer executive queries on market structure using RAG over structured game data and vector embeddings.  
Technologies used: Python, ChromaDB with SQLite, retrieval and summarization agents, notebook-driven workflows.  
Scope: retrieval pipeline with lightweight vector store for exploratory industry analysis.

P4. [Multi-Agent Travel Assistant System](https://github.com/Ohara124c41/agentic_ai/tree/main/multi-agent_travel_assistant_system)  
High-level goal: prototype a travel concierge that plans itineraries, evaluates booking options, and coordinates customer messaging across specialized agents.  
Technologies used: Python, framework-agnostic multi-agent orchestration patterns, OpenAI-compatible models.  
Scope: modular assistants for planning, evaluation, and communications across a shared state.

### Building Regulation-Aware Multi-Agent Systems

P1. [Orphan Finder: Rare Disease Variant-to-Therapy Matchmaker](https://github.com/Ohara124c41/orphan-finder-variant-to-therapy-matcher)  
High-level goal: rank candidate variants with ClinVar, synthesize PMID-backed evidence from PubMed, and match to ClinicalTrials.gov trials to produce a clinician-style, auditable brief.  
Technologies used: Python 3.10, three-agent orchestration (Orchestrator, VariantPrioritizer, EvidenceSynthesizer, TrialMatcher), ClinVar queries, PubMed E-utilities, ClinicalTrials.gov API, JSONL run logging.  
Scope: integrates three public biomedical sources with an auditable JSONL trace of tool calls and selections, producing a report.md and ranked_variants.csv per run for reproducibility.

### Building Agents with Core Bioinformatics Tools

P1. [UdaciScan - An AI Research Agent for Drug-Repurposing Insights](https://github.com/Ohara124c41/building_agents_core_bioinformatics_tools)  
High-level goal: answer drug repurposing questions by retrieving PubMed abstracts, gating on retrieval confidence, extracting and ranking candidate drugs, and emitting reproducible briefs with traces.  
Technologies used: Python 3.10+, ChromaDB vector store, OpenAI-compatible endpoint via Vocareum, PubMed and NCBI integrations, YAML configuration, CLI runner with JSON/Markdown outputs.  
Scope: end-to-end CLI and notebook workflows with vector search, ranked outputs, and artifact logging.

### Generative AI

P1. [AI Photo Editing & Inpainting](https://github.com/Ohara124c41/generative_ai/tree/main/ai_photo_editing_inpainting)  
High-level goal: build an interactive app for targeted inpainting and style edits on user-uploaded photos.  
Technologies used: Python, Streamlit, diffusers or Stable Diffusion inpainting, Pillow, OpenCV.

P2. [Custom Architecture Framework Chatbot](https://github.com/Ohara124c41/generative_ai/tree/main/custom_architecture_framework_chatbot)  
High-level goal: create an LLM assistant that answers questions about a proprietary architecture governance framework using RAG.  
Technologies used: LangChain, OpenAI via Vocareum, ChromaDB or FAISS, pandas, Python notebooks and CLI.  
Scope: RAG pipeline with configurable ingestion and vector store, designed for governance corpus Q&A and reproducible CLI runs.

P3. [Lightweight Fine-Tuning of a Foundational Model](https://github.com/Ohara124c41/generative_ai/tree/main/lightweight_fine-tuning_foundational_model)  
High-level goal: fine-tune DistilBERT for sentiment classification with parameter-efficient techniques.  
Technologies used: Hugging Face Transformers, PEFT/LoRA, PyTorch, datasets, scikit-learn metrics.

P4. [HomeMatch - Personalized Real Estate Agent](https://github.com/Ohara124c41/generative_ai/tree/main/personalized_real-estate_agent)  
High-level goal: generate synthetic listings, store text and image embeddings, collect buyer preferences, and return GPT-personalized property narratives.  
Technologies used: LangChain, LangChain-OpenAI, ChromaDB, CLIP and Sentence Transformers, Pillow, pandas.

### Deep Reinforcement Learning

P1. [Navigation](https://github.com/Ohara124c41/DRLND-Navigation)  
High-level goal: train an agent to collect target items and avoid distractors in a Unity environment to maximize cumulative reward.  
Technologies used: PyTorch, DQN/Double DQN with experience replay and target networks, Unity ML-Agents environment.

P2. [Continuous Control](https://github.com/Ohara124c41/DRLND-Continuous_Control)  
High-level goal: learn a continuous control policy to keep a robotic arm’s end-effector at target positions in the Reacher environment.  
Technologies used: PyTorch, actor-critic (DDPG/TD3), Ornstein-Uhlenbeck noise, Unity ML-Agents environment.

P3. [Collaboration and Competition](https://github.com/Ohara124c41/DRLND-Collaborate-n-Competition)  
High-level goal: train two cooperative agents to sustain rallies in a shared tennis-like environment under multi-agent dynamics.  
Technologies used: PyTorch, multi-agent DDPG or MADDPG variants, parameter noise/exploration strategies, Unity multi-agent environment.

### Deep Learning

P1. [GAN Face Generator](https://github.com/Ohara124c41/deep_learning/tree/main/generative_adversarial_network_face_generator)  
High-level goal: synthesize realistic human faces from random latent vectors by training a GAN on CelebA.  
Technologies used: PyTorch, torchvision, DCGAN/WGAN-GP training loop with gradient penalty, NumPy, Matplotlib, Jupyter.

P2. [Landmark Classification & Social Media Tagging](https://github.com/Ohara124c41/deep_learning/tree/main/landmark_classification_tagging_social_media)  
High-level goal: identify which of 50 global landmarks appears in a user photo to enable automatic social-media tagging.  
Technologies used: PyTorch CNNs, transfer learning with pretrained backbones, PIL, numpy/pandas, Matplotlib/Seaborn, helper training utilities. 

P3. [MNIST Handwritten Digit Classifier](https://github.com/Ohara124c41/deep_learning/tree/main/MNIST_handwritten_digits)  
High-level goal: reach greater than 98 percent accuracy on MNIST with a compact MLP and documented training pipeline.  
Technologies used: PyTorch, torchvision datasets, BatchNorm and dropout, AdamW with LR scheduling, Jupyter.

P4. [Text Translation & Sentiment Analysis with Transformers](https://github.com/Ohara124c41/deep_learning/tree/main/text_translation_sentiment_analysis_transformers)  
High-level goal: translate multilingual movie reviews to English and attach sentiment labels to produce a unified analytics-ready CSV.  
Technologies used: Hugging Face Transformers, MarianMT translation, DistilBERT sentiment pipeline, pandas, PyTorch, sacremoses. 

### Computer Vision

P1. [Facial Keypoint Detection](https://github.com/Ohara124c41/CVND-Facial_Keypoint_Detection)  
High-level goal: localize facial landmarks to support downstream tasks such as tracking, pose estimation, or expression analysis.  
Technologies used: PyTorch CNNs, data augmentation, MSE loss, OpenCV-based visualization.

P2. [Image Captioning](https://github.com/Ohara124c41/CVND-Image_Captioning)  
High-level goal: generate natural language descriptions from images using an encoder-decoder architecture.  
Technologies used: CNN encoder (e.g., ResNet) with RNN/LSTM decoder, PyTorch, beam search or greedy decoding.

P3. [Landmark Detection & Robot Tracking](https://github.com/Ohara124c41/CVND-SLAM)  
High-level goal: detect landmarks and estimate robot pose in a mapping task using probabilistic state estimation.  
Technologies used: EKF-SLAM or particle filtering, NumPy, OpenCV, Python visualization.

### Natural Language Processing

P2. [Machine Translation](https://github.com/Ohara124c41/NLP-Machine_Translation/blob/master/machineTranslation/machineTranslation/machine_translation.ipynb)  
High-level goal: translate sentences from English to French with a sequence-to-sequence neural model.  
Technologies used: PyTorch, RNN/LSTM/GRU encoder-decoder with attention, tokenization, batching.

### Machine Learning with Pytorch

P1. [Finding Donors](https://github.com/Ohara124c41/MLND-Finding_Donors)  
High-level goal: predict potential charitable donors from census-like attributes to optimize outreach.  
Technologies used: scikit-learn (SVM, RandomForest, GradientBoosting), pandas, feature scaling, cross-validation.

P2. [Create Your Own Image Classifier](https://github.com/Ohara124c41/ML_PytorchND/tree/main/Create_Your_Own_Image_Classifier)  
High-level goal: implement transfer learning for image recognition with configurable hyperparameters and saved checkpoints.  
Technologies used: PyTorch, torchvision models, command-line training and inference utilities.

P3. [Identifying Customer Segments](https://github.com/Ohara124c41/ML_PytorchND/tree/main/Identifying_Customer_Segments)  
High-level goal: discover latent customer groups to support targeted marketing and product decisions.  
Technologies used: PCA for dimensionality reduction, KMeans clustering, scikit-learn, pandas.

### Machine Learning

P1. [Predicting Boston Housing Prices](https://github.com/Ohara124c41/MLND-Predicting_Boston_Housing_Pricing)  
High-level goal: model median home value as a function of neighborhood features and evaluate generalization.  
Technologies used: scikit-learn regressors, grid search, learning curves, cross-validation.

P2. [Finding Donors](https://github.com/Ohara124c41/MLND-Finding_Donors)  
High-level goal: rank prospective donors to improve campaign efficiency and ROI.  
Technologies used: scikit-learn classifiers, feature engineering, model selection and evaluation metrics.

P3. [Titanic Survival Exploration](https://github.com/Ohara124c41/MLND-Titanic_Survival_Exploration)  
High-level goal: predict passenger survival to illustrate supervised learning pipelines and error analysis.  
Technologies used: decision trees, ensemble baselines, scikit-learn, pandas.

P4. [Smart Cab](https://github.com/Ohara124c41/MLND-Smartcab)  
High-level goal: learn a driving policy that obeys rules and reaches destinations in a simulated city.  
Technologies used: Q-learning, epsilon-greedy exploration, Python simulation.

P5. [Customer Segments](https://github.com/Ohara124c41/MLND-Customer_Segments)  
High-level goal: segment customers to enable differentiated strategies for retention and growth.  
Technologies used: PCA, clustering (KMeans), scikit-learn, pandas.

P6. [Dog Breed Classifier](https://github.com/Ohara124c41/MLND-Dog_Breed_Classifier)  
High-level goal: detect whether an image contains a dog or human and classify the dog breed.  
Technologies used: CNN transfer learning (e.g., VGG, ResNet), PyTorch, data augmentation.

P7. [Plagiarism Detection](https://github.com/Ohara124c41/MLNDT-Beta-Plagiarism_Detection)  
High-level goal: identify potential plagiarism by measuring document similarity against known sources.  
Technologies used: engineered text similarity features, SVM or Logistic Regression, scikit-learn.

P8. [Capstone: Deep Traffic (MIT)](https://github.com/Ohara124c41/MLND-Capstone-DeepTraffic)  
High-level goal: learn a high-throughput driving policy in a browser-based multi-lane traffic simulator.  
Technologies used: deep reinforcement learning policy networks in JavaScript, simulator-provided APIs, hyperparameter search.

### GitHub Copilot with JavaScript

P1. [GitHub Copilot Starter](https://github.com/Ohara124c41/github-copilot-starter)  
High-level goal: refactor a simple Sudoku game to a modern React and Next.js web app, adding generator, timer, hints, difficulty selection, validation, and persistence of top scores.  
Technologies used: JavaScript, React, Next.js, HTML, CSS, optional Tailwind or modular CSS, localStorage for leaderboard.

### AI Programming 

P1. [Image Classification for a City Dog Show](https://github.com/Ohara124c41/AI_programming/tree/main/image_classification_city_dog_show)  
High-level goal: classify images of dogs and humans and identify dog breeds using a pre-trained convolutional neural network.  
Technologies used: Python, PyTorch, torchvision, NumPy, PIL, transfer learning.

P2. [Create Your Own Image Classifier](https://github.com/Ohara124c41/AI_programming/tree/main/create_your_own_image_classifier)  
High-level goal: build and train a flower classifier and expose a command-line interface for training and inference.  
Technologies used: PyTorch, torchvision, argparse, PIL, checkpointing for transfer learning.

---

<a id="autonomous-systems"></a>
## 🤖 School of Autonomous Systems ✈️

### Sensor Fusion

P1. [LiDAR Obstacle Detection](https://github.com/Ohara124c41/SFND_Lidar_Obstacle_Detection)  
High-level goal: separate ground from obstacles and cluster objects in 3D point clouds.  
Technologies used: C++, PCL, RANSAC plane fitting, KD-Tree Euclidean clustering.

P2. [2D Object Tracking](https://github.com/Ohara124c41/SFND_2D_Feature_Tracking)  
High-level goal: detect and track keypoints across frames to measure motion and build correspondences.  
Technologies used: C++, OpenCV (detectors/descriptors), descriptor matching.

P3. [3D Object Tracking](https://github.com/Ohara124c41/SFND_3D_Object_Tracking)  
High-level goal: associate camera detections with lidar points and track objects over time.  
Technologies used: C++, OpenCV, PCL, bounding box association, time-to-collision estimation.

P4. [Radar Target Generation and Detection](https://github.com/Ohara124c41/SFND-RADAR-Target_Generation-n-Detection)  
High-level goal: simulate FMCW radar signals and implement CFAR to detect targets.  
Technologies used: MATLAB/Octave signal processing, CFAR, range-Doppler maps.

P5. [Unscented Kalman Filter](https://github.com/Ohara124c41/SFND_Unscented_Kalman_Filter)  
High-level goal: estimate object state under nonlinear dynamics using UKF on lidar and radar measurements.  
Technologies used: C++, Eigen, CTRV process model, sigma-point filtering.

### Autonomous Flight (Flying Cars)

P1. [Backyard Flyer](https://github.com/Ohara124c41/FCND-Drone-Backyard_Flyer)  
High-level goal: plan and execute a simple waypoint mission to validate control and state estimation.  
Technologies used: Python, drone simulator APIs, waypoint planning.

P2. [3D Motion Planner](https://github.com/Ohara124c41/FCND-Drone-3D_Motion_Planning)  
High-level goal: compute collision-free global paths in a mapped environment.  
Technologies used: Python, occupancy grids, A* search, heuristic tuning.

P3. [Building a Controller](https://github.com/Ohara124c41/FCND-Drone-Building_a_Controller)  
High-level goal: design attitude and position controllers for a quadrotor to track trajectories.  
Technologies used: C++ or Python control loops, PID/linear controllers, simulator integration.

P4. [Building an Estimator](https://github.com/Ohara124c41/FCND-Drone-Building_an_Estimator)  
High-level goal: estimate vehicle state from noisy IMU and GPS using probabilistic filtering.  
Technologies used: C++ Extended Kalman Filter, sensor fusion, tuning and validation.

### Robotics

P1. [Autonomous Search Rover](https://github.com/Ohara124c41/RoboND-Autonomous_Search_Rover)  
High-level goal: perceive obstacles and navigable terrain to autonomously explore and retrieve samples.  
Technologies used: Python, OpenCV perception pipeline, decision logic, telemetry logging.

P2. [Kinematics (Kuka-KR210)](https://github.com/Ohara124c41/RoboND-Kinematics-Kuka-KR210)  
High-level goal: compute forward and inverse kinematics for a 6-DOF manipulator and validate poses.  
Technologies used: Python, ROS, DH parameterization, SymPy for IK.

P3. [3D Perception (PR2)](https://github.com/Ohara124c41/RoboND-PR2-3D_Perception)  
High-level goal: segment scenes and recognize tabletop objects for pick-and-place tasks.  
Technologies used: ROS, PCL (voxel grid, RANSAC), clustering, SVM or template matching.

P4. [Follow Me](https://github.com/Ohara124c41/RoboND-Follow_Me)  
High-level goal: detect and follow a target person in video streams.  
Technologies used: semantic segmentation or object detection CNNs, ROS nodes for perception and control.

P5. [Robotic Inference](https://github.com/Ohara124c41/RSEND-Robotic_Inference)  
High-level goal: apply deep learning inference for robotic perception or manipulation tasks.  
Technologies used: TensorFlow or PyTorch models, ROS integration, inference optimization.

P6. [Where am I?](https://github.com/Ohara124c41/RSEND-Localization-WhereAmI)  
High-level goal: localize a robot within a known map using probabilistic techniques.  
Technologies used: ROS AMCL or particle filter localization, sensor models, map server.

P7. [Map My World](https://github.com/Ohara124c41/RSEND-Map_My_World)  
High-level goal: build a map while estimating pose from onboard sensors.  
Technologies used: ROS with RTAB-Map or SLAM backends, bagging and visualization tools.

P8. [Deep Reinforcement Learning Arm (Kuka KR-210)](https://github.com/Ohara124c41/RSEND-KUKA-DeepRL_Arm)  
High-level goal: learn a control policy for a manipulator to reach or grasp under uncertainty.  
Technologies used: PyTorch RL algorithms, robotics simulator interfaces, reward shaping.

P9. [Home Service Robot](https://github.com/Ohara124c41/RSEND-Home_Service_Robot)  
High-level goal: perform autonomous pick-and-deliver tasks in a house map.  
Technologies used: ROS Navigation stack (mapping, localization, planning), Gazebo, custom state machine.

---

<a id="data-science"></a>
## 🔬 School of Data Science 💾

### Data Architect

P1. [Design an HR Database](https://github.com/Ohara124c41/DataArchitectND/tree/main/Design_an_HR_Database)  
High-level goal: elicit business requirements and design a normalized relational schema for HR operations.  
Technologies used: ER modeling, PostgreSQL 9.5 DDL, access control and backup considerations.

P2. [Design a Data Warehouse](https://github.com/Ohara124c41/DataArchitectND/tree/main/Design_a_Data_Warehouse)  
High-level goal: model analytical workloads with star schemas and implement an end-to-end ETL pipeline.  
Technologies used: dimensional modeling, PostgreSQL, SQL-based ETL, performance tuning.

### Data Analyst

P1. [Investigate a Dataset](https://github.com/Ohara124c41/data_analyst/tree/main/investigate_dataset)  
High-level goal: explore and analyze a firearm and census dataset, articulate key questions, and document visual findings.  
Technologies used: Jupyter Notebook, pandas, NumPy, Matplotlib, exploratory data analysis workflow.

P2. [Cybersecurity Data Wrangling](https://github.com/Ohara124c41/data_analyst/tree/main/cybersecurity_data_wrangling)  
High-level goal: gather, assess, clean, and analyze Kaggle Cyber Security Attacks and MITRE ATT&CK datasets, including an optional graph export.  
Technologies used: pandas, NumPy, programmatic data collection, data quality and tidiness fixes, NetworkX for graph representation, GraphQL.

P3. [Communicate Nuclear IAEA Data Findings](https://github.com/Ohara124c41/data_analyst/tree/main/communicate_nuclear_IAEA_data)  
High-level goal: produce exploratory and explanatory visualizations on nuclear infrastructure and electricity statistics using merged country-level data.  
Technologies used: Jupyter Notebook, pandas, Matplotlib, HTML exports for presentation-ready deliverables. 

### Statistics for Data Analysis

P1. [Salary Prediction via Linear Regression](https://github.com/Ohara124c41/statistics_for_data_analysis/tree/main/model_salary_data_linear_regression)  
High-level goal: build, fit, and interpret a salary prediction model using linear regression with clear diagnostics and interpretation.  
Technologies used: Python, pandas, NumPy, statsmodels or scikit-learn linear models, Matplotlib, Jupyter.

P2. [A/B Test Results Analysis](https://github.com/Ohara124c41/statistics_for_data_analysis/tree/main/analyze_a-b_test_results)  
High-level goal: evaluate an online experiment for conversion differences using hypothesis tests and effect size.  
Technologies used: pandas, SciPy statistical tests, confidence intervals, Matplotlib, Jupyter.

P3. [Basketball Scoring Probabilities](https://github.com/Ohara124c41/statistics_for_data_analysis/tree/main/calculate_basketball_scoring_probabilities)  
High-level goal: model and compute scoring probabilities for game scenarios with worked, notebook-based examples.  
Technologies used: probability rules, combinatorics, NumPy, pandas, Jupyter.

P4. [Health and Sleep Quality Description](https://github.com/Ohara124c41/statistics_for_data_analysis/tree/main/describe_health_sleep_quality_data)  
High-level goal: explore and describe health and sleep quality data to surface descriptive insights and visual patterns.  
Technologies used: pandas, descriptive statistics, Matplotlib or Seaborn visualization, Jupyter.

### Data Foundations

P1. [Flight Delays and Cancellations](https://github.com/Ohara124c41/DFND-Flight-Delays-and-Cancellations)  
High-level goal: analyze air travel performance and factors driving delays and cancellations.  
Technologies used: SQL queries, relational joins and aggregations, data visualization.

---

<a id="business"></a>
## 💼 School of Business

### Agentic AI for Business Leaders

P1. [Agentic Expense Reporting System](https://github.com/Ohara124c41/AIBLND-Delivering_an_ML-AI_Strategy/tree/master/Agentic-AI-for-Business-Leaders)  
High-level goal: automate receipt parsing, policy checks, and payment decisions through a multi-agent workflow.  
Technologies used: OCR and data extraction, spreadsheet APIs, rule evaluation, payment service integration.

### Artificial Intelligence for Business Leaders

P1. [Delivering an ML/AI Strategy](https://github.com/Ohara124c41/AIBLND-Delivering_an_ML-AI_Strategy)  
High-level goal: define a practical ML/AI strategy aligned with product and organizational constraints.  
Technologies used: strategic planning artifacts, data maturity assessment, roadmap and KPI definition.

### Artificial Intelligence for Product Managers

P1. [Create a Medical Image Data Annotation Job](https://github.com/Ohara124c41/AIPMND-AI_Product_Manager/tree/master/P1-Create_a_Medical_Image_Data_Annotation_Job)  
High-level goal: specify a labeling task with clear guidelines to ensure high-quality medical image annotations.  
Technologies used: labeling tool configuration, JSON schemas, quality control workflows.

P2. [Build a Model with Google AutoML](https://github.com/Ohara124c41/AIPMND-AI_Product_Manager/tree/master/P2-Build_a_Model_with_Google_AutoML)  
High-level goal: train and evaluate a classification model using a managed AutoML service.  
Technologies used: Google Cloud AutoML, dataset curation, evaluation metrics and model export.

P3. [Capstone: AI for Space](https://github.com/Ohara124c41/AIPMND-AI_Product_Manager/tree/master/P3-Capstone_Proposal-AI_for_Space)  
High-level goal: propose an AI product concept for space applications with problem framing and feasibility.  
Technologies used: product requirement definition, risk and stakeholder analysis, experimental roadmap.

### Project Management

P2. [Design Sprint](https://github.com/Ohara124c41/PMND-Project_Management_Nanodegree/tree/master/C2-Design_Sprint)  
High-level goal: validate a product idea through a structured multi-day sprint workflow.  
Technologies used: sprint artifacts, user journey mapping, prototyping and feedback capture.

### Digital Marketing

P2. [Facebook Campaign](https://github.com/Ohara124c41/DMND-Facebook_Campaign)  
High-level goal: plan, launch, and analyze a paid social campaign against business objectives.  
Technologies used: Facebook Ads Manager, audience targeting, KPI tracking and reporting.

P5. [Udacity Enterprise Portfolio](https://github.com/Ohara124c41/DMND-Digital_Marketing)  
High-level goal: present a cohesive digital marketing portfolio for enterprise stakeholders.  
Technologies used: portfolio structuring, analytics summaries, content curation.

---

<a id="programming"></a>
## 💻 School of Programming and Software Development 🧑‍💻

### Virtual Reality

P1. [Carnival](https://github.com/Ohara124c41/VRND/tree/master/Project-1-Carnival)  
High-level goal: build interactive VR mini-games to demonstrate physics and interaction design.  
Technologies used: Unity, C#, VR interaction frameworks.

P2. [Design an Apartment](https://github.com/Ohara124c41/VRND)  
High-level goal: create a navigable VR interior scene with lighting and materials suitable for presence.  
Technologies used: Unity, C#, scene layout and optimization.

P3. [Maze](https://github.com/Ohara124c41/VRND/tree/master/Project-3-Maze)  
High-level goal: implement a VR maze experience with locomotion and collision handling.  
Technologies used: Unity, C#, VR controller input and physics.

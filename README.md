# Towards Edge-Assisted Trajectory Prediction for Connected Autonomous Vehicles
Mehdi Salim Benhelal, Badii Jouaber, Hossam Afif, Hassine Moungla

## Abstact

Trajectory prediction has been identified as a challenging critical task for achieving a full autonomy of the connected and autonomous vehicles (CAVs), despite advancement of Communication technologies  only few studies includes the connectivity aspect, thus we introduce a novel Edge-Assisted clustering architecture that takes advantage of recent deep learning models and the evolution of edge technologies to achieve better forecasting. first the historical positions of the target vehicle are fed to the base models of all CAVs in the scene, resulting in multiple generated predictions then each are sent to an edge server where trajectories clustering is performed using DBSCAN algorithm to obtain multiple partitions with similar trajectories, the largest cluster is averaged then broadcast to all CAVs in the scene, our proposed method achieves state-of-the-art results on the real-world trajectory prediction nuScenes vehicles dataset, obtaining better predictions up to 21%, we also demonstrate the robustness of our method against single agent's system failures succeeding to get satisfactory results due to it’s ability to detect outliers. System  practicality is studied under the current 5G/6G capabilities.


<img src="https://user-images.githubusercontent.com/130567644/236206993-097f1088-af9d-442f-bd14-2c98b7e42bdb.png" width=50% height=50%>

## Setting up and testing
To set up the environment and dataset follow the same steps used in https://github.com/StanfordASL/Trajectron-plus-plus eccv2020 paper, the dataset can be found at https://www.nuscenes.org/nuscenes

To test the codebase, you can run the following command in your terminal under Towards-Edge-Assisted-Trajectory-Prediction-for-Connected-Autonomous-Vehicles directory:


python clustering.py --model nuScenes/models/yesdynamics --checkpoint=12 --model2 nuScenes/models/nodynamics --checkpoint2=12 --model3 nuScenes/models/plusmap --checkpoint3=12 --epsilon=10 --data processed/nuScenes_test_full.pkl --output_path nuScenes/results --output_tag clustering --node_type VEHICLE
--prediction_horizon 2


This command will execute the clustering script and produce predictions based on the given models and checkpoints. The --model and --model2 arguments specify the dynamic and non-dynamic models, while --model3 specifies the map model. The --checkpoint, --checkpoint2, and --checkpoint3 arguments specify the respective checkpoints to use. The --epsilon argument sets the epsilon value for the clustering algorithm. The --data argument specifies the input data file, while --output_path and --output_tag specify the output file and tag, respectively.


The output should be: 


![image](https://user-images.githubusercontent.com/130567644/236200982-22b3b080-a5e2-4a83-9785-7cffad42bdd7.png)
## Credits
This codebase is built upon Trajectron++: Dynamically-Feasible Trajectory Forecasting With Heterogeneous Data by Tim Salzmann*, Boris Ivanovic*, Punarjay Chakravarty, and Marco Pavone (* denotes equal contribution) ECCV2020 paper. We would like to thank the authors for their valuable contribution to the field of autonomous vehicle trajectory prediction and for making their work publicly available.

# Towards-Edge-Assisted-Trajectory-Prediction-for-Connected-Autonomous-Vehicles
Codebase of the paper: Towards Edge-Assisted Trajectory Prediction for Connected Autonomous Vehicles
## Paper abstract 
Trajectory prediction has been identified as a challenging critical task for achieving a full autonomy of the connected and autonomous vehicles (CAVs), despite advancement of Communication technologies  only few studies includes the connectivity aspect, thus
we introduce a novel Edge-Assisted clustering architecture
that takes advantage of recent deep learning models and the evolution of edge technologies to achieve better forecasting. first the historical positions of the target vehicle are fed to the base models of all CAVs in the scene, resulting in multiple generated predictions then each are sent to an edge server where trajectories clustering is performed using DBSCAN algorithm to obtain multiple partitions with similar trajectories, the largest cluster is averaged then broadcast to all CAVs in the scene, our proposed method achieves state-of-the-art results on the real-world trajectory prediction nuScenes vehicles dataset, obtaining better predictions up to 21%, we also demonstrate the robustness of our method against single agent's system failures succeeding to get satisfactory results due to it’s ability to detect outliers. System  practicality is studied under the current 5G/6G capabilities.

## Setting up and testing
To set up the environement for testing you should follow the same steps as in https://github.com/StanfordASL/Trajectron-plus-plus eccv2020 paper the nuScenes dataset can be found at: https://www.nuscenes.org/nuscenes

To test the code base you can execute the fellowing command

python clustering.py --model nuScenes/models/yesdynamics --checkpoint=12 --model2 nuScenes/models/nodynamics --checkpoint2=12 --model3 nuScenes/models/plusmap --checkpoint3=12 --epsilon=10 --data processed/nuScenes_test_full.pkl --output_path nuScenes/results --output_tag clustering --node_type VEHICLE --prediction_horizon 8

This command will execute the clustering script and produce predictions based on the given models and checkpoints. The --model and --model2 arguments specify the dynamic and non-dynamic models, while --model3 specifies the map model. The --checkpoint, --checkpoint2, and --checkpoint3 arguments specify the respective checkpoints to use. The --epsilon argument sets the epsilon value for the clustering algorithm. The --data argument specifies the input data file, while --output_path and --output_tag specify the output file and tag, respectively. The --node_type argument specifies the node type to use, while --prediction_horizon sets the prediction horizon.

This will output the following result:

![image](https://user-images.githubusercontent.com/130567644/236187217-2b93c6f8-c04f-4d9b-91d5-99f3bd0c1481.png)




.

## Credits
This codebase is built upon  Trajectron++: Dynamically-Feasible Trajectory Forecasting With Heterogeneous Data by Tim Salzmann*, Boris Ivanovic*, Punarjay Chakravarty, and Marco Pavone (* denotes equal contribution) ECCV2020 paper. We would like to thank the authors for their valuable contribution to the field of autonomous vehicle trajectory prediction and for making their work publicly available. If you use this codebase for your own research, please consider citing the Trajectron++ 


We hope you find this repository useful! If you have any questions or feedback, please don't hesitate to contact us

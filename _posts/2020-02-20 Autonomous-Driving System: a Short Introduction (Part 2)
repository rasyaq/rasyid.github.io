After briefly introduced the history/background and the levels of autonomous driving in this article, I will continue to briefly discuss, a very light and to the best of my knowledge -which is of course still limited-, the overall system needed for autonomous driving.
There are at least four points that may enable a car to perform autonomous driving task: Sensing, Localization/Mapping, Planning, and Cooperating.

Sensing
Let's start with Sensing. An autonomous driving car should be able to sense its surrounding.
By sensing, it means the car is aware of what and where is the road, lane, cars, pedestrians, traffic sign, and other possible objects or obstacles likely be.
To sense its surroundings, a car needs sensor(s). Up to now, there is no perfect sensor capable to sense everything, everywhere, at anytime.
Nevertheless, three popular sensors used by autonomous driving companies are camera, radar, and lidar. So let's talk briefly about these three superheroes.
Camera
Camera's superpower is just like a human eye, "seeing". It has the ability to distinguish/categorize objects, but its kryptonite is (at least now) bad weather conditions. Night time is also a challenge, but still solvable to some extent.
Radar
Radar's superpower is in its ability to precisely locate and measure object location and its velocity, more robust on day and/or night, on various weather conditions. Its weakness point is however, to categorize different objects (although it is possible, but limited) semantically as powerful as what camera can do.
Lidar
Lidar's superpower is in its semi-combined ability of camera (to construct a precise object shape to classify its type) and radar (to precisely identify location and measure object's velocity). However, its kryptonite's is unfortunately its own price,which may weaken its adoption rate compared to camera and radar which is relatively cheaper.
Lastly, since each sensor has its own power and limitation, the best way to obtain better performance is to combine several if not all the available sensor. This is usually called as sensor fusion.


Localization and Mapping
A self-driving car needs to figure out where are they (localization) and where are the others (destination, other objects). This method is popularly known as simultaneous localization and mapping (SLAM).
Mapping in SLAM usually refers to a real-time/online mapping the surroundings/unknown environment when the car is in driving mode. This allows the car to update the environment information and make decision by short term prediction dynamically over time (potential collision with other vehicles or pedestrian). Although the results of SLAM somehow can be recorded as an offline map (and  its localization recording refers to car history trajectory),  another aspect of offline mapping on the other hands, provides an extremely detailed information which is mostly static traffic/road environment and most likely fixed over time (intersection location, how many traffic lights will be passed before reaching the destination, lanes, etc).Offline mapping for autonomous driving is commonly called high definition (HD) map. It provides detailed information up to centimeter-level precision.

Both online and offline mapping is necessary and leads to the following topic: Planning.


Planning
Planning is arguably one of the most complex tasks in autonomous driving other than sensing. It requires the best possible decision and action to reach the driving destination and guarantee safety given various driving/traffic scenarios.
I generally divide planning into two categories:
1. Global Planning
In global planning, the main task is to estimate the best possible and efficient path from point A (say, home) to point B (say, office). Not only the path (where to go) which is being planned, but the vehicle action (what to do and when) as well. In global planning, HD map plays a major role. For example, if we know in the next 1 km there will be a traffic light, or in one particular road there will be a speed limit traffic sign, the vehicle will adjust the path and action based on that information.
2. Local Planning
Local planning is the mind of self driving car safety. Its main task is to ensure safety during the driving where many potentially unpredictable situations may appear which could endanger drivers, passengers, and pedestrians.
Basically local planning can be approached in at least two ways: pre-programmed rules, and a machine learning based approach (human-driver behavior mimicking and self-learning).
Pre-programmed rules basically consider all possible traffic scenarios and write the rules (specific or general) in the code to decide which action to be chosen given a traffic scenario.
Machine learning approach on the other hand, instead of writing the rule one by one up to enormous amount of rules, we let the computer learns from the data (suppose to be a large amount of data), and fit it to a complex enough mathematical model/architecture designed by the engineer beforehand. At this moment, neural-network based method (now popularly known as deep learning) is taking the main stage in the machine learning field.
Human-driver behavior mimicking/cloning how the human driver will act given a particular traffic scenario. Self-learning on the other hand, let the computer learn by itself how to choose the best possible action, after a large number of trials, by giving the algorithm reward if the action is chosen to be aligned with the goal, say, no collision occurs. Self-learning or reward based learning is also called reinforcement learning.

Cooperating
In order to increase safety to the highest level possible, cooperation between different entities in the road/traffic ecosystem is necessary. Cooperation is expected to reduce the probability of collision/crash during driving on the road.
As a basic form of cooperation, we need it between vehicles (called V2V) to communicate and exchange information regarding its state and action.
In practice, not only cooperation between vehicles are necessary, but also between vehicle and everything surrounding it, so called V2X.


Summary
That is a brief and light review on overall system of autonomous driving to ensure safety and comfortability.
The level 5 of autonomous driving level may still be felt distant to reach, but we are going there for sure. In near future, it will not be surprising if there is another creative/breakthrough way to accelerate the feasibility of autonomous driving in the future.. very very possibly, nearer than we thought.

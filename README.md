# GenAI
**House Price Prediction**  
**Blue scatter points** → Actual data points from our dataset.  
**Red surface** → Regression plane showing predicted house prices based on size and bedrooms.  
This helps visualize how the model fits the data in three dimensions.  
This 3D plot gives us a clear picture of how the regression model predicts house prices based on two features. 

**MultVariate Linear Regression**  
**Intercept** → Baseline sales when all budgets are zero.  
**Coefficients** → Contribution of each budget channel to sales.  
Example: If TV coefficient = 0.05, then every extra $1 in TV budget increases sales by 0.05 units (holding other budgets constant).  
**Prediction** → You can input new budget allocations and get estimated sales.  
This regression model quantifies how each advertising channel contributes to product sales. Typically, TV and Radio budgets have stronger positive effects, while Newspaper often shows weaker or inconsistent influence.  

**Pima Neural Network**  
	•	Input layer: 8 neurons (for the 8 features in the dataset: pregnancies, glucose, blood pressure, etc.).  
	•	Hidden layers:  
	◦	First hidden layer: 12 neurons, ReLU activation.  
	◦	Second hidden layer: 8 neurons, ReLU activation.  
	•	Output layer: 1 neuron with sigmoid activation (predicts probability of diabetes).  
	•	Loss function: Binary cross‑entropy (since it’s a classification problem).  
	•	Optimizer: Adam (efficient gradient descent).  
The plot_model function generates a diagram (pima_nn.png) showing the neural network structure with layer names and shapes.  
This program sets up a basic feed‑forward neural network for diabetes prediction. You can train it with model.fit(X, y, epochs=100, batch_size=10) to actually learn from the dataset, and the diagram will give you a clear visualization of the architecture.  

**Predic Employee Attrition Using KNN Algorithm**  
**LabelEncoder** → Converts JobRole into numeric values so KNN can process it.  
**Train test split** → Splits data into training and testing sets (70/30).  
**KNN model** → Uses n_neighbors=3 (you can tune this hyperparameter).  
**Evaluation** → Confusion matrix and classification report show precision, recall, and F1-score.  
**Prediction** → You can input a new employee’s details to predict attrition.  
**StandardScaler** → Transforms each feature to have mean = 0 and standard deviation = 1.  
This KNN classifier gives you a baseline model for predicting attrition. With more data, you can tune the number of neighbors (k), scale features (important for KNN), and test different distance metrics to improve accuracy.  

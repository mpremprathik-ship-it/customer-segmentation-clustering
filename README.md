*# Customer Segmentation Using Clustering*



*## 1. Project Overview*



*Customer Segmentation Using Clustering is an unsupervised machine learning project that groups customers according to their age, annual income, and spending behaviour.*



*The project helps identify different customer groups so businesses can better understand their customers and plan targeted marketing campaigns, promotions, and customer retention strategies.*



*## 2. Objectives*



*- Understand and validate the customer dataset.*

*- Perform exploratory data analysis (EDA).*

*- Select relevant features for clustering.*

*- Scale numerical features before applying clustering algorithms.*

*- Apply K-Means, Agglomerative Clustering, and DBSCAN.*

*- Evaluate clustering results using suitable metrics.*

*- Visualize customer groups and interpret their characteristics.*

*- Provide a dashboard for exploring results and assigning a cluster to a new customer.*



*## 3. Dataset*



*\*\*Dataset:\*\* Mall Customers*



*\*\*Source file:\*\* `data/raw/Mall\_Customers.csv`*



*The dataset contains 200 customer records and five columns:*



*| Column | Description |*

*|---|---|*

*| CustomerID | Unique customer identifier |*

*| Gender | Customer gender |*

*| Age | Customer age |*

*| Annual Income (k$) | Annual income in thousands of dollars |*

*| Spending Score (1-100) | Spending score on a scale of 1 to 100 |*



*The clustering features are:*

*- Age*

*- Annual Income (k$)*

*- Spending Score (1-100)*



*CustomerID is an identifier and is not used as a clustering feature.*



*## 4. Methodology*



*1. \*\*Data understanding:\*\* Inspect the dataset structure, summary statistics, and feature distributions.*

*2. \*\*Data quality checks:\*\* Check missing values, duplicate records, and potential outliers.*

*3. \*\*Exploratory data analysis:\*\* Study customer characteristics and relationships between features.*

*4. \*\*Feature selection:\*\* Select age, annual income, and spending score.*

*5. \*\*Feature scaling:\*\* Standardize numerical features so differences in their scales do not dominate distance calculations.*

*6. \*\*Clustering:\*\* Apply K-Means, Agglomerative Clustering, and DBSCAN.*

*7. \*\*Model evaluation:\*\* Use Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Score where applicable.*

*8. \*\*Cluster interpretation:\*\* Examine customer groups and their characteristics.*

*9. \*\*Visualization:\*\* Display results through interactive charts and a Streamlit dashboard.*



*## 5. Algorithms Used*



*### K-Means*



*K-Means groups customers into a selected number of clusters. The Elbow Method and Silhouette Score are used to help select an appropriate number of clusters.*



*### Agglomerative Clustering*



*Agglomerative Clustering builds groups hierarchically by merging similar observations.*



*### DBSCAN*



*DBSCAN identifies dense groups and can label observations that do not belong to a dense group as noise.*



*## 6. Evaluation Results*



*The following results were obtained from the current model evaluation run.*



*| Metric | K-Means (K=5) | Agglomerative | DBSCAN |*

*|---|---:|---:|---:|*

*| Silhouette Score | 0.4166 | 0.3900 | 0.4817 |*

*| Davies-Bouldin Index | 0.8746 | 0.9163 | 0.6386 |*

*| Calinski-Harabasz Score | 125.1009 | 107.8266 | 117.8049 |*

*| Clusters reported | 5 | 5 | 6 |*

*| Noise observations | 0 | 0 | 60 |*



*\*\*Important evaluation note:\*\* The DBSCAN metrics were calculated on the 140 observations assigned to clusters, excluding its 60 noise observations. The K-Means and Agglomerative metrics were calculated on all 200 observations. Therefore, these scores are not a directly comparable, like-for-like ranking of the algorithms.*



*For the K-Means experiment, the highest Silhouette Score among the tested values K=2 to K=10 was 0.4284 at K=6. K=5 obtained 0.4166. The final choice of K should also consider the Elbow Method, cluster profiles, and business interpretability.*



*## 7. Dashboard Features*



*The Streamlit dashboard provides:*



*- \*\*Overview:\*\* Dataset summary and key charts.*

*- \*\*Data Exploration:\*\* Interactive visualizations of customer features.*

*- \*\*Model Comparison:\*\* Evaluation results for the clustering algorithms.*

*- \*\*Customer Segments:\*\* Cluster visualizations and customer profiles.*

*- \*\*New Customer:\*\* Assign a new customer's scaled feature values to a K-Means cluster.*

*- \*\*CSV Export:\*\* Download the segmented customer data.*



*## 8. Technologies Used*



*- Python*

*- Pandas*

*- NumPy*

*- Scikit-learn*

*- Matplotlib*

*- Seaborn*

*- Plotly*

*- Streamlit*

*- Jupyter Notebook*

*- Git and GitHub*



*## 9. Project Structure*



*```text*

*customer-segmentation/*

*├── app/*

*│   └── app.py*

*├── data/*

*│   └── raw/*

*│       └── Mall\_Customers.csv*

*├── notebooks/*

*│   ├── Dataset\_Understanding.ipynb*

*│   ├── 02\_Data\_Preprocessing.ipynb*

*│   ├── eda\_feature\_selection\_scaling.ipynb*

*│   ├── 05\_Elbow\_Silhouette.ipynb*

*│   ├── k\_means.ipynb*

*│   ├── Cluster\_Analysis.ipynb*

*│   └── Visualisation\_08.ipynb*

*├── requirements.txt*

*├── .gitignore*

*└── README.md*

*```*



*Additional project folders or helper files may be present in the local repository.*



*## 10. Installation and Execution*



*### Prerequisites*



*- Python installed*

*- Git installed*

*- The project repository cloned locally*



*### Setup*



*1. Clone the repository:*



&#x20;  *```bash*

&#x20;  *git clone https://github.com/prasannajampa/customer-segmentation-clustering.git*

&#x20;  *cd customer-segmentation-clustering*

&#x20;  *```*



*2. Create a virtual environment:*



&#x20;  *```bash*

&#x20;  *python -m venv venv*

&#x20;  *```*



*3. Activate it on Windows PowerShell:*



&#x20;  *```powershell*

&#x20;  *.\\venv\\Scripts\\Activate.ps1*

&#x20;  *```*



*4. Install dependencies:*



&#x20;  *```bash*

&#x20;  *pip install -r requirements.txt*

&#x20;  *```*



*5. Ensure the dataset is available at `data/raw/Mall\_Customers.csv`.*



*6. Start the dashboard from the repository root:*



&#x20;  *```bash*

&#x20;  *streamlit run app/app.py*

&#x20;  *```*



*7. Open the local URL displayed in the terminal.*



*## 11. Limitations and Future Improvements*



*- The dataset contains only 200 customers, so findings may not generalize to every business.*

*- Cluster results depend on feature selection, scaling, and algorithm parameters.*

*- DBSCAN labels 60 observations as noise in the current experiment; its parameters and results need further investigation.*

*- The final number of K-Means clusters should be justified using both evaluation metrics and customer-profile interpretation.*

*- Future improvements could include testing additional clustering methods, evaluating cluster stability, and using larger real-world datasets.*



*## 12. Conclusion*



*This project demonstrates how unsupervised machine learning can be used to explore customer behaviour and discover customer segments. It combines data quality checks, exploratory analysis, feature scaling, clustering, evaluation metrics, interactive visualizations, and a dashboard for examining customer groups.*



*The results can support customer-focused marketing and service decisions, while further validation is needed before applying the segments to real business decisions.*



*## 13. License and Acknowledgements*



*Add the dataset source and license details here after verifying the dataset's original source and applicable terms.*




# AI-Detector

## Project Flow:
1. Data Collection 
2. Preparing Dataset -> a. AI-Text Generation, b. Data Cleaning
3. Model Building & Evaluation -> a. Feature Engineering, b. Exploratory Data Analysis & Visualisation, c. ML Modelling and Evaluation

## Business use case:
The advanced text-generating AI models poses a dual-edged sword: while they are efficient in content creation and in simplifying & automating many tasks, they also raise concerns over the authenticity of the content. The ability to identify AI-generated texts is important for:

- Academic Integrity: Preventing plagiarism and ensuring students’ work is genuinely their own.
- Content Verification: Helping publishers and content platforms in maintaining trustworthiness.
- Information Quality: Ensuring that readers and consumers can rely on the genuineness of content, which is crucial for building and maintaining trust in digital and informational ecosystems.

Our goal is to address these challenges by developing a solution that can differentiate between AI-generated and human-written texts.


## Data Collection & Preparation:
_FOLDER: 1. Data Collection & 2. Preparing Dataset_

**1. Data Collection**: Has a collection of Jupyter notebooks for each website to scrape human written content. Used BeautifulSoup, requests and selenium to collect data based on the requirements. 

**2. Preparing Dataset**: Has the notebooks for the AI text generation from the human scraped text and data cleaning. 

### Approach
To create a balanced dataset for training our machine learning model, we collected 50% of our dataset with human written text from various websites: 
1. Al Jazeera
2. CNN
3. The Conversation
4. The Guardian
5. Project Gutenberg
6. IMDb 
7. Nature 
8. Open Culture
9. PubMed
10. TechCrunch
11. Wikipedia 
12. Yahoo 

For each article, we scraped the content, including the title and the body text from different HTML webpages of each website, making sure to adhere to the terms of service and copyright policies as indicated by the robots.txt file of each site. The subsequent 50% of our data was generated using AI. During this process, we engaged in prompt engineering to produce text derived from the original articles, ensuring thematic similarity across both classes in our dataset. These texts were generated using the GPT-2, a pre-trained text generation model from HuggingFace, crafted to mimic the style and depth of human writing. By executing this strategy, we succeeded in creating a balanced dataset that will aid in our classification task. In total, we generated 2065 rows of text data, each duly labeled.

![image](https://github.com/revanthkrishnamg/AI-Detector/assets/149286080/0f48dcbd-b333-4091-80ff-742c6ff2c5b8)

## Modeling:
_FOLDER: 3. Model Building & Evaluation_
- a. **Feature Engineering.ipynb**: Created features here to predict the output. 
- b. **Exploratory Data Analysis & Visualisation.ipynb**: Performed EDAV to check how the newly features created are and performed some visualisation.
- c. **ML Modelling and Evaluation.ipynb**: Performed ML classification modelling and found the best model.

## Feature Engineering Plan
This plan outlines the key features to be engineered for distinguishing between human and AI-generated texts, focusing on linguistic, readability, n-gram, emotional, and semantic coherence. These features aim to capture the differentiation between AI and human writing styles and content.

1. **Linguistic Features**:
   - **Syntactic Complexity**: Measures the complexity of sentence structure.
   - **Lexical Richness**: Measures the variety of vocabulary used.
   - **Burstiness**: Measures variations in word frequencies within the text.
   - **Perplexity**: Measures how predictable a text is based on the probabilities assigned by a language model (GPT-2).
   - **Semantic Coherence**: Measures how logically connected the ideas in the text are.

2. **Readability Feature**:
   - **Readability scores**: Determine how easy or difficult a text is to understand.

3. **N-Gram Feature**:
   - **Unique N-grams**: Identifies and counts unique sequences of words.

4. **Sentiment Feature**:
   - **Sentiment analysis**: Evaluates the emotional tone of the text.

By using these specialized methods, we aim to build a robust system that can discern the intricate details of human-written texts and those created by AI, enhancing our understanding of artificial content.

## ML Modeling and Evaluation

### Model Building:
For the machine learning modeling phase, we employed several classification models to evaluate their performance in distinguishing between AI-generated and human-written texts. The process includes:

- **Model Selection**: We used a variety of models including Logistic Regression, Decision Trees, Random Forest, AdaBoost, XGBoost, SVM, and others to explore different algorithmic approaches.
- **Feature Engineering Integration**: Features like syntactic complexity, lexical richness, readability score, sentiment analysis, burstiness, semantic coherence, and n-gram features were integrated into the model training process to provide a robust set of predictors.
- **Model Training**: Each model was trained on the prepared dataset, utilizing techniques like cross-validation to ensure the models do not overfit and can generalize well to new, unseen data.
- **Evaluation Metrics**: Models were evaluated based on accuracy, precision, recall, F1-score, and ROC-AUC scores to ensure a comprehensive analysis of performance across various aspects.

### Hyperparameter Tuning:
To optimize the models, hyperparameter tuning was conducted using techniques like GridSearchCV. This process helps in fine-tuning the models to achieve the best performance by adjusting parameters such as learning rates, number of estimators, depth of trees, etc.

### Model Evaluation:
- **Cross-Validation**: Employed k-fold cross-validation to assess the effectiveness of models. This method helps in understanding how the models perform across different subsets of the dataset, thus providing insight into their stability and reliability.
- **Performance Metrics**: Utilized various metrics to evaluate the models:
  - **Accuracy** measures the overall correctness of the model.
  - **Precision** assesses the accuracy of positive predictions.
  - **Recall** evaluates the ability of a model to find all the relevant cases (all AI-generated texts).
  - **F1 Score** provides a balance between precision and recall. It is particularly useful when the class distribution is uneven.
  - **ROC-AUC Score** helps in measuring the capability of the model to distinguish between the classes.

### Insights from Modeling:
- **General Observations**: Certain models like XGBoost and SVM showed superior performance owing to their ability to handle non-linear data and maintain accuracy even with less feature engineering.
- **Feature Importance**: Analysis of feature importance revealed that certain features, such as perplexity and n-grams, were more influential in predicting AI-generated texts, indicating their strong discriminative power in the context of AI vs. human text classification.
- **Challenges and Limitations**: Addressed potential overfitting issues and the challenge of balancing recall and precision, which are critical when it is equally important to minimize false negatives and false positives.

### Future Work:
- **Model Enhancement**: Suggestions for further refinement of models through more extensive hyperparameter tuning and testing newer, more sophisticated machine learning algorithms.
- **Deployment Strategy**: Outline potential strategies for deploying the model into a production environment, where it can be used to automatically screen and verify text content for authenticity.

By continually refining our models and incorporating more diverse data, we aim to improve the robustness and accuracy of our AI-detector, ensuring it remains effective against evolving AI text generation technologies.

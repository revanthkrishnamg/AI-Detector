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

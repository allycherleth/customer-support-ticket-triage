# Customer IT Support Ticket Triage & Text Analytics

## Business Problem
IT help desks rely heavily on manual triage to categorize incoming tickets, assess urgency, and route issues to the appropriate support queue. Misrouted or miscategorized tickets trigger costly handoff delays and slow down resolution times. 

Using a dataset of **16,338 English customer support tickets**, this project explores whether natural language processing (NLP) and machine learning can automate ticket classification (`Incident`, `Request`, `Problem`, or `Change`), laying the analytical foundation for a **Smart-Ticket Triage Tool**.

## Methodology & Key Results
* **Text Preprocessing (Python / NLTK):** Combined subject and body fields, stripped escape characters, applied lemmatization, and built a custom domain stop-word list to remove help-desk filler words.
* **Topic Modeling (Gensim LDA vs. GenAI):** Identified 4 core operational themes (`Software & System Issues`, `Analytics, SaaS & Integration Inquiries`, `Healthcare Data Security`, and `Digital Marketing Strategy`). Benchmarked LDA explainability against ChatGPT topic extraction.
* **Predictive Classification (TensorFlow / Keras):** Built a neural network comparing structured metadata features against TF-IDF text features:
  * **Baseline Model (Metadata Only):** `45.3%` test accuracy
  * **Text-Enhanced Model (Metadata + TF-IDF):** `82.9%` test accuracy (**+37.7% lift**)

## Repository Contents
* `support_ticket_nlp_pipeline.ipynb` — Full Python code for preprocessing, LDA coherence testing, neural network training, and model evaluation.
* `Customer_Support_Text_Analytics_Report.pdf` — Executive report detailing business questions, methodology, and routing recommendations.
* `data/` — Contains the exact Kaggle dataset snapshot (`aa_dataset-tickets-multi-lang-5-2-50-version.csv`) used to ensure full reproducibility.

## Dataset & Acknowledgments
* **Dataset:** [Multilingual Customer Support Tickets (Kaggle)](https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets)
* **Project Team:** Ally Herleth, Adriana Guerrero, Tammy Marley, and Arielle Shaver
* **AI Transparency:** Generative AI tools (Claude and ChatGPT) were utilized during development to troubleshoot code syntax, refine preprocessing justifications, and benchmark topic labeling. All code, data transformations, and model evaluations were independently executed, validated, and interpreted.

# Counter-Strike AI-Tool

## Problem Statement
The competitive nature of Counter-Strike (CS) necessitates tools that provide players and analysts with insights to improve their gameplay strategies. Currently, there is a lack of predictive analytics and educational tools that can help players make informed decisions during the game. Additionally, new players often find it challenging to adapt and learn all the intricacies of the game, which can be overwhelming and hinder their progress.

## Context
Counter-Strike is one of the most popular and competitive games in the esports industry, attracting millions of players worldwide. Despite its popularity, the game's complexity can be daunting for newcomers, while seasoned players are continuously seeking ways to gain a competitive edge. Leveraging AI technology can address these challenges by providing predictive analytics and interactive learning tools to improve the gaming experience for all players.

## Research Questions
1. How can machine learning be used to predict the outcome of Counter-Strike rounds based on in-game data?
2. What impact can a chatbot, powered by natural language processing and image recognition, have on assisting players with game-related queries and identifying in-game items?
3. How do predictive insights and interactive educational tools affect the performance and engagement of new and experienced Counter-Strike players?

## Hypotheses
1. **Predictive Accuracy Hypothesis:** Implementing a machine learning-based predictive model will achieve high accuracy in forecasting round outcomes, thereby providing valuable strategic insights to players.
2. **Learning Curve Hypothesis:** An interactive chatbot that answers Counter-Strike-related questions and identifies game elements will significantly reduce the learning curve for new players, making the game more accessible and enjoyable.
3. **Engagement and Performance Hypothesis:** The combination of real-time predictive analytics and interactive learning tools will enhance player engagement and improve overall performance, particularly for new players adapting to the game and experienced players seeking strategic improvements.

## Motivation
Our motivation is to bridge this gap by developing an AI-based system that predicts round outcomes and offers a chatbot assistant for CS players. By providing predictive insights and interactive learning tools, we aim to enhance the gaming experience for both new and experienced players. For new players, our tools ease the learning curve by offering easy access to game information and strategic advice. For experienced players, the predictor provides in-depth analytics to refine their strategies and improve their gameplay.

## Theoretical Foundation
The prediction model leverages machine learning algorithms to analyze various in-game factors such as the number of players alive, weapon types, and remaining round time. The chatbot uses natural language processing (NLP) and image recognition (CNN) technologies to interact with users and identify weapons and skins from uploaded images.

## Argumentation of Choices
We chose to focus on Counter-Strike due to its large player base and the strategic depth of the game, which benefits significantly from predictive analytics. Machine learning models were selected for their ability to handle complex, multi-variable data sets and provide accurate predictions. NLP and image recognition were integrated into the chatbot to enhance user interaction and provide detailed information about game elements.

## Design
The project is divided into two main components:

1. **Prediction System:** A backend service that collects data from the users' choices in the frontend, processes it using the random forest classifier model, and predicts the outcome of the round.
2. **Chatbot Assistant:** An interactive chatbot that answers CS-related questions and identifies weapons from user-uploaded images.

## System Architecture

- **Backend:** Implemented using Python and the library FastApi. The machine learning models are developed using Scikit-learn (Forrest model), Tensorflow/Keras (CNN), Langchain (NLP) and OpenAI/Langchain (LLM)
- **Frontend:** Implemented using React with Next.js framework written in TypeScript for a robust and scalable user interface.
- **Data Collection:** Round data is collected through a combination of manual input from frontend and automated scripts handling Kaggle data. Images and skins information are collected with UiPath from [CSGO Skins](https://www.csgoskins.gg). General knowledge from the game Counter-Strike is collected from scraping [Wikipedia](https://en.wikipedia.org/wiki/Counter-Strike).

## Model Training
Historical game data is used to train the predictive models, ensuring they can accurately forecast round outcomes. Several images for each weapon in the game are used to train the image recognition models. Skin data from CSV files, and text from Wikipedia is used to train the natural language processing model.

## Code Structure
The project repository contains the following directories:

### Backend

- `config/`: Contains settings such as directory paths and environment variables.
- `data/`: Includes all the data used to train machine learning models.
- `src/`: Contains all the main code for the project.
  - `src/bots/`: Contains both of the chatbots (CNN and NLP) in the form of text and image responses.
  - `src/models/`: Includes files for training the CNN and NLP models, and stores the trained models.
  - `src/predictors/`: Contains the trainer and model for the Random Forest Classifier.
  - `src/scrapers/`: Contains files to scrape websites for data needed to train NLP.
  - `src/types/`: Contains all the types needed to make FastApi and Python code robust and work together.
- `main.py`: This file contains the code for FastApi.

### Frontend

- `public/`: Contains all images used on the website.
- `src/`: Contains all code to handle the project.
  - `src/app`: Contains all pages for the website.
  - `src/components`: Includes all components used on the pages.
  - `src/styles`: Contains all styling for handling HTML code by using Tailwind.
  - `src/utils`: Contains utilities that are used in pages.

The project repository is divided into two parts:

- **Backend Repository:** Contains API and all the AI functionality used on the frontend.
- **Frontend Repository:** Contains the Next.js application interface to use the AI functionalities.

## Artefacts

- **Predictive Models:** Trained machine learning models for round outcome prediction.
- **Chatbot Models:** Trained deep learning models for chatbot.
- **Chatbot Interface:** A web-based chatbot capable of answering questions and recognizing images.
- **Predictive Interface:** A web-based UI for specifying a round and showing the outcome.
- **Documentation:** Detailed project documentation, including this README and additional technical documentation.

## Outcomes

- **Predictive Accuracy:** Our models achieve a high accuracy rate in predicting round outcomes and giving precise answers for the chatbot.
- **User Engagement:** The chatbot provides an interactive and informative experience, enhancing user knowledge about CS.
- **Learning Support:** New players can quickly adapt and learn about the game through the chatbot’s educational features, while experienced players can benefit from strategic insights.

## Implementation Instructions

### Prerequisites

- Python 3.10 or higher
- scikit-learn
- Keras
- TensorFlow
- Langchain
- OpenAI API key
- All dependencies listed in `requirements.txt`

### Installation

#### Backend

1. Clone the repository:
   ```bash
   git clone https://github.com/bentsen/counter-strike-predicter-api.git
   cd counter-strike-predicter-api
2. Create Virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows use `venv\Scripts\activate`
3. Install the dependencies
   ```bash
   pip install -r requirements.txt

#### Frontend

1. Clone the repository
   ```bash
   git clone https://github.com/bentsen/counter-strike-predicter.git
   cd counter-strike-predicter
2. Intall dependencies
   ```bash
   npm install
3. Run server
   ```bash
   npm run dev

## Usage

  - **Round Prediction:** Follow the website stepper guidance to select the wanted data for a specific round.
  - **Chatbot:**  Interact with the chatbot via the web interface to ask questions or upload images of weapons/skins.

## Repositories
The complete project code can be found on our GitHub repositories:
- [Backend Repository](https://github.com/bentsen/counter-strike-predicter-api)
- [Frontend Repository](https://github.com/bentsen/counter-strike-predicter)


## Conclusion
Our AI project aims to enhance the Counter-Strike gaming experience by leveraging machine learning for predicting round outcomes and an interactive chatbot for game-related question and item identification. 
These tools are designed to reduce the learning curve for new players and provide strategic insights for experienced players. By improving prediction accuracy and engagement, our AI-powered solutions will make Counter-Strike more accessible and enjoyable for all players, significantly impacting the competitive gaming landscape.
   
   


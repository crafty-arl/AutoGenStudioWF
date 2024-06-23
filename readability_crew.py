import os
from dotenv import load_dotenv
import streamlit as st
import textstat
from crewai import Agent, Task, Crew, Process

# Load environment variables from .env file
load_dotenv()

# Now you can access the OPENAI_API_KEY environment variable
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Streamlit UI
st.title('Readability Analysis')

# Text input
user_text = st.text_area("Enter text to analyze readability:")

if st.button('Analyze Readability'):
    if user_text:
        # Calculate readability scores directly
        flesch_reading_ease = textstat.flesch_reading_ease(user_text)
        smog_index = textstat.smog_index(user_text)
        flesch_kincaid_grade = textstat.flesch_kincaid_grade(user_text)
        coleman_liau_index = textstat.coleman_liau_index(user_text)
        automated_readability_index = textstat.automated_readability_index(user_text)
        dale_chall_readability_score = textstat.dale_chall_readability_score(user_text)
        difficult_words = textstat.difficult_words(user_text)
        linsear_write_formula = textstat.linsear_write_formula(user_text)
        gunning_fog_index = textstat.gunning_fog(user_text)
        text_standard = textstat.text_standard(user_text)

        scores = {
            "Flesch Reading Ease": flesch_reading_ease,
            "SMOG Index": smog_index,
            "Flesch-Kincaid Grade Level": flesch_kincaid_grade,
            "Coleman-Liau Index": coleman_liau_index,
            "Automated Readability Index": automated_readability_index,
            "Dale-Chall Readability Score": dale_chall_readability_score,
            "Difficult Words": difficult_words,
            "Linsear Write Formula": linsear_write_formula,
            "Gunning Fog Index": gunning_fog_index,
            "Text Standard": text_standard
        }

        st.write("Readability Analysis Result:")
        for key, value in scores.items():
            st.write(f"{key}: {value}")

        # Prepare text and scores for agents to analyze
        analysis_text = user_text
        analysis_scores = scores

        # Create the Readability Analysis Agent
        readability_agent = Agent(
            role='Readability Analyzer',
            goal='Analyze readability scores and explain why the text received these scores.',
            verbose=True,
            memory=True,
            max_iter=1,
            allow_delegation=False,
            backstory=(
                "You are an expert in text analysis, specializing in readability metrics. "
                "Your extensive knowledge and tools enable you to provide detailed insights "
                "into the complexity and accessibility of any text. Your primary mission is "
                "to ensure that the readability analysis is accurate and informative, using "
                "state-of-the-art methods to evaluate the text."
            ),
            system_template=(
                "System: You are {role}. Your goal is {goal}. Your backstory: {backstory}. "
                "You will analyze the readability scores of the given text and provide detailed explanations. "
                "Ensure that you only process the initial text and scores provided."
            ),
            prompt_template=(
                "Prompt: Please analyze the readability scores of the following text: {text}. "
                "Explain why the text received these scores: {scores}."
            ),
            response_template=(
                "Response: The readability analysis for the given text is as follows: {readability_scores}. "
                "The text received these scores because: {{ .Response }}."
            )
        )

        # Define Tasks
        analysis_task = Task(
            description='Analyze the readability scores and explain why the text received these scores.',
            expected_output='Detailed explanation of readability scores.',
            agent=readability_agent,
            handler=lambda inputs: {"text": analysis_text, "scores": analysis_scores}
        )

        # Form the Crew
        crew = Crew(
            agents=[readability_agent],
            tasks=[analysis_task],
            process=Process.sequential
        )

        # Kickoff the Crew with the analysis text and scores
        result = crew.kickoff(inputs={'text': analysis_text, 'scores': analysis_scores})
        st.write("Detailed Analysis Result:")
        st.write(result)
    else:
        st.write("Please enter some text to analyze.")

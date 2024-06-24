import os
from dotenv import load_dotenv
import time
import streamlit as st
import streamlit_shadcn_ui as ui
import textstat
import plotly.express as px
from crewai import Agent, Task, Crew

# Function to load environment variables
def load_env_variable(var_name):
    value = os.getenv(var_name)
    if value is None:
        st.error(f"Environment variable '{var_name}' not found.")
    return value

# Load environment variables from .env file
load_dotenv()

# Streamlit UI configuration
st.set_page_config(page_title='Readable IQ', page_icon='📖')
st.title('Readable IQ')

# Sidebar for OpenAI API key input and instructions
st.sidebar.title("Configuration")
st.sidebar.markdown("""
### How to get your OpenAI API Key:
1. Go to [OpenAI API](https://beta.openai.com/signup/).
2. Sign up for an account if you don't have one.
3. Log in to your OpenAI account.
4. Navigate to the API section.
5. Generate a new API key.
6. Copy the key and paste it in the text box below.
""")
openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key

# Text input using streamlit_shadcn_ui
user_text = ui.textarea(default_value="Copy / Paste Text You Need to Analyze", placeholder="Enter longer text", key="textarea1")

# Regular Streamlit button for analysis
analyze_button_clicked = st.button('Analyze Readability')

# Initialize the analysis state
if 'analyze_triggered' not in st.session_state:
    st.session_state['analyze_triggered'] = False

if analyze_button_clicked:
    st.session_state['analyze_triggered'] = True

analyze_triggered = st.session_state['analyze_triggered'] and user_text.strip() != ""

# Perform the analysis if triggered
if analyze_triggered:
    progress_text = "Analyzing readability. Please wait."
    my_bar = st.progress(0, text=progress_text)

    with st.spinner("Analyzing readability..."):
        try:
            # Simulate progress for readability analysis
            for percent_complete in range(0, 101, 20):
                time.sleep(0.1)
                my_bar.progress(percent_complete, text=progress_text)

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
                "FRE": flesch_reading_ease,
                "SMOG": smog_index,
                "FKG": flesch_kincaid_grade,
                "CLI": coleman_liau_index,
                "ARI": automated_readability_index,
                "DCRS": dale_chall_readability_score,
                "DW": difficult_words,
                "LWF": linsear_write_formula,
                "GFI": gunning_fog_index,
                "TS": text_standard
            }

            # Finalize progress bar
            my_bar.progress(100, text="Analysis complete!")
            time.sleep(0.5)
            my_bar.empty()

            # Display the Readability Scores
            st.header("Readability Scores")
            # Display scores as metric cards
            cols = st.columns(3)
            for i, (key, value) in enumerate(scores.items()):
                with cols[i % 3]:
                    ui.metric_card(title=key, content=str(value), description=f"Score for {key}")

            # Create Plotly bar graph
            score_labels = list(scores.keys())
            score_values = [value for value in scores.values()]
            fig = px.bar(x=score_values, y=score_labels, orientation='h', labels={'x':'Score', 'y':'Metrics'})
            fig.update_layout(title='Readability Scores', xaxis_title='Score', yaxis_title='Metrics')

            st.plotly_chart(fig)

            # Prepare data for Crew AI
            analysis_result_str = "\n".join([f"{key}: {value}" for key, value in scores.items()])

            # Define system message templates
            readability_system_message_template = """
Role: Readability Analyzer
Goal: Analyze readability scores and explain why the text received these scores.
Verbose: True
Memory: True
Max Iterations: 15
Allow Delegation: False

Backstory:
You are an expert in text analysis, specializing in readability metrics. Your extensive knowledge and tools enable you to provide detailed insights into the complexity and accessibility of any text. Your primary mission is to ensure that the readability analysis is accurate and informative, using state-of-the-art methods to evaluate the text.

The text to analyze is:
{analysis_text}

The readability scores are:
{analysis_result_str}

Please provide a detailed report in the following format:
Report Name: {analysis_topic} Report
Report Scores:
  - FRE: {flesch_reading_ease} (Explanation: {explanation})
  - SMOG: {smog_index} (Explanation: {explanation})
  - FKG: {flesch_kincaid_grade} (Explanation: {explanation})
  - CLI: {coleman_liau_index} (Explanation: {explanation})
  - ARI: {automated_readability_index} (Explanation: {explanation})
  - DCRS: {dale_chall_readability_score} (Explanation: {explanation})
  - DW: {difficult_words} (Explanation: {explanation})
  - LWF: {linsear_write_formula} (Explanation: {explanation})
  - GFI: {gunning_fog_index} (Explanation: {explanation})
  - TS: {text_standard} (Explanation: {explanation})
Explanation of Scores:
  Provide real examples from the provided text that correspond to the readability scores.
Suggestions:
  If necessary, provide suggestions for improving the readability of the text.
Conclusion:
  Provide a conclusion summarizing the overall readability of the text.
"""

            readability_system_message = readability_system_message_template.format(
                analysis_text=user_text.replace("\n", "\\n").replace('"', '\\"'),
                analysis_result_str=analysis_result_str.replace("\n", "\\n").replace('"', '\\"'),
                flesch_reading_ease=flesch_reading_ease,
                smog_index=smog_index,
                flesch_kincaid_grade=flesch_kincaid_grade,
                coleman_liau_index=coleman_liau_index,
                automated_readability_index=automated_readability_index,
                dale_chall_readability_score=dale_chall_readability_score,
                difficult_words=difficult_words,
                linsear_write_formula=linsear_write_formula,
                gunning_fog_index=gunning_fog_index,
                text_standard=text_standard,
                explanation="Explanation placeholder",  # Added placeholder for explanation
                analysis_topic="Report name placeholder"
            )

            # Create the Readability Analysis Agent with system message
            readability_agent = Agent(
                name="Readability Analyzer",
                role="Readability Analyzer",
                goal="Analyze readability scores and explain why the text received these scores.",
                verbose=True,
                memory=True,
                max_iter=15,
                allow_delegation=False,
                backstory=readability_system_message
            )

            # Define the Task for the Readability Analysis Agent
            analysis_task = Task(
                name="Readability Analysis",
                description="Analyze the readability scores and explain why the text received these scores.",
                expected_output="A detailed report explaining the readability scores and highlighting the reasons behind each score, including examples from the text.",
                agent=readability_agent
            )

            # Form the Crew
            crew = Crew(
                agents=[readability_agent],
                tasks=[analysis_task],
            )

            # Kickoff the Crew with the concatenated input string
            input_string = f"Text: {user_text}\nReadability Scores:\n{analysis_result_str}"
            result = crew.kickoff(inputs={'text': input_string})

            # Display the result
            st.header("Readable IQ Report")
            if isinstance(result, str):
                st.markdown(result.replace("\\n", "\n"))
            else:
                st.write("Unexpected result format:", result)

        except Exception as e:
            st.error(f"An error occurred: {e}")

# Instructions for migrating and deploying the app
st.sidebar.title("Deployment Instructions")
st.sidebar.markdown("""
### To migrate this app to GitHub and deploy on Streamlit:
1. **Clone the repository or create a new one:**
    - Create a new repository on GitHub.
    - Clone the repository to your local machine.
    - Add your project files to the repository.

2. **Push the code to GitHub:**
    ```bash
    git add .
    git commit -m "Initial commit"
    git push origin main
    ```

3. **Deploy on Streamlit:
    - Go to [Streamlit Sharing](https://share.streamlit.io/).
    - Click on "New app".
    - Connect your GitHub account and select the repository.
    - Choose the branch and the main Python file (e.g., `readstats.py`).
    - Click "Deploy".
""")

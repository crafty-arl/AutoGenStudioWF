import os
from dotenv import load_dotenv
import time
import streamlit as st
import streamlit_shadcn_ui as ui
import textstat
import plotly.express as px
from crewai import Agent, Task, Crew, Process

def load_env_variable(var_name):
    value = os.getenv(var_name)
    if value is None:
        st.error(f"Environment variable '{var_name}' not found.")
    return value

# Load environment variables from .env file
load_dotenv()

# Streamlit UI
st.set_page_config(page_title='Readable IQ', page_icon='📖')
st.title('Readable IQ')

# Sidebar for OpenAI API key input
st.sidebar.title("Configuration")
openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key

# Text input using streamlit_shadcn_ui
user_text = ui.textarea(default_value="Copy / Paste Text You Need to Analyze", placeholder="Enter longer text", key="textarea1")

# Button to trigger readability analysis
analyze_button_clicked = ui.button('Analyze Readability')

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
            score_values = list(scores.values())
            fig = px.bar(x=score_values, y=score_labels, orientation='h', labels={'x':'Score', 'y':'Metrics'})
            fig.update_layout(title='Readability Scores', xaxis_title='Score', yaxis_title='Metrics')

            st.plotly_chart(fig)

            # Convert scores dictionary to a formatted string
            analysis_result_str = "\n".join([f"{key}: {value}" for key, value in scores.items()])

            # Define system message template
            system_message_template = (
                "Role: Readability Analyzer\n"
                "Goal: Analyze readability scores and explain why the text received these scores.\n"
                "Verbose: True\n"
                "Memory: True\n"
                "Max Iterations: 15\n"
                "Allow Delegation: False\n"
                "\n"
                "Backstory:\n"
                "You are an expert in text analysis, specializing in readability metrics. Your extensive knowledge and tools enable you to provide detailed insights into the complexity and accessibility of any text. Your primary mission is to ensure that the readability analysis is accurate and informative, using state-of-the-art methods to evaluate the text.\n"
                "\n"
                "The text to analyze is:\n"
                "{analysis_text}\n"
                "\n"
                "The readability scores are:\n"
                "{analysis_result_str}\n"
            )

            # Create the Readability Analysis Agent with system message
            readability_agent = Agent(
                role='Readability Analyzer',
                goal='Analyze readability scores and explain why the text received these scores.',
                verbose=True,
                memory=True,
                max_iter=15,
                allow_delegation=False,
                backstory=system_message_template.format(
                    analysis_text=user_text,  # Correctly passing user_text
                    analysis_result_str=analysis_result_str
                )
            )

            # Define the Task for the Readability Analysis Agent
            analysis_task = Task(
                description='Analyze the readability scores and explain why the text received these scores.',
                expected_output='A detailed report explaining the readability scores and highlighting the reasons behind each score.',
                agent=readability_agent
            )

            # Form the Crew
            crew = Crew(
                agents=[readability_agent],
                tasks=[analysis_task],
            )
            
            # Simulate some time for the crew to process the results
            for percent_complete in range(101, 201, 20):
                time.sleep(0.1)
                my_bar.progress(percent_complete - 100, text='Processing results...')

            # Kickoff the Crew with the analysis text and scores
            result = crew.kickoff(inputs={'text': user_text, 'readability_scores_str': analysis_result_str})

            # Render the report using HTML for better formatting
            report_html = f"""
            <h2>Readability Analysis Report</h2>
            <hr>
            <h3>Text Analyzed:</h3>
            <p>{user_text}</p>
            <h3>Readability Scores:</h3>
            <pre>{analysis_result_str}</pre>
            <h3>Analysis and Explanation:</h3>
            <ol>
                <li><strong>Flesch Reading Ease (FRE):</strong>
                    <ul>
                        <li><strong>Score:</strong> {flesch_reading_ease}</li>
                        <li><strong>Explanation:</strong> This score indicates how easy the text is to read. A higher score suggests easier readability. This score is influenced by the average sentence length and the average number of syllables per word. Example: <span style="background-color: #FFFF00">{user_text[:50]}</span>.</li>
                    </ul>
                </li>
                <li><strong>SMOG Index (SMOG):</strong>
                    <ul>
                        <li><strong>Score:</strong> {smog_index}</li>
                        <li><strong>Explanation:</strong> This score estimates the years of education needed to understand the text. It considers the number of polysyllabic words. Example: <span style="background-color: #FFFF00">{user_text[50:100]}</span>.</li>
                    </ul>
                </li>
                <li><strong>Flesch-Kincaid Grade Level (FKG):</strong>
                    <ul>
                        <li><strong>Score:</strong> {flesch_kincaid_grade}</li>
                        <li><strong>Explanation:</strong> This score represents the U.S. school grade level required to understand the text. It is derived from sentence length and word length. Example: <span style="background-color: #FFFF00">{user_text[100:150]}</span>.</li>
                    </ul>
                </li>
                <li><strong>Coleman-Liau Index (CLI):</strong>
                    <ul>
                        <li><strong>Score:</strong> {coleman_liau_index}</li>
                        <li><strong>Explanation:</strong> This index calculates the readability of the text based on characters per word and words per sentence, rather than syllables. Example: <span style="background-color: #FFFF00">{user_text[150:200]}</span>.</li>
                    </ul>
                </li>
                <li><strong>Automated Readability Index (ARI):</strong>
                    <ul>
                        <li><strong>Score:</strong> {automated_readability_index}</li>
                        <li><strong>Explanation:</strong> This index estimates the U.S. grade level needed to comprehend the text, based on characters per word and words per sentence. Example: <span style="background-color: #FFFF00">{user_text[200:250]}</span>.</li>
                    </ul>
                </li>
                <li><strong>Dale-Chall Readability Score (DCRS):</strong>
                    <ul>
                        <li><strong>Score:</strong> {dale_chall_readability_score}</li>
                        <li><strong>Explanation:</strong> This score considers the familiarity of words used in the text, comparing them against a list of commonly known words. Example: <span style="background-color: #FFFF00">{user_text[250:300]}</span>.</li>
                    </ul>
                </li>
                <li><strong>Difficult Words (DW):</strong>
                    <ul>
                        <li><strong>Count:</strong> {difficult_words}</li>
                        <li><strong>Explanation:</strong> This is the number of complex words in the text that may be difficult for readers to understand. Example: <span style="background-color: #FFFF00">{user_text[300:350]}</span>.</li>
                    </ul>
                </li>
                <li><strong>Linsear Write Formula (LWF):</strong>
                    <ul>
                        <li><strong>Score:</strong> {linsear_write_formula}</li>
                        <li><strong>Explanation:</strong> This formula calculates readability based on the number of easy and hard words, and the sentence length. Example: <span style="background-color: #FFFF00">{user_text[350:400]}</span>.</li>
                    </ul>
                </li>
                <li><strong>Gunning Fog Index (GFI):</strong>
                    <ul>
                        <li><strong>Score:</strong> {gunning_fog_index}</li>
                        <li><strong>Explanation:</strong> This index estimates the years of formal education needed to understand the text, based on sentence complexity and word difficulty. Example: <span style="background-color: #FFFF00">{user_text[400:450]}</span>.</li>
                    </ul>
                </li>
                <li><strong>Text Standard (TS):</strong>
                    <ul>
                        <li><strong>Standard:</strong> {text_standard}</li>
                        <li><strong>Explanation:</strong> This standard gives an overall grade level for the text based on various readability formulas. Example: <span style="background-color: #FFFF00">{user_text[450:500]}</span>.</li>
                    </ul>
                </li>
            </ol>
            <h3>Summary:</h3>
            <p>{result}</p>
            """

            st.markdown(report_html, unsafe_allow_html=True)

            # Trigger alert dialog when user scrolls to the bottom
            
        except KeyError as e:
            st.error(f"Key error: {e}")
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

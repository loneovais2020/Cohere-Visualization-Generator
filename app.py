import streamlit as st
import os
import glob
from langchain_cohere.chat_models import ChatCohere
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import Tool, AgentExecutor
from langchain_experimental.utilities import PythonREPL
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_cohere.react_multi_hop.agent import create_cohere_react_agent
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# Function to clean all .png files in the current directory
def clean_png_files():
    png_files = glob.glob("*.png")
    for file in png_files:
        try:
            os.remove(file)
        except Exception as e:
            st.error(f"Error removing file {file}: {e}")

# Initialize session state for API keys
if 'cohere_api_key' not in st.session_state:
    st.session_state['cohere_api_key'] = ''
if 'tavily_api_key' not in st.session_state:
    st.session_state['tavily_api_key'] = ''

# Streamlit application
st.title("AI-Powered Visualization Generator")

# Sidebar for API key inputs
st.sidebar.header("API Keys")
cohere_api_key = st.sidebar.text_input("Enter Cohere API Key:", type="password", value=st.session_state['cohere_api_key'])
tavily_api_key = st.sidebar.text_input("Enter Tavily API Key:", type="password", value=st.session_state['tavily_api_key'])

if st.sidebar.button("Submit API Keys"):
    if cohere_api_key and tavily_api_key:
        # Store API keys in session state
        st.session_state['cohere_api_key'] = cohere_api_key
        st.session_state['tavily_api_key'] = tavily_api_key
        st.sidebar.success("API keys saved successfully!")
    else:
        st.sidebar.error("Please enter both API keys.")

# Main content
if st.session_state['cohere_api_key'] and st.session_state['tavily_api_key']:
    # Initialize the Cohere model
    chat = ChatCohere(model="command-r-plus", temperature=0.7, api_key=st.session_state['cohere_api_key'])

    # Initialize internet search tool
    internet_search = TavilySearchResults(api_key=st.session_state['tavily_api_key'])
    internet_search.name = "internet_search"
    internet_search.description = "Returns a list of relevant documents from the internet."
    class TavilySearchInput(BaseModel):
        query: str = Field(description="Internet query engine.")
    internet_search.args_schema = TavilySearchInput

    # Initialize Python REPL tool
    python_repl = PythonREPL()
    repl_tool = Tool(
        name="python_repl",
        description="Executes python code and returns the result.",
        func=python_repl.run,
    )
    repl_tool.name = "python_interpreter"
    class ToolInput(BaseModel):
        code: str = Field(description="Python code execution.")
    repl_tool.args_schema = ToolInput

    # Create the agent
    prompt = ChatPromptTemplate.from_template("{input}")
    agent = create_cohere_react_agent(
        llm=chat,
        tools=[internet_search, repl_tool],
        prompt=prompt,
    )
    agent_executor = AgentExecutor(agent=agent, tools=[internet_search, repl_tool], verbose=True)

    user_input = st.text_input("Enter your query:")

    if st.button("Submit Query"):
        with st.spinner("Processing..."):
            try:
                # Clean all .png files in the current directory
                clean_png_files()
                
                #prompt
                prompt =  f"""{user_input} Determine the most appropriate chart type, unless a specific type is requested. Create a chart using Matplotlib that effectively visualizes the data. Use a clean, modern style with a light background. Choose visually appealing colors that complement each other and enhance data clarity. Ensure lines or shapes are distinct and easy to differentiate.
                Include a clear, descriptive title that summarizes the main point of the data. Label axes appropriately and use easy-to-read font sizes. If applicable, add a legend to explain different data series or categories. Use subtle gridlines if they improve data interpretation without cluttering the visual. Also include the labels, if required
                Adjust the layout to prevent overlapping elements and ensure all labels are fully visible. Fine-tune the chart's size for optimal viewing. If dealing with time-based data, consider appropriate time intervals on the x-axis."""
                # Execute the agent and generate the chart
                response = agent_executor.invoke({"input": prompt})
                
                st.success("Done!")
                
                # Display the newly generated chart(s)
                st.header("Displaying The Chart:", divider=True)
                new_png_files = glob.glob("*.png")
                if not new_png_files:
                    st.warning("No charts were generated.")
                for png_file in new_png_files:
                    st.image(png_file, caption=png_file)

                st.header("References Used:", divider=True)

                # Initialize an empty list to collect all URLs
                urls = []

                # Iterate through the 'citations' in the response
                for citation in response.get('citations', []):
                    for document in citation.documents:
                        if 'url' in document:
                            urls.append(document['url'])

                # Print all collected URLs
                if urls:
                    for url in urls:
                        st.write(url)
                else:
                    st.warning("No references were found in the response.")

            except KeyError as ke:
                st.error(f"Missing expected key in the response: {ke}")
            except Exception as e:
                st.error(f"An error occurred: {e}")
else:
    st.info("Please enter your API keys in the sidebar and click 'Submit API Keys' to proceed.")
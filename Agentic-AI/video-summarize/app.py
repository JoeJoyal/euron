import streamlit as st
from phi.agent import Agent 
from phi.model.google import gemini
from phi.tools.duckduckgo import DuckDuckGo # Web search
from google.generativeai import upload_file, get_file
import google.generativeai as genai

import time
from pathlib import Path

import tempfile

from dotenv import load_dotenv
load_dotenv()

import os

API_KEY = os.getenv("GOOGLE_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)
from typing import Optional
import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urlparse
import time
from smolagents import CodeAgent, tool
import os
import openai

class AzureModel:
    def __init__(self, api_key: str, endpoint: str, deployment_name: str):
        """
        Initialize Azure OpenAI model connection.
        
        Args:
            api_key: Azure OpenAI API key
            endpoint: Azure endpoint URL
            deployment_name: Model deployment name in Azure
        """
        openai.api_type = "azure"
        openai.api_base = endpoint
        openai.api_version = "2024-02-15-preview"
        openai.api_key = api_key
        self.deployment_name = deployment_name
    
    def __call__(self, prompt: str) -> str:
        """
        Make the class callable as required by smolagents.
        
        Args:
            prompt: Input prompt for the model
            
        Returns:
            str: Generated response
        """
        try:
            response = openai.ChatCompletion.create(
                engine=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are a professional business proposal writer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=800
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Azure API Error: {str(e)}")
            return f"Error generating content: {str(e)}"

@tool
def scrape_website(url: str) -> dict:
    """
    Scrapes relevant information from a company website.

    Args:
        url: The complete URL of the website to scrape (e.g., 'https://example.com')

    Returns:
        dict: A dictionary containing the scraped information with the following keys:
            - company_name (str): The name of the company
            - meta_description (str): Website meta description
            - about_content (str): Content from the about section
            - contact_emails (list): List of email addresses found
            - website (str): Original URL
            - error (str, optional): Error message if scraping fails
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        domain = urlparse(url).netloc
        company_name = soup.title.string if soup.title else domain.split('.')[0]
        
        meta_desc = ""
        meta_tag = soup.find('meta', attrs={'name': 'description'})
        if meta_tag:
            meta_desc = meta_tag.get('content', '')
            
        about_content = ""
        about_section = soup.find(lambda tag: tag.name in ['div', 'section'] and 
                                any(word in tag.get('class', []) + [tag.get('id', '')] 
                                    for word in ['about', 'company', 'mission']))
        if about_section:
            about_content = about_section.get_text(strip=True)
            
        contact_pattern = r'[\w\.-]+@[\w\.-]+\.\w+'
        emails = re.findall(contact_pattern, response.text)
        
        return {
            'company_name': company_name,
            'meta_description': meta_desc,
            'about_content': about_content,
            'contact_emails': emails,
            'website': url
        }
    except Exception as e:
        return {
            'error': f"Error scraping website: {str(e)}",
            'website': url
        }

class CustomCodeAgent(CodeAgent):
    def run(self, prompt: str) -> str:
        """
        Override run method to handle website processing.

        Args:
            prompt: Input prompt containing the website URL

        Returns:
            str: Generated proposal or error message
        """
        try:
            url = prompt.replace("Generate a proposal for ", "").strip()
            company_info = scrape_website(url)
            
            if 'error' in company_info:
                return f"Error: {company_info['error']}"
                
            proposal_prompt = f"""
            Based on the following company information, write a professional and personalized service proposal email:
            
            Company Name: {company_info['company_name']}
            Website: {company_info['website']}
            Description: {company_info['meta_description']}
            About: {company_info['about_content']}
            
            Write a proposal that:
            1. Shows understanding of their business
            2. Identifies potential pain points
            3. Proposes relevant solutions
            4. Includes a clear call to action
            5. Maintains a professional yet warm tone
            my name is Krishna and iam from imaigen ai we are a technology company that specializes in AI and ML solutions.
            Format the email with a subject line, greeting, body, and signature.
            """
            
            return self.model(proposal_prompt)
            
        except Exception as e:
            return f"Error running agent: {str(e)}"

class ProposalAgent:
    def __init__(self, azure_api_key: str, azure_endpoint: str, deployment_name: str):
        """
        Initialize the proposal agent with Azure credentials.

        Args:
            azure_api_key: Azure OpenAI API key
            azure_endpoint: Azure endpoint URL
            deployment_name: Model deployment name
        """
        self.model = AzureModel(
            api_key=azure_api_key,
            endpoint=azure_endpoint,
            deployment_name=deployment_name
        )
        self.agent = CustomCodeAgent(
            tools=[scrape_website],
            model=self.model
        )
    
    def generate_proposal_for_website(self, url: str) -> str:
        """
        Generate a proposal for the given website.

        Args:
            url: The website URL to generate a proposal for

        Returns:
            str: Generated proposal or error message
        """
        try:
            result = self.agent.run(f"Generate a proposal for {url}")
            return result
        except Exception as e:
            return f"Error generating proposal: {str(e)}"

def main():
    # Get Azure credentials from environment variables
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    
    if not all([azure_api_key, azure_endpoint, deployment_name]):
        print("Please set the required Azure OpenAI environment variables:")
        print("AZURE_OPENAI_API_KEY")
        print("AZURE_OPENAI_ENDPOINT")
        print("AZURE_OPENAI_DEPLOYMENT_NAME")
        return
    
    agent = ProposalAgent(
        azure_api_key=azure_api_key,
        azure_endpoint=azure_endpoint,
        deployment_name=deployment_name
    )
    
    url = input("Enter the website URL to generate a proposal for: ")
    proposal = agent.generate_proposal_for_website(url)
    print("\nGenerated Proposal:")
    print("="* 80)
    print(proposal)

if __name__ == "__main__":
    main()
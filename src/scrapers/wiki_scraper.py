import requests
from bs4 import BeautifulSoup
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def scrape_wikipedia():
    url = 'https://en.wikipedia.org/wiki/Counter-Strike'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        logger.error(f"Failed to retrieve page: Status code {response.status_code}")
        return ""

    soup = BeautifulSoup(response.content, 'html.parser')

    # Locate the introductory section
    intro_paragraphs = soup.find_all('p', limit=3)
    if not intro_paragraphs:
        logger.error("No introductory paragraphs found")
        return ""

    # Extract the text from the introductory paragraphs
    intro_text = " ".join([para.get_text(strip=True) for para in intro_paragraphs])

    if intro_text:
        logger.info("Wikipedia data scraped successfully.")
    else:
        logger.info("No introductory text found.")

    return intro_text


if __name__ == "__main__":
    intro_text = scrape_wikipedia()
    logger.info(f"Scraped Intro Text: {intro_text}")

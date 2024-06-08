import cloudscraper
from bs4 import BeautifulSoup
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

scraper = cloudscraper.create_scraper()

def scrape_hltv_matches():
    url = 'https://www.hltv.org/matches'
    try:
        response = scraper.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to retrieve page: {e}")
        return []

    soup = BeautifulSoup(response.content, 'html.parser')

    matches = []
    for item in soup.find_all('div', class_='upcomingMatch'):
        try:
            time = item.find('div', class_='matchTime').text.strip()
            teams = [team.text.strip() for team in item.find_all('div', class_='matchTeamName')]
            event = item.find('div', class_='matchEventName').text.strip()
            match_url = item.find('a', class_='match a-reset')['href']
            match_details = scrape_match_details(f'https://www.hltv.org{match_url}')
            matches.append({
                'time': time,
                'teams': teams,
                'event': event,
                'details': match_details
            })
        except AttributeError as e:
            logger.warning(f"Error processing an item: {e}")
            continue

    return matches

def scrape_match_details(url):
    try:
        response = scraper.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to retrieve match details: {e}")
        return "Details not available"

    soup = BeautifulSoup(response.content, 'html.parser')
    try:
        content = soup.find('div', class_='match-page').text.strip()
    except AttributeError as e:
        logger.warning(f"Error processing match details: {e}")
        content = "Details not available"

    return content

if __name__ == "__main__":
    data = scrape_hltv_matches()
    logger.info(f"Scraped Data: {data}")

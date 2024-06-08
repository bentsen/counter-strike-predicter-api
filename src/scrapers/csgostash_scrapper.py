import requests
from bs4 import BeautifulSoup
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def scrape_csgo_stash():
    base_url = "https://csgostash.com"
    response = requests.get(base_url)

    if response.status_code != 200:
        logger.error(f"Failed to retrieve page: Status code {response.status_code}")
        return []

    soup = BeautifulSoup(response.content, "html.parser")

    weapons_data = []

    try:
        items = soup.find_all("div", class_="well result-box nomargin")
        logger.info(f"Found {len(items)} items")

        for item in items:
            try:
                if item.find("script"):  # Skip divs that contain scripts
                    continue

                name_tag = item.find("h3").find("a")
                description_tag = item.find("div", class_="quality")
                weapon_url = name_tag['href'] if name_tag else ""

                if not name_tag:
                    name_tag = item.find("h3")
                if not description_tag:
                    description_tag = item.find("div", "details-link")

                if name_tag and description_tag:
                    name = name_tag.text.strip()
                    description = description_tag.text.strip()

                    # Correct URL construction
                    if weapon_url.startswith("http"):
                        weapon_url = weapon_url
                    else:
                        weapon_url = base_url + weapon_url

                    # Fetch weapon details
                    weapon_response = requests.get(weapon_url)
                    weapon_soup = BeautifulSoup(weapon_response.content, "html.parser")

                    # Updated way to find skins
                    skin_items = weapon_soup.find_all("div", class_="well result-box")
                    skins = []
                    for skin_item in skin_items:
                        skin_name_tag = skin_item.find("h3")
                        if skin_name_tag:
                            skin_name = skin_name_tag.text.strip()
                            skins.append(skin_name)

                    weapons_data.append({
                        "name": name,
                        "description": description,
                        "skins": skins
                    })
                else:
                    logger.warning("Missing name or description in one of the items")
                    logger.debug("Item HTML content: %s", item.prettify())
            except Exception as e:
                logger.error(f"Error processing an item: {e}")
                logger.debug("Item HTML content: %s", item.prettify())

        logger.info("CSGO Stash data scraped successfully.")
    except Exception as e:
        logger.error(f"Error scraping CSGO Stash: {e}")

    return weapons_data

if __name__ == "__main__":
    data = scrape_csgo_stash()
    logger.info(f"Scraped Data: {data}")

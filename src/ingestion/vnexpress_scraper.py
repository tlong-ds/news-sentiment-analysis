import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import time
from datetime import datetime
import re

from src.config import (
    START_DATE, END_DATE, DATA_DIR,
    VNEXPRESS_CATEGORIES, VNEXPRESS_BASE_URL
)

class VnExpressScraper:
    def __init__(self, start_date=START_DATE, end_date=END_DATE):
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
        self.output_file = os.path.join(DATA_DIR, "news_VN_vnexpress.csv")
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

    def parse_date(self, date_str):
        """Parses VnExpress date format: 'January 3, 2024 | 09:00 am GMT+7'"""
        try:
            clean_date = date_str.split("|")[0].strip()
            return datetime.strptime(clean_date, "%B %d, %Y")
        except Exception:
            return None

    def extract_article_details(self, url):
        """Fetches and extracts article details from a URL."""
        try:
            resp = requests.get(url, headers=self.headers, timeout=10)
            if resp.status_code != 200: return None
            soup = BeautifulSoup(resp.content, "html.parser")
            
            title = soup.select_one(".title-detail").get_text(strip=True) if soup.select_one(".title-detail") else ""
            date_raw = soup.select_one(".date").get_text(strip=True) if soup.select_one(".date") else ""
            pub_date = self.parse_date(date_raw)
            
            body_div = soup.select_one(".fck_detail") or soup.select_one(".content-detail")
            if body_div:
                for div in body_div.select(".list_link_project, .box-embed, .table, .description"):
                    div.decompose()
                body_text = body_div.get_text(separator=" ", strip=True)
            else:
                body_text = ""
                
            return {
                "url": url,
                "title": title,
                "date": pub_date,
                "date_raw": date_raw,
                "body": body_text
            }
        except Exception:
            return None

    def run(self):
        """Starts the scraping process."""
        print(f"\n--- VnExpress Business Archive Scraper ({self.start_date.date()} to {self.end_date.date()}) ---")
        
        # Load existing IDs to avoid duplicates
        existing_ids = set()
        if os.path.exists(self.output_file):
            try:
                df_tmp = pd.read_csv(self.output_file)
                if "url" in df_tmp.columns:
                    existing_ids = {re.search(r"-(\d+)\.html", u).group(1) for u in df_tmp["url"] if re.search(r"-(\d+)\.html", u)}
            except Exception as e:
                print(f"Error loading existing IDs: {e}")

        for cat in VNEXPRESS_CATEGORIES:
            print(f"\n[*] Starting category: {cat}")
            page = 1
            reached_end = False
            
            while not reached_end:
                cat_url = f"{VNEXPRESS_BASE_URL}/{cat}-p{page}"
                print(f"    Scanning Page {page}...")
                
                try:
                    resp = requests.get(cat_url, headers=self.headers, timeout=15)
                    if resp.status_code != 200:
                        print(f"    [!] Failed to load page {page}. Stopping category.")
                        break
                        
                    soup = BeautifulSoup(resp.content, "html.parser")
                    articles = soup.select(".title-news a")
                    
                    if not articles:
                        print("    [!] No more articles found in this category.")
                        break
                    
                    page_records = []
                    for a in articles:
                        link = a.get("href", "")
                        if link.startswith("/"): link = VNEXPRESS_BASE_URL + link
                        
                        match = re.search(r"-(\d+)\.html", link)
                        if not match: continue
                        story_id = match.group(1)
                        
                        if story_id in existing_ids:
                            continue
                        
                        data = self.extract_article_details(link)
                        if not data: continue
                        
                        if data["date"] and data["date"] < self.start_date:
                            print(f"    [!] Reached date {data['date'].date()}. Stopping category.")
                            reached_end = True
                            break
                        
                        if data["date"] and data["date"] <= self.end_date:
                            page_records.append(data)
                            existing_ids.add(story_id)
                        
                        time.sleep(0.3)
                    
                    if page_records:
                        df = pd.DataFrame(page_records)
                        df.to_csv(self.output_file, mode='a', header=not os.path.exists(self.output_file), index=False)
                        print(f"    [+] Saved {len(page_records)} new articles.")
                    
                    page += 1
                    time.sleep(1.0)
                except Exception as e:
                    print(f"    ❌ Error on page {page}: {e}")
                    break

if __name__ == "__main__":
    scraper = VnExpressScraper()
    scraper.run()

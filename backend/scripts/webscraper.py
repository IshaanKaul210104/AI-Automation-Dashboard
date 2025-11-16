import requests, os, re, datetime, json, csv
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

def is_article_link(href):
    # Common patterns for article/blog/news URLs
    patterns = ["news", "article", "blog", "story", "post", "202", "20"]
    return any(p in href.lower() for p in patterns)

def extract_articles(soup, base_url):
    articles = []

    # 1️⃣ Prefer <article> tags
    for art in soup.find_all("article"):
        title_tag = art.find(["h1", "h2", "h3", "a"])
        if title_tag:
            title = clean_text(title_tag.get_text())
            href = title_tag.get("href")
            if href:
                full_url = urljoin(base_url, href)
                articles.append({"title": title, "link": full_url})

    # 2️⃣ Fallback: <a> tags with article-like hrefs
    if not articles:
        for a in soup.find_all("a", href=True):
            href = a["href"]
            title = clean_text(a.get_text())
            if is_article_link(href) and len(title.split()) > 3:
                full_url = urljoin(base_url, href)
                articles.append({"title": title, "link": full_url})

    # 3️⃣ Fallback: headings that look like titles
    if not articles:
        for h in soup.find_all(["h1", "h2", "h3", "h4"]):
            text = clean_text(h.get_text())
            if len(text.split()) > 3:
                articles.append({"title": text, "link": base_url})

    # Deduplicate
    seen = set()
    unique_articles = []
    for art in articles:
        if art["title"] not in seen:
            unique_articles.append(art)
            seen.add(art["title"])

    return unique_articles

def run(params=None, timestamp=None):
    url = params.get("url", "https://www.theverge.com") if params else "https://www.theverge.com"

    print(f"Scraping articles from {url}...")

    try:
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        res.raise_for_status()
    except Exception as e:
        return {"count": 0, "error": str(e)}

    soup = BeautifulSoup(res.text, "lxml")
    articles = extract_articles(soup, url)

    # Save output
    output_dir = os.path.join("outputs", "webscraper", timestamp or datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "articles.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)
        
    csv_path = os.path.join(output_dir, "articles.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["title", "link"])
        writer.writeheader()
        writer.writerows(articles)

    return {
            "count": len(articles),
            "output": output_dir.replace("\\\\", "/"),
            "csv_file": csv_path.replace("\\\\", "/")
        }
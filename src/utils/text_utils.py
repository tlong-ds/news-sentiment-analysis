from bs4 import BeautifulSoup


def clean_html(html_content):
    """Strips HTML tags and removes script/style elements."""
    if not html_content:
        return ""

    soup = BeautifulSoup(html_content, "html.parser")

    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()

    text = soup.get_text(separator=" ", strip=True)

    # Basic sanity check to avoid placeholders
    if "The text version of this document is not available" in text:
        return "[TEXT_UNAVAILABLE]"

    return text

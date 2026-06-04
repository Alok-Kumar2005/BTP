import re
import math
import string
from collections import Counter
from typing import Dict, Any


COMMON_TLDS = {
    "com", "net", "org", "edu", "gov", "io", "co", "us", "uk",
    "de", "fr", "ru", "cn", "br", "in", "au", "ca", "jp",
    "eu", "info", "biz", "mobi", "name", "pro", "pw", "cc", "tv",
}
 
SUSPICIOUS_TLDS = {"pw", "cc", "biz", "xyz", "top", "club", "online", "site", "icu"}

COMMON_WORDS = {
    "google", "facebook", "twitter", "amazon", "microsoft", "apple",
    "youtube", "instagram", "linkedin", "netflix", "reddit", "yahoo",
    "mail", "web", "news", "blog", "shop", "store", "cloud", "service",
    "login", "secure", "bank", "pay", "update", "support", "help",
    "download", "upload", "file", "image", "video", "media", "data",
    "admin", "api", "cdn", "static", "img", "js", "css",
}

def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    length = len(s)
    return -sum((c/length)*math.log2(c / length) for c in counts.values())

def _parse_domain(domain: str):
    domain = domain.lower().strip().rstrip(".")   # lower, remove white space, and remove dots from end
    parts = domain.split(".")
    if len(parts) >= 3:
        tld = parts[-1]
        sld = parts[-2]
        subdomains = parts[:-2]
    elif len(parts) == 2:
        tld = parts[-1]
        sld = parts[0]
        subdomains = []
    else:
        tld = ""
        sld = domain
        subdomains = []
    return subdomains, sld, tld

def _ngram(s: str, n: int)-> Dict[str, float]:
    if len(s) < n:
        return {}
    grams = [s[i: i+n] for i in range(len(s)-n+1)]
    total = len(grams)
    count = Counter(grams)
    return {g: c/total for g, c in count.items()}

def _ngram_entropy(s, n):
    freq = _ngram(s, n)
    if  not freq:
        return 0.0
    return -sum(v* math.log2(v) for v in freq.values() if v > 0)

def _consonant_vowel_ratio(s: str) -> float:
    vowels = set("aeiou")
    consonants = set(string.ascii_lowercase) - vowels
    v = sum(1 for c in s if c in vowels)
    c = sum(1 for c in s if c in consonants)
    return c / (v + 1e-9)

def _longest_constant(s: str)->int:
    vowels = set("aeiou")
    maxi = curr = 0
    for c in s.lower():
        if c.isalpha() and c not in vowels:
            curr = curr+1
            maxi = max(maxi, curr)
        else:
            curr = 0
    return maxi

def _has_meaningful_word(s: str) -> int:
    """1 if any common word appears as a substring, else 0."""
    s_lower = s.lower()
    return int(any(w in s_lower for w in COMMON_WORDS))


def extract_features(domain: str)-> Dict[str, Any]:
    domain = domain.lower().strip().rstrip(".")
    subdomains, sld, tld = _parse_domain(domain)
    full_name = domain

    labels = domain.split(".")
    subdomain_str = ".".join(subdomains) if subdomains else ""
    registrable = f"{sld}.{tld}" if tld else sld

    # structural
    f_domain_len = len(domain)
    f_sld_len = len(sld)
    f_subdomain_len = len(subdomain_str)
    f_label_count = len(labels)
    f_max_label_len = max(len(l) for l in labels) if labels else 0
    f_avg_label_len = sum(len(l) for l in labels) / len(labels) if labels else 0
    f_total_label_ratio = len(domain) / 253.0
    f_per_label_ratio = f_max_label_len / 63.0

    ## charracter ratio
    alpha = sum(1 for c in domain if c.isalpha())
    digits = sum(1 for c in domain if c.isdigit())
    upper = sum(1 for c in domain if c.isupper())
    hyphens = domain.count("-")
    total_chars = len(domain.replace(".", "")) ## not counting dots
    f_digit_ratio = digits / (total_chars + 1e-9)
    f_upper_ratio = upper / (total_chars + 1e-9)
    f_hyphen_ratio = hyphens / (total_chars + 1e-9)
    f_alpha_ratio = alpha / (total_chars + 1e-9)

    ## secondry level char ratio
    sld_digits = sum(1 for c in sld if c.isdigit())
    sld_alpha = sum(1 for c in sld if c.isalpha())
    f_sld_digit_ratio = sld_digits / (len(sld) + 1e-9)
    f_sld_alpha_ratio = sld_alpha / (len(sld) + 1e-9)

    ## entropy calculation
    f_entropy_full = _shannon_entropy(domain.replace(".", ""))  ## not calculting for dots
    f_entropy_sld = _shannon_entropy(sld)
    f_entropy_subdomain = _shannon_entropy(subdomain_str) if subdomain_str else 0.0

    f_bigram_entropy_sld = _ngram_entropy(sld, 2)
    f_trigram_entropy_sld = _ngram_entropy(sld, 3)
    f_bigram_entropy_full = _ngram_entropy(domain.replace(".", ""), 2)

    ## lingustic
    f_consonant_vowel_ratio = _consonant_vowel_ratio(sld)
    f_longest_consonant_run = _longest_constant(sld)
    f_has_meaningful_word = _has_meaningful_word(sld)
    f_unique_char_ratio = len(set(sld)) / (len(sld) + 1e-9)

    ## tld features
    f_is_common_tld = int(tld in COMMON_TLDS)
    f_is_suspicious_tld = int(tld in SUSPICIOUS_TLDS)
    f_tld_len = len(tld)

    ## subdomain features
    f_has_subdomain = int(len(subdomains) > 0)
    f_subdomain_count = len(subdomains)
 
    #3 numeric features
    f_has_digits = int(digits > 0)
    f_numeric_percentage = digits / (len(domain) + 1e-9)
    return {
        # Structural
        "domain_len": f_domain_len,
        "sld_len": f_sld_len,
        "subdomain_len": f_subdomain_len,
        "label_count": f_label_count,
        "max_label_len": f_max_label_len,
        "avg_label_len": f_avg_label_len,
        "total_label_ratio": f_total_label_ratio,
        "per_label_ratio": f_per_label_ratio,
        # Character ratios
        "digit_ratio": f_digit_ratio,
        "upper_ratio": f_upper_ratio,
        "hyphen_ratio": f_hyphen_ratio,
        "alpha_ratio": f_alpha_ratio,
        "sld_digit_ratio": f_sld_digit_ratio,
        "sld_alpha_ratio": f_sld_alpha_ratio,
        # Entropy
        "entropy_full": f_entropy_full,
        "entropy_sld": f_entropy_sld,
        "entropy_subdomain": f_entropy_subdomain,
        # N-gram entropy
        "bigram_entropy_sld": f_bigram_entropy_sld,
        "trigram_entropy_sld": f_trigram_entropy_sld,
        "bigram_entropy_full": f_bigram_entropy_full,
        # Linguistic
        "consonant_vowel_ratio": f_consonant_vowel_ratio,
        "longest_consonant_run": f_longest_consonant_run,
        "has_meaningful_word": f_has_meaningful_word,
        "unique_char_ratio": f_unique_char_ratio,
        # TLD
        "is_common_tld": f_is_common_tld,
        "is_suspicious_tld": f_is_suspicious_tld,
        "tld_len": f_tld_len,
        # Subdomain
        "has_subdomain": f_has_subdomain,
        "subdomain_count": f_subdomain_count,
        # Numeric
        "has_digits": f_has_digits,
        "numeric_percentage": f_numeric_percentage,
    }

def get_feature_values():
    dummy = extract_features("www.takshpatel.com")
    print(len(dummy))
    for key, value in dummy.items():
        print(key, value)

def get_feature_names():
    dummy = extract_features("www.takshpatel.com")
    return list(dummy.keys())



if __name__ == "__main__":
    # print(_longest_constant("sdfrtyhj"))
    print(get_feature_names())
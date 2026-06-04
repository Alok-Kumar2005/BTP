import pandas as pd
import time

class Layer1Filter:
    def __init__(self, csv_path: str = "data/dns_data.csv"):
        self.csv_path = csv_path
        self.legit_domains = set()
        self.dga_domains = set()
        self._load_data()

    def _load_data(self):
        start_time = time.time()
        try:
            df = pd.read_csv(self.csv_path, header=None, names=["label", "family", "domain"])
            df["domain"] = df["domain"].astype(str).str.strip().str.lower()
            self.legit_domains = set(df[df["label"] == "legit"]["domain"])
            self.dga_domains = set(df[df["label"] == "dga"]["domain"])
            
            elapsed = time.time() - start_time
            print(f"[Layer 1] Loaded {len(self.legit_domains)} legit and {len(self.dga_domains)} DGA domains in {elapsed:.2f} seconds.")
            
        except Exception as e:
            print(f"[Layer 1] Error loading CSV: {e}")

    def check_domain(self, domain: str) -> str:
        clean_domain = domain.strip().lower()
        
        if clean_domain in self.legit_domains:
            return "legit"
        elif clean_domain in self.dga_domains:
            return "dga"
        else:
            return "unknown"

if __name__ == "__main__":
    dns_filter = Layer1Filter("data/dns_data.csv")
    test_queries = [
        "mortiscontrastatim.com",   # Should be DGA
        "mzltrack.com",             # Should be Legit
        "unknown-domain-123.com",   # Should be Unknown
        "RJYUOSMHFNAEDLYG.EU"       # Should be DGA (Testing uppercase resilience)
    ]
    
    print("\n--- Processing Live Traffic ---")
    for q in test_queries:
        result = dns_filter.check_domain(q)
        print(f"Domain: {q:25} | Layer 1 Result: {result}")
import math
from features.extractor import extract_features

class AdvancedAnalytics:
    def __init__(self, threshold: float = 0.25):
        self.threshold = threshold

    def calculate_anomaly_index(self, domain: str, network_stats: dict = None) -> float:
        features = extract_features(domain)
        network_stats = network_stats or {}
        i_r = network_stats.get("high_request_frequency", 0.0)

        i_h = network_stats.get("high_subdomain_frequency", 0.0)
        raw_entropy = max(features["entropy_subdomain"], features["entropy_sld"])
        i_e = min(raw_entropy / 5.17, 1.0) 
        i_d = 0.0
        if features["has_meaningful_word"] == 0:
            i_d += 0.4
        if features["consonant_vowel_ratio"] > 3.0:
            i_d += 0.3
        ngram_score = min((features["bigram_entropy_full"] + features["trigram_entropy_sld"]) / 8.0, 1.0)
        i_d = min(i_d + (ngram_score * 0.3), 1.0)

        A = (i_r + i_h + i_e + i_d) / 4.0
        return A

    def assess_domain(self, domain: str, layer2_score: float) -> dict:
        anomaly_index = self.calculate_anomaly_index(domain)
        
        is_covert_channel = anomaly_index > self.threshold

        return {
            "domain": domain,
            "layer2_score": round(layer2_score, 3),
            "anomaly_index": round(anomaly_index, 3),
            "final_verdict": "Malicious (Covert Channel)" if is_covert_channel else "Legit (False Positive)",
            "is_malicious": is_covert_channel
        }
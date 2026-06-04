from pipeline.train import load_model, decision_scores
from pipeline.database import Layer1Filter 
from pipeline.layer3 import AdvancedAnalytics

def run_full_pipeline():
    layer1 = Layer1Filter("data/dns_data.csv")
    clf, scaler = load_model()
    layer3 = AdvancedAnalytics(threshold=0.25)
    test_domains = [
        "www.google.com", 
        "mortiscontrastatim.com", 
        "y2tyfol9hiuw5hzwe2hnusbzm1qz51545315830.nuid.imrworldwide.com",
        "xj3k9s8d7f6g5h4j3k2l1.com", 
        "update.cisc0.net" 
    ]
    
    print("\n" + "="*70)
    print(f"{'DOMAIN':<40} | {'FINAL VERDICT':<25}")
    print("="*70)
    
    for domain in test_domains:
        l1_result = layer1.check_domain(domain)
        
        if l1_result == "legit":
            print(f"{domain:<40} | Blocked? NO  (Layer 1 Whitelist)")
            continue
        elif l1_result == "dga":
            print(f"{domain:<40} | Blocked? YES (Layer 1 Blacklist)")
            continue
            
        score = decision_scores([domain], clf, scaler)[0]
        
        if score > 0:
            print(f"{domain:<40} | Blocked? NO  (Layer 2 Normal)")
            continue
        
        l3_result = layer3.assess_domain(domain, score)
        
        verdict = l3_result['final_verdict']
        print(f"{domain:<40} | Blocked? {'YES' if l3_result['is_malicious'] else 'NO '} (L3 Index: {l3_result['anomaly_index']})")

if __name__ == "__main__":
    run_full_pipeline()
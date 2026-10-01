from itertools import combinations

def get_support(itemset, transactions):
    """Calculates the support of an itemset across all transactions."""
    count = sum(1 for t in transactions if itemset.issubset(t))
    return count / len(transactions)

def get_frequent_1_itemsets(transactions, min_support):
    """Finds all frequent 1-itemsets (L1)."""
    item_counts = {}
    for t in transactions:
        for item in t:
            item_counts[item] = item_counts.get(item, 0) + 1
            
    num_transactions = len(transactions)
    l1 = {}
    for item, count in item_counts.items():
        support = count / num_transactions
        if support >= min_support:
            l1[frozenset([item])] = support
    return l1

def apriori_gen(prev_frequent_itemsets, k):
    """
    Generates candidate k-itemsets (Ck) from frequent (k-1)-itemsets (Lk-1)
    and prunes candidates containing any non-frequent (k-1)-subsets.
    """
    candidates = set()
    itemsets_list = list(prev_frequent_itemsets.keys())
    n = len(itemsets_list)
    
    for i in range(n):
        for j in range(i + 1, n):
            union_set = itemsets_list[i].union(itemsets_list[j])
            if len(union_set) == k:
                # Prune step: all (k-1) subsets must be frequent
                all_subsets_frequent = True
                for sub in combinations(union_set, k - 1):
                    if frozenset(sub) not in prev_frequent_itemsets:
                        all_subsets_frequent = False
                        break
                if all_subsets_frequent:
                    candidates.add(union_set)
    return candidates

def find_all_frequent_itemsets(transactions, min_support):
    """Finds all frequent itemsets of all sizes."""
    all_frequent = {}
    
    # Step 1: L1
    current_l = get_frequent_1_itemsets(transactions, min_support)
    k = 2
    
    while current_l:
        all_frequent.update(current_l)
        candidates = apriori_gen(current_l, k)
        
        # Count support for candidates
        current_l = {}
        for candidate in candidates:
            support = get_support(candidate, transactions)
            if support >= min_support:
                current_l[candidate] = support
        k += 1
        
    return all_frequent

def generate_association_rules(frequent_itemsets, min_confidence):
    """
    Generates association rules X -> Y from frequent itemsets:
    Confidence(X -> Y) = Support(X U Y) / Support(X)
    """
    rules = []
    
    for itemset, support_xy in frequent_itemsets.items():
        if len(itemset) < 2:
            continue
            
        # Generate non-empty proper subsets as antecedent X
        for r in range(1, len(itemset)):
            for antecedent in combinations(itemset, r):
                X = frozenset(antecedent)
                Y = itemset - X
                
                support_x = frequent_itemsets.get(X)
                if support_x is not None and support_x > 0:
                    confidence = support_xy / support_x
                    if confidence >= min_confidence:
                        rules.append({
                            'X': X,
                            'Y': Y,
                            'support': support_xy,
                            'confidence': confidence
                        })
    return rules

def main():
    # Example input values (replace or read via input() as needed)
    n = 6
    A = []
    for i in range(n):
        ts = A.append(list(input(f"Enter the {i+1} transaction: ").strip().split()))
    
    min_support = 0.3 
    min_confidence = 1

    # Convert transaction contents to sets
    transactions = [set(t) for t in A]

    # Run Apriori
    frequent_itemsets = find_all_frequent_itemsets(transactions, min_support)
    rules = generate_association_rules(frequent_itemsets, min_confidence)

    # Output Frequent Itemsets
    print("=" * 50)
    print(f"FREQUENT ITEMSETS (min_support = {min_support})")
    print("=" * 50)
    for itemset, supp in sorted(frequent_itemsets.items(), key=lambda x: (len(x[0]), -x[1])):
        items_str = ", ".join(sorted(itemset))
        print(f"{{{items_str}}} -> Support: {supp:.2f}")

    # Output Association Rules
    print("\n" + "=" * 50)
    print(f"ASSOCIATION RULES (min_confidence = {min_confidence})")
    print("=" * 50)
    for rule in sorted(rules, key=lambda r: (-r['confidence'], -r['support'])):
        x_str = "{" + ", ".join(sorted(rule['X'])) + "}"
        y_str = "{" + ", ".join(sorted(rule['Y'])) + "}"
        print(f"{x_str} => {y_str} | Support: {rule['support']:.2f} | Confidence: {rule['confidence']:.2f}")

if __name__ == "__main__":
    main()
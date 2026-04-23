from rapidfuzz import fuzz

def fuzzy_compare(text1, text2):
    if not text1 or not text2:
        return 0
    return fuzz.token_sort_ratio(text1, text2)
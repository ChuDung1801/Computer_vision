from text_utils import normalize_text, extract_volume
from matching import fuzzy_compare


# ===== 1. VERIFY LABEL =====
def verify_label(detected_labels, ground_truth_labels):
    detected_set = set(detected_labels)
    gt_set = set(ground_truth_labels)

    if detected_set == gt_set:
        return True, "All labels match"

    missing = gt_set - detected_set
    extra = detected_set - gt_set

    reason = ""
    if missing:
        reason += f"Missing: {list(missing)} "
    if extra:
        reason += f"Unexpected: {list(extra)}"

    return False, reason.strip()


# ===== 2. VERIFY TEXT (FUZZY) =====
def verify_text(ocr_text, ground_truth_text, threshold=80):
    ocr_text_norm = normalize_text(ocr_text)
    gt_text_norm = normalize_text(ground_truth_text)

    score = fuzzy_compare(ocr_text_norm, gt_text_norm)

    if score >= threshold:
        return True, f"Text match (score={score})"
    else:
        return False, f"Text mismatch (score={score})"


# ===== 3. VERIFY VOLUME =====
def verify_volume(ocr_text, expected_volume):
    extracted = extract_volume(ocr_text)

    if extracted is None:
        return False, "No volume found"

    if extracted.replace(" ", "") == expected_volume.replace(" ", ""):
        return True, "Volume match"
    else:
        return False, f"Expected {expected_volume}, got {extracted}"


# ===== 4. MAIN VERIFY =====
def verify_all(detected_labels, ground_truth, ocr_text=None):
    results = {}

    # label check
    label_ok, label_reason = verify_label(
        detected_labels,
        ground_truth["labels"]
    )
    results["label"] = (label_ok, label_reason)

    # volume check (optional)
    if "volume" in ground_truth and ocr_text:
        vol_ok, vol_reason = verify_volume(
            ocr_text,
            ground_truth["volume"]
        )
        results["volume"] = (vol_ok, vol_reason)

    # text check (optional)
    if "text" in ground_truth and ocr_text:
        text_ok, text_reason = verify_text(
            ocr_text,
            ground_truth["text"]
        )
        results["text"] = (text_ok, text_reason)

    # final decision
    final_ok = all(v[0] for v in results.values())

    return {
        "final_result": "OK" if final_ok else "FAIL",
        "details": results
    }
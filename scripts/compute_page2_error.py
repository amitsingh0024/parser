import re
import unicodedata

def normalize_text(text):
    text = unicodedata.normalize('NFC', text)
    # Remove markdown tags, formatting, and excess whitespace
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'#+\s*', '', text)
    text = re.sub(r'[\r\t]', ' ', text)
    text = re.sub(r' +', ' ', text)
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    return '\n'.join(lines)

def levenshtein_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]

def compute_cer(gt, pred):
    dist = levenshtein_distance(gt, pred)
    cer = dist / max(1, len(gt))
    return dist, cer

def compute_wer(gt, pred):
    w_gt = gt.split()
    w_pred = pred.split()
    dist = levenshtein_distance(w_gt, w_pred)
    wer = dist / max(1, len(w_gt))
    return dist, wer

gt_page2 = """1. किस शब्द में 'वि' उपसर्ग नहीं है ?
(1) विशाखा
(2) विश्वास
(3) विधान
(4) विद्या
2. किस विकल्प में 'उपसर्ग और प्रत्यय - दोनों का प्रयोग हुआ है ?
(1) विशिष्टता
(2) प्राथमिकता
(3) वैश्विकता
(4) नैतिकता
3. किस शब्द में एक से अधिक उपसर्ग हैं ?
(1) उपस्थिति
(2) अवतरण
(3) अभिभाषण
(4) व्याकरण
4. इनमें से कौन सा समास-विग्रह सही नहीं है ?
(1) धर्मविमुख = धर्म से विमुख
(2) आपबीती = आप पर बीती
(3) हवनसामग्री = हवन की सामग्री
(4) जलधारा = जल की धारा
5. कौन सा समास-विग्रह गलत है ?
(1) चौराहा = चार हैं जो रास्ते
(2) तपोधन = तप (तपस्या) ही धन है जिसका
(3) नीलोत्पल = नीला है जो उत्पल
(4) राष्ट्रभक्ति = राष्ट्र के लिए भक्ति
6. इनमें कौन सा सामासिक पद सही नहीं है ?
(1) युद्ध में स्थिर है जो = युधिष्ठिर
(2) जितना संभव हो = यथासाध्य
(3) जीवन पर्यन्त = आजीवन
(4) शीत और उष्ण = शीतोष्ण
7. कौन सा समास-विग्रह गलत है ?
(1) कालीमिर्च = काली है जो मिर्च
(2) सभामंडप = सभा के लिए मंडप
(3) मनमाना = मन से माना हुआ
(4) बलिपशु = पशु की बलि
8. इनमें कौन सा सामासिक पद सही नहीं है ?
(1) चार हैं आनन जिसके = चतुरानन
(2) लम्बा है उदर जिसका = लम्बोदर
(3) दश हैं मुख जिसके = दशानन
(4) तीन हैं नेत्र जिसके = त्रिनेत्र
9. कौन सा समास-विग्रह सही है ?
(1) घुड़दौड़ = घोड़ों के लिए दौड़
(2) चिड़ीमार = चिड़ियों को मारने वाला
(3) गंगास्नान = गंगा का स्नान
(4) रणनिमंत्रण = रण का निमंत्रण
01 (Gen. Hindi & GK)"""

with open("outputs/web_results/6e296c57/page_2/result.md", "r", encoding="utf-8") as f:
    pred_raw = f.read()

# Extract only the content text from result.md (removing <PAGE>, <BLOCK>...<FORK> prefixes)
clean_pred_lines = []
for line in pred_raw.splitlines():
    if "<FORK>" in line:
        clean_pred_lines.append(line.split("<FORK>")[-1].strip())
    elif not line.startswith("<"):
        clean_pred_lines.append(line.strip())
pred_clean = "\n".join([l for l in clean_pred_lines if l])

norm_gt = normalize_text(gt_page2)
norm_pred = normalize_text(pred_clean)

char_edits, cer = compute_cer(norm_gt, norm_pred)
word_edits, wer = compute_wer(norm_gt, norm_pred)
char_acc = max(0.0, 1.0 - cer) * 100
word_acc = max(0.0, 1.0 - wer) * 100

print(f"GT Length: {len(norm_gt)} chars, {len(norm_gt.split())} words")
print(f"Pred Length: {len(norm_pred)} chars, {len(norm_pred.split())} words")
print(f"Character Edits: {char_edits}")
print(f"Character Error Rate (CER): {cer * 100:.2f}%")
print(f"Character Accuracy: {char_acc:.2f}%")
print(f"Word Edits: {word_edits}")
print(f"Word Error Rate (WER): {wer * 100:.2f}%")
print(f"Word Accuracy: {word_acc:.2f}%")

# Let's inspect differences
import difflib
print("\n--- DIFF (First 40 lines) ---")
diff = list(difflib.unified_diff(norm_gt.splitlines(), norm_pred.splitlines(), lineterm=''))
for line in diff[:40]:
    print(line)
